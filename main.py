#!/usr/bin/env python
"""
EAGLE: Efficient Alignment of Generalized Latent Embeddings
Main script for training EAGLE models and running comparative analysis

python main.py --mode all --comprehensive-attribution
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple

# Survival analysis imports
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# EAGLE imports
from eagle import (
    UnifiedPipeline,
    UnifiedSurvivalModel,
    UnifiedSurvivalDataset,
    UnifiedClinicalProcessor,
    get_text_extractor,
    ModelConfig,
    GBM_CONFIG,
    IPMN_CONFIG,
    NSCLC_CONFIG,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BaselineModels:
    """Baseline survival models for comparison"""

    @staticmethod
    def run_rsf(X_train, y_train, X_test, y_test):
        """Random Survival Forest"""
        # Create structured array for sksurv
        y_train_struct = np.array(
            [(bool(e), t) for e, t in zip(y_train["event"], y_train["time"])],
            dtype=[("event", bool), ("time", float)],
        )
        y_test_struct = np.array(
            [(bool(e), t) for e, t in zip(y_test["event"], y_test["time"])],
            dtype=[("event", bool), ("time", float)],
        )

        # Train RSF
        rsf = RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=6,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        rsf.fit(X_train, y_train_struct)

        # Get risk scores using cumulative hazard
        median_time = np.median(y_test["time"])
        chf = rsf.predict_cumulative_hazard_function(X_test)
        risk_scores = np.array([chf_i(median_time) for chf_i in chf])

        # Calculate C-index
        c_index = concordance_index_censored(
            y_test_struct["event"], y_test_struct["time"], risk_scores
        )[0]

        return c_index, risk_scores

    @staticmethod
    def run_coxph(X_train, y_train, X_test, y_test):
        """Cox Proportional Hazards"""
        # Create structured array
        y_train_struct = np.array(
            [(bool(e), t) for e, t in zip(y_train["event"], y_train["time"])],
            dtype=[("event", bool), ("time", float)],
        )
        y_test_struct = np.array(
            [(bool(e), t) for e, t in zip(y_test["event"], y_test["time"])],
            dtype=[("event", bool), ("time", float)],
        )

        # Apply PCA if needed
        if X_train.shape[1] > 30:
            pca = PCA(n_components=30, random_state=42)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
        else:
            X_train_pca = X_train
            X_test_pca = X_test

        # Train Cox model
        try:
            cox = CoxPHSurvivalAnalysis(alpha=0.1, n_iter=100)
            cox.fit(X_train_pca, y_train_struct)
            risk_scores = cox.predict(X_test_pca)

            c_index = concordance_index_censored(
                y_test_struct["event"], y_test_struct["time"], risk_scores
            )[0]

            return c_index, risk_scores
        except Exception as e:
            logging.warning(f"CoxPH failed: {e}")
            return np.nan, np.zeros(len(X_test))

    @staticmethod
    def run_deepsurv(X_train, y_train, X_test, y_test):
        """Simple DeepSurv implementation"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Apply PCA if too many features (similar to CoxPH)
        if X_train.shape[1] > 100:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=min(100, X_train.shape[0] // 2), random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)

        # Simple neural network
        class DeepSurvNet(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(32, 16),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(16, 1),
                )

            def forward(self, x):
                return self.net(x)

        model = DeepSurvNet(X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train
        model.train()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(100):  # Increased epochs
            optimizer.zero_grad()
            risk_pred = model(X_train_t).squeeze()

            # Simple Cox loss
            sorted_idx = torch.argsort(torch.tensor(y_train["time"]))
            sorted_risk = risk_pred[sorted_idx]
            sorted_event = torch.tensor(
                y_train["event"][sorted_idx], dtype=torch.float32
            ).to(device)

            # Clip predictions to prevent overflow
            sorted_risk = torch.clamp(sorted_risk, min=-10, max=10)

            exp_risk = torch.exp(sorted_risk)
            risk_sum = torch.cumsum(exp_risk.flip(0), 0).flip(0)
            loss = -torch.mean(
                sorted_event * (sorted_risk - torch.log(risk_sum + 1e-7))
            )

            # Check for NaN
            if torch.isnan(loss):
                logging.warning("NaN loss encountered in DeepSurv")
                return np.nan, np.zeros(len(X_test))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 10:
                    break

        # Evaluate
        model.eval()
        with torch.no_grad():
            risk_scores = model(X_test_t).squeeze().cpu().numpy()

            # Check for NaN in predictions
            if np.any(np.isnan(risk_scores)):
                logging.warning("NaN predictions in DeepSurv")
                return np.nan, np.zeros(len(X_test))

        try:
            c_index = concordance_index_censored(
                y_test["event"].astype(bool), y_test["time"], risk_scores
            )[0]
        except Exception as e:
            logging.warning(f"DeepSurv C-index calculation failed: {e}")
            return np.nan, risk_scores

        return c_index, risk_scores


def load_embeddings_data(
    data_path: str, dataset_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings from parquet file"""
    df = pd.read_parquet(data_path)

    # Extract embeddings based on file type
    if "eagle_embeddings" in df.columns:
        # EAGLE embeddings
        embeddings = np.array(df["eagle_embeddings"].tolist())
    elif "medgemma" in data_path:
        # MedGemma files have dataset-specific structures
        if dataset_name == "GBM":
            # Use combined embeddings for GBM MedGemma
            if "combined_embeddings" in df.columns:
                embeddings = []
                for emb in df["combined_embeddings"]:
                    if isinstance(emb, bytes):
                        embeddings.append(np.frombuffer(emb, dtype=np.float32))
                    else:
                        embeddings.append(np.array(emb))

                # Pad to max size to handle variable shapes
                max_size = max(emb.shape[0] for emb in embeddings)
                padded_embeddings = []
                for emb in embeddings:
                    if emb.shape[0] < max_size:
                        padded = np.pad(
                            emb, (0, max_size - emb.shape[0]), mode="constant"
                        )
                        padded_embeddings.append(padded)
                    else:
                        padded_embeddings.append(emb)
                embeddings = np.array(padded_embeddings)

        elif dataset_name == "IPMN":
            # Use fused embeddings for IPMN MedGemma
            if "multimodal_ct_fused_embeddings" in df.columns:
                embeddings = []
                for emb in df["multimodal_ct_fused_embeddings"]:
                    if isinstance(emb, bytes):
                        embeddings.append(np.frombuffer(emb, dtype=np.float32))
                    else:
                        embeddings.append(np.array(emb))

                # Pad to max size
                max_size = max(emb.shape[0] for emb in embeddings)
                padded_embeddings = []
                for emb in embeddings:
                    if emb.shape[0] < max_size:
                        padded = np.pad(
                            emb, (0, max_size - emb.shape[0]), mode="constant"
                        )
                        padded_embeddings.append(padded)
                    else:
                        padded_embeddings.append(emb)
                embeddings = np.array(padded_embeddings)

        else:  # NSCLC
            # For NSCLC, use contrast embeddings as primary
            if "multimodal_contrast_embeddings" in df.columns:
                embeddings = []
                for emb in df["multimodal_contrast_embeddings"]:
                    if pd.notna(emb):
                        if isinstance(emb, bytes):
                            embeddings.append(np.frombuffer(emb, dtype=np.float32))
                        else:
                            embeddings.append(np.array(emb))
                    else:
                        embeddings.append(None)

                # Filter out None values and get valid embeddings
                valid_embeddings = [e for e in embeddings if e is not None]
                valid_indices = [i for i, e in enumerate(embeddings) if e is not None]

                if valid_embeddings:
                    # Pad to max size
                    max_size = max(emb.shape[0] for emb in valid_embeddings)
                    padded_embeddings = []
                    for emb in valid_embeddings:
                        if emb.shape[0] < max_size:
                            padded = np.pad(
                                emb, (0, max_size - emb.shape[0]), mode="constant"
                            )
                            padded_embeddings.append(padded)
                        else:
                            padded_embeddings.append(emb)
                    embeddings = np.array(padded_embeddings)

                    # Return filtered dataframe indices for survival data
                    df = df.iloc[valid_indices]
                else:
                    raise ValueError("No valid embeddings found")

    else:
        # Unimodal files - use primary modality only for baseline comparison
        primary_col = None

        if dataset_name == "GBM":
            primary_col = "mri_embeddings"
        elif dataset_name == "IPMN":
            primary_col = "ct_embeddings"
        else:  # NSCLC
            primary_col = "ct_contrast_embeddings"

        # Process primary embeddings
        if primary_col and primary_col in df.columns:
            embeddings = []
            valid_indices = []

            for idx, emb in enumerate(df[primary_col]):
                if pd.notna(emb):
                    if isinstance(emb, bytes):
                        embeddings.append(np.frombuffer(emb, dtype=np.float32))
                    else:
                        embeddings.append(np.array(emb))
                    valid_indices.append(idx)

            if embeddings:
                # Pad to max size to handle variable shapes
                max_size = max(emb.shape[0] for emb in embeddings)
                padded_embeddings = []
                for emb in embeddings:
                    if emb.shape[0] < max_size:
                        padded = np.pad(
                            emb, (0, max_size - emb.shape[0]), mode="constant"
                        )
                        padded_embeddings.append(padded)
                    else:
                        padded_embeddings.append(emb)
                embeddings = np.array(padded_embeddings)

                # Filter dataframe to match valid embeddings
                df = df.iloc[valid_indices]
            else:
                raise ValueError(f"No valid embeddings found in {primary_col}")
        else:
            raise ValueError(f"Primary embedding column {primary_col} not found")

    # Extract survival data based on dataset (using potentially filtered df)
    if dataset_name == "GBM":
        y_time = df["survival_time_in_months"].values
        y_event = (df["vital_status_desc"] == "DEAD").astype(int).values
    elif dataset_name == "IPMN":
        y_time = df["survival_time_days"].values / 30.44  # Convert to months
        y_event = (df["vital_status"] == "DEAD").astype(int).values
    else:  # NSCLC
        y_time = df["SURVIVAL_TIME_IN_MONTHS"].values
        y_event = df["event"].values

    # Filter out any NaN values in survival data
    valid_mask = ~(np.isnan(y_time) | np.isnan(y_event))
    if not np.all(valid_mask):
        logging.warning(
            f"Filtering out {np.sum(~valid_mask)} samples with NaN survival data"
        )
        embeddings = embeddings[valid_mask]
        y_time = y_time[valid_mask]
        y_event = y_event[valid_mask]

    # Ensure positive survival times
    if np.any(y_time <= 0):
        logging.warning(
            f"Found {np.sum(y_time <= 0)} non-positive survival times, setting to 0.1"
        )
        y_time[y_time <= 0] = 0.1

    return embeddings, y_time, y_event


def generate_baseline_visualizations(results_df: pd.DataFrame):
    """Generate comprehensive visualizations for baseline model comparison"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create baseline figures directory
    baseline_dir = Path("results/baseline_figures")
    baseline_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    # 1. Grouped bar plot comparing all models across embeddings
    plt.figure(figsize=(14, 8))

    # Prepare data for plotting
    plot_df = results_df.pivot(
        index=["Dataset", "Model"], columns="Embedding", values="Mean_C_Index"
    )
    plot_df = plot_df.reset_index()

    # Create grouped bar plot
    x = np.arange(len(results_df["Dataset"].unique()))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, model in enumerate(["RSF", "CoxPH", "DeepSurv"]):
        ax = axes[idx]
        model_data = results_df[results_df["Model"] == model]

        for i, embedding in enumerate(["Unimodal", "MedGemma", "EAGLE"]):
            emb_data = model_data[model_data["Embedding"] == embedding]
            values = []
            errors = []

            for dataset in ["GBM", "IPMN", "NSCLC"]:
                dataset_data = emb_data[emb_data["Dataset"] == dataset]
                if not dataset_data.empty:
                    values.append(dataset_data["Mean_C_Index"].values[0])
                    errors.append(dataset_data["Std_C_Index"].values[0])
                else:
                    values.append(0)
                    errors.append(0)

            bars = ax.bar(
                x + i * width - width,
                values,
                width,
                label=embedding,
                yerr=errors,
                capsize=5,
                alpha=0.8,
            )

            # Add value labels on bars
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        ax.set_xlabel("Dataset", fontsize=12)
        if idx == 0:
            ax.set_ylabel("C-Index", fontsize=12)
        ax.set_title(f"{model} Performance", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["GBM", "IPMN", "NSCLC"])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Baseline Model Performance Across Embeddings", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(baseline_dir / "baseline_comparison_by_model.png", bbox_inches="tight")
    plt.close()

    # 2. Heatmap of performance
    plt.figure(figsize=(10, 8))

    # Create pivot table for heatmap
    heatmap_data = results_df.pivot_table(
        index=["Dataset", "Model"], columns="Embedding", values="Mean_C_Index"
    )

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.6,
        vmin=0.4,
        vmax=0.8,
        cbar_kws={"label": "C-Index"},
    )

    plt.title("Model Performance Heatmap", fontsize=16, fontweight="bold")
    plt.xlabel("Embedding Type", fontsize=12)
    plt.ylabel("Dataset / Model", fontsize=12)
    plt.tight_layout()
    plt.savefig(baseline_dir / "baseline_performance_heatmap.png", bbox_inches="tight")
    plt.close()

    # 3. Box plot showing performance distribution by embedding type
    plt.figure(figsize=(12, 8))

    # Create box plot
    ax = sns.boxplot(
        data=results_df, x="Embedding", y="Mean_C_Index", hue="Model", palette="Set2"
    )

    # Add strip plot for individual points
    sns.stripplot(
        data=results_df,
        x="Embedding",
        y="Mean_C_Index",
        hue="Model",
        dodge=True,
        palette="Set2",
        alpha=0.6,
        size=8,
        ax=ax,
    )

    plt.xlabel("Embedding Type", fontsize=12)
    plt.ylabel("C-Index", fontsize=12)
    plt.title(
        "Performance Distribution by Embedding Type", fontsize=16, fontweight="bold"
    )
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        baseline_dir / "baseline_performance_distribution.png", bbox_inches="tight"
    )
    plt.close()

    # 4. Improvement plot showing relative gains
    plt.figure(figsize=(12, 8))

    # Calculate improvements relative to Unimodal baseline
    improvement_data = []
    for dataset in results_df["Dataset"].unique():
        for model in results_df["Model"].unique():
            baseline_perf = results_df[
                (results_df["Dataset"] == dataset)
                & (results_df["Model"] == model)
                & (results_df["Embedding"] == "Unimodal")
            ]["Mean_C_Index"].values

            if len(baseline_perf) > 0:
                baseline = baseline_perf[0]

                for embedding in ["MedGemma", "EAGLE"]:
                    emb_perf = results_df[
                        (results_df["Dataset"] == dataset)
                        & (results_df["Model"] == model)
                        & (results_df["Embedding"] == embedding)
                    ]["Mean_C_Index"].values

                    if len(emb_perf) > 0:
                        improvement = ((emb_perf[0] - baseline) / baseline) * 100
                        improvement_data.append(
                            {
                                "Dataset": dataset,
                                "Model": model,
                                "Embedding": embedding,
                                "Improvement": improvement,
                            }
                        )

    improvement_df = pd.DataFrame(improvement_data)

    # Create grouped bar plot for improvements
    fig, ax = plt.subplots(figsize=(12, 8))

    improvement_pivot = improvement_df.pivot(
        index=["Dataset", "Model"], columns="Embedding", values="Improvement"
    )

    improvement_pivot.plot(kind="bar", ax=ax, width=0.8)

    plt.xlabel("Dataset / Model", fontsize=12)
    plt.ylabel("Improvement over Unimodal (%)", fontsize=12)
    plt.title(
        "Relative Performance Improvement over Unimodal Baseline",
        fontsize=16,
        fontweight="bold",
    )
    plt.legend(title="Embedding")
    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(baseline_dir / "baseline_improvement_analysis.png", bbox_inches="tight")
    plt.close()

    # 5. Dataset-specific performance plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, dataset in enumerate(["GBM", "IPMN", "NSCLC"]):
        ax = axes[idx]
        dataset_data = results_df[results_df["Dataset"] == dataset]

        # Create grouped bar plot
        pivot_data = dataset_data.pivot(
            columns="Model", index="Embedding", values="Mean_C_Index"
        )
        pivot_data.plot(kind="bar", ax=ax, width=0.8)

        ax.set_xlabel("Embedding Type", fontsize=12)
        ax.set_ylabel("C-Index", fontsize=12)
        ax.set_title(f"{dataset} Performance", fontsize=14, fontweight="bold")
        ax.legend(title="Model")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3)

    plt.suptitle("Dataset-Specific Model Performance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        baseline_dir / "baseline_dataset_specific_performance.png", bbox_inches="tight"
    )
    plt.close()

    # 6. Summary statistics table as figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    # Create summary table
    summary_data = []
    for embedding in ["Unimodal", "MedGemma", "EAGLE"]:
        emb_data = results_df[results_df["Embedding"] == embedding]
        if not emb_data.empty:
            summary_data.append(
                [
                    embedding,
                    f"{emb_data['Mean_C_Index'].mean():.3f} ± {emb_data['Mean_C_Index'].std():.3f}",
                    f"{emb_data['Mean_C_Index'].max():.3f}",
                    f"{emb_data['Mean_C_Index'].min():.3f}",
                    f"{emb_data.loc[emb_data['Mean_C_Index'].idxmax(), 'Dataset']} / {emb_data.loc[emb_data['Mean_C_Index'].idxmax(), 'Model']}",
                ]
            )

    table = ax.table(
        cellText=summary_data,
        colLabels=[
            "Embedding",
            "Mean C-Index",
            "Max C-Index",
            "Min C-Index",
            "Best Performing",
        ],
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.2, 0.15, 0.15, 0.35],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.title(
        "Baseline Model Performance Summary", fontsize=16, fontweight="bold", pad=20
    )
    plt.savefig(baseline_dir / "baseline_summary_table.png", bbox_inches="tight")
    plt.close()

    logging.info(f"Baseline visualizations saved to {baseline_dir}")


def generate_combined_comparison_plots(all_results_df: pd.DataFrame):
    """Generate comparison plots between EAGLE and baseline models"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create combined figures directory
    combined_dir = Path("results/combined_figures")
    combined_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    # 1. Overall performance comparison - EAGLE vs best baseline
    plt.figure(figsize=(14, 8))

    # Find best baseline for each dataset
    best_results = []
    for dataset in ["GBM", "IPMN", "NSCLC"]:
        # Get EAGLE performance
        eagle_data = all_results_df[
            (all_results_df["Dataset"] == dataset)
            & (all_results_df["Model"] == "EAGLE")
        ]
        if not eagle_data.empty:
            best_results.append(
                {
                    "Dataset": dataset,
                    "Model": "EAGLE",
                    "C-Index": eagle_data["Mean_C_Index"].values[0],
                    "Std": eagle_data["Std_C_Index"].values[0],
                    "Type": "EAGLE",
                }
            )

        # Get best baseline (using EAGLE embeddings)
        baseline_data = all_results_df[
            (all_results_df["Dataset"] == dataset)
            & (all_results_df["Model"] != "EAGLE")
            & (all_results_df["Embedding"] == "EAGLE")
        ]
        if not baseline_data.empty:
            best_baseline_idx = baseline_data["Mean_C_Index"].idxmax()
            best_baseline = baseline_data.loc[best_baseline_idx]
            best_results.append(
                {
                    "Dataset": dataset,
                    "Model": f"{best_baseline['Model']} (EAGLE emb)",
                    "C-Index": best_baseline["Mean_C_Index"],
                    "Std": best_baseline["Std_C_Index"],
                    "Type": "Best Baseline",
                }
            )

    best_df = pd.DataFrame(best_results)

    # Create grouped bar plot
    x = np.arange(len(["GBM", "IPMN", "NSCLC"]))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))

    eagle_data = best_df[best_df["Type"] == "EAGLE"]
    baseline_data = best_df[best_df["Type"] == "Best Baseline"]

    bars1 = ax.bar(
        x - width / 2,
        eagle_data["C-Index"],
        width,
        yerr=eagle_data["Std"],
        label="EAGLE Model",
        color="#2E7D32",
        capsize=5,
    )
    bars2 = ax.bar(
        x + width / 2,
        baseline_data["C-Index"],
        width,
        yerr=baseline_data["Std"],
        label="Best Baseline (EAGLE embeddings)",
        color="#1976D2",
        capsize=5,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_ylabel("C-Index", fontsize=14)
    ax.set_title(
        "EAGLE vs Best Baseline Model Performance", fontsize=16, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["GBM", "IPMN", "NSCLC"])
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(combined_dir / "eagle_vs_best_baseline.png", bbox_inches="tight")
    plt.close()

    # 2. Comprehensive model comparison across all embeddings
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for idx, dataset in enumerate(["GBM", "IPMN", "NSCLC"]):
        ax = axes[idx]

        # Get data for this dataset
        dataset_data = all_results_df[all_results_df["Dataset"] == dataset].copy()

        # Separate EAGLE from baselines
        eagle_perf = (
            dataset_data[dataset_data["Model"] == "EAGLE"]["Mean_C_Index"].values[0]
            if not dataset_data[dataset_data["Model"] == "EAGLE"].empty
            else 0
        )

        # Get baseline performances
        baseline_data = dataset_data[dataset_data["Model"] != "EAGLE"]

        # Create plot data
        plot_data = []
        models = ["RSF", "CoxPH", "DeepSurv"]
        embeddings = ["Unimodal", "MedGemma", "EAGLE"]

        for model in models:
            for embedding in embeddings:
                perf_data = baseline_data[
                    (baseline_data["Model"] == model)
                    & (baseline_data["Embedding"] == embedding)
                ]
                if not perf_data.empty:
                    plot_data.append(
                        {
                            "Model": f"{model}\n({embedding})",
                            "C-Index": perf_data["Mean_C_Index"].values[0],
                            "Std": perf_data["Std_C_Index"].values[0],
                        }
                    )

        # Add EAGLE
        if eagle_perf > 0:
            plot_data.append(
                {
                    "Model": "EAGLE\n(Full Model)",
                    "C-Index": eagle_perf,
                    "Std": dataset_data[dataset_data["Model"] == "EAGLE"][
                        "Std_C_Index"
                    ].values[0],
                }
            )

        plot_df = pd.DataFrame(plot_data)

        # Create bar plot
        x_pos = np.arange(len(plot_df))
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_df)))

        bars = ax.bar(
            x_pos,
            plot_df["C-Index"],
            yerr=plot_df["Std"],
            capsize=5,
            color=colors,
            alpha=0.8,
        )

        # Highlight EAGLE
        if "EAGLE\n(Full Model)" in plot_df["Model"].values:
            eagle_idx = plot_df[plot_df["Model"] == "EAGLE\n(Full Model)"].index[0]
            bars[eagle_idx].set_color("#2E7D32")
            bars[eagle_idx].set_alpha(1.0)

        # Add value labels
        for bar, val in zip(bars, plot_df["C-Index"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xlabel("Model (Embedding)", fontsize=12)
        ax.set_ylabel("C-Index", fontsize=12)
        ax.set_title(f"{dataset}", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_df["Model"], rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Comprehensive Model Performance Comparison", fontsize=18, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        combined_dir / "comprehensive_model_comparison.png", bbox_inches="tight"
    )
    plt.close()

    # 3. Performance improvement visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    improvement_data = []
    for dataset in ["GBM", "IPMN", "NSCLC"]:
        # Get baseline unimodal performance (best among RSF, CoxPH, DeepSurv)
        baseline_unimodal = all_results_df[
            (all_results_df["Dataset"] == dataset)
            & (all_results_df["Model"] != "EAGLE")
            & (all_results_df["Embedding"] == "Unimodal")
        ]["Mean_C_Index"].max()

        # Get EAGLE performance
        eagle_perf = all_results_df[
            (all_results_df["Dataset"] == dataset)
            & (all_results_df["Model"] == "EAGLE")
        ]["Mean_C_Index"].values

        if len(eagle_perf) > 0 and baseline_unimodal > 0:
            improvement = (
                (eagle_perf[0] - baseline_unimodal) / baseline_unimodal
            ) * 100
            improvement_data.append(
                {
                    "Dataset": dataset,
                    "Improvement": improvement,
                    "Baseline": baseline_unimodal,
                    "EAGLE": eagle_perf[0],
                }
            )

    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)

        bars = ax.bar(
            imp_df["Dataset"],
            imp_df["Improvement"],
            color=["#2E7D32" if x > 0 else "#D32F2F" for x in imp_df["Improvement"]],
            alpha=0.8,
        )

        # Add value labels
        for bar, val, base, eagle in zip(
            bars, imp_df["Improvement"], imp_df["Baseline"], imp_df["EAGLE"]
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                0,
                f"({base:.3f} → {eagle:.3f})",
                ha="center",
                va="top",
                fontsize=10,
            )

        ax.set_xlabel("Dataset", fontsize=14)
        ax.set_ylabel("Improvement over Best Unimodal Baseline (%)", fontsize=14)
        ax.set_title("EAGLE Performance Improvement", fontsize=16, fontweight="bold")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(combined_dir / "eagle_performance_improvement.png", bbox_inches="tight")
    plt.close()

    # 4. Summary table
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("tight")
    ax.axis("off")

    # Create comprehensive summary
    summary_data = []
    for dataset in ["GBM", "IPMN", "NSCLC"]:
        # EAGLE performance
        eagle_data = all_results_df[
            (all_results_df["Dataset"] == dataset)
            & (all_results_df["Model"] == "EAGLE")
        ]
        if not eagle_data.empty:
            eagle_cindex = eagle_data["Mean_C_Index"].values[0]

            # Best baseline with EAGLE embeddings
            best_baseline_eagle = all_results_df[
                (all_results_df["Dataset"] == dataset)
                & (all_results_df["Model"] != "EAGLE")
                & (all_results_df["Embedding"] == "EAGLE")
            ]
            if not best_baseline_eagle.empty:
                best_idx = best_baseline_eagle["Mean_C_Index"].idxmax()
                best_model = best_baseline_eagle.loc[best_idx, "Model"]
                best_cindex = best_baseline_eagle.loc[best_idx, "Mean_C_Index"]
            else:
                best_model = "N/A"
                best_cindex = 0

            # Best unimodal baseline
            best_unimodal = all_results_df[
                (all_results_df["Dataset"] == dataset)
                & (all_results_df["Model"] != "EAGLE")
                & (all_results_df["Embedding"] == "Unimodal")
            ]
            if not best_unimodal.empty:
                unimodal_cindex = best_unimodal["Mean_C_Index"].max()
            else:
                unimodal_cindex = 0

            summary_data.append(
                [
                    dataset,
                    f"{eagle_cindex:.3f}",
                    f"{best_model}: {best_cindex:.3f}",
                    f"{unimodal_cindex:.3f}",
                    f"{((eagle_cindex - unimodal_cindex) / unimodal_cindex * 100):.1f}%"
                    if unimodal_cindex > 0
                    else "N/A",
                ]
            )

    table = ax.table(
        cellText=summary_data,
        colLabels=[
            "Dataset",
            "EAGLE C-Index",
            "Best Baseline (EAGLE emb)",
            "Best Unimodal",
            "Improvement",
        ],
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.15, 0.25, 0.15, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor("#2E7D32")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style cells
    for i in range(1, len(summary_data) + 1):
        table[(i, 0)].set_facecolor("#E8F5E9")
        table[(i, 0)].set_text_props(weight="bold")

    plt.title(
        "EAGLE vs Baseline Models Summary", fontsize=18, fontweight="bold", pad=20
    )
    plt.savefig(combined_dir / "model_comparison_summary.png", bbox_inches="tight")
    plt.close()

    logging.info(f"Combined comparison visualizations saved to {combined_dir}")


def run_baseline_comparison(args):
    """Run baseline model comparison across different embeddings"""
    logging.info("Running baseline model comparison...")

    results = []
    datasets = ["GBM", "IPMN", "NSCLC"]
    embedding_types = {"unimodal": "Unimodal", "medgemma": "MedGemma", "eagle": "EAGLE"}
    models = ["RSF", "CoxPH", "DeepSurv"]

    for dataset in datasets:
        logging.info(f"\nProcessing {dataset}...")

        for emb_file, emb_name in embedding_types.items():
            data_path = f"data/{dataset}/{emb_file}.parquet"

            # Skip EAGLE embeddings if they don't exist yet
            if emb_file == "eagle" and not Path(data_path).exists():
                continue

            logging.info(f"  Loading {emb_name} embeddings...")
            try:
                X, y_time, y_event = load_embeddings_data(data_path, dataset)
            except Exception as e:
                logging.warning(f"  Failed to load {data_path}: {e}")
                continue

            # Prepare data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Cross-validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for model_name in models:
                cv_scores = []

                for fold, (train_idx, test_idx) in enumerate(
                    skf.split(X_scaled, y_event)
                ):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train = {"time": y_time[train_idx], "event": y_event[train_idx]}
                    y_test = {"time": y_time[test_idx], "event": y_event[test_idx]}

                    # Run model
                    if model_name == "RSF":
                        c_index, _ = BaselineModels.run_rsf(
                            X_train, y_train, X_test, y_test
                        )
                    elif model_name == "CoxPH":
                        c_index, _ = BaselineModels.run_coxph(
                            X_train, y_train, X_test, y_test
                        )
                    else:  # DeepSurv
                        c_index, _ = BaselineModels.run_deepsurv(
                            X_train, y_train, X_test, y_test
                        )

                    if not np.isnan(c_index):
                        cv_scores.append(c_index)

                if cv_scores:
                    result = {
                        "Dataset": dataset,
                        "Embedding": emb_name,
                        "Model": model_name,
                        "Mean_C_Index": np.mean(cv_scores),
                        "Std_C_Index": np.std(cv_scores),
                        "N_Features": X.shape[1],
                    }
                    results.append(result)
                    logging.info(
                        f"    {model_name}: {result['Mean_C_Index']:.4f} ± {result['Std_C_Index']:.4f}"
                    )

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/baseline_comparison.csv", index=False)

    # Generate baseline comparison visualizations
    generate_baseline_visualizations(results_df)

    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON")
    print("=" * 80)

    pivot_table = results_df.pivot_table(
        index=["Dataset", "Model"], columns="Embedding", values="Mean_C_Index"
    ).round(4)

    print(pivot_table)

    return results_df


def train_eagle_models(args):
    """Train EAGLE models on all datasets"""
    logging.info("Training EAGLE models...")

    datasets = ["GBM", "IPMN", "NSCLC"]
    configs = {"GBM": GBM_CONFIG, "IPMN": IPMN_CONFIG, "NSCLC": NSCLC_CONFIG}

    results = []

    for dataset in datasets:
        logging.info(f"\nTraining EAGLE on {dataset}...")

        # Create output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(f"results/{dataset}/{timestamp}")
        output_dirs = {
            "base": output_dir,
            "models": output_dir / "models",
            "results": output_dir / "results",
            "figures": output_dir / "figures",
            "attribution": output_dir / "attribution",
            "logs": output_dir / "logs",
        }

        for dir_path in output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Get configuration
        config = configs[dataset]
        model_config = ModelConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

        # Train model
        pipeline = UnifiedPipeline(config, model_config, output_dirs=output_dirs)

        # Enable comprehensive attribution if requested
        if args.comprehensive_attribution:
            # Run with comprehensive attribution
            eagle_results, risk_df, stats = pipeline.run(
                n_folds=5,
                n_risk_groups=3,
                enable_attribution=True,
                enable_comprehensive_attribution=True,
            )
        else:
            # Run with simple attribution only
            eagle_results, risk_df, stats = pipeline.run(
                n_folds=5, n_risk_groups=3, enable_attribution=args.analyze_attribution
            )

        # Store results
        results.append(
            {
                "Dataset": dataset,
                "Embedding": "EAGLE Model",
                "Model": "EAGLE",
                "Mean_C_Index": eagle_results["mean_cindex"],
                "Std_C_Index": eagle_results["std_cindex"],
                "N_Features": "Multi-modal",
            }
        )

        logging.info(
            f"  EAGLE C-index: {eagle_results['mean_cindex']:.4f} ± {eagle_results['std_cindex']:.4f}"
        )

        # Generate visualizations
        logging.info(f"  Generating visualizations for {dataset}...")
        from eagle.viz import (
            plot_km_curves,
            create_comprehensive_plots,
            plot_dataset_specific,
        )

        # Save risk dataframe
        risk_df.to_csv(output_dirs["results"] / "risk_scores.csv", index=False)

        # Generate Kaplan-Meier curves
        km_path = output_dirs["figures"] / "kaplan_meier_curves.png"
        plot_km_curves(
            risk_df,
            title=f"{dataset} Risk-Stratified Survival Curves",
            save_path=str(km_path),
        )

        # Generate comprehensive plots
        create_comprehensive_plots(risk_df, output_dir=str(output_dirs["figures"]))

        # Generate dataset-specific plots
        plot_dataset_specific(risk_df, dataset, output_dir=str(output_dirs["figures"]))

        # Generate attribution plots if enabled
        if (
            args.analyze_attribution or args.comprehensive_attribution
        ) and "imaging_contribution" in risk_df.columns:
            logging.info(f"  Generating attribution visualizations for {dataset}...")
            from eagle.attribution import (
                plot_modality_contributions,
                plot_patient_level_attribution,
                plot_comprehensive_attribution_comparison,
                create_comprehensive_attribution_report,
            )

            # Plot modality contributions
            attr_path = output_dirs["attribution"] / "modality_contributions.pdf"
            plot_modality_contributions(
                risk_df, save_path=str(attr_path), dataset_name=dataset
            )

            # Plot patient-level attribution
            patient_attr_path = (
                output_dirs["attribution"] / "patient_level_attribution.pdf"
            )
            plot_patient_level_attribution(risk_df, save_path=str(patient_attr_path))

            # If comprehensive attribution was used, generate additional visualizations
            if args.comprehensive_attribution and "simple_imaging" in risk_df.columns:
                logging.info(
                    f"  Generating comprehensive attribution analysis for {dataset}..."
                )

                # Comprehensive comparison plot
                comp_path = (
                    output_dirs["attribution"]
                    / "comprehensive_attribution_comparison.pdf"
                )
                plot_comprehensive_attribution_comparison(
                    risk_df, save_path=str(comp_path), dataset_name=dataset
                )

                # Detailed report
                create_comprehensive_attribution_report(
                    risk_df,
                    output_dir=str(output_dirs["attribution"]),
                    dataset_name=dataset,
                )

        # Generate EAGLE embeddings
        if args.generate_embeddings:
            logging.info(f"  Generating EAGLE embeddings for {dataset}...")
            generate_eagle_embeddings(dataset, output_dir / "models")

    return results


def generate_eagle_embeddings(dataset: str, model_dir: Path):
    """Generate EAGLE embeddings from trained model"""
    # Get configuration
    configs = {"GBM": GBM_CONFIG, "IPMN": IPMN_CONFIG, "NSCLC": NSCLC_CONFIG}
    config = configs[dataset]

    # Load data
    df = pd.read_parquet(config.data_path)
    pipeline = UnifiedPipeline(config)
    df_filtered = pipeline._filter_data(df)

    # Initialize processors
    clinical_processor = UnifiedClinicalProcessor(config)
    clinical_processor.fit(df_filtered)
    text_extractor = get_text_extractor(config.name)

    # Create dataset
    dataset_obj = UnifiedSurvivalDataset(
        df_filtered, config, clinical_processor, text_extractor
    )

    # Load model
    model_path = model_dir / "best_model_fold1.pth"
    if not model_path.exists():
        model_path = model_dir / "fold1.pth"

    num_clinical = dataset_obj.clinical_features.shape[1]
    num_text_features = len(text_extractor.get_feature_names()) if text_extractor else 0

    model = UnifiedSurvivalModel(
        dataset_config=config,
        model_config=ModelConfig(),
        num_clinical_features=num_clinical,
        num_text_features=num_text_features,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Extract embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    eagle_embeddings = []
    with torch.no_grad():
        for idx in range(len(dataset_obj)):
            data = dataset_obj[idx]

            # Prepare inputs
            imaging = data["imaging_features"].unsqueeze(0).to(device)
            clinical = data["clinical_features"].unsqueeze(0).to(device)
            text_embeddings = {
                k: v.unsqueeze(0).to(device) for k, v in data["text_embeddings"].items()
            }

            # Get fused features
            fused_features = model.get_fused_features(
                imaging, clinical, text_embeddings
            )
            eagle_embeddings.append(fused_features.cpu().numpy().squeeze())

    # Save embeddings
    output_df = df_filtered.copy()
    output_df["eagle_embeddings"] = eagle_embeddings
    output_df["eagle_embedding_shape"] = [emb.shape for emb in eagle_embeddings]

    output_path = f"data/{dataset}/eagle.parquet"
    output_df.to_parquet(output_path, index=False)
    logging.info(f"    Saved EAGLE embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="EAGLE: Multimodal Survival Analysis")

    # Main modes
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "baseline", "all"],
        help="Mode: train EAGLE only, run baselines only, or all",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--analyze-attribution", action="store_true", help="Run attribution analysis"
    )
    parser.add_argument(
        "--comprehensive-attribution",
        action="store_true",
        help="Run comprehensive attribution analysis with all three methods (Simple, Gradient, Integrated Gradients)",
    )
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        default=True,
        help="Generate EAGLE embeddings after training",
    )

    args = parser.parse_args()

    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)

    if args.mode in ["train", "all"]:
        # Train EAGLE models
        eagle_results = train_eagle_models(args)

        if args.mode == "all":
            # Run baselines after training
            baseline_results = run_baseline_comparison(args)

            # Combine results
            all_results = pd.DataFrame(eagle_results)
            if not baseline_results.empty:
                all_results = pd.concat(
                    [baseline_results, pd.DataFrame(eagle_results)], ignore_index=True
                )

            # Save combined results
            all_results.to_csv("results/all_results.csv", index=False)

            # Generate combined comparison visualizations
            generate_combined_comparison_plots(all_results)

            # Print final summary
            print("\n" + "=" * 80)
            print("COMPLETE ANALYSIS SUMMARY")
            print("=" * 80)

            for dataset in ["GBM", "IPMN", "NSCLC"]:
                dataset_results = all_results[all_results["Dataset"] == dataset]
                if not dataset_results.empty:
                    best_idx = dataset_results["Mean_C_Index"].idxmax()
                    best = dataset_results.loc[best_idx]
                    print(f"\n{dataset}:")
                    print(
                        f"  Best: {best['Model']} + {best['Embedding']} (C-Index: {best['Mean_C_Index']:.4f})"
                    )

    elif args.mode == "baseline":
        # Run baselines only
        run_baseline_comparison(args)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
