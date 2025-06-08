"""
Visualization utilities for EAGLE
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict
from lifelines import KaplanMeierFitter


def plot_km_curves(
    risk_df: pd.DataFrame, title: str = None, save_path: str = "km_curves.png"
) -> Dict:
    """Plot Kaplan-Meier curves by risk groups"""
    if title is None:
        title = "Risk-Stratified Survival Curves"

    plt.figure(figsize=(12, 8))

    # Colors for risk groups
    colors = ["#2E7D32", "#FFA726", "#D32F2F", "#7B1FA2"]

    kmf_dict = {}
    risk_groups = sorted(risk_df["risk_group"].unique())

    for i, group in enumerate(risk_groups):
        group_data = risk_df[risk_df["risk_group"] == group]

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=group_data["survival_time"],
            event_observed=group_data["event"],
            label=f"{group} (n={len(group_data)})",
        )

        kmf.plot_survival_function(color=colors[i], linewidth=2.5, ci_show=True)
        kmf_dict[group] = kmf

    plt.xlabel("Time (months)", fontsize=14)
    plt.ylabel("Survival Probability", fontsize=14)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return kmf_dict


def create_comprehensive_plots(risk_df: pd.DataFrame, output_dir: str = "."):
    """Create comprehensive visualization plots"""
    import os

    # Risk score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(
        risk_df["risk_score"], bins=30, alpha=0.7, color="steelblue", edgecolor="black"
    )
    plt.xlabel("Risk Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Risk Scores", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_score_distribution.png"), dpi=300)
    plt.close()

    # Risk score vs survival time scatter
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        risk_df["risk_score"],
        risk_df["survival_time"],
        c=risk_df["event"],
        cmap="RdYlBu",
        alpha=0.6,
        s=50,
    )
    plt.colorbar(scatter, label="Event (0=Censored, 1=Death)")
    plt.xlabel("Risk Score", fontsize=12)
    plt.ylabel("Survival Time (months)", fontsize=12)
    plt.title("Risk Score vs Survival Time", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_vs_survival_scatter.png"), dpi=300)
    plt.close()

    # Risk groups distribution
    if "risk_group" in risk_df.columns:
        plt.figure(figsize=(10, 6))
        risk_groups = risk_df["risk_group"].value_counts().sort_index()
        colors = ["#2E7D32", "#FFA726", "#D32F2F", "#7B1FA2"][: len(risk_groups)]

        bars = plt.bar(range(len(risk_groups)), risk_groups.values, color=colors)
        plt.xticks(range(len(risk_groups)), risk_groups.index, rotation=45)
        plt.xlabel("Risk Group", fontsize=12)
        plt.ylabel("Number of Patients", fontsize=12)
        plt.title("Patient Distribution by Risk Group", fontsize=14, fontweight="bold")

        # Add count labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "risk_group_distribution.png"), dpi=300)
        plt.close()

    # Box plot of risk scores by event status
    plt.figure(figsize=(8, 6))
    risk_df_plot = risk_df.copy()
    risk_df_plot["Event Status"] = risk_df_plot["event"].map(
        {0: "Censored", 1: "Death"}
    )

    sns.boxplot(
        data=risk_df_plot,
        x="Event Status",
        y="risk_score",
        palette=["lightblue", "salmon"],
    )
    plt.ylabel("Risk Score", fontsize=12)
    plt.title("Risk Score Distribution by Event Status", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_by_event_status.png"), dpi=300)
    plt.close()


def plot_dataset_specific(
    risk_df: pd.DataFrame, dataset_name: str, output_dir: str = "."
):
    """Create dataset-specific plots"""
    import os

    if dataset_name == "NSCLC":
        # Stage-specific survival if available
        stage_cols = [col for col in risk_df.columns if col.startswith("STAGE_")]
        if stage_cols:
            for col in stage_cols[:2]:  # Plot first two stage columns
                plt.figure(figsize=(12, 8))

                stage_groups = risk_df.groupby(col)
                colors = plt.cm.Set1(np.linspace(0, 1, len(stage_groups)))

                for i, (stage, group) in enumerate(stage_groups):
                    if len(group) >= 5:
                        kmf = KaplanMeierFitter()
                        kmf.fit(
                            durations=group["survival_time"],
                            event_observed=group["event"],
                            label=f"{stage} (n={len(group)})",
                        )
                        kmf.plot_survival_function(color=colors[i], linewidth=2)

                plt.xlabel("Time (months)", fontsize=12)
                plt.ylabel("Survival Probability", fontsize=12)
                plt.title(f"NSCLC Survival by {col}", fontsize=14, fontweight="bold")
                plt.legend(loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"survival_by_{col.lower()}.png"), dpi=300
                )
                plt.close()

    elif dataset_name == "GBM":
        # Plot by molecular markers if available
        molecular_cols = ["mgmt_methylated", "idh_mutant", "1p19q_codeleted"]
        available_cols = [col for col in molecular_cols if col in risk_df.columns]

        if available_cols:
            fig, axes = plt.subplots(
                1, len(available_cols), figsize=(6 * len(available_cols), 5)
            )
            if len(available_cols) == 1:
                axes = [axes]

            for i, col in enumerate(available_cols):
                risk_df_plot = risk_df.copy()
                risk_df_plot[col] = risk_df_plot[col].map(
                    {0: "Negative", 1: "Positive"}
                )
                sns.boxplot(
                    data=risk_df_plot,
                    x=col,
                    y="risk_score",
                    ax=axes[i],
                    palette=["lightcoral", "lightgreen"],
                )
                axes[i].set_title(f"Risk Score by {col.replace('_', ' ').title()}")
                axes[i].set_ylabel("Risk Score")
                axes[i].grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "gbm_molecular_markers.png"), dpi=300)
            plt.close()