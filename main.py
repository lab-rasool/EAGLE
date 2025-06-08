#!/usr/bin/env python
"""
EAGLE: Efficient Alignment of Generalized Latent Embeddings
Main script for training EAGLE models and running comparative analysis
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
    UnifiedPipeline, UnifiedSurvivalModel, UnifiedSurvivalDataset,
    UnifiedClinicalProcessor, UnifiedTrainer, UnifiedRiskStratification,
    get_text_extractor, ModelConfig, DatasetConfig,
    GBM_CONFIG, IPMN_CONFIG, NSCLC_CONFIG,
    GBM_MEDGEMMA_CONFIG, IPMN_MEDGEMMA_CONFIG, NSCLC_MEDGEMMA_CONFIG,
    plot_km_curves, create_comprehensive_plots,
    ModalityAttributionAnalyzer, plot_modality_contributions,
    plot_patient_level_attribution, create_attribution_report
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BaselineModels:
    """Baseline survival models for comparison"""
    
    @staticmethod
    def run_rsf(X_train, y_train, X_test, y_test):
        """Random Survival Forest"""
        # Create structured array for sksurv
        y_train_struct = np.array(
            [(bool(e), t) for e, t in zip(y_train['event'], y_train['time'])],
            dtype=[('event', bool), ('time', float)]
        )
        y_test_struct = np.array(
            [(bool(e), t) for e, t in zip(y_test['event'], y_test['time'])],
            dtype=[('event', bool), ('time', float)]
        )
        
        # Train RSF
        rsf = RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=6,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )
        rsf.fit(X_train, y_train_struct)
        
        # Get risk scores using cumulative hazard
        median_time = np.median(y_test['time'])
        chf = rsf.predict_cumulative_hazard_function(X_test)
        risk_scores = np.array([chf_i(median_time) for chf_i in chf])
        
        # Calculate C-index
        c_index = concordance_index_censored(
            y_test_struct['event'], y_test_struct['time'], risk_scores
        )[0]
        
        return c_index, risk_scores
    
    @staticmethod
    def run_coxph(X_train, y_train, X_test, y_test):
        """Cox Proportional Hazards"""
        # Create structured array
        y_train_struct = np.array(
            [(bool(e), t) for e, t in zip(y_train['event'], y_train['time'])],
            dtype=[('event', bool), ('time', float)]
        )
        y_test_struct = np.array(
            [(bool(e), t) for e, t in zip(y_test['event'], y_test['time'])],
            dtype=[('event', bool), ('time', float)]
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
                y_test_struct['event'], y_test_struct['time'], risk_scores
            )[0]
            
            return c_index, risk_scores
        except Exception as e:
            logging.warning(f"CoxPH failed: {e}")
            return np.nan, np.zeros(len(X_test))
    
    @staticmethod
    def run_deepsurv(X_train, y_train, X_test, y_test):
        """Simple DeepSurv implementation"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
                    torch.nn.Linear(16, 1)
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = DeepSurvNet(X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(100):  # Increased epochs
            optimizer.zero_grad()
            risk_pred = model(X_train_t).squeeze()
            
            # Simple Cox loss
            sorted_idx = torch.argsort(torch.tensor(y_train['time']))
            sorted_risk = risk_pred[sorted_idx]
            sorted_event = torch.tensor(
                y_train['event'][sorted_idx], 
                dtype=torch.float32
            ).to(device)
            
            # Clip predictions to prevent overflow
            sorted_risk = torch.clamp(sorted_risk, min=-10, max=10)
            
            exp_risk = torch.exp(sorted_risk)
            risk_sum = torch.cumsum(exp_risk.flip(0), 0).flip(0)
            loss = -torch.mean(sorted_event * (sorted_risk - torch.log(risk_sum + 1e-7)))
            
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
                y_test['event'].astype(bool), y_test['time'], risk_scores
            )[0]
        except Exception as e:
            logging.warning(f"DeepSurv C-index calculation failed: {e}")
            return np.nan, risk_scores
        
        return c_index, risk_scores


def load_embeddings_data(data_path: str, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load embeddings from parquet file"""
    df = pd.read_parquet(data_path)
    
    # Extract embeddings based on file type
    if 'eagle_embeddings' in df.columns:
        # EAGLE embeddings
        embeddings = np.array(df['eagle_embeddings'].tolist())
    elif "medgemma" in data_path:
        # MedGemma files have dataset-specific structures
        if dataset_name == "GBM":
            # Use combined embeddings for GBM MedGemma
            if 'combined_embeddings' in df.columns:
                embeddings = []
                for emb in df['combined_embeddings']:
                    if isinstance(emb, bytes):
                        embeddings.append(np.frombuffer(emb, dtype=np.float32))
                    else:
                        embeddings.append(np.array(emb))
                
                # Pad to max size to handle variable shapes
                max_size = max(emb.shape[0] for emb in embeddings)
                padded_embeddings = []
                for emb in embeddings:
                    if emb.shape[0] < max_size:
                        padded = np.pad(emb, (0, max_size - emb.shape[0]), mode='constant')
                        padded_embeddings.append(padded)
                    else:
                        padded_embeddings.append(emb)
                embeddings = np.array(padded_embeddings)
                    
        elif dataset_name == "IPMN":
            # Use fused embeddings for IPMN MedGemma
            if 'multimodal_ct_fused_embeddings' in df.columns:
                embeddings = []
                for emb in df['multimodal_ct_fused_embeddings']:
                    if isinstance(emb, bytes):
                        embeddings.append(np.frombuffer(emb, dtype=np.float32))
                    else:
                        embeddings.append(np.array(emb))
                
                # Pad to max size
                max_size = max(emb.shape[0] for emb in embeddings)
                padded_embeddings = []
                for emb in embeddings:
                    if emb.shape[0] < max_size:
                        padded = np.pad(emb, (0, max_size - emb.shape[0]), mode='constant')
                        padded_embeddings.append(padded)
                    else:
                        padded_embeddings.append(emb)
                embeddings = np.array(padded_embeddings)
                    
        else:  # NSCLC
            # For NSCLC, use contrast embeddings as primary
            if 'multimodal_contrast_embeddings' in df.columns:
                embeddings = []
                for emb in df['multimodal_contrast_embeddings']:
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
                            padded = np.pad(emb, (0, max_size - emb.shape[0]), mode='constant')
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
            primary_col = 'mri_embeddings'
        elif dataset_name == "IPMN":
            primary_col = 'ct_embeddings'
        else:  # NSCLC
            primary_col = 'ct_contrast_embeddings'
        
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
                        padded = np.pad(emb, (0, max_size - emb.shape[0]), mode='constant')
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
        y_time = df['survival_time_in_months'].values
        y_event = (df['vital_status_desc'] == 'DEAD').astype(int).values
    elif dataset_name == "IPMN":
        y_time = df['survival_time_days'].values / 30.44  # Convert to months
        y_event = (df['vital_status'] == 'DEAD').astype(int).values
    else:  # NSCLC
        y_time = df['SURVIVAL_TIME_IN_MONTHS'].values
        y_event = df['event'].values
    
    # Filter out any NaN values in survival data
    valid_mask = ~(np.isnan(y_time) | np.isnan(y_event))
    if not np.all(valid_mask):
        logging.warning(f"Filtering out {np.sum(~valid_mask)} samples with NaN survival data")
        embeddings = embeddings[valid_mask]
        y_time = y_time[valid_mask]
        y_event = y_event[valid_mask]
    
    # Ensure positive survival times
    if np.any(y_time <= 0):
        logging.warning(f"Found {np.sum(y_time <= 0)} non-positive survival times, setting to 0.1")
        y_time[y_time <= 0] = 0.1
    
    return embeddings, y_time, y_event


def run_baseline_comparison(args):
    """Run baseline model comparison across different embeddings"""
    logging.info("Running baseline model comparison...")
    
    results = []
    datasets = ["GBM", "IPMN", "NSCLC"]
    embedding_types = {
        "unimodal": "Unimodal",
        "medgemma": "MedGemma",
        "eagle": "EAGLE"
    }
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
                
                for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_event)):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train = {'time': y_time[train_idx], 'event': y_event[train_idx]}
                    y_test = {'time': y_time[test_idx], 'event': y_event[test_idx]}
                    
                    # Run model
                    if model_name == "RSF":
                        c_index, _ = BaselineModels.run_rsf(X_train, y_train, X_test, y_test)
                    elif model_name == "CoxPH":
                        c_index, _ = BaselineModels.run_coxph(X_train, y_train, X_test, y_test)
                    else:  # DeepSurv
                        c_index, _ = BaselineModels.run_deepsurv(X_train, y_train, X_test, y_test)
                    
                    if not np.isnan(c_index):
                        cv_scores.append(c_index)
                
                if cv_scores:
                    result = {
                        'Dataset': dataset,
                        'Embedding': emb_name,
                        'Model': model_name,
                        'Mean_C_Index': np.mean(cv_scores),
                        'Std_C_Index': np.std(cv_scores),
                        'N_Features': X.shape[1]
                    }
                    results.append(result)
                    logging.info(f"    {model_name}: {result['Mean_C_Index']:.4f} ± {result['Std_C_Index']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/baseline_comparison.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE MODEL COMPARISON")
    print("="*80)
    
    pivot_table = results_df.pivot_table(
        index=['Dataset', 'Model'],
        columns='Embedding',
        values='Mean_C_Index'
    ).round(4)
    
    print(pivot_table)
    
    return results_df


def train_eagle_models(args):
    """Train EAGLE models on all datasets"""
    logging.info("Training EAGLE models...")
    
    datasets = ["GBM", "IPMN", "NSCLC"]
    configs = {
        "GBM": GBM_CONFIG,
        "IPMN": IPMN_CONFIG,
        "NSCLC": NSCLC_CONFIG
    }
    
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
            "logs": output_dir / "logs"
        }
        
        for dir_path in output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get configuration
        config = configs[dataset]
        model_config = ModelConfig(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Train model
        pipeline = UnifiedPipeline(config, model_config, output_dirs=output_dirs)
        eagle_results, risk_df, stats = pipeline.run(
            n_folds=5,
            n_risk_groups=3,
            enable_attribution=args.analyze_attribution
        )
        
        # Store results
        results.append({
            'Dataset': dataset,
            'Embedding': 'EAGLE Model',
            'Model': 'EAGLE',
            'Mean_C_Index': eagle_results['mean_cindex'],
            'Std_C_Index': eagle_results['std_cindex'],
            'N_Features': 'Multi-modal'
        })
        
        logging.info(f"  EAGLE C-index: {eagle_results['mean_cindex']:.4f} ± {eagle_results['std_cindex']:.4f}")
        
        # Generate visualizations
        logging.info(f"  Generating visualizations for {dataset}...")
        from eagle.viz import plot_km_curves, create_comprehensive_plots, plot_dataset_specific
        
        # Save risk dataframe
        risk_df.to_csv(output_dirs["results"] / "risk_scores.csv", index=False)
        
        # Generate Kaplan-Meier curves
        km_path = output_dirs["figures"] / "kaplan_meier_curves.png"
        plot_km_curves(risk_df, title=f"{dataset} Risk-Stratified Survival Curves", save_path=str(km_path))
        
        # Generate comprehensive plots
        create_comprehensive_plots(risk_df, output_dir=str(output_dirs["figures"]))
        
        # Generate dataset-specific plots
        plot_dataset_specific(risk_df, dataset, output_dir=str(output_dirs["figures"]))
        
        # Generate attribution plots if enabled
        if args.analyze_attribution and "imaging_contribution" in risk_df.columns:
            logging.info(f"  Generating attribution visualizations for {dataset}...")
            from eagle.attribution import plot_modality_contributions, plot_patient_level_attribution
            
            # Plot modality contributions
            attr_path = output_dirs["attribution"] / "modality_contributions.png"
            plot_modality_contributions(risk_df, save_path=str(attr_path), dataset_name=dataset)
            
            # Plot patient-level attribution
            patient_attr_path = output_dirs["attribution"] / "patient_level_attribution.png"
            plot_patient_level_attribution(risk_df, save_path=str(patient_attr_path))
        
        # Generate EAGLE embeddings
        if args.generate_embeddings:
            logging.info(f"  Generating EAGLE embeddings for {dataset}...")
            generate_eagle_embeddings(dataset, output_dir / "models")
    
    return results


def generate_eagle_embeddings(dataset: str, model_dir: Path):
    """Generate EAGLE embeddings from trained model"""
    # Get configuration
    configs = {
        "GBM": GBM_CONFIG,
        "IPMN": IPMN_CONFIG,
        "NSCLC": NSCLC_CONFIG
    }
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
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Extract embeddings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    eagle_embeddings = []
    with torch.no_grad():
        for idx in range(len(dataset_obj)):
            data = dataset_obj[idx]
            
            # Prepare inputs
            imaging = data["imaging_features"].unsqueeze(0).to(device)
            clinical = data["clinical_features"].unsqueeze(0).to(device)
            text_embeddings = {k: v.unsqueeze(0).to(device) for k, v in data["text_embeddings"].items()}
            
            # Get fused features
            fused_features = model.get_fused_features(imaging, clinical, text_embeddings)
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
    parser.add_argument("--mode", type=str, default="all",
                       choices=["train", "baseline", "all"],
                       help="Mode: train EAGLE only, run baselines only, or all")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--analyze-attribution", action="store_true",
                       help="Run attribution analysis")
    parser.add_argument("--generate-embeddings", action="store_true", default=True,
                       help="Generate EAGLE embeddings after training")
    
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
                all_results = pd.concat([baseline_results, pd.DataFrame(eagle_results)], ignore_index=True)
            
            # Save combined results
            all_results.to_csv('results/all_results.csv', index=False)
            
            # Print final summary
            print("\n" + "="*80)
            print("COMPLETE ANALYSIS SUMMARY")
            print("="*80)
            
            for dataset in ["GBM", "IPMN", "NSCLC"]:
                dataset_results = all_results[all_results['Dataset'] == dataset]
                if not dataset_results.empty:
                    best_idx = dataset_results['Mean_C_Index'].idxmax()
                    best = dataset_results.loc[best_idx]
                    print(f"\n{dataset}:")
                    print(f"  Best: {best['Model']} + {best['Embedding']} (C-Index: {best['Mean_C_Index']:.4f})")
    
    elif args.mode == "baseline":
        # Run baselines only
        run_baseline_comparison(args)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()