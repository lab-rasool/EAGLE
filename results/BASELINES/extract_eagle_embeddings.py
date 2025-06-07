#!/usr/bin/env python
"""
Extract embeddings from trained EAGLE models and train baseline survival models
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging
from datetime import datetime
import json
import sys
import os

# Add EAGLE to path if needed
sys.path.append('/proj/rasool_lab_projects/Aakash/EAGLE')

from eagle import (
    UnifiedSurvivalModel, UnifiedSurvivalDataset, UnifiedClinicalProcessor,
    get_text_extractor, GBM_CONFIG, IPMN_CONFIG, NSCLC_CONFIG, ModelConfig
)

# Import baseline utilities
sys.path.append('/proj/rasool_lab_projects/Aakash/EAGLE/results/BASELINES')
from baseline_utils import prepare_data
from baseline_coxph import train_coxph, evaluate_coxph, prepare_survival_data
from baseline_rsf import train_rsf, evaluate_rsf
from baseline_deepsurv import DeepSurv, SurvivalDataset, train_deepsurv

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')


class EAGLEEmbeddingExtractor:
    """Extract embeddings from trained EAGLE models"""
    
    def __init__(self, model_path: str, dataset_config, model_config):
        self.model_path = model_path
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model_and_data(self, data_path: str):
        """Load the trained model and prepare dataset"""
        # Load and prepare data
        df = pd.read_parquet(data_path)
        df = self._filter_data(df)
        
        # Initialize processors
        clinical_processor = UnifiedClinicalProcessor(self.dataset_config)
        clinical_processor.fit(df)
        text_extractor = get_text_extractor(self.dataset_config.name)
        
        # Create dataset
        dataset = UnifiedSurvivalDataset(
            df, self.dataset_config, clinical_processor, text_extractor
        )
        
        # Initialize model
        num_clinical = dataset.clinical_features.shape[1]
        num_text_features = len(text_extractor.get_feature_names()) if text_extractor else 0
        
        model = UnifiedSurvivalModel(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            num_clinical_features=num_clinical,
            num_text_features=num_text_features,
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model, dataset, df
    
    def _filter_data(self, df):
        """Filter data to remove invalid samples"""
        if self.dataset_config.name == "GBM":
            # Filter based on survival time and event
            df = df.dropna(subset=[self.dataset_config.survival_time_col])
            df = df[df[self.dataset_config.survival_time_col] > 0]
            
            # Convert event column
            if self.dataset_config.event_col == "vital_status_desc":
                df = df.dropna(subset=[self.dataset_config.event_col])
                df['event'] = (df[self.dataset_config.event_col] == "DEAD").astype(int)
            
        elif self.dataset_config.name == "IPMN":
            # Filter based on survival time and event
            df = df.dropna(subset=[self.dataset_config.survival_time_col])
            df = df[df[self.dataset_config.survival_time_col] > 0]
            
            # Convert event column
            if self.dataset_config.event_col == "vital_status":
                df = df.dropna(subset=[self.dataset_config.event_col])
                df['event'] = (df[self.dataset_config.event_col] == "DEAD").astype(int)
        
        elif self.dataset_config.name == "NSCLC":
            # Filter based on survival time and event
            df = df.dropna(subset=[self.dataset_config.survival_time_col])
            df = df[df[self.dataset_config.survival_time_col] > 0]
            
            # Event column should already be binary
            df = df.dropna(subset=[self.dataset_config.event_col])
        
        return df.reset_index(drop=True)
    
    def extract_embeddings(self, model, dataset):
        """Extract EAGLE embeddings for all samples"""
        embeddings = []
        survival_times = []
        events = []
        
        # Create data loader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                imaging = batch["imaging_features"].to(self.device)
                text_emb = {k: v.to(self.device) for k, v in batch["text_embeddings"].items()}
                clinical = batch["clinical_features"].to(self.device)
                text_feat = batch["text_features"].to(self.device)
                
                # Get modality embeddings
                modality_embeddings = model.get_modality_embeddings(
                    imaging, text_emb, clinical, text_feat
                )
                
                # Extract fused features before final prediction
                # We'll manually run the forward pass up to fusion
                batch_size = imaging.shape[0]
                device = imaging.device
                
                # Encode each modality
                imaging_encoded = model.imaging_encoder(imaging)
                text_encoded = model._encode_text(text_emb, batch_size, device)
                clinical_encoded = model.clinical_encoder(clinical)
                
                # Project to common dimension
                imaging_proj = model.imaging_projection(imaging_encoded).unsqueeze(1)
                text_proj = model.text_projection(text_encoded).unsqueeze(1)
                clinical_proj = model.clinical_projection(clinical_encoded).unsqueeze(1)
                
                # Apply cross-attention if enabled
                if model.model_config.use_cross_attention:
                    # Create multimodal sequences for attention
                    img_text_sequence = torch.cat([imaging_proj, text_proj], dim=1)
                    img_clin_sequence = torch.cat([imaging_proj, clinical_proj], dim=1)
                    
                    # Apply attention fusion
                    img_text_attended = model.imaging_text_attention(img_text_sequence)
                    img_clin_attended = model.imaging_clinical_attention(img_clin_sequence)
                    
                    # Pool attended features
                    img_text_pooled = img_text_attended.mean(dim=1)
                    img_clin_pooled = img_clin_attended.mean(dim=1)
                    
                    # Combine attended features
                    fused_features = torch.cat(
                        [img_text_pooled, img_clin_pooled, clinical_proj.squeeze(1)], dim=1
                    )
                else:
                    # Simple concatenation
                    fused_features = torch.cat(
                        [imaging_proj.squeeze(1), text_proj.squeeze(1), clinical_proj.squeeze(1)], dim=1
                    )
                
                # Apply fusion layers to get final embeddings
                final_embeddings = model.fusion_layers(fused_features)
                
                # Store embeddings and targets
                embeddings.append(final_embeddings.cpu().numpy())
                survival_times.append(batch["survival_time"].numpy())
                events.append(batch["event"].numpy())
        
        # Concatenate all batches
        embeddings = np.vstack(embeddings)
        survival_times = np.concatenate(survival_times)
        events = np.concatenate(events)
        
        return embeddings, survival_times, events


def run_baseline_experiments(X, y_time, y_event, dataset_name, output_dir, n_folds=5):
    """Run CoxPH, RSF, and DeepSurv experiments with extracted embeddings"""
    
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare data
    X_scaled = prepare_data(X, scale=True)
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 1. Cox Proportional Hazards
    logging.info("Running Cox Proportional Hazards...")
    coxph_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_event)):
        try:
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_time_train, y_time_val = y_time[train_idx], y_time[val_idx]
            y_event_train, y_event_val = y_event[train_idx], y_event[val_idx]
            
            # Clean data
            X_train, y_time_train, y_event_train = prepare_survival_data(
                X_train, y_time_train, y_event_train
            )
            X_val, y_time_val, y_event_val = prepare_survival_data(
                X_val, y_time_val, y_event_val
            )
            
            model = train_coxph(X_train, y_time_train, y_event_train, penalizer=0.1)
            c_index, _ = evaluate_coxph(model, X_val, y_time_val, y_event_val)
            coxph_scores.append(c_index)
            logging.info(f"CoxPH Fold {fold + 1}: {c_index:.4f}")
            
        except Exception as e:
            logging.warning(f"CoxPH Fold {fold + 1} failed: {str(e)}")
    
    results['coxph'] = {
        'scores': coxph_scores,
        'mean': np.mean(coxph_scores) if coxph_scores else 0,
        'std': np.std(coxph_scores) if coxph_scores else 0
    }
    
    # 2. Random Survival Forest
    logging.info("Running Random Survival Forest...")
    rsf_scores = []
    
    # Create structured array for sksurv
    y_structured = np.array([(bool(e), t) for e, t in zip(y_event, y_time)],
                           dtype=[('event', bool), ('time', float)])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_event)):
        try:
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_structured[train_idx], y_structured[val_idx]
            
            model = train_rsf(X_train, y_train, n_estimators=200)
            c_index, _ = evaluate_rsf(model, X_val, y_val)
            rsf_scores.append(c_index)
            logging.info(f"RSF Fold {fold + 1}: {c_index:.4f}")
            
        except Exception as e:
            logging.warning(f"RSF Fold {fold + 1} failed: {str(e)}")
    
    results['rsf'] = {
        'scores': rsf_scores,
        'mean': np.mean(rsf_scores) if rsf_scores else 0,
        'std': np.std(rsf_scores) if rsf_scores else 0
    }
    
    # 3. DeepSurv
    logging.info("Running DeepSurv...")
    deepsurv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_event)):
        try:
            # Create datasets
            train_dataset = SurvivalDataset(
                X_scaled[train_idx], y_time[train_idx], y_event[train_idx]
            )
            val_dataset = SurvivalDataset(
                X_scaled[val_idx], y_time[val_idx], y_event[val_idx]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Create and train model
            model = DeepSurv(input_dim=X_scaled.shape[1], hidden_dims=[256, 128, 64])
            model, val_cindex = train_deepsurv(
                model, train_loader, val_loader, 
                n_epochs=100, lr=1e-4, device=device
            )
            
            deepsurv_scores.append(val_cindex)
            logging.info(f"DeepSurv Fold {fold + 1}: {val_cindex:.4f}")
            
        except Exception as e:
            logging.warning(f"DeepSurv Fold {fold + 1} failed: {str(e)}")
    
    results['deepsurv'] = {
        'scores': deepsurv_scores,
        'mean': np.mean(deepsurv_scores) if deepsurv_scores else 0,
        'std': np.std(deepsurv_scores) if deepsurv_scores else 0
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"eagle_embeddings_{dataset_name}_{timestamp}_results.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'embedding_source': 'EAGLE',
            'results': results,
            'timestamp': timestamp
        }, f, indent=2)
    
    return results, results_file


def main():
    parser = argparse.ArgumentParser(description="Extract EAGLE embeddings and train baseline models")
    parser.add_argument("--dataset", type=str, required=True, choices=["GBM", "IPMN", "NSCLC"])
    parser.add_argument("--model-dir", type=str, required=True, 
                       help="Directory containing trained EAGLE models (e.g., results/GBM/2025-06-05_18-13-59/models)")
    parser.add_argument("--data-path", type=str, default=None, 
                       help="Custom path to dataset (uses config default if not provided)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="eagle_baseline_results")
    parser.add_argument("--best-fold-only", action="store_true", 
                       help="Use only the best performing fold model")
    
    args = parser.parse_args()
    
    # Setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.output_dir) / f"eagle_baselines_{args.dataset}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Get dataset config
    if args.dataset == "GBM":
        config = GBM_CONFIG
    elif args.dataset == "IPMN":
        config = IPMN_CONFIG
    else:  # NSCLC
        config = NSCLC_CONFIG
    
    if args.data_path:
        config.data_path = args.data_path
    
    # Default model config (should match training)
    model_config = ModelConfig()
    
    # Find model files
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    model_files = list(model_dir.glob("fold*.pth"))
    if not model_files:
        model_files = list(model_dir.glob("*.pth"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    logging.info(f"Found {len(model_files)} model files")
    
    # Use best fold or average across folds
    if args.best_fold_only and len(model_files) > 1:
        # Find best model (assuming best_model.pth exists or use fold1.pth)
        best_model = model_dir / "best_model.pth"
        if best_model.exists():
            model_files = [best_model]
        else:
            model_files = [model_files[0]]  # Use first fold
        logging.info(f"Using single model: {model_files[0]}")
    
    all_embeddings = []
    all_survival_times = []
    all_events = []
    
    # Extract embeddings from each model
    for i, model_file in enumerate(model_files):
        logging.info(f"Processing model {i+1}/{len(model_files)}: {model_file.name}")
        
        try:
            # Create extractor
            extractor = EAGLEEmbeddingExtractor(str(model_file), config, model_config)
            
            # Load model and data
            model, dataset, df = extractor.load_model_and_data(config.data_path)
            
            # Extract embeddings
            embeddings, survival_times, events = extractor.extract_embeddings(model, dataset)
            
            if args.best_fold_only or len(model_files) == 1:
                # Use embeddings from this model only
                all_embeddings = embeddings
                all_survival_times = survival_times
                all_events = events
                break
            else:
                # Store for averaging
                all_embeddings.append(embeddings)
                all_survival_times.append(survival_times)
                all_events.append(events)
                
        except Exception as e:
            logging.error(f"Failed to process {model_file}: {str(e)}")
            continue
    
    # Average embeddings if using multiple models
    if len(model_files) > 1 and not args.best_fold_only:
        logging.info("Averaging embeddings across models...")
        all_embeddings = np.mean(all_embeddings, axis=0)
        # Use survival times and events from first model (should be identical)
        all_survival_times = all_survival_times[0]
        all_events = all_events[0]
    
    logging.info(f"Final embeddings shape: {all_embeddings.shape}")
    logging.info(f"Samples: {len(all_survival_times)}")
    logging.info(f"Events: {np.sum(all_events)} ({np.mean(all_events):.2%})")
    
    # Save embeddings
    embeddings_file = Path(args.output_dir) / f"eagle_embeddings_{args.dataset}_{timestamp}.npz"
    np.savez(embeddings_file, 
             embeddings=all_embeddings,
             survival_times=all_survival_times, 
             events=all_events)
    logging.info(f"Embeddings saved to: {embeddings_file}")
    
    # Run baseline experiments
    logging.info("\nRunning baseline survival models with EAGLE embeddings...")
    results, results_file = run_baseline_experiments(
        all_embeddings, all_survival_times, all_events, 
        args.dataset, args.output_dir, args.n_folds
    )
    
    # Print results
    print("\n" + "="*60)
    print(f"EAGLE Embedding Baseline Results for {args.dataset}")
    print("="*60)
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    print(f"Number of samples: {len(all_survival_times)}")
    print(f"Event rate: {np.mean(all_events):.2%}")
    print()
    
    for method_name, method_results in results.items():
        if method_results['scores']:
            print(f"{method_name.upper()}:")
            print(f"  Mean C-index: {method_results['mean']:.4f} ± {method_results['std']:.4f}")
            print(f"  Per-fold: {[f'{s:.4f}' for s in method_results['scores']]}")
        else:
            print(f"{method_name.upper()}: FAILED")
        print()
    
    print(f"Results saved to: {results_file}")
    print(f"Embeddings saved to: {embeddings_file}")


if __name__ == "__main__":
    main()