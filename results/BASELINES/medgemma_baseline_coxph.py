"""
Cox Proportional Hazards baseline for MedGemma embeddings
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from medgemma_baseline_utils import (
    load_medgemma_gbm_data, 
    load_medgemma_ipmn_data, 
    load_medgemma_nsclc_data, 
    prepare_medgemma_data
)


def reduce_dimensions(X, n_components=100):
    """Reduce dimensions using PCA for CoxPH stability"""
    pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
    X_reduced = pca.fit_transform(X)
    explained_var = np.sum(pca.explained_variance_ratio_)
    logging.info(f"PCA reduced to {X_reduced.shape[1]} components, explaining {explained_var:.2%} variance")
    return X_reduced, pca


def prepare_survival_data(X, y_time, y_event):
    """Prepare data for survival analysis, removing any problematic samples"""
    # Remove samples with NaN or infinite values
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y_time) & np.isfinite(y_event)
    valid_mask &= (y_time > 0)  # Ensure positive survival times
    
    X_clean = X[valid_mask]
    y_time_clean = y_time[valid_mask]
    y_event_clean = y_event[valid_mask]
    
    if np.sum(~valid_mask) > 0:
        logging.warning(f"Removed {np.sum(~valid_mask)} samples with invalid values")
    
    return X_clean, y_time_clean, y_event_clean


def train_coxph(X_train, y_time_train, y_event_train, penalizer=0.1, l1_ratio=0):
    """Train Cox Proportional Hazards model"""
    # Clean the data first
    X_train, y_time_train, y_event_train = prepare_survival_data(X_train, y_time_train, y_event_train)
    
    # Create DataFrame for lifelines
    train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    train_df['time'] = y_time_train
    train_df['event'] = y_event_train
    
    # Ensure no NaN values
    if train_df.isnull().any().any():
        logging.warning("NaN values detected, filling with zeros")
        train_df = train_df.fillna(0)
    
    # Fit Cox model
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(train_df, duration_col='time', event_col='event', show_progress=False)
    
    return cph


def evaluate_coxph(model, X_test, y_time_test, y_event_test):
    """Evaluate Cox model"""
    # Clean the data first
    X_test, y_time_test, y_event_test = prepare_survival_data(X_test, y_time_test, y_event_test)
    
    # Create DataFrame for prediction
    test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    
    # Get partial hazards (risk scores)
    risk_scores = model.predict_partial_hazard(test_df).values
    
    # Calculate C-index
    c_index = concordance_index(y_time_test, -risk_scores, y_event_test)
    
    return c_index, risk_scores


def main():
    parser = argparse.ArgumentParser(description="CoxPH baseline for MedGemma embeddings")
    parser.add_argument("--dataset", type=str, required=True, choices=["GBM", "IPMN", "NSCLC"])
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--n-components", type=int, default=100, help="PCA components")
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument("--l1-ratio", type=float, default=0, help="0 for L2, 1 for L1")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="medgemma_baseline_results")
    
    args = parser.parse_args()
    
    # Setup logging
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.output_dir) / f"medgemma_coxph_{args.dataset}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting MedGemma CoxPH baseline for {args.dataset}")
    
    # Load MedGemma data
    try:
        if args.dataset == "GBM":
            X, y_time, y_event = load_medgemma_gbm_data(args.data_path)
        elif args.dataset == "IPMN":
            X, y_time, y_event = load_medgemma_ipmn_data(args.data_path)
        else:  # NSCLC
            X, y_time, y_event = load_medgemma_nsclc_data(args.data_path)
    except Exception as e:
        logging.error(f"Error loading MedGemma data: {str(e)}")
        return
    
    logging.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Prepare data
    X = prepare_medgemma_data(X, scale=True)
    
    # Clean data before PCA
    X, y_time, y_event = prepare_survival_data(X, y_time, y_event)
    
    # Reduce dimensions for stability
    X_reduced, pca = reduce_dimensions(X, n_components=args.n_components)
    
    # Convert to contiguous arrays
    X_reduced = np.ascontiguousarray(X_reduced)
    y_time = np.ascontiguousarray(y_time)
    y_event = np.ascontiguousarray(y_event)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    cv_scores = []
    successful_folds = 0
    
    indices = np.arange(len(X_reduced))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y_event)):
        logging.info(f"Fold {fold + 1}/{args.n_folds}")
        
        X_train = X_reduced[train_idx]
        X_val = X_reduced[val_idx]
        y_time_train = y_time[train_idx]
        y_time_val = y_time[val_idx]
        y_event_train = y_event[train_idx]
        y_event_val = y_event[val_idx]
        
        try:
            # Train model
            model = train_coxph(X_train, y_time_train, y_event_train, 
                              penalizer=args.penalizer, l1_ratio=args.l1_ratio)
            
            # Evaluate
            c_index, _ = evaluate_coxph(model, X_val, y_time_val, y_event_val)
            cv_scores.append(c_index)
            successful_folds += 1
            logging.info(f"Fold {fold + 1} C-index: {c_index:.4f}")
            
        except Exception as e:
            logging.warning(f"Fold {fold + 1} failed: {str(e)}")
    
    if successful_folds == 0:
        logging.error("All folds failed!")
        return
    
    # Results summary
    results = {
        "model": "Cox Proportional Hazards (MedGemma)",
        "dataset": args.dataset,
        "n_samples": len(X),
        "n_features_original": X.shape[1],
        "n_features_reduced": X_reduced.shape[1],
        "n_components_pca": args.n_components,
        "penalizer": args.penalizer,
        "l1_ratio": args.l1_ratio,
        "successful_folds": successful_folds,
        "total_folds": args.n_folds,
        "cv_scores": cv_scores,
        "mean_cindex": np.mean(cv_scores),
        "std_cindex": np.std(cv_scores),
        "timestamp": timestamp,
        "embedding_type": "MedGemma"
    }
    
    # Save results
    results_file = Path(args.output_dir) / f"medgemma_coxph_{args.dataset}_{timestamp}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nResults saved to {results_file}")
    logging.info(f"Mean C-index: {results['mean_cindex']:.4f} ± {results['std_cindex']:.4f}")
    logging.info(f"Successful folds: {successful_folds}/{args.n_folds}")


if __name__ == "__main__":
    main()