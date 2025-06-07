"""
Random Survival Forest baseline for MedGemma embeddings
"""

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
from medgemma_baseline_utils import (
    load_medgemma_gbm_data, 
    load_medgemma_ipmn_data, 
    load_medgemma_nsclc_data, 
    prepare_medgemma_data
)


def train_rsf(X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=10):
    """Train Random Survival Forest"""
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=6,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )
    
    rsf.fit(X_train, y_train)
    return rsf


def evaluate_rsf(model, X_test, y_test):
    """Evaluate RSF model"""
    # Get risk scores (negative survival function at median time)
    risk_scores = model.predict(X_test)
    
    # Calculate C-index
    c_index = concordance_index_censored(
        y_test['event'],
        y_test['time'],
        -risk_scores  # Negative because higher risk = lower survival
    )[0]
    
    return c_index, risk_scores


def main():
    parser = argparse.ArgumentParser(description="RSF baseline for MedGemma embeddings")
    parser.add_argument("--dataset", type=str, required=True, choices=["GBM", "IPMN", "NSCLC"])
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="medgemma_baseline_results")
    
    args = parser.parse_args()
    
    # Setup logging
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.output_dir) / f"medgemma_rsf_{args.dataset}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting MedGemma RSF baseline for {args.dataset}")
    
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
    
    # Create structured array for sksurv
    y = np.array([(bool(e), t) for e, t in zip(y_event, y_time)],
                 dtype=[('event', bool), ('time', float)])
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_event)):
        logging.info(f"Fold {fold + 1}/{args.n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = train_rsf(X_train, y_train, n_estimators=args.n_estimators, max_depth=args.max_depth)
        
        # Evaluate
        c_index, _ = evaluate_rsf(model, X_val, y_val)
        cv_scores.append(c_index)
        logging.info(f"Fold {fold + 1} C-index: {c_index:.4f}")
    
    # Results summary
    results = {
        "model": "Random Survival Forest (MedGemma)",
        "dataset": args.dataset,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "cv_scores": cv_scores,
        "mean_cindex": np.mean(cv_scores),
        "std_cindex": np.std(cv_scores),
        "timestamp": timestamp,
        "embedding_type": "MedGemma"
    }
    
    # Save results
    results_file = Path(args.output_dir) / f"medgemma_rsf_{args.dataset}_{timestamp}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nResults saved to {results_file}")
    logging.info(f"Mean C-index: {results['mean_cindex']:.4f} ± {results['std_cindex']:.4f}")


if __name__ == "__main__":
    main()