"""
DeepSurv baseline for MedGemma embeddings
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
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


class SurvivalDataset(Dataset):
    """Dataset for survival analysis"""
    def __init__(self, X, y_time, y_event):
        self.X = torch.FloatTensor(X)
        self.y_time = torch.FloatTensor(y_time)
        self.y_event = torch.FloatTensor(y_event)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_time[idx], self.y_event[idx]


class DeepSurv(nn.Module):
    """DeepSurv neural network"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer (single neuron for risk score)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def cox_loss(risk_scores, times, events):
    """Negative log partial likelihood for Cox model"""
    # Sort by time
    sorted_idx = torch.argsort(times, descending=True)
    sorted_risks = risk_scores[sorted_idx].squeeze()
    sorted_events = events[sorted_idx]
    
    # Compute log partial likelihood
    max_risk = sorted_risks.max()
    exp_risks = torch.exp(sorted_risks - max_risk)
    cumsum_exp_risks = torch.cumsum(exp_risks, dim=0)
    
    log_likelihood = sorted_risks - torch.log(cumsum_exp_risks + 1e-7) - max_risk
    log_likelihood = log_likelihood * sorted_events
    
    return -log_likelihood.sum() / (sorted_events.sum() + 1e-7)


def train_deepsurv(model, train_loader, val_loader, n_epochs=100, lr=1e-4, device='cuda'):
    """Train DeepSurv model"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_c_index = 0
    patience_counter = 0
    patience = 20
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        total_loss = 0
        
        for X_batch, y_time_batch, y_event_batch in train_loader:
            X_batch = X_batch.to(device)
            y_time_batch = y_time_batch.to(device)
            y_event_batch = y_event_batch.to(device)
            
            optimizer.zero_grad()
            risk_scores = model(X_batch)
            loss = cox_loss(risk_scores, y_time_batch, y_event_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_risks = []
            val_times = []
            val_events = []
            
            with torch.no_grad():
                for X_batch, y_time_batch, y_event_batch in val_loader:
                    X_batch = X_batch.to(device)
                    risk_scores = model(X_batch)
                    val_risks.extend(risk_scores.cpu().numpy().flatten())
                    val_times.extend(y_time_batch.numpy())
                    val_events.extend(y_event_batch.numpy())
            
            # Calculate C-index
            c_index = concordance_index(val_times, -np.array(val_risks), val_events)
            scheduler.step(-c_index)  # We want to maximize C-index
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Val C-index={c_index:.4f}")
            
            # Early stopping
            if c_index > best_c_index:
                best_c_index = c_index
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break
    
    return model, best_c_index


def main():
    parser = argparse.ArgumentParser(description="DeepSurv baseline for MedGemma embeddings")
    parser.add_argument("--dataset", type=str, required=True, choices=["GBM", "IPMN", "NSCLC"])
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="medgemma_baseline_results")
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(args.output_dir) / f"medgemma_deepsurv_{args.dataset}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting MedGemma DeepSurv baseline for {args.dataset}")
    logging.info(f"Device: {device}")
    
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
    
    # Ensure all data is numpy arrays, not pandas Series
    X = np.ascontiguousarray(X)
    y_time = np.ascontiguousarray(y_time)
    y_event = np.ascontiguousarray(y_event)
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    # Create indices for cross-validation
    indices = np.arange(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y_event)):
        logging.info(f"\nFold {fold + 1}/{args.n_folds}")
        
        # Split data using numpy indexing
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_time_train = y_time[train_idx]
        y_time_val = y_time[val_idx]
        y_event_train = y_event[train_idx]
        y_event_val = y_event[val_idx]
        
        # Create datasets and loaders
        train_dataset = SurvivalDataset(X_train, y_time_train, y_event_train)
        val_dataset = SurvivalDataset(X_val, y_time_val, y_event_val)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Initialize model
        model = DeepSurv(
            input_dim=X.shape[1],
            hidden_dims=args.hidden_dims,
            dropout=args.dropout
        )
        
        try:
            # Train model
            model, best_c_index = train_deepsurv(
                model, train_loader, val_loader, 
                n_epochs=args.n_epochs, lr=args.lr, device=device
            )
            
            cv_scores.append(best_c_index)
            logging.info(f"Fold {fold + 1} C-index: {best_c_index:.4f}")
            
        except Exception as e:
            logging.warning(f"Fold {fold + 1} failed: {str(e)}")
    
    if not cv_scores:
        logging.error("All folds failed!")
        return
    
    # Results summary
    results = {
        "model": "DeepSurv (MedGemma)",
        "dataset": args.dataset,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "successful_folds": len(cv_scores),
        "total_folds": args.n_folds,
        "cv_scores": cv_scores,
        "mean_cindex": np.mean(cv_scores),
        "std_cindex": np.std(cv_scores),
        "timestamp": timestamp,
        "embedding_type": "MedGemma"
    }
    
    # Save results
    results_file = Path(args.output_dir) / f"medgemma_deepsurv_{args.dataset}_{timestamp}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"\nResults saved to {results_file}")
    logging.info(f"Mean C-index: {results['mean_cindex']:.4f} ± {results['std_cindex']:.4f}")


if __name__ == "__main__":
    main()