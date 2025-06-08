"""
Training utilities for EAGLE
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Tuple
from lifelines.utils import concordance_index
from .models import UnifiedSurvivalModel, ModelConfig


class UnifiedTrainer:
    """Unified trainer for all datasets"""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

    def train_model(
        self,
        model: UnifiedSurvivalModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ModelConfig,
        save_path: str = "best_model.pth",
    ) -> Tuple[UnifiedSurvivalModel, float]:
        """Train the model"""
        model = model.to(self.device)

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )

        best_val_cindex = 0
        patience_counter = 0

        for epoch in range(config.num_epochs):
            # Training
            model.train()
            train_losses = []

            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"
            ):
                # Move to device
                imaging = batch["imaging_features"].to(self.device)
                text_emb = {
                    k: v.to(self.device) for k, v in batch["text_embeddings"].items()
                }
                clinical = batch["clinical_features"].to(self.device)
                text_feat = batch["text_features"].to(self.device)
                survival_time = batch["survival_time"].to(self.device).float()
                event = batch["event"].to(self.device).float()

                # Forward pass
                risk_scores, aux_outputs = model(imaging, text_emb, clinical, text_feat)

                # Compute loss
                loss = self._cox_loss(risk_scores, survival_time, event)

                # Add auxiliary losses if available
                if config.use_auxiliary_tasks and "event_pred" in aux_outputs:
                    event_loss = nn.BCELoss()(
                        aux_outputs["event_pred"].squeeze(), event.float()
                    )
                    loss = loss + 0.1 * event_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            val_cindex = self.evaluate(model, val_loader)

            # Update scheduler
            scheduler.step(val_cindex)

            # Logging
            avg_loss = np.mean(train_losses)
            logging.info(
                f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val C-index={val_cindex:.4f}"
            )

            # Early stopping
            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    logging.info("Early stopping triggered")
                    break

        # Load best model
        model.load_state_dict(torch.load(save_path))
        return model, best_val_cindex

    def _cox_loss(
        self, risk_scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        """Cox partial likelihood loss"""
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

    def evaluate(self, model: UnifiedSurvivalModel, loader: DataLoader) -> float:
        """Evaluate model using C-index"""
        model.eval()

        all_risks = []
        all_times = []
        all_events = []

        with torch.no_grad():
            for batch in loader:
                imaging = batch["imaging_features"].to(self.device)
                text_emb = {
                    k: v.to(self.device) for k, v in batch["text_embeddings"].items()
                }
                clinical = batch["clinical_features"].to(self.device)
                text_feat = batch["text_features"].to(self.device)

                risk_scores, _ = model(imaging, text_emb, clinical, text_feat)

                all_risks.extend(risk_scores.cpu().numpy().flatten())
                all_times.extend(batch["survival_time"].numpy())
                all_events.extend(batch["event"].numpy())

        # Calculate C-index
        try:
            c_index = concordance_index(all_times, -np.array(all_risks), all_events)
        except:
            c_index = 0.5

        return c_index
