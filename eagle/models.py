"""
Model architectures for EAGLE with attribution support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging
from .data import DatasetConfig


@dataclass
class ModelConfig:
    """Configuration for model architecture and training"""

    # Architecture
    imaging_encoder_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    text_encoder_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    clinical_encoder_dims: List[int] = field(default_factory=lambda: [64, 32])
    fusion_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    # Training parameters
    dropout: float = 0.3
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    patience: int = 15

    # Attention settings
    use_cross_attention: bool = True
    attention_heads: int = 8

    # Auxiliary tasks
    use_auxiliary_tasks: bool = True


class AttentionFusion(nn.Module):
    """Attention-based fusion module with weight tracking"""

    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        self.last_attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, attn_weights = self.attention(x, x, x, need_weights=True)
        self.last_attention_weights = attn_weights
        return self.norm(x + attn_output)


class UnifiedSurvivalModel(nn.Module):
    """Unified survival prediction model supporting multiple datasets with attribution"""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        num_clinical_features: int,
        num_text_features: int = 0,
    ):
        super().__init__()

        self.dataset_config = dataset_config
        self.model_config = model_config

        # Build component encoders
        self._build_imaging_encoder()
        self._build_text_encoder()
        self._build_clinical_encoder(num_clinical_features)

        # Build projection layers for fusion
        self._build_projection_layers()

        # Build attention modules if enabled
        if self.model_config.use_cross_attention:
            self._build_attention_modules()

        # Build fusion layers
        self._build_fusion_layers()

        # Build output layers
        self._build_output_layers()

        # For tracking intermediate outputs
        self.last_encoder_outputs = {}
        self.last_attention_weights = {}

    def _build_imaging_encoder(self):
        """Build imaging encoder layers"""
        # Determine input dimension based on dataset
        if self.dataset_config.name == "NSCLC":
            input_dim = self.dataset_config.imaging_embedding_dim * 2
        else:
            input_dim = self.dataset_config.imaging_embedding_dim

        layers = []
        prev_dim = input_dim

        for dim in self.model_config.imaging_encoder_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(self.model_config.dropout),
                ]
            )
            prev_dim = dim

        self.imaging_encoder = nn.Sequential(*layers)
        self.imaging_output_dim = prev_dim

    def _build_text_encoder(self):
        """Build text encoder layers"""
        # Determine input dimension based on dataset
        if self.dataset_config.name == "NSCLC":
            input_dim = self.dataset_config.text_embedding_dim
        elif self.dataset_config.name == "GBM":
            base_dim = self.dataset_config.text_embedding_dim
            multiplier = 3 if self.dataset_config.has_treatment_text else 2
            input_dim = base_dim * multiplier
        else:  # IPMN
            input_dim = self.dataset_config.text_embedding_dim * 2

        layers = []
        prev_dim = input_dim

        for dim in self.model_config.text_encoder_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(self.model_config.dropout),
                ]
            )
            prev_dim = dim

        self.text_encoder = nn.Sequential(*layers)
        self.text_output_dim = prev_dim

    def _build_clinical_encoder(self, num_clinical_features: int):
        """Build clinical encoder layers"""
        layers = []
        prev_dim = num_clinical_features

        for dim in self.model_config.clinical_encoder_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(self.model_config.dropout),
                ]
            )
            prev_dim = dim

        self.clinical_encoder = nn.Sequential(*layers)
        self.clinical_output_dim = prev_dim

    def _build_projection_layers(self):
        """Build projection layers to standardize dimensions for fusion"""
        max_dim = max(
            self.imaging_output_dim, self.text_output_dim, self.clinical_output_dim
        )

        self.imaging_projection = nn.Linear(self.imaging_output_dim, max_dim)
        self.text_projection = nn.Linear(self.text_output_dim, max_dim)
        self.clinical_projection = nn.Linear(self.clinical_output_dim, max_dim)

        self.projection_dim = max_dim

    def _build_attention_modules(self):
        """Build attention modules if enabled"""
        self.imaging_text_attention = AttentionFusion(
            input_dim=self.projection_dim,
            num_heads=self.model_config.attention_heads,
            dropout=self.model_config.dropout,
        )

        self.imaging_clinical_attention = AttentionFusion(
            input_dim=self.projection_dim,
            num_heads=self.model_config.attention_heads,
            dropout=self.model_config.dropout,
        )

    def _build_fusion_layers(self):
        """Build fusion layers"""
        fusion_input_dim = self.projection_dim * 3

        layers = []
        prev_dim = fusion_input_dim

        for dim in self.model_config.fusion_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(self.model_config.dropout),
                ]
            )
            prev_dim = dim

        self.fusion_layers = nn.Sequential(*layers)
        self.fusion_output_dim = prev_dim

    def _build_output_layers(self):
        """Build output layers"""
        # Main survival prediction head
        self.survival_head = nn.Sequential(
            nn.Linear(self.fusion_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.model_config.dropout),
            nn.Linear(64, 1),
        )

        # Auxiliary task heads if enabled
        if self.model_config.use_auxiliary_tasks:
            self.event_head = nn.Sequential(
                nn.Linear(self.fusion_output_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        imaging_features,
        text_embeddings,
        clinical_features,
        text_features,
        return_attention_weights: bool = False,
    ):
        """Forward pass with optional attention weight return"""
        # Get batch size and device from any available input
        batch_size = imaging_features.shape[0]
        device = imaging_features.device

        # Encode each modality and store outputs
        imaging_encoded = self.imaging_encoder(imaging_features)
        self.last_encoder_outputs["imaging"] = imaging_encoded

        text_encoded = self._encode_text(text_embeddings, batch_size, device)
        self.last_encoder_outputs["text"] = text_encoded

        clinical_encoded = self.clinical_encoder(clinical_features)
        self.last_encoder_outputs["clinical"] = clinical_encoded

        # Project to common dimension
        imaging_proj = self.imaging_projection(imaging_encoded).unsqueeze(1)
        text_proj = self.text_projection(text_encoded).unsqueeze(1)
        clinical_proj = self.clinical_projection(clinical_encoded).unsqueeze(1)

        # Apply cross-attention if enabled
        if self.model_config.use_cross_attention:
            # Create multimodal sequences for attention
            img_text_sequence = torch.cat([imaging_proj, text_proj], dim=1)
            img_clin_sequence = torch.cat([imaging_proj, clinical_proj], dim=1)

            # Apply attention fusion
            img_text_attended = self.imaging_text_attention(img_text_sequence)
            img_clin_attended = self.imaging_clinical_attention(img_clin_sequence)

            # Store attention weights
            self.last_attention_weights["imaging_text"] = (
                self.imaging_text_attention.last_attention_weights
            )
            self.last_attention_weights["imaging_clinical"] = (
                self.imaging_clinical_attention.last_attention_weights
            )

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
                [
                    imaging_proj.squeeze(1),
                    text_proj.squeeze(1),
                    clinical_proj.squeeze(1),
                ],
                dim=1,
            )

        # Fusion layers
        fused = self.fusion_layers(fused_features)

        # Main prediction
        risk_scores = self.survival_head(fused)

        # Auxiliary outputs
        aux_outputs = {}
        if self.model_config.use_auxiliary_tasks:
            aux_outputs["event_pred"] = self.event_head(fused)

        if return_attention_weights and self.model_config.use_cross_attention:
            aux_outputs["attention_weights"] = self.last_attention_weights

        return risk_scores, aux_outputs

    def _encode_text(self, text_embeddings, batch_size=None, device=None):
        """Encode text embeddings"""
        if isinstance(text_embeddings, dict):
            # For NSCLC, we use only clinical embeddings
            if self.dataset_config.name == "NSCLC":
                if "clinical" in text_embeddings:
                    combined_text = text_embeddings["clinical"]
                else:
                    # Use provided batch size or infer from text embeddings
                    bs = batch_size or (
                        list(text_embeddings.values())[0].shape[0]
                        if text_embeddings
                        else 1
                    )
                    dev = device or (
                        next(iter(text_embeddings.values())).device
                        if text_embeddings
                        else "cpu"
                    )
                    combined_text = torch.zeros(
                        bs, self.dataset_config.text_embedding_dim, device=dev
                    )
            else:
                # For GBM/IPMN, concatenate different text embeddings with padding for missing ones
                # Use provided batch size or infer from text embeddings
                bs = batch_size or (
                    list(text_embeddings.values())[0].shape[0] if text_embeddings else 1
                )
                embed_dim = self.dataset_config.text_embedding_dim
                dev = device or (
                    next(iter(text_embeddings.values())).device
                    if text_embeddings
                    else "cpu"
                )

                if self.dataset_config.name == "GBM":
                    # GBM expects radiology, pathology, and optionally treatment
                    expected_keys = ["radiology", "pathology"]
                    if self.dataset_config.has_treatment_text:
                        expected_keys.append("treatment")
                else:  # IPMN
                    # IPMN expects radiology and pathology
                    expected_keys = ["radiology", "pathology"]

                # Create embeddings list with padding for missing ones
                text_list = []
                for key in expected_keys:
                    if key in text_embeddings:
                        text_list.append(text_embeddings[key])
                    else:
                        # Pad with zeros if embedding is missing
                        zeros = torch.zeros(bs, embed_dim, device=dev)
                        text_list.append(zeros)

                combined_text = torch.cat(text_list, dim=1)

                # Verify the shape
                if combined_text.shape[1] != len(expected_keys) * embed_dim:
                    raise ValueError(
                        f"Text embedding dimension mismatch. Expected {len(expected_keys) * embed_dim}, got {combined_text.shape[1]}"
                    )
        else:
            combined_text = text_embeddings

        return self.text_encoder(combined_text)

    def get_modality_embeddings(
        self, imaging_features, text_embeddings, clinical_features, text_features
    ) -> Dict[str, torch.Tensor]:
        """Get encoded representations for each modality"""
        # Get batch size and device
        batch_size = imaging_features.shape[0]
        device = imaging_features.device

        # Encode modalities
        imaging_encoded = self.imaging_encoder(imaging_features)
        text_encoded = self._encode_text(text_embeddings, batch_size, device)
        clinical_encoded = self.clinical_encoder(clinical_features)

        return {
            "imaging": imaging_encoded,
            "text": text_encoded,
            "clinical": clinical_encoded,
        }

    def get_fused_features(self, imaging_features, clinical_features, text_embeddings):
        """Extract fused features without final prediction layers"""
        # Get batch size and device
        batch_size = imaging_features.shape[0]
        device = imaging_features.device

        # Encode each modality
        imaging_encoded = self.imaging_encoder(imaging_features)
        text_encoded = self._encode_text(text_embeddings, batch_size, device)
        clinical_encoded = self.clinical_encoder(clinical_features)

        # Project to common dimension
        imaging_proj = self.imaging_projection(imaging_encoded).unsqueeze(1)
        text_proj = self.text_projection(text_encoded).unsqueeze(1)
        clinical_proj = self.clinical_projection(clinical_encoded).unsqueeze(1)

        # Apply cross-attention if enabled
        if self.model_config.use_cross_attention:
            # Create multimodal sequences for attention
            img_text_sequence = torch.cat([imaging_proj, text_proj], dim=1)
            img_clin_sequence = torch.cat([imaging_proj, clinical_proj], dim=1)

            # Apply attention fusion
            img_text_attended = self.imaging_text_attention(img_text_sequence)
            img_clin_attended = self.imaging_clinical_attention(img_clin_sequence)

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
                [
                    imaging_proj.squeeze(1),
                    text_proj.squeeze(1),
                    clinical_proj.squeeze(1),
                ],
                dim=1,
            )

        # Apply fusion layers
        fused = self.fusion_layers(fused_features)

        return fused
