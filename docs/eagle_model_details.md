# EAGLE Model Architecture - Detailed Documentation

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Input Processing](#input-processing)
4. [Encoder Architectures](#encoder-architectures)
5. [Attention Fusion Mechanism](#attention-fusion-mechanism)
6. [Loss Functions and Objectives](#loss-functions-and-objectives)
7. [Training Pipeline](#training-pipeline)
8. [Attribution Analysis](#attribution-analysis)
9. [Mathematical Formulations](#mathematical-formulations)
10. [Implementation Details](#implementation-details)
11. [Configuration and Hyperparameters](#configuration-and-hyperparameters)

## Overview

EAGLE (Efficient Alignment of Generalized Latent Embeddings) is a sophisticated multimodal deep learning framework designed for cancer patient survival prediction. The model integrates three distinct data modalities:

- **Imaging data**: Pre-extracted embeddings from medical images (CT/MRI)
- **Clinical data**: Structured patient features (demographics, lab values, staging)
- **Text data**: Natural language reports (radiology, pathology, treatment notes)

The framework employs attention-based fusion mechanisms to learn cross-modal interactions and generate unified patient representations for survival analysis.

## Model Architecture

### High-Level Architecture

```
Inputs:
├── Imaging Embeddings (1024/2048 dim)
├── Text Embeddings (768×N dim)
├── Clinical Features (variable dim)
└── Text Features (dataset-specific)
    ↓
Modality-Specific Encoders:
├── Imaging Encoder → [512, 256, 128]
├── Text Encoder → [512, 256, 128]
└── Clinical Encoder → [64, 32]
    ↓
Feature Projection Layer (to common dimension)
    ↓
Attention-Based Fusion:
├── Imaging-Text Attention
└── Imaging-Clinical Attention
    ↓
Fusion Network → [256, 128, 64]
    ↓
Output Heads:
├── Survival Head → Risk Score
└── Event Head → Binary Event Prediction
```

### Core Components

#### 1. **ModelConfig Dataclass**
Defines all architectural and training hyperparameters:

```python
@dataclass
class ModelConfig:
    # Architecture
    imaging_encoder_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    text_encoder_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    clinical_encoder_dims: List[int] = field(default_factory=lambda: [64, 32])
    fusion_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # Fusion settings
    use_attention_fusion: bool = True
    attention_heads: int = 8
    
    # Training parameters
    dropout: float = 0.3
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    early_stopping_patience: int = 15
    
    # Loss weights
    auxiliary_loss_weight: float = 0.1
```

#### 2. **UnifiedSurvivalModel Class**
The main model class that orchestrates all components:

```python
class UnifiedSurvivalModel(nn.Module):
    def __init__(self, config: ModelConfig, dataset_name: str):
        # Initialize encoders for each modality
        # Set up attention fusion modules
        # Create output heads
```

## Input Processing

### Dataset-Specific Input Dimensions

#### NSCLC (Non-Small Cell Lung Cancer)
- **Imaging**: 2048 dimensions (1024 contrast CT + 1024 non-contrast CT)
- **Text Embeddings**: 768 dimensions (clinical notes only)
- **Clinical Features**: Variable (staging, demographics, lab values)

#### GBM (Glioblastoma)
- **Imaging**: 1024 dimensions (MRI embeddings)
- **Text Embeddings**: 
  - With treatment notes: 2304 dimensions (768×3 modalities)
  - Without treatment notes: 1536 dimensions (768×2 modalities)
- **Clinical Features**: Variable (age, MGMT status, extent of resection)

#### IPMN (Intraductal Papillary Mucinous Neoplasm)
- **Imaging**: 1024 dimensions (CT embeddings)
- **Text Embeddings**: 1536 dimensions (768×2 for radiology + pathology)
- **Clinical Features**: Variable (cyst characteristics, lab values)

### Feature Preprocessing

1. **Normalization**: All inputs are normalized to have zero mean and unit variance
2. **Missing Value Handling**: 
   - Clinical features: Imputed with median/mode
   - Text embeddings: Zero-filled for missing reports
3. **Batch Processing**: Dynamic batching with padding for variable-length sequences

## Encoder Architectures

### Imaging Encoder

```python
def build_imaging_encoder(input_dim, hidden_dims, dropout):
    layers = []
    in_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        in_dim = hidden_dim
    
    return nn.Sequential(*layers)
```

**Architecture Details:**
- Input: Pre-extracted embeddings from RadImageNet or similar
- Hidden layers: [512, 256, 128] with BatchNorm and Dropout
- Activation: ReLU throughout
- Output: 128-dimensional representation

### Text Encoder

**Architecture Details:**
- Input: Concatenated embeddings from multiple text sources
- Processing: Similar to imaging encoder but handles higher input dimensions
- Hidden layers: [512, 256, 128]
- Special handling for variable number of text modalities

### Clinical Encoder

**Architecture Details:**
- Input: Tabular clinical features (normalized)
- Smaller architecture: [64, 32] due to lower input dimensionality
- Includes feature-specific preprocessing based on dataset

## Attention Fusion Mechanism

### AttentionFusion Module

```python
class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
```

### Fusion Process

1. **Feature Projection**: All modality embeddings projected to common dimension
2. **Sequence Formation**: Features arranged as sequences for attention
3. **Cross-Modal Attention**: 
   ```python
   # Imaging-Text Attention
   imaging_seq = imaging_features.unsqueeze(1)  # [B, 1, D]
   text_seq = text_features.unsqueeze(1)        # [B, 1, D]
   combined_seq = torch.cat([imaging_seq, text_seq], dim=1)  # [B, 2, D]
   
   attended_features, attention_weights = self.attention(
       combined_seq, combined_seq, combined_seq
   )
   ```
4. **Residual Connection**: Original features added back
5. **Pooling**: Mean pooling over sequence dimension

### Attention Weight Storage

- Attention weights stored for attribution analysis
- Accessible via `model.imaging_text_attention_weights`
- Shape: [batch_size, num_heads, seq_len, seq_len]

## Loss Functions and Objectives

### Primary Loss: Cox Partial Likelihood

The main survival prediction loss based on Cox proportional hazards model:

```python
def cox_partial_likelihood_loss(risk_scores, survival_times, events):
    # Sort by survival time
    sorted_indices = torch.argsort(survival_times, descending=True)
    sorted_risk = risk_scores[sorted_indices]
    sorted_events = events[sorted_indices]
    
    # Compute log partial likelihood
    max_risk = sorted_risk.max()
    risk_exp = torch.exp(sorted_risk - max_risk)
    
    log_likelihood = []
    for i in range(len(sorted_risk)):
        if sorted_events[i] == 1:
            risk_sum = risk_exp[i:].sum()
            log_lik = sorted_risk[i] - max_risk - torch.log(risk_sum)
            log_likelihood.append(log_lik)
    
    # Negative log likelihood
    loss = -torch.stack(log_likelihood).mean()
    return loss
```

**Key Properties:**
- Handles censored data appropriately
- Numerically stable with max-risk normalization
- Focuses on ranking patients by risk

### Auxiliary Loss: Event Prediction

Binary cross-entropy loss for predicting event occurrence:

```python
event_loss = F.binary_cross_entropy(
    event_predictions, 
    events.float(), 
    reduction='mean'
)
```

### Combined Loss

```python
total_loss = cox_loss + config.auxiliary_loss_weight * event_loss
```

Default auxiliary weight: 0.1

## Training Pipeline

### Training Loop (UnifiedTrainer)

```python
class UnifiedTrainer:
    def train_epoch(self, model, dataloader, optimizer):
        model.train()
        epoch_losses = []
        
        for batch in dataloader:
            # Forward pass
            risk_scores, event_preds = model(
                batch['imaging_features'],
                batch['text_embeddings'],
                batch['clinical_features'],
                batch['text_features']
            )
            
            # Compute losses
            cox_loss = cox_partial_likelihood_loss(
                risk_scores, 
                batch['survival_time'], 
                batch['event']
            )
            event_loss = F.binary_cross_entropy(
                event_preds, 
                batch['event'].float()
            )
            
            # Combined loss
            total_loss = cox_loss + 0.1 * event_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
        return np.mean(epoch_losses)
```

### Training Configuration

1. **Optimizer**: AdamW with weight decay
   ```python
   optimizer = torch.optim.AdamW(
       model.parameters(),
       lr=config.learning_rate,
       weight_decay=config.weight_decay
   )
   ```

2. **Learning Rate Scheduler**: ReduceLROnPlateau
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, 
       mode='max',
       patience=5,
       factor=0.5
   )
   ```

3. **Early Stopping**: Based on validation C-index
   - Patience: 15 epochs (default)
   - Saves best model checkpoint

4. **Gradient Clipping**: Max norm = 1.0 for stability

### Cross-Validation

- **Strategy**: 5-fold stratified cross-validation
- **Stratification**: By event status to maintain class balance
- **Model Saving**: Best model saved for each fold

## Attribution Analysis

### Modality Attribution Methods

#### 1. **Attention-Based Attribution**
```python
def get_attention_attribution(self):
    # Extract attention weights
    imaging_text_weights = self.imaging_text_attention_weights
    imaging_clinical_weights = self.imaging_clinical_attention_weights
    
    # Average across heads and normalize
    modality_importance = {
        'imaging': weights[:, :, 0, :].mean(),
        'text': weights[:, :, 1, :].mean(),
        'clinical': weights[:, :, 2, :].mean()
    }
```

#### 2. **Gradient-Based Attribution**
```python
def compute_gradient_attribution(model, inputs, target_idx):
    model.eval()
    inputs.requires_grad_(True)
    
    # Forward pass
    output = model(*inputs)
    
    # Compute gradients
    output[target_idx].backward()
    
    # Attribution scores
    attribution = inputs.grad.abs().mean(dim=-1)
    return attribution
```

#### 3. **Integrated Gradients**
Provides more robust attribution by integrating gradients along a path:

```python
def integrated_gradients(model, inputs, baseline, steps=50):
    # Create interpolated inputs
    alphas = torch.linspace(0, 1, steps)
    
    integrated_grads = 0
    for alpha in alphas:
        interpolated = baseline + alpha * (inputs - baseline)
        interpolated.requires_grad_(True)
        
        output = model(interpolated)
        output.backward()
        
        integrated_grads += interpolated.grad
    
    # Average and scale
    attribution = (inputs - baseline) * integrated_grads / steps
    return attribution
```

### Patient-Level Analysis

The framework provides detailed patient-level attribution:

1. **Top/Bottom Patient Identification**: Based on risk scores
2. **Modality Contribution Analysis**: For each patient
3. **Feature Importance**: Within each modality
4. **Visualization**: Heatmaps and bar charts

## Mathematical Formulations

### Cox Proportional Hazards Model

The hazard function for patient i at time t:

```
h_i(t) = h_0(t) × exp(β^T x_i)
```

Where:
- h_0(t): Baseline hazard function
- β: Model parameters (learned)
- x_i: Patient i's features

### Partial Likelihood

```
L(β) = ∏_{i:δ_i=1} [exp(β^T x_i) / Σ_{j∈R_i} exp(β^T x_j)]
```

Where:
- δ_i: Event indicator (1 if event occurred)
- R_i: Risk set at time t_i (patients still at risk)

### Negative Log Partial Likelihood (Loss)

```
Loss = -Σ_{i:δ_i=1} [β^T x_i - log(Σ_{j∈R_i} exp(β^T x_j))]
```

### Concordance Index (C-index)

Evaluation metric for survival models:

```
C-index = P(risk_i > risk_j | T_i < T_j, δ_i = 1)
```

Measures the probability that the model correctly ranks patient risk.

### Attention Mechanism

Multi-head attention computation:

```
Attention(Q,K,V) = softmax(QK^T / √d_k) × V
```

Where:
- Q: Query matrix
- K: Key matrix  
- V: Value matrix
- d_k: Key dimension (for scaling)

## Implementation Details

### Memory Optimization

1. **Gradient Accumulation**: For large batch training
2. **Mixed Precision Training**: FP16 computation where applicable
3. **Efficient Data Loading**: 
   - Pre-computed embeddings stored in parquet format
   - Lazy loading for large datasets

### Computational Efficiency

1. **Batch Processing**: Dynamic batching for variable-length inputs
2. **Parallel Encoding**: Modalities processed independently
3. **Cached Embeddings**: Reuse computed embeddings during evaluation

### Code Organization

```
eagle/
├── models.py          # Model architecture definitions
├── train.py           # Training logic and utilities
├── data.py            # Dataset and data processing
├── eval.py            # Evaluation and metrics
├── attribution.py     # Attribution analysis
├── viz.py            # Visualization utilities
└── __init__.py       # Main pipeline orchestration
```

### Key Design Patterns

1. **Modular Architecture**: Each component is self-contained
2. **Configuration-Driven**: All hyperparameters in config objects
3. **Dataset Abstraction**: Unified interface for different cancers
4. **Reproducibility**: Fixed seeds and deterministic operations

## Configuration and Hyperparameters

### Default Configuration

```python
default_config = ModelConfig(
    # Architecture
    imaging_encoder_dims=[512, 256, 128],
    text_encoder_dims=[512, 256, 128],
    clinical_encoder_dims=[64, 32],
    fusion_dims=[256, 128, 64],
    
    # Attention
    use_attention_fusion=True,
    attention_heads=8,
    
    # Training
    dropout=0.3,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_epochs=100,
    early_stopping_patience=15,
    
    # Loss
    auxiliary_loss_weight=0.1
)
```

### Dataset-Specific Configurations

#### NSCLC Configuration
- Higher dropout (0.35) for regularization
- Smaller batch size (24) due to memory constraints
- Lower learning rate (5e-5) for stability
- Adapted to handle dual CT inputs

#### IPMN Configuration
- Smaller encoder architectures ([256, 128])
- Optimized for smaller dataset size
- Special handling for cyst-specific features

#### GBM Configuration
- Standard configuration works well
- Optional treatment note integration
- MGMT status as key clinical feature

### Hyperparameter Tuning Recommendations

1. **Learning Rate**: Start with 1e-4, reduce if unstable
2. **Dropout**: 0.2-0.4 range, higher for smaller datasets
3. **Batch Size**: Limited by GPU memory and dataset size
4. **Architecture Depth**: Deeper for complex relationships
5. **Attention Heads**: 4-8 heads typically sufficient

### Performance Optimization Settings

```python
# For faster training
fast_config = ModelConfig(
    imaging_encoder_dims=[256, 128],
    text_encoder_dims=[256, 128],
    clinical_encoder_dims=[32],
    fusion_dims=[128, 64],
    batch_size=64,
    max_epochs=50
)

# For best performance
best_config = ModelConfig(
    imaging_encoder_dims=[512, 384, 256, 128],
    text_encoder_dims=[512, 384, 256, 128],
    clinical_encoder_dims=[128, 64, 32],
    fusion_dims=[384, 256, 128, 64],
    attention_heads=12,
    dropout=0.25,
    learning_rate=5e-5,
    max_epochs=200
)
```

## Advanced Features

### Multi-Task Learning
The model simultaneously learns:
1. **Primary Task**: Survival time ranking (Cox regression)
2. **Auxiliary Task**: Binary event prediction
3. **Implicit Tasks**: Modality alignment through attention

### Ensemble Predictions
- Models from each CV fold are saved
- Final predictions can use ensemble averaging
- Uncertainty estimation through prediction variance

### Extension Points
The architecture supports easy extensions:
1. **New Modalities**: Add new encoder branches
2. **Different Fusion Methods**: Replace attention module
3. **Additional Auxiliary Tasks**: Add more output heads
4. **Custom Loss Functions**: Modify training objectives

### Future Enhancements
Potential improvements being considered:
1. **Transformer-based encoders** for better feature extraction
2. **Contrastive learning** for modality alignment
3. **Uncertainty quantification** for risk predictions
4. **Time-dependent covariates** support
5. **Multi-event modeling** for competing risks