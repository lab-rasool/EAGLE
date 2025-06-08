# 🔬 Advanced EAGLE Usage

This tutorial covers advanced features and customization options in EAGLE.

## Table of Contents

1. [Custom Datasets](#custom-datasets)
2. [Model Architecture Customization](#model-architecture-customization)
3. [Advanced Attribution Analysis](#advanced-attribution-analysis)
4. [Performance Optimization](#performance-optimization)
5. [Ensemble Methods](#ensemble-methods)
6. [Custom Visualizations](#custom-visualizations)

## Custom Datasets

### Preparing Your Data

EAGLE expects data in parquet format with specific columns. Here's how to prepare your custom dataset:

```python
import pandas as pd
import numpy as np

# Load your data
df = pd.DataFrame({
    'PatientID': patient_ids,
    'SurvivalTime': survival_times,  # In days or months
    'Event': events,  # 0 for censored, 1 for event
    'Age': ages,
    'Gender': genders,
    # ... other clinical features
})

# Add embedding columns
# Imaging embeddings (e.g., from a pretrained CNN)
imaging_embeddings = np.load('imaging_embeddings.npy')
for i in range(imaging_embeddings.shape[1]):
    df[f'imaging_feat_{i}'] = imaging_embeddings[:, i]

# Text embeddings (e.g., from BERT/GatorTron)
text_embeddings = np.load('text_embeddings.npy')
for i in range(text_embeddings.shape[1]):
    df[f'text_feat_{i}'] = text_embeddings[:, i]

# Save as parquet
df.to_parquet('data/custom_dataset.parquet')
```

### Creating Custom Configuration

```python
from eagle import DatasetConfig, UnifiedPipeline
from eagle.data import UnifiedClinicalProcessor

# Define clinical feature processor
def custom_clinical_processor(df):
    """Custom preprocessing for clinical features"""
    # Normalize age
    df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['Stage', 'Grade'])
    
    return df

# Define text feature extractor
def custom_text_extractor(row):
    """Extract specific text features"""
    features = []
    
    # Example: Extract keywords from reports
    report = row.get('radiology_report', '')
    features.extend([
        float('tumor' in report.lower()),
        float('metastasis' in report.lower()),
        float('progression' in report.lower())
    ])
    
    return features

# Create configuration
custom_config = DatasetConfig(
    name="CustomCancer",
    data_path="data/custom_dataset.parquet",
    imaging_modality="MRI",
    imaging_embedding_dim=2048,  # Your embedding size
    clinical_features=['Age', 'Gender', 'Stage_*', 'Grade_*'],
    text_columns=['radiology_report', 'pathology_report'],
    survival_time_col='SurvivalTime',
    event_col='Event',
    patient_col='PatientID',
    clinical_preprocessor=custom_clinical_processor,
    text_extractor=custom_text_extractor
)

# Run pipeline
pipeline = UnifiedPipeline(custom_config)
results, risk_df, stats = pipeline.run()
```

## Model Architecture Customization

### Custom Encoder Networks

```python
import torch
import torch.nn as nn
from eagle import ModelConfig, UnifiedSurvivalModel

class CustomAttentionEncoder(nn.Module):
    """Custom encoder with self-attention"""
    def __init__(self, input_dim, hidden_dims, n_heads=8):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dims[i], n_heads)
            for i in range(len(hidden_dims))
        ])
        
        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i] * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dims[i] * 4, hidden_dims[i])
            )
            for i in range(len(hidden_dims))
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dims[i])
            for i in range(len(hidden_dims))
        ])
        
    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        
        # Apply attention blocks
        for attn, ff, ln in zip(self.attention_layers, 
                                self.ff_layers, 
                                self.layer_norms):
            # Self-attention with residual
            x_attn, _ = attn(x, x, x)
            x = ln(x + x_attn)
            
            # Feed-forward with residual
            x = ln(x + ff(x))
            
        return x

# Use custom encoder in model
model_config = ModelConfig()
model = UnifiedSurvivalModel(model_config)

# Replace default encoder
model.imaging_encoder = CustomAttentionEncoder(
    input_dim=2048,
    hidden_dims=[512, 256, 128]
)
```

### Custom Fusion Strategies

```python
class GatedMultimodalFusion(nn.Module):
    """Gated fusion mechanism for multimodal data"""
    def __init__(self, modality_dims, output_dim):
        super().__init__()
        
        # Gates for each modality
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.Sigmoid()
            )
            for dim in modality_dims
        ])
        
        # Transformation layers
        self.transforms = nn.ModuleList([
            nn.Linear(dim, output_dim)
            for dim in modality_dims
        ])
        
    def forward(self, *modalities):
        # Apply gating mechanism
        gated_features = []
        for modality, gate, transform in zip(modalities, 
                                            self.gates, 
                                            self.transforms):
            g = gate(modality)
            transformed = transform(modality)
            gated_features.append(g * transformed)
        
        # Combine gated features
        return sum(gated_features)
```

## Advanced Attribution Analysis

### Integrated Gradients

```python
from eagle import ModalityAttributionAnalyzer
import torch

class AdvancedAttributionAnalyzer(ModalityAttributionAnalyzer):
    """Extended attribution analyzer with integrated gradients"""
    
    def integrated_gradients(self, patient_idx, n_steps=50):
        """Compute integrated gradients for patient"""
        # Get patient data
        data = self.dataset[patient_idx]
        
        # Create baseline (zeros)
        baseline = {
            'imaging': torch.zeros_like(data['imaging']),
            'clinical': torch.zeros_like(data['clinical']),
            'text': torch.zeros_like(data['text'])
        }
        
        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, n_steps)
        gradients = []
        
        for alpha in alphas:
            # Interpolated input
            interp_input = {
                k: baseline[k] + alpha * (data[k] - baseline[k])
                for k in ['imaging', 'clinical', 'text']
            }
            
            # Forward pass with gradient
            interp_input = {k: v.requires_grad_(True) for k, v in interp_input.items()}
            output = self.model(**interp_input)
            
            # Compute gradients
            grads = torch.autograd.grad(output, interp_input.values())
            gradients.append(grads)
        
        # Average gradients and compute attribution
        avg_gradients = [
            torch.stack([g[i] for g in gradients]).mean(0)
            for i in range(3)
        ]
        
        # Compute integrated gradients
        attributions = {
            'imaging': (avg_gradients[0] * (data['imaging'] - baseline['imaging'])).sum(),
            'clinical': (avg_gradients[1] * (data['clinical'] - baseline['clinical'])).sum(),
            'text': (avg_gradients[2] * (data['text'] - baseline['text'])).sum()
        }
        
        # Normalize
        total = sum(attributions.values())
        return {k: v / total for k, v in attributions.items()}
```

### SHAP-based Attribution

```python
import shap
import numpy as np

def compute_shap_values(model, dataset, sample_size=100):
    """Compute SHAP values for model interpretation"""
    
    # Create background dataset
    background_idx = np.random.choice(len(dataset), sample_size)
    background_data = [dataset[i] for i in background_idx]
    
    # Define prediction function
    def predict_fn(inputs):
        # inputs is numpy array, convert to appropriate format
        predictions = []
        for inp in inputs:
            # Split input back into modalities
            imaging = torch.tensor(inp[:2048])  # Adjust dimensions
            clinical = torch.tensor(inp[2048:2048+10])
            text = torch.tensor(inp[2048+10:])
            
            with torch.no_grad():
                pred = model(imaging, clinical, text)
            predictions.append(pred.item())
        
        return np.array(predictions)
    
    # Create explainer
    explainer = shap.KernelExplainer(predict_fn, background_data)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(background_data)
    
    return shap_values
```

## Performance Optimization

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

class OptimizedTrainer:
    """Trainer with mixed precision support"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.scaler = GradScaler()
        
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(
                    batch['imaging'],
                    batch['clinical'],
                    batch['text']
                )
                loss = self.compute_loss(outputs, batch['time'], batch['event'])
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
```

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup_distributed(rank, world_size):
    """Setup distributed training"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_distributed(rank, world_size, dataset_config, model_config):
    setup_distributed(rank, world_size)
    
    # Create model and move to GPU
    model = UnifiedSurvivalModel(model_config)
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    # Create distributed sampler
    dataset = UnifiedSurvivalDataset(dataset_config.data_path, dataset_config)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    
    dataloader = DataLoader(dataset, sampler=sampler, 
                          batch_size=model_config.batch_size)
    
    # Train model
    trainer = UnifiedTrainer(model_config, dataset_config, f"output_rank_{rank}")
    trainer.train(dataloader)
```

## Ensemble Methods

### Model Ensemble

```python
class EnsemblePredictor:
    """Ensemble of multiple EAGLE models"""
    
    def __init__(self, model_paths, config):
        self.models = []
        for path in model_paths:
            model = UnifiedSurvivalModel(config)
            model.load_state_dict(torch.load(path))
            model.eval()
            self.models.append(model)
    
    def predict(self, imaging, clinical, text, method='mean'):
        """Ensemble prediction"""
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(imaging, clinical, text)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        if method == 'mean':
            return predictions.mean(dim=0)
        elif method == 'median':
            return predictions.median(dim=0)[0]
        elif method == 'weighted':
            # Weight by validation performance
            weights = torch.tensor([m.val_cindex for m in self.models])
            weights = weights / weights.sum()
            return (predictions * weights.view(-1, 1, 1)).sum(dim=0)
```

### Cross-Validation Ensemble

```python
def create_cv_ensemble(dataset_config, model_config, n_folds=5):
    """Create ensemble from cross-validation models"""
    
    # Train models
    pipeline = UnifiedPipeline(dataset_config, model_config)
    results, risk_df, stats = pipeline.run(n_folds=n_folds)
    
    # Load all fold models
    model_paths = [f"results/{dataset_config.name}/models/best_model_fold{i}.pth" 
                   for i in range(1, n_folds+1)]
    
    ensemble = EnsemblePredictor(model_paths, model_config)
    
    # Evaluate ensemble
    dataset = UnifiedSurvivalDataset(dataset_config.data_path, dataset_config)
    ensemble_predictions = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        pred = ensemble.predict(
            data['imaging'].unsqueeze(0),
            data['clinical'].unsqueeze(0),
            data['text'].unsqueeze(0)
        )
        ensemble_predictions.append(pred.item())
    
    return ensemble_predictions
```

## Custom Visualizations

### Interactive Visualizations

```python
import plotly.graph_objects as go
import plotly.express as px

def create_interactive_survival_plot(risk_df):
    """Create interactive Kaplan-Meier curves"""
    
    fig = go.Figure()
    
    for group in ['Low', 'Medium', 'High']:
        group_df = risk_df[risk_df['risk_group'] == group]
        
        # Compute survival function
        times, survival_prob = kaplan_meier_estimator(
            group_df['event'].values,
            group_df['survival_time'].values
        )
        
        fig.add_trace(go.Scatter(
            x=times,
            y=survival_prob,
            mode='lines',
            name=f'{group} Risk',
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title='Interactive Survival Curves by Risk Group',
        xaxis_title='Time (days)',
        yaxis_title='Survival Probability',
        hovermode='x unified'
    )
    
    return fig

def create_3d_risk_visualization(risk_df, stats):
    """3D visualization of risk scores with modality contributions"""
    
    fig = px.scatter_3d(
        risk_df,
        x='imaging_contribution',
        y='clinical_contribution',
        z='text_contribution',
        color='risk_score',
        size='survival_time',
        hover_data=['PatientID', 'event'],
        title='3D Risk Score Visualization'
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Imaging Contribution',
            yaxis_title='Clinical Contribution',
            zaxis_title='Text Contribution'
        )
    )
    
    return fig
```

### Custom Attribution Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

def create_feature_importance_heatmap(attribution_results, top_k=20):
    """Create detailed feature importance heatmap"""
    
    # Extract feature-level attributions
    feature_importance = {}
    
    for patient_id, attrs in attribution_results.items():
        for feature, importance in attrs['features'].items():
            if feature not in feature_importance:
                feature_importance[feature] = []
            feature_importance[feature].append(importance)
    
    # Average importance per feature
    avg_importance = {
        feat: np.mean(imps) for feat, imps in feature_importance.items()
    }
    
    # Select top features
    top_features = sorted(avg_importance.items(), 
                         key=lambda x: abs(x[1]), 
                         reverse=True)[:top_k]
    
    # Create heatmap data
    heatmap_data = []
    for patient_id in list(attribution_results.keys())[:50]:  # First 50 patients
        patient_data = []
        for feat, _ in top_features:
            patient_data.append(
                attribution_results[patient_id]['features'].get(feat, 0)
            )
        heatmap_data.append(patient_data)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        np.array(heatmap_data).T,
        xticklabels=[f'P{i}' for i in range(len(heatmap_data))],
        yticklabels=[feat for feat, _ in top_features],
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Feature Importance'}
    )
    plt.title('Feature Importance Across Patients')
    plt.tight_layout()
    
    return plt.gcf()
```

## Advanced Configuration

### Hyperparameter Optimization

```python
import optuna
from functools import partial

def objective(trial, dataset_config):
    """Optuna objective for hyperparameter tuning"""
    
    # Suggest hyperparameters
    model_config = ModelConfig(
        learning_rate=trial.suggest_loguniform('lr', 1e-5, 1e-3),
        dropout=trial.suggest_uniform('dropout', 0.1, 0.5),
        batch_size=trial.suggest_categorical('batch_size', [8, 16, 32]),
        imaging_encoder_dims=[
            trial.suggest_int('img_dim1', 128, 1024, step=128),
            trial.suggest_int('img_dim2', 64, 512, step=64),
            trial.suggest_int('img_dim3', 32, 256, step=32)
        ]
    )
    
    # Run training
    pipeline = UnifiedPipeline(dataset_config, model_config)
    results, _, _ = pipeline.run(n_folds=3)  # Fewer folds for speed
    
    return results['mean_cindex']

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(
    partial(objective, dataset_config=GBM_CONFIG),
    n_trials=50
)

print(f"Best parameters: {study.best_params}")
print(f"Best C-index: {study.best_value:.3f}")
```

## Production Deployment

### Model Export

```python
def export_model_for_production(model_path, output_path):
    """Export model for production deployment"""
    
    # Load model
    model = UnifiedSurvivalModel(model_config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create example input
    example_input = {
        'imaging': torch.randn(1, 2048),
        'clinical': torch.randn(1, 10),
        'text': torch.randn(1, 768)
    }
    
    # Trace model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save
    traced_model.save(output_path)
    
    # Also save configuration
    import json
    config_dict = {
        'model_config': model_config.__dict__,
        'dataset_config': dataset_config.__dict__
    }
    with open(output_path.replace('.pt', '_config.json'), 'w') as f:
        json.dump(config_dict, f)
```

### Batch Inference

```python
class BatchInferenceEngine:
    """Efficient batch inference for production"""
    
    def __init__(self, model_path, batch_size=100):
        self.model = torch.jit.load(model_path)
        self.batch_size = batch_size
        
    def predict_batch(self, data_path, output_path):
        """Run batch predictions on new data"""
        
        # Load data
        df = pd.read_parquet(data_path)
        
        # Process in batches
        predictions = []
        
        for i in range(0, len(df), self.batch_size):
            batch_df = df.iloc[i:i+self.batch_size]
            
            # Extract features
            imaging = torch.tensor(
                batch_df[[f'imaging_feat_{j}' for j in range(2048)]].values
            )
            clinical = torch.tensor(
                batch_df[['age', 'gender', 'stage']].values
            )
            text = torch.tensor(
                batch_df[[f'text_feat_{j}' for j in range(768)]].values
            )
            
            # Predict
            with torch.no_grad():
                batch_pred = self.model(imaging, clinical, text)
            
            predictions.extend(batch_pred.cpu().numpy())
        
        # Save results
        df['risk_score'] = predictions
        df[['PatientID', 'risk_score']].to_csv(output_path, index=False)
```

## Conclusion

These advanced techniques enable you to:
- Adapt EAGLE to any survival prediction task
- Optimize performance for your specific use case
- Deploy models in production environments
- Gain deeper insights through advanced attribution

For more examples and updates, visit the [EAGLE GitHub repository](https://github.com/lab-rasool/EAGLE).