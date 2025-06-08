# 🦅 EAGLE API Reference

This document provides a comprehensive reference for the EAGLE (Efficient Alignment of Generalized Latent Embeddings) API.

## Table of Contents

- [Core Classes](#core-classes)
  - [UnifiedPipeline](#unifiedpipeline)
  - [UnifiedSurvivalModel](#unifiedsurvivalmodel)
  - [UnifiedSurvivalDataset](#unifiedsurvivaldataset)
- [Configuration Classes](#configuration-classes)
  - [DatasetConfig](#datasetconfig)
  - [ModelConfig](#modelconfig)
- [Training & Evaluation](#training--evaluation)
  - [UnifiedTrainer](#unifiedtrainer)
  - [UnifiedRiskStratification](#unifiedriskstratification)
- [Attribution Analysis](#attribution-analysis)
  - [ModalityAttributionAnalyzer](#modalityattributionanalyzer)
- [Visualization](#visualization)
- [Utility Functions](#utility-functions)

---

## Core Classes

### UnifiedPipeline

The main entry point for running EAGLE experiments.

```python
eagle.UnifiedPipeline(dataset_config, model_config=None)
```

**Parameters:**
- `dataset_config` (DatasetConfig): Configuration for the dataset
- `model_config` (ModelConfig, optional): Model configuration. If None, uses default settings

**Methods:**

#### run()
```python
run(n_folds=5, n_risk_groups=3, enable_attribution=False, top_patients=5)
```

Runs the complete pipeline including training, evaluation, and visualization.

**Parameters:**
- `n_folds` (int): Number of cross-validation folds
- `n_risk_groups` (int): Number of risk groups for stratification
- `enable_attribution` (bool): Whether to perform attribution analysis
- `top_patients` (int): Number of top/bottom patients to analyze

**Returns:**
- `results` (dict): Dictionary containing performance metrics
- `risk_df` (pd.DataFrame): DataFrame with patient risk scores and groups
- `attribution_stats` (dict): Attribution analysis results (if enabled)

**Example:**
```python
from eagle import UnifiedPipeline, GBM_CONFIG

pipeline = UnifiedPipeline(GBM_CONFIG)
results, risk_df, stats = pipeline.run(enable_attribution=True)
```

---

### UnifiedSurvivalModel

The core neural network model for multimodal survival prediction.

```python
eagle.models.UnifiedSurvivalModel(config)
```

**Parameters:**
- `config` (ModelConfig): Model configuration object

**Architecture:**
- Separate encoders for imaging, clinical, and text modalities
- Attention-based fusion mechanism
- Cox proportional hazards output layer

**Methods:**

#### forward()
```python
forward(imaging_features, clinical_features, text_features, return_attention=False)
```

**Parameters:**
- `imaging_features` (torch.Tensor): Imaging embeddings [batch_size, imaging_dim]
- `clinical_features` (torch.Tensor): Clinical features [batch_size, clinical_dim]
- `text_features` (torch.Tensor): Text embeddings [batch_size, text_dim]
- `return_attention` (bool): Whether to return attention weights

**Returns:**
- `risk_scores` (torch.Tensor): Predicted risk scores [batch_size, 1]
- `attention_weights` (dict, optional): Attention weights for each modality

---

### UnifiedSurvivalDataset

PyTorch dataset for loading multimodal survival data.

```python
eagle.data.UnifiedSurvivalDataset(data_path, config, indices=None)
```

**Parameters:**
- `data_path` (str): Path to the parquet file
- `config` (DatasetConfig): Dataset configuration
- `indices` (list, optional): Subset of indices to use

**Methods:**

#### \_\_getitem\_\_()
Returns a dictionary containing:
- `imaging`: Imaging embeddings
- `clinical`: Processed clinical features
- `text`: Text embeddings
- `time`: Survival time
- `event`: Event indicator
- `patient_id`: Patient identifier

---

## Configuration Classes

### DatasetConfig

Configuration for dataset-specific settings.

```python
@dataclass
class DatasetConfig:
    name: str
    data_path: str
    imaging_modality: str
    imaging_embedding_dim: int
    clinical_features: List[str]
    text_columns: List[str]
    survival_time_col: str
    event_col: str
    patient_col: str = 'PatientID'
    clinical_preprocessor: Optional[Callable] = None
    text_extractor: Optional[Callable] = None
```

**Pre-defined Configurations:**
- `GBM_CONFIG`: Glioblastoma dataset configuration
- `IPMN_CONFIG`: Pancreatic cyst dataset configuration
- `NSCLC_CONFIG`: Lung cancer dataset configuration
- `GBM_MEDGEMMA_CONFIG`: GBM with MedGemma embeddings
- `IPMN_MEDGEMMA_CONFIG`: IPMN with MedGemma embeddings
- `NSCLC_MEDGEMMA_CONFIG`: NSCLC with MedGemma embeddings

**Example:**
```python
from eagle import DatasetConfig

custom_config = DatasetConfig(
    name="MyDataset",
    data_path="path/to/data.parquet",
    imaging_modality="MRI",
    imaging_embedding_dim=1000,
    clinical_features=["age", "gender", "stage"],
    text_columns=["report1", "report2"],
    survival_time_col="survival_months",
    event_col="death_status"
)
```

---

### ModelConfig

Configuration for model architecture and training.

```python
@dataclass
class ModelConfig:
    # Architecture
    imaging_encoder_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    clinical_encoder_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    text_encoder_dims: List[int] = field(default_factory=lambda: [256, 128])
    fusion_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 200
    patience: int = 20
    
    # Regularization
    dropout: float = 0.3
    weight_decay: float = 1e-4
    
    # Advanced options
    use_batch_norm: bool = True
    use_cross_attention: bool = False
    attention_heads: int = 4
    auxiliary_tasks: bool = False
```

**Example:**
```python
from eagle import ModelConfig

config = ModelConfig(
    imaging_encoder_dims=[1024, 512, 256],
    learning_rate=5e-5,
    num_epochs=150,
    dropout=0.4
)
```

---

## Training & Evaluation

### UnifiedTrainer

Handles the training process with cross-validation.

```python
eagle.train.UnifiedTrainer(model_config, dataset_config, output_dir)
```

**Methods:**

#### train_fold()
```python
train_fold(train_dataset, val_dataset, fold)
```

Trains a single fold and returns the best model.

#### cross_validate()
```python
cross_validate(dataset, n_folds=5)
```

Performs k-fold cross-validation and returns results for all folds.

---

### UnifiedRiskStratification

Performs risk stratification and evaluation.

```python
eagle.eval.UnifiedRiskStratification(models, dataset, config)
```

**Methods:**

#### stratify_patients()
```python
stratify_patients(n_groups=3)
```

Stratifies patients into risk groups based on predicted scores.

**Returns:**
- `risk_df` (pd.DataFrame): DataFrame with columns:
  - `PatientID`: Patient identifier
  - `risk_score`: Predicted risk score
  - `risk_group`: Assigned risk group
  - `survival_time`: Actual survival time
  - `event`: Event indicator

#### evaluate()
```python
evaluate()
```

Computes evaluation metrics including C-index.

---

## Attribution Analysis

### ModalityAttributionAnalyzer

Analyzes the contribution of each modality to predictions.

```python
eagle.attribution.ModalityAttributionAnalyzer(model, dataset, device='cuda')
```

**Methods:**

#### analyze_patient()
```python
analyze_patient(patient_idx, method='attention')
```

Analyzes modality contributions for a specific patient.

**Parameters:**
- `patient_idx` (int): Index of the patient to analyze
- `method` (str): Attribution method ('attention', 'gradient', 'integrated_gradients')

**Returns:**
- Dictionary with modality contributions (sum to 1.0)

#### analyze_cohort()
```python
analyze_cohort(sample_size=None, method='attention')
```

Analyzes average modality contributions across the cohort.

**Returns:**
- Dictionary with average contributions for each modality

#### create_attribution_report()
```python
create_attribution_report(risk_df, output_dir, top_k=5)
```

Creates a comprehensive attribution report with visualizations.

---

## Visualization

### Core Visualization Functions

```python
from eagle.viz import (
    plot_survival_curves,
    plot_risk_distribution,
    plot_modality_contributions,
    plot_patient_level_attribution,
    create_comprehensive_visualization
)
```

#### plot_survival_curves()
```python
plot_survival_curves(risk_df, save_path=None)
```

Creates Kaplan-Meier survival curves for each risk group.

#### plot_risk_distribution()
```python
plot_risk_distribution(risk_df, save_path=None)
```

Plots the distribution of risk scores.

#### plot_modality_contributions()
```python
plot_modality_contributions(attribution_df, save_path=None)
```

Visualizes modality contributions across patients.

#### create_comprehensive_visualization()
```python
create_comprehensive_visualization(risk_df, output_dir)
```

Creates all standard visualizations in the specified directory.

---

## Utility Functions

### Data Processing

```python
from eagle.data import (
    load_and_preprocess_data,
    create_stratified_folds,
    normalize_clinical_features
)
```

#### load_and_preprocess_data()
```python
load_and_preprocess_data(config)
```

Loads data according to the dataset configuration.

#### create_stratified_folds()
```python
create_stratified_folds(dataset, n_folds=5, random_state=42)
```

Creates stratified cross-validation folds based on event status.

### Model Utilities

```python
from eagle.models import (
    load_checkpoint,
    save_checkpoint,
    count_parameters
)
```

#### load_checkpoint()
```python
load_checkpoint(path, model, device='cuda')
```

Loads a model checkpoint from disk.

#### save_checkpoint()
```python
save_checkpoint(model, path, epoch, optimizer_state=None)
```

Saves a model checkpoint to disk.

---

## Example: Complete Workflow

```python
from eagle import (
    UnifiedPipeline, 
    DatasetConfig, 
    ModelConfig,
    ModalityAttributionAnalyzer
)

# 1. Configure dataset
dataset_config = DatasetConfig(
    name="CustomCancer",
    data_path="data/custom_cancer.parquet",
    imaging_modality="CT",
    imaging_embedding_dim=2048,
    clinical_features=["age", "sex", "stage", "grade"],
    text_columns=["radiology_report"],
    survival_time_col="os_months",
    event_col="death"
)

# 2. Configure model
model_config = ModelConfig(
    imaging_encoder_dims=[1024, 512, 256],
    clinical_encoder_dims=[64, 32],
    learning_rate=5e-5,
    dropout=0.35
)

# 3. Run pipeline
pipeline = UnifiedPipeline(dataset_config, model_config)
results, risk_df, stats = pipeline.run(
    n_folds=5,
    enable_attribution=True
)

# 4. Analyze results
print(f"C-index: {results['mean_cindex']:.3f} ± {results['std_cindex']:.3f}")
print(f"High-risk group size: {(risk_df['risk_group'] == 'High').sum()}")
print(f"Median survival by group:")
for group in ['Low', 'Medium', 'High']:
    median_surv = risk_df[risk_df['risk_group'] == group]['survival_time'].median()
    print(f"  {group}: {median_surv:.1f} months")
```

---

## Advanced Topics

### Custom Encoders

To implement a custom encoder:

```python
import torch.nn as nn

class CustomImageEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)
```

### Custom Loss Functions

To use a custom survival loss:

```python
def custom_survival_loss(risk_scores, times, events):
    # Implement your custom loss
    return loss

# Use in model configuration
model_config.loss_function = custom_survival_loss
```

### Batch Processing

For large datasets:

```python
from eagle import batch_process_patients

results = batch_process_patients(
    model_path="path/to/model.pth",
    data_path="path/to/large_dataset.parquet",
    batch_size=100,
    output_path="predictions.csv"
)
```

---

## Performance Tips

1. **GPU Memory**: For large embedding dimensions, reduce batch size
2. **Training Speed**: Use mixed precision training with `torch.cuda.amp`
3. **Attribution Analysis**: For large cohorts, use sampling with `sample_size` parameter
4. **Cross-validation**: Use fewer folds (3 instead of 5) for faster experimentation

---

## Troubleshooting

See the [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.