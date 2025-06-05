<div align="center">
  <img src="docs/EAGLE.png" alt="EAGLE Logo" width="150px" height="150px">
  
  # 🦅 EAGLE
  ## Efficient Alignment of Generalized Latent Embeddings
  
  *A unified multimodal survival prediction framework with interpretable attribution analysis*
  
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  
</div>

---

## 🎯 Overview

EAGLE is a state-of-the-art multimodal survival prediction framework that seamlessly integrates imaging, clinical, and textual data to provide robust survival predictions with interpretable attribution analysis. Built for healthcare researchers and practitioners, EAGLE offers a unified approach to handle diverse medical datasets while maintaining transparency through advanced attribution techniques.

### ✨ Key Features

- 🔬 **Multimodal Integration**: Seamlessly combines imaging (MRI, CT), clinical data, and text reports
- 🎯 **Unified Architecture**: Single framework supporting multiple datasets (GBM, IPMN, NSCLC)
- 🔍 **Attribution Analysis**: Advanced interpretability with modality-level and feature-level contributions
- 📊 **Risk Stratification**: Automatic patient stratification into risk groups with survival analysis
- 🏥 **Clinical Ready**: Designed for real-world clinical research applications
- 🚀 **Easy to Use**: Simple API with comprehensive examples and documentation

## 🏗️ Architecture

EAGLE employs a sophisticated attention-based fusion architecture:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Imaging   │    │   Clinical   │    │    Text     │
│   Encoder   │    │   Encoder    │    │   Encoder   │
└─────┬───────┘    └──────┬───────┘    └─────┬───────┘
      │                   │                  │
      └───────────────────┼──────────────────┘
                          │
                   ┌──────▼──────┐
                   │  Attention  │
                   │   Fusion    │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │  Survival   │
                   │ Prediction  │
                   └─────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/EAGLE.git
cd EAGLE

# Install dependencies
pip install -r requirements.txt

# Install EAGLE in development mode
pip install -e .
```

### Basic Usage

```python
from eagle import UnifiedPipeline, GBM_CONFIG, ModelConfig

# Configure your analysis
model_config = ModelConfig(
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100
)

# Create and run pipeline
pipeline = UnifiedPipeline(GBM_CONFIG, model_config)
results, risk_df, stats = pipeline.run(
    n_folds=5,
    n_risk_groups=3,
    enable_attribution=True
)

print(f"Mean C-index: {results['mean_cindex']:.4f}")
```

### Command Line Interface

```bash
# Run analysis on GBM dataset with attribution
python main.py --dataset GBM --analyze-attribution

# Custom configuration
python main.py --dataset NSCLC \
               --epochs 150 \
               --batch-size 24 \
               --lr 5e-5 \
               --analyze-attribution \
               --top-patients 10
```

## 📊 Supported Datasets

| Dataset | Modalities | Features | Use Case |
|---------|------------|----------|----------|
| **GBM** | MRI + Clinical + Reports | Age, Gender, Radiology/Pathology Reports | Glioblastoma Survival |
| **IPMN** | CT + Clinical + Reports | Demographics, Imaging Features | Pancreatic Cyst Analysis |
| **NSCLC** | CT + Clinical + Reports | TNM Staging, Treatment History | Lung Cancer Prognosis |

## 🔍 Attribution Analysis

EAGLE provides comprehensive interpretability through multiple attribution methods:

### Modality-Level Attribution
```python
from eagle import ModalityAttributionAnalyzer

analyzer = ModalityAttributionAnalyzer(model, dataset)
contributions = analyzer.analyze_cohort()

# Visualize modality contributions
plot_modality_contributions(risk_df, save_path="modality_contrib.png")
```

### Patient-Level Analysis
```python
# Analyze individual patient
patient_result = analyzer.analyze_patient(patient_idx=42)
plot_patient_level_attribution(patient_result, save_path="patient_analysis.png")
```

### Feature Importance
```python
# Generate comprehensive attribution report
attribution_summary = create_attribution_report(
    risk_df,
    output_dir="attribution_results/",
    dataset_name="GBM"
)
```

## 📈 Results & Visualization

EAGLE automatically generates comprehensive visualizations:

- **Kaplan-Meier Curves**: Risk-stratified survival analysis
- **ROC Curves**: Time-dependent AUC analysis  
- **Attribution Heatmaps**: Feature and modality importance
- **Risk Distribution**: Patient risk score distributions
- **Calibration Plots**: Model calibration assessment

## 🛠️ Configuration

### Dataset Configuration
```python
from eagle import DatasetConfig

custom_config = DatasetConfig(
    name="MyDataset",
    data_path="path/to/data.parquet",
    imaging_modality="MRI",
    imaging_embedding_dim=1000,
    clinical_features=["age", "gender", "stage"],
    survival_time_col="survival_months",
    event_col="status"
)
```

### Model Configuration
```python
from eagle import ModelConfig

model_config = ModelConfig(
    imaging_encoder_dims=[512, 256, 128],
    text_encoder_dims=[256, 128],
    clinical_encoder_dims=[128, 64, 32],
    fusion_dims=[256, 128, 64],
    dropout=0.35,
    batch_size=24,
    learning_rate=5e-5,
    use_cross_attention=True,
    attention_heads=8
)
```

## 📁 Project Structure

```
EAGLE/
├── eagle/
│   ├── __init__.py          # Main API
│   ├── data.py              # Data processing & datasets
│   ├── models.py            # Model architectures
│   ├── train.py             # Training utilities
│   ├── eval.py              # Evaluation & risk stratification
│   ├── attribution.py       # Attribution analysis
│   └── viz.py               # Visualization utilities
├── main.py                  # Example usage script
├── requirements.txt         # Dependencies
├── docs/                    # Documentation
└── examples/                # Usage examples
```

## 🎯 Use Cases

### Clinical Research
- **Survival Prediction**: Robust multi-modal survival modeling
- **Risk Stratification**: Identify high-risk patient populations
- **Biomarker Discovery**: Understand modality contributions

### Model Interpretability
- **Feature Attribution**: Understand which features drive predictions
- **Modality Analysis**: Compare importance of different data types
- **Patient-Level Insights**: Personalized prediction explanations

### Healthcare Applications
- **Treatment Planning**: Inform clinical decision making
- **Patient Counseling**: Provide interpretable risk assessments
- **Quality Assurance**: Validate model predictions with attributions


## 📝 Citation

If you use EAGLE in your research, please cite:

```bibtex
@article{eagle2024,
  title={EAGLE: Efficient Alignment of Generalized Latent Embeddings for Multimodal Survival Prediction},
  author={Your Name and Contributors},
  journal={Medical Image Analysis},
  year={2024},
  volume={XX},
  pages={XX-XX}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>Built with ❤️ for advancing healthcare through AI</strong>
</div>