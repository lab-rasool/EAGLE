<div align="center">
  
  <img src="docs/EAGLE.png" alt="EAGLE Logo" width="200px">
  
  # 🦅 EAGLE
  ## **E**fficient **A**lignment of **G**eneralized **L**atent **E**mbeddings
  
  <p align="center">
    <strong>A State-of-the-Art Multimodal Survival Prediction Framework</strong>
  </p>
  
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
  
  <p align="center">
    <a href="#-key-features">Features</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-documentation">Documentation</a> •
    <a href="#-citation">Citation</a>
  </p>
  
</div>

---

## 🎯 Overview

EAGLE is a multimodal deep learning framework designed for survival prediction in cancer patients. By integrating **imaging**, **clinical**, and **textual** data through attention-based fusion, EAGLE provides a survival predictions with interpretability through attribution analysis.

### 🔬 Why EAGLE?

- **🏆 State-of-the-Art Performance**: Achieves superior C-index scores across multiple cancer types
- **🔍 Interpretable AI**: Attribution analysis reveals which modalities drive predictions
- **⚡ Efficient Architecture**: 99.96% dimensionality reduction while maintaining competitive performance
- **🏥 Clinical Ready**: Designed with healthcare practitioners in mind, providing actionable insights
- **📊 Comprehensive Evaluation**: Built-in comparison with traditional survival models (RSF, CoxPH, DeepSurv)

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🧬 Multimodal Integration
- Seamless fusion of imaging embeddings (MRI/CT)
- Clinical feature processing with standardization
- Automated text feature extraction from reports
- Attention-based modality fusion

</td>
<td width="50%">

### 📈 Advanced Analytics
- Risk stratification into clinically meaningful groups
- Kaplan-Meier survival analysis
- Time-dependent AUC evaluation
- Comprehensive performance metrics

</td>
</tr>
<tr>
<td width="50%">

### 🔍 Interpretability
- Patient-level attribution analysis
- Modality contribution visualization
- Feature importance rankings
- Cohort-level insights

</td>
<td width="50%">

### 🚀 Production Ready
- Modular, extensible architecture
- Comprehensive logging and checkpointing
- Cross-validation support
- Automatic visualization generation

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 📋 Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/lab-rasool/EAGLE.git
cd EAGLE

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🎯 Basic Usage

```bash
# Run survival analysis on GBM dataset
python main.py --dataset GBM

# Enable attribution analysis for interpretability
python main.py --dataset NSCLC --comprehensive-attribution

# Run with custom configuration
python main.py --dataset IPMN \
               --epochs 150 \
               --batch-size 24 \
               --lr 5e-5 \
               --comprehensive-attribution
```

### 🔬 Advanced Usage

```bash
# Compare with baseline models
python main.py --mode baseline --dataset GBM

# Run complete analysis (EAGLE + all baselines)
python main.py --mode all --comprehensive-attribution

# Use MedGemma embeddings
python main.py --dataset NSCLC \
               --data-path data/NSCLC/medgemma.parquet
```

---

## 📚 Detailed Examples

### 🐍 Python API

```python
from eagle import UnifiedPipeline, GBM_CONFIG, ModelConfig

# Configure model
model_config = ModelConfig(
    imaging_encoder_dims=[512, 256, 128],
    clinical_encoder_dims=[128, 64, 32],
    text_encoder_dims=[256, 128],
    fusion_dims=[256, 128, 64],
    dropout=0.35,
    batch_size=32,
    learning_rate=1e-4,
    num_epochs=100
)

# Create pipeline
pipeline = UnifiedPipeline(GBM_CONFIG, model_config)

# Run analysis
results, risk_df, stats = pipeline.run(
    n_folds=5,
    n_risk_groups=3,
    enable_attribution=True
)

# Display results
print(f"Mean C-index: {results['mean_cindex']:.4f}")
print(f"Std C-index: {results['std_cindex']:.4f}")
```

### 📊 Attribution Analysis

```python
from eagle import ModalityAttributionAnalyzer

# Analyze modality contributions
analyzer = ModalityAttributionAnalyzer(model, dataset)
contributions = analyzer.analyze_cohort()

# Analyze specific patient
patient_attr = analyzer.analyze_patient(patient_idx=42)
print(f"Imaging contribution: {patient_attr['imaging']:.2%}")
print(f"Clinical contribution: {patient_attr['clinical']:.2%}")
print(f"Text contribution: {patient_attr['text']:.2%}")
```

### 🎨 Custom Dataset

```python
from eagle import DatasetConfig, UnifiedPipeline

# Define custom dataset configuration
custom_config = DatasetConfig(
    name="MyDataset",
    data_path="path/to/data.parquet",
    imaging_modality="MRI",
    imaging_embedding_dim=1000,
    clinical_features=["age", "gender", "stage", "biomarker"],
    text_columns=["radiology_report", "pathology_report"],
    survival_time_col="survival_months",
    event_col="status",
    patient_col="patient_id"
)

# Run pipeline
pipeline = UnifiedPipeline(custom_config, model_config)
results, risk_df, stats = pipeline.run()
```

---

## 📁 Project Structure

```
EAGLE/
│
├── 📂 eagle/                    # Core library
│   ├── __init__.py             # Main API and pipeline
│   ├── data.py                 # Data loading and preprocessing
│   ├── models.py               # Neural network architectures
│   ├── train.py                # Training logic
│   ├── eval.py                 # Evaluation and metrics
│   ├── attribution.py          # Interpretability analysis
│   └── viz.py                  # Visualization utilities
│
├── 📂 data/                     # Dataset directory
│   ├── GBM/                    # Glioblastoma data
│   ├── IPMN/                   # Pancreatic cyst data
│   └── NSCLC/                  # Lung cancer data
│
├── 📂 results/                  # Output directory
│   └── [Dataset]/[Timestamp]/  # Experiment results
│
├── 📄 main.py                   # CLI interface
├── 📄 requirements.txt          # Dependencies
└── 📄 README.md                # This file
```

---

## 🔧 Configuration

### Dataset Configuration

EAGLE supports three cancer datasets out of the box:

| Dataset | Cancer Type | Imaging | Key Features |
|---------|------------|---------|--------------|
| **GBM** | Glioblastoma | MRI | MGMT status, age, gender |
| **IPMN** | Pancreatic Cysts | CT | Cyst size, location, morphology |
| **NSCLC** | Lung Cancer | CT | TNM staging, histology, smoking |

### Model Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Imaging   │     │   Clinical   │     │    Text     │
│  Embeddings │     │   Features   │     │ Embeddings  │
└──────┬──────┘     └──────┬───────┘     └─────┬───────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Imaging   │     │   Clinical   │     │    Text     │
│   Encoder   │     │   Encoder    │     │   Encoder   │
└──────┬──────┘     └──────┬───────┘     └─────┬───────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Attention  │
                    │   Fusion    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Survival  │
                    │ Prediction  │
                    └─────────────┘
```

---

## 📈 Output Structure

Each experiment generates a comprehensive set of outputs:

```
results/[Dataset]/[Timestamp]/
├── 📊 figures/                 # Visualizations
│   ├── kaplan_meier_curves.png
│   ├── risk_distribution.png
│   └── risk_vs_survival.png
├── 🧠 models/                  # Trained models
│   ├── best_model_fold1.pth
│   └── ...
├── 📈 results/                 # Metrics and predictions
│   └── risk_scores.csv
├── 🔍 attribution/             # Interpretability
│   ├── modality_contributions.png
│   └── patient_attribution.csv
└── 📝 run_info.txt            # Experiment configuration
```

---

## 🛠️ Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset to use (GBM, IPMN, NSCLC) | Required |
| `--mode` | Run mode (eagle, baseline, all) | eagle |
| `--epochs` | Number of training epochs | 200 |
| `--batch-size` | Batch size for training | 16 |
| `--lr` | Learning rate | 1e-4 |
| `--comprehensive-attribution` | Enable attribution analysis | False |
| `--top-patients` | Number of patients for detailed analysis | 5 |
| `--output-dir` | Output directory | results/ |

---

## 🔬 Research Applications

EAGLE has been designed for various research applications:

- **🏥 Clinical Decision Support**: Risk stratification for treatment planning
- **🧬 Biomarker Discovery**: Understanding which features drive outcomes
- **📊 Comparative Studies**: Built-in baseline comparisons
- **🔍 Interpretable AI Research**: Advanced attribution methods
- **🎯 Precision Medicine**: Patient-specific risk assessment

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black eagle/
isort eagle/

# Run linting
flake8 eagle/

# Run tests (when available)
pytest
```

---

## 📚 Documentation

For detailed documentation, please visit our [Documentation Site](https://eagle-docs.readthedocs.io).

### Quick Links
- [API Reference](docs/api.md)
- [Tutorials](docs/tutorials/)
- [FAQ](docs/faq.md)
- [Troubleshooting](docs/troubleshooting.md)

---

## 📝 Citation

If you use EAGLE in your research, please cite our paper:

```bibtex
... Pending publication ...
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Medical imaging embeddings powered by [RadImageNet](https://www.radimagenet.com/)
- Clinical text embeddings via [GatorTron](https://www.nature.com/articles/s41746-022-00742-2)
- Multimodal embeddings from [MedGemma](https://deepmind.google/models/gemma/medgemma/)

---

<div align="center">
  <p>
    <strong>Built with ❤️ for advancing healthcare through interpretable AI</strong>
  </p>
  <p>
    <a href="https://github.com/lab-rasool/EAGLE">GitHub</a> •
    <a href="https://eagle-docs.readthedocs.io">Documentation</a> •
    <a href="https://github.com/lab-rasool/EAGLE/issues">Issues</a> •
    <a href="https://github.com/lab-rasool/EAGLE/discussions">Discussions</a>
  </p>
</div>