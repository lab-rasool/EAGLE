# 📊 EAGLE Datasets Overview

The EAGLE framework includes three comprehensive cancer datasets, each with multimodal data integration combining imaging, clinical features, and text reports for survival prediction.

## 🎯 Dataset Summary

<div align="center">

| **Dataset** | **Cancer Type** | **Patients** | **Event Rate** | **Median Survival** | **Imaging** |
|-------------|-----------------|--------------|----------------|---------------------|-------------|
| [GBM](GBM.md) | Glioblastoma | 160 | 95.6% | 13.0 months | MRI |
| [IPMN](IPMN.md) | Pancreatic Cysts | 171 | 48.5% | 6.53 years | CT |
| [NSCLC](NSCLC.md) | Lung Cancer | 580 | 51.7% | 35.0 months | CT |

</div>

---

## 📈 Performance Comparison

### EAGLE Model Results

<div align="center">

| **Dataset** | **C-index** | **Best Baseline** | **Improvement** |
|-------------|-------------|-------------------|-----------------|
| GBM | 0.637 ± 0.087 | 0.589 (CoxPH w/ MedGemma) | Superior |
| IPMN | 0.679 ± 0.029 | 0.776 (RSF) | Competitive |
| NSCLC | 0.598 ± 0.021 | 0.722 (CoxPH) | Competitive |

</div>

Key Achievement: **99.96-99.98% dimensionality reduction** while maintaining competitive performance.

---

## 🧬 Multimodal Components

All datasets include three modalities:

### 1. **Imaging Data**
- **GBM**: MRI sequences (T1, T2, FLAIR, DWI)
- **IPMN**: Triple-phase CT (arterial, venous, non-contrast)
- **NSCLC**: Contrast-enhanced and non-contrast CT

### 2. **Clinical Features**
- Demographics (age, gender, race)
- Disease-specific markers
- Laboratory values
- Treatment information

### 3. **Text Reports**
- Radiology reports (78.9-100% coverage)
- Pathology reports (96.2-98.2% coverage)
- Embedded using GatorTron language model

---

## 🎯 Clinical Applications

Each dataset addresses specific clinical challenges:

### GBM - Brain Cancer
- **Challenge**: Uniformly poor prognosis requiring risk stratification
- **Application**: Identify patients for aggressive treatment vs. palliative care
- **Key Features**: Age, extent of resection, MGMT status

### IPMN - Pancreatic Cysts
- **Challenge**: Distinguish benign from malignant potential
- **Application**: Guide surveillance intervals and surgical decisions
- **Key Features**: Cyst size, main duct involvement, CA 19-9

### NSCLC - Lung Cancer
- **Challenge**: Heterogeneous disease with varied outcomes
- **Application**: Personalize treatment based on risk profile
- **Key Features**: TNM stage, histology, smoking history

---

## 📊 Dataset Statistics

### Patient Demographics

<div align="center">

| **Feature** | **GBM** | **IPMN** | **NSCLC** |
|-------------|---------|----------|-----------|
| Mean Age | 62.6 years | N/A* | 69.6 years |
| Male Ratio | 60% | N/A* | N/A* |
| Smoking History | N/A | 42.7% | 81.9% |
| Family History | N/A | 66.7% | N/A |

*Demographics not directly available in processed data

</div>

### Data Completeness

<div align="center">

| **Data Type** | **GBM** | **IPMN** | **NSCLC** |
|---------------|---------|----------|-----------|
| Imaging | 100% | 100% | 74% (contrast CT) |
| Radiology Reports | 100% | 78.9% | N/A |
| Pathology Reports | 96.2% | 98.2% | N/A |
| Clinical Features | 90.6%+ | 91.8%+ | ~53% (staging) |

</div>

---

## 🔧 Technical Specifications

### Embedding Dimensions

<div align="center">

| **Modality** | **GBM** | **IPMN** | **NSCLC** |
|--------------|---------|----------|-----------|
| Imaging | (155, 1000) | 1024 | (2, 1000) |
| Text Reports | 1024 | 1024 | N/A |
| Clinical | Variable | Variable | 1024 |
| EAGLE Output | 64 | 64 | 64 |

</div>

### File Formats
- **Storage**: Parquet format
- **Embeddings**: Pre-extracted using specialized models
- **Text**: GatorTron embeddings (biomedical language model)
- **Images**: RadImageNet features

---

## 🚀 Quick Start

### Loading a Dataset

```python
from eagle import GBM_CONFIG, IPMN_CONFIG, NSCLC_CONFIG, UnifiedPipeline

# Choose your dataset
config = GBM_CONFIG  # or IPMN_CONFIG, NSCLC_CONFIG

# Create pipeline
pipeline = UnifiedPipeline(config)

# Run analysis
results, risk_df, stats = pipeline.run(
    n_folds=5,
    enable_attribution=True
)
```

### Dataset Paths

```python
# Default paths
GBM_PATH = "data/GBM/unimodal.parquet"
IPMN_PATH = "data/IPMN/unimodal.parquet"
NSCLC_PATH = "data/NSCLC/unimodal.parquet"

# Alternative embeddings
GBM_MEDGEMMA = "data/GBM/medgemma.parquet"
IPMN_MEDGEMMA = "data/IPMN/medgemma.parquet"
NSCLC_MEDGEMMA = "data/NSCLC/medgemma.parquet"
```

---

## 📈 Research Impact

These datasets enable:

1. **Benchmarking**: Compare multimodal vs. unimodal approaches
2. **Method Development**: Test new architectures and fusion strategies
3. **Clinical Translation**: Develop deployable risk prediction tools
4. **Interpretability**: Understand which modalities drive predictions

### Published Results

Using these datasets, EAGLE demonstrates:
- Effective multimodal fusion
- Massive dimensionality reduction (>99%)
- Interpretable predictions via attribution
- Competitive performance with traditional methods

---

## 📚 Citation

When using these datasets, please cite:

```bibtex
@article{eagle2024,
  title={EAGLE: Efficient Alignment of Generalized Latent Embeddings for 
         Multimodal Survival Prediction},
  author={Your Name and Contributors},
  journal={Medical Image Analysis},
  year={2024}
}
```

---

## 🤝 Contributing

To contribute additional datasets:
1. Ensure multimodal data availability
2. Include survival outcomes
3. Follow EAGLE data format
4. Document thoroughly

See [Contributing Guidelines](../../CONTRIBUTING.md) for details.

---

For detailed information about each dataset, click on the dataset names in the table above or navigate to:
- [GBM Dataset Documentation](GBM.md)
- [IPMN Dataset Documentation](IPMN.md)
- [NSCLC Dataset Documentation](NSCLC.md)
