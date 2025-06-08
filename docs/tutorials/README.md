# 📚 EAGLE Tutorials

Welcome to the EAGLE tutorials! These guides will help you master the framework from basics to advanced usage.

## 🎯 Tutorial Overview

### 1. [Getting Started](getting_started.md)
**For beginners and first-time users**
- Installation and setup
- Running your first analysis
- Understanding outputs
- Basic Python API usage
- Attribution analysis basics

**Time:** 20-30 minutes

### 2. [Advanced Usage](advanced_usage.md)
**For experienced users and researchers**
- Custom dataset preparation
- Model architecture customization
- Advanced attribution techniques
- Performance optimization
- Ensemble methods
- Production deployment

**Time:** 45-60 minutes

## 🗺️ Learning Path

```mermaid
graph LR
    A[Installation] --> B[First Analysis]
    B --> C[Understanding Results]
    C --> D[Attribution Analysis]
    D --> E[Custom Datasets]
    E --> F[Model Customization]
    F --> G[Advanced Features]
```

## 💡 Quick Tips

1. **Start with the provided datasets** (GBM, IPMN, NSCLC) before using custom data
2. **Always enable attribution analysis** for clinical interpretability
3. **Use the Python API** for more control than command-line interface
4. **Save your configurations** for reproducible experiments
5. **Check the visualizations** - they provide intuitive insights

## 📊 Example Datasets

EAGLE comes with three pre-configured cancer datasets:

| Dataset | Type | Features | Best For |
|---------|------|----------|----------|
| **GBM** | Brain Cancer | MRI + Clinical + Reports | Learning basics |
| **IPMN** | Pancreatic | CT + Clinical + Reports | Risk stratification |
| **NSCLC** | Lung Cancer | CT + Clinical + Reports | Survival prediction |

## 🛠️ Common Workflows

### Research Workflow
1. Start with baseline comparison (`--mode all`)
2. Analyze attribution for best model
3. Iterate on model configuration
4. Generate publication-ready figures

### Clinical Workflow
1. Train on your cohort data
2. Enable attribution analysis
3. Examine high-risk patients
4. Validate with clinical outcomes

### Development Workflow
1. Create custom dataset configuration
2. Implement custom encoders if needed
3. Optimize hyperparameters
4. Deploy with batch inference

## 📚 Additional Resources

- [API Reference](../api.md) - Detailed function documentation
- [FAQ](../faq.md) - Common questions answered
- [Troubleshooting](../troubleshooting.md) - Solutions to common issues
- [GitHub Examples](https://github.com/lab-rasool/EAGLE/tree/main/examples) - Code examples

## 🤝 Getting Help

If you need help:
1. Check the relevant tutorial section
2. Review the FAQ
3. Search [GitHub Issues](https://github.com/lab-rasool/EAGLE/issues)
4. Open a new issue with a minimal example

## 🎓 Tutorial Notebooks

Interactive Jupyter notebooks are coming soon:
- `01_basic_usage.ipynb` - Interactive getting started
- `02_custom_data.ipynb` - Preparing your own data
- `03_attribution_deep_dive.ipynb` - Advanced interpretability
- `04_model_optimization.ipynb` - Hyperparameter tuning

Stay tuned for updates!

---

Happy learning! The EAGLE framework is designed to be both powerful and accessible. These tutorials will help you unlock its full potential for your survival prediction tasks.