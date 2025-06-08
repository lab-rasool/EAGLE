# 🚀 Getting Started with EAGLE

This tutorial will walk you through your first EAGLE analysis, from installation to interpreting results.

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- A CUDA-capable GPU (recommended but not required)
- At least 16GB of RAM
- Basic familiarity with Python and deep learning concepts

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/lab-rasool/EAGLE.git
cd EAGLE
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv eagle_env
source eagle_env/bin/activate  # On Windows: eagle_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Your First EAGLE Analysis

Let's run a survival analysis on the GBM (Glioblastoma) dataset.

### Step 1: Basic Run

The simplest way to start is using the command line:

```bash
python main.py --dataset GBM
```

This command will:
- Load the GBM dataset with pre-computed embeddings
- Train the EAGLE model using 5-fold cross-validation
- Generate risk scores and stratify patients
- Create visualizations in the `results/` directory

### Step 2: Understanding the Output

After the run completes, you'll find results in:
```
results/GBM/[timestamp]/
├── models/          # Trained model checkpoints
├── results/         # Risk scores and metrics
├── figures/         # Visualizations
└── run_info.txt     # Configuration used
```

Key files to examine:
- `results/risk_scores.csv`: Patient-level predictions
- `figures/kaplan_meier_curves.png`: Survival curves by risk group
- `run_info.txt`: Exact configuration for reproducibility

### Step 3: Interpreting Results

The console output will show:
```
=== Final Results ===
Mean C-index: 0.599 ± 0.062
```

The C-index (concordance index) measures how well the model ranks patients by risk:
- 0.5 = Random performance
- 0.7+ = Good performance
- 0.8+ = Excellent performance

## Adding Attribution Analysis

To understand which data modalities drive predictions:

```bash
python main.py --dataset GBM --analyze-attribution
```

This adds:
- Modality contribution analysis
- Patient-level attribution
- Feature importance visualizations

New outputs in `attribution/`:
- `modality_contributions.png`: Shows average importance of imaging vs clinical vs text data
- `patient_level_attribution.png`: Individual patient analysis
- `top_bottom_patients_analysis.csv`: Detailed analysis of extreme cases

## Python API Example

For more control, use the Python API:

```python
from eagle import UnifiedPipeline, GBM_CONFIG, ModelConfig

# Custom model configuration
model_config = ModelConfig(
    learning_rate=5e-5,
    num_epochs=150,
    dropout=0.35
)

# Create and run pipeline
pipeline = UnifiedPipeline(GBM_CONFIG, model_config)
results, risk_df, stats = pipeline.run(enable_attribution=True)

# Examine results
print(f"C-index: {results['mean_cindex']:.3f}")
print(f"\nRisk group distribution:")
print(risk_df['risk_group'].value_counts())

# High-risk patients
high_risk = risk_df[risk_df['risk_group'] == 'High']
print(f"\nHigh-risk patients: {len(high_risk)}")
print(f"Median survival: {high_risk['survival_time'].median():.1f} days")
```

## Comparing with Baselines

EAGLE includes traditional survival models for comparison:

```bash
# Run baseline models only
python main.py --mode baseline --dataset GBM

# Run both EAGLE and baselines
python main.py --mode all --dataset GBM
```

This generates comparative results showing EAGLE's performance against:
- Random Survival Forest (RSF)
- Cox Proportional Hazards (CoxPH)
- DeepSurv

## Tips for Success

1. **Start Simple**: Use default settings first, then customize
2. **Monitor Training**: Check the console output for training progress
3. **Examine Visualizations**: The generated plots provide intuitive insights
4. **Use Attribution**: Always enable attribution for clinical interpretability
5. **Save Results**: The timestamp-based output ensures you never lose results

## Next Steps

- Try different datasets (IPMN, NSCLC)
- Experiment with model configurations
- Read the [Advanced Tutorial](advanced_usage.md) for custom datasets
- Explore the [API Reference](../api.md) for detailed documentation

## Common Issues

If you encounter problems:
1. Ensure data files exist in the `data/` directory
2. Check GPU memory if using large batch sizes
3. Verify all dependencies are installed correctly
4. See the [Troubleshooting Guide](../troubleshooting.md)

## Getting Help

- Check the [FAQ](../faq.md) for common questions
- Open an issue on [GitHub](https://github.com/lab-rasool/EAGLE/issues)
- Review the example scripts in the repository

---

Congratulations! You've completed your first EAGLE analysis. The framework's power lies in its ability to combine multiple data types while maintaining interpretability - crucial for clinical applications.