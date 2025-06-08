# ❓ EAGLE FAQ (Frequently Asked Questions)

## General Questions

### What is EAGLE?

EAGLE (Efficient Alignment of Generalized Latent Embeddings) is a multimodal deep learning framework for survival prediction. It combines imaging, clinical, and textual data to predict patient outcomes while providing interpretable insights through attribution analysis.

### What makes EAGLE different from other survival models?

EAGLE offers several unique advantages:
- **Multimodal Integration**: Seamlessly combines three data modalities
- **Interpretability**: Built-in attribution analysis explains predictions
- **Efficiency**: 99.96% dimensionality reduction while maintaining performance
- **Flexibility**: Works with any cancer type or survival prediction task
- **Comprehensive**: Includes baseline comparisons and extensive visualizations

### What types of cancer does EAGLE support?

EAGLE comes pre-configured for three cancer types:
- **GBM** (Glioblastoma) - Brain cancer
- **IPMN** (Intraductal Papillary Mucinous Neoplasms) - Pancreatic cysts
- **NSCLC** (Non-Small Cell Lung Cancer)

However, EAGLE can be adapted to any cancer type or survival prediction task with appropriate data.

### Do I need a GPU to run EAGLE?

A GPU is strongly recommended but not required:
- **With GPU**: Training takes 30-60 minutes per dataset
- **Without GPU**: Training may take several hours
- **Minimum GPU**: 8GB VRAM (e.g., RTX 2070, GTX 1080)
- **Recommended GPU**: 16GB+ VRAM (e.g., RTX 3090, A100)

## Data Questions

### What data format does EAGLE expect?

EAGLE expects data in **parquet format** with:
- Pre-computed embeddings (not raw images)
- Clinical features as columns
- Survival time and event columns
- Patient identifier column

Example structure:
```
PatientID | SurvivalTime | Event | Age | Gender | imaging_feat_0 | ... | text_feat_0 | ...
```

### How do I prepare imaging embeddings?

You need to pre-extract features from images:

```python
# Example using a pretrained model
import torch
import torchvision.models as models
from torchvision import transforms

# Load pretrained model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final layer
model.eval()

# Extract features
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with torch.no_grad():
    image = transform(pil_image).unsqueeze(0)
    features = model(image).squeeze().numpy()  # 2048-dimensional vector
```

### Can I use raw images instead of embeddings?

Not directly. EAGLE expects pre-computed embeddings because:
1. It allows flexibility in choosing the best feature extractor
2. Reduces computational requirements during training
3. Enables using domain-specific models (e.g., RadImageNet for medical images)

### What if I don't have text data?

You can run EAGLE without text data by:
1. Creating dummy text embeddings (zeros)
2. Adjusting the model to give zero weight to text modality
3. Or using a custom configuration that excludes text

```python
# Option 1: Dummy embeddings
df['text_feat_0'] = 0
# ... repeat for all text features

# Option 2: Custom config without text
config = DatasetConfig(
    # ... other settings
    text_columns=[],  # Empty list
    text_extractor=lambda x: []  # Return empty features
)
```

## Model Questions

### What is the C-index and what values are good?

The C-index (concordance index) measures how well the model ranks patients by risk:
- **0.5**: Random performance (no predictive ability)
- **0.6-0.7**: Fair to good performance
- **0.7-0.8**: Good to excellent performance
- **0.8+**: Excellent performance (rare in survival analysis)

EAGLE typically achieves C-index values between 0.59-0.67 depending on the dataset.

### How many epochs should I train for?

Default is 200 epochs with early stopping:
- Most models converge within 50-150 epochs
- Early stopping prevents overfitting
- For quick experiments, try 50-100 epochs
- For final results, use the default 200

### What does attribution analysis tell me?

Attribution analysis reveals:
- **Modality Importance**: Which data type (imaging/clinical/text) drives predictions
- **Patient-Level Insights**: Why specific patients are high/low risk
- **Feature Importance**: Which specific features are most predictive
- **Clinical Validation**: Whether the model focuses on clinically relevant factors

### Can I use EAGLE for classification instead of survival?

While EAGLE is designed for survival prediction, you can adapt it for classification:
1. Modify the output layer to use sigmoid/softmax
2. Change the loss function to binary/multi-class cross-entropy
3. Adjust evaluation metrics accordingly

## Technical Questions

### How do I save and load trained models?

Models are automatically saved during training:
```python
# Models saved to: results/[Dataset]/[Timestamp]/models/

# To load a model:
import torch
from eagle import UnifiedSurvivalModel, ModelConfig

model = UnifiedSurvivalModel(ModelConfig())
model.load_state_dict(torch.load('path/to/best_model_fold1.pth'))
model.eval()
```

### Can I use different embeddings for different patients?

Yes, as long as the dimensionality is consistent:
- All imaging embeddings must have the same size
- All text embeddings must have the same size
- Different embedding models can be used per patient if needed

### How do I handle missing data?

EAGLE handles missing data through:
1. **Clinical features**: Imputation in the preprocessor
2. **Text features**: Empty strings handled by text extractor
3. **Imaging**: Must have embeddings for all patients

For custom handling:
```python
def custom_preprocessor(df):
    # Handle missing clinical data
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Stage'].fillna('Unknown', inplace=True)
    return df
```

### Why is my model not improving?

Common issues and solutions:
1. **Learning rate too high/low**: Try values between 1e-5 and 1e-3
2. **Insufficient data**: EAGLE works best with 100+ patients
3. **Poor embeddings**: Ensure embeddings capture relevant features
4. **Class imbalance**: Check event rate (should be 20-80%)
5. **Wrong features**: Verify clinical features are relevant

## Performance Questions

### How can I speed up training?

Several options:
1. **Reduce batch size**: Uses less memory but may be slower
2. **Use mixed precision**: Add `--mixed-precision` flag
3. **Fewer epochs**: Use `--epochs 100`
4. **Fewer folds**: Use 3-fold instead of 5-fold CV
5. **GPU upgrade**: Use a faster GPU

### How much memory do I need?

Memory requirements depend on:
- **RAM**: 16GB minimum, 32GB recommended
- **GPU VRAM**: 8GB minimum, 16GB+ recommended
- **Disk space**: 10GB for data and results

Reduce memory usage by:
- Decreasing batch size
- Using smaller embedding dimensions
- Processing data in chunks

### Can I run EAGLE on a cluster?

Yes, EAGLE supports distributed training:
```python
# Launch with multiple GPUs
python -m torch.distributed.launch --nproc_per_node=4 main.py --dataset GBM
```

## Results Questions

### What visualizations does EAGLE generate?

EAGLE automatically creates:
- **Kaplan-Meier curves**: Survival by risk group
- **Risk distributions**: Score histograms
- **Risk vs. survival scatter**: Individual patient plots
- **Attribution heatmaps**: Modality importance
- **Performance metrics**: C-index per fold

### How do I interpret the risk groups?

Risk groups (Low/Medium/High) are created by:
1. Ranking patients by predicted risk score
2. Dividing into tertiles (or custom groups)
3. Higher scores indicate higher risk of event

Typical interpretation:
- **Low risk**: Longer expected survival
- **Medium risk**: Intermediate prognosis
- **High risk**: Shorter expected survival, may need aggressive treatment

### Can I export results for further analysis?

Yes, all results are saved as CSV files:
```python
# Risk scores: results/[Dataset]/[Timestamp]/results/risk_scores.csv
# Contains: PatientID, risk_score, risk_group, survival_time, event

# Load for analysis
import pandas as pd
risk_df = pd.read_csv('path/to/risk_scores.csv')
```

## Troubleshooting Questions

### Why am I getting NaN losses?

Common causes:
1. **Learning rate too high**: Try reducing by 10x
2. **Numerical instability**: Enable gradient clipping
3. **Data issues**: Check for NaN/infinite values in embeddings
4. **Batch size too small**: Increase to at least 8

### Why is attribution analysis slow?

Attribution analysis can be memory-intensive:
- Process fewer patients: `--top-patients 3`
- Use simpler method: Attention-based instead of integrated gradients
- Run on GPU: Ensure CUDA is available
- Reduce batch size for attribution

### How do I debug poor performance?

Debugging steps:
1. **Check data quality**: Verify embeddings are meaningful
2. **Examine loss curves**: Look for overfitting/underfitting
3. **Review attributions**: Ensure model focuses on relevant features
4. **Compare with baselines**: Run `--mode baseline`
5. **Try different configs**: Adjust architecture, dropout, learning rate

## Advanced Questions

### Can I use EAGLE for other diseases?

Yes! EAGLE is disease-agnostic. You need:
1. Survival outcomes (time-to-event data)
2. Relevant imaging embeddings
3. Clinical features
4. Optional: Text data

Examples: Cardiovascular disease, kidney disease, COVID-19 outcomes

### How do I contribute to EAGLE?

We welcome contributions:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

See [CONTRIBUTING.md](https://github.com/lab-rasool/EAGLE/blob/main/CONTRIBUTING.md) for details.

### Where can I get more help?

- **Documentation**: Check the [tutorials](tutorials/) and [API reference](api.md)
- **GitHub Issues**: Search or create [issues](https://github.com/lab-rasool/EAGLE/issues)
- **Discussions**: Join our [community discussions](https://github.com/lab-rasool/EAGLE/discussions)
- **Email**: Contact the maintainers (see GitHub profile)

---

Don't see your question? Please open an issue on GitHub and we'll add it to the FAQ!