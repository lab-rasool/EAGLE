# 🔧 EAGLE Troubleshooting Guide

This guide helps you resolve common issues when using EAGLE.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data Loading Errors](#data-loading-errors)
- [Training Problems](#training-problems)
- [Memory Issues](#memory-issues)
- [Performance Issues](#performance-issues)
- [Attribution Errors](#attribution-errors)
- [Visualization Problems](#visualization-problems)
- [Common Error Messages](#common-error-messages)

---

## Installation Issues

### 🔴 Error: "No module named 'eagle'"

**Problem**: Python can't find the EAGLE module.

**Solutions**:
1. Ensure you're in the EAGLE directory:
   ```bash
   cd /path/to/EAGLE
   ```

2. Add EAGLE to Python path:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:/path/to/EAGLE"
   ```

3. Or run scripts from the EAGLE root directory:
   ```bash
   python main.py --dataset GBM  # Not python eagle/main.py
   ```

### 🔴 Error: "torch not compiled with CUDA enabled"

**Problem**: PyTorch installed without GPU support.

**Solution**:
```bash
# Uninstall CPU-only PyTorch
pip uninstall torch torchvision

# Install CUDA-enabled PyTorch (check your CUDA version first)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 🔴 Dependency conflicts

**Problem**: Package version conflicts.

**Solution**:
```bash
# Create fresh environment
python -m venv eagle_env_fresh
source eagle_env_fresh/bin/activate  # Windows: eagle_env_fresh\Scripts\activate

# Install exact versions
pip install -r requirements.txt --no-cache-dir
```

---

## Data Loading Errors

### 🔴 Error: "FileNotFoundError: data/GBM/unimodal.parquet"

**Problem**: Data files not found.

**Solutions**:
1. Check data directory structure:
   ```bash
   ls -la data/GBM/
   # Should show: unimodal.parquet, medgemma.parquet, eagle.parquet
   ```

2. Use absolute paths:
   ```python
   config = DatasetConfig(
       data_path="/absolute/path/to/data/GBM/unimodal.parquet",
       # ... other settings
   )
   ```

3. Download missing data files from the dataset source.

### 🔴 Error: "KeyError: 'PatientID'"

**Problem**: Expected column missing from data.

**Solutions**:
1. Check column names:
   ```python
   import pandas as pd
   df = pd.read_parquet('data/GBM/unimodal.parquet')
   print(df.columns.tolist())
   ```

2. Update configuration with correct column names:
   ```python
   config = DatasetConfig(
       patient_col='patient_id',  # or whatever your column is named
       survival_time_col='os_days',
       event_col='death_event'
   )
   ```

### 🔴 Error: "ValueError: Cannot determine embedding dimensions"

**Problem**: Embedding columns not properly formatted.

**Solution**:
Ensure embedding columns follow naming convention:
```python
# Imaging embeddings: imaging_feat_0, imaging_feat_1, ...
# Text embeddings: text_feat_0, text_feat_1, ...

# Fix column names:
for i in range(2048):
    df[f'imaging_feat_{i}'] = imaging_embeddings[:, i]
```

---

## Training Problems

### 🔴 Loss becomes NaN

**Problem**: Numerical instability during training.

**Solutions**:

1. **Reduce learning rate**:
   ```bash
   python main.py --dataset GBM --lr 1e-5  # Default is 1e-4
   ```

2. **Enable gradient clipping**:
   ```python
   # In custom training loop
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **Check for invalid data**:
   ```python
   # Check for NaN/Inf in data
   assert not df.isnull().any().any(), "Data contains NaN"
   assert not np.isinf(df.select_dtypes(include=[np.number])).any().any(), "Data contains Inf"
   ```

4. **Increase batch size**:
   ```bash
   python main.py --dataset GBM --batch-size 32
   ```

### 🔴 Model not improving (C-index stuck)

**Problem**: Model not learning effectively.

**Solutions**:

1. **Check class balance**:
   ```python
   event_rate = df['Event'].mean()
   print(f"Event rate: {event_rate:.2%}")
   # Should be between 20-80%
   ```

2. **Try different learning rates**:
   ```bash
   # Grid search
   for lr in 1e-5 5e-5 1e-4 5e-4; do
       python main.py --dataset GBM --lr $lr --epochs 50
   done
   ```

3. **Adjust model architecture**:
   ```python
   config = ModelConfig(
       imaging_encoder_dims=[1024, 512, 256],  # Larger network
       dropout=0.5,  # More regularization
       use_batch_norm=True
   )
   ```

4. **Verify data quality**:
   - Ensure embeddings are meaningful (not random)
   - Check survival times are reasonable
   - Verify clinical features are normalized

### 🔴 Early stopping triggered too early

**Problem**: Model stops training before convergence.

**Solution**:
```python
config = ModelConfig(
    patience=50,  # Default is 20
    min_epochs=50  # Minimum epochs before early stopping
)
```

---

## Memory Issues

### 🔴 CUDA out of memory

**Problem**: GPU memory exhausted.

**Solutions**:

1. **Reduce batch size**:
   ```bash
   python main.py --dataset GBM --batch-size 8  # or even 4
   ```

2. **Use gradient accumulation**:
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = compute_loss(batch) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Clear cache periodically**:
   ```python
   torch.cuda.empty_cache()
   ```

4. **Use mixed precision**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   with autocast():
       output = model(batch)
       loss = criterion(output, target)
   ```

### 🔴 RAM out of memory

**Problem**: System memory exhausted.

**Solutions**:

1. **Process data in chunks**:
   ```python
   # Instead of loading all data at once
   chunk_size = 1000
   for chunk in pd.read_parquet('data.parquet', chunksize=chunk_size):
       process_chunk(chunk)
   ```

2. **Reduce number of workers**:
   ```python
   dataloader = DataLoader(dataset, num_workers=0)  # Single-threaded
   ```

3. **Use data sampling for testing**:
   ```python
   # Use subset for debugging
   indices = np.random.choice(len(dataset), size=100, replace=False)
   subset = Subset(dataset, indices)
   ```

---

## Performance Issues

### 🔴 Training is very slow

**Problem**: Training takes too long.

**Solutions**:

1. **Verify GPU usage**:
   ```python
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Current device: {torch.cuda.current_device()}")
   print(f"Device name: {torch.cuda.get_device_name()}")
   ```

2. **Profile the code**:
   ```python
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   # Your training code
   profiler.disable()
   
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(10)  # Top 10 time-consuming functions
   ```

3. **Use faster data loading**:
   ```python
   # Increase number of workers
   dataloader = DataLoader(
       dataset,
       batch_size=32,
       num_workers=4,
       pin_memory=True,
       persistent_workers=True
   )
   ```

### 🔴 Inference is slow

**Problem**: Predictions take too long.

**Solutions**:

1. **Use batch prediction**:
   ```python
   # Instead of one-by-one
   model.eval()
   with torch.no_grad():
       all_predictions = []
       for batch in dataloader:
           predictions = model(batch)
           all_predictions.append(predictions)
   ```

2. **Export to TorchScript**:
   ```python
   traced_model = torch.jit.trace(model, example_input)
   torch.jit.save(traced_model, 'model_traced.pt')
   ```

---

## Attribution Errors

### 🔴 AttributeError in attribution analysis

**Problem**: Attribution analysis fails.

**Solutions**:

1. **Ensure model is in eval mode**:
   ```python
   model.eval()
   analyzer = ModalityAttributionAnalyzer(model, dataset)
   ```

2. **Check gradient computation**:
   ```python
   # Enable gradients for attribution
   for param in model.parameters():
       param.requires_grad = True
   ```

3. **Use simpler attribution method**:
   ```bash
   # Use attention-based instead of gradient-based
   python main.py --dataset GBM --analyze-attribution --attribution-method attention
   ```

### 🔴 Attribution takes too long

**Problem**: Attribution analysis is slow.

**Solutions**:

1. **Analyze fewer patients**:
   ```bash
   python main.py --dataset GBM --analyze-attribution --top-patients 3
   ```

2. **Use sampling**:
   ```python
   # Sample patients for attribution
   sample_indices = np.random.choice(len(dataset), size=50)
   analyzer.analyze_cohort(sample_indices=sample_indices)
   ```

---

## Visualization Problems

### 🔴 Plots not generated

**Problem**: Visualization files missing.

**Solutions**:

1. **Check matplotlib backend**:
   ```python
   import matplotlib
   matplotlib.use('Agg')  # For server environments
   import matplotlib.pyplot as plt
   ```

2. **Verify output directory**:
   ```python
   import os
   output_dir = 'results/GBM/figures'
   os.makedirs(output_dir, exist_ok=True)
   ```

3. **Check for errors in console**:
   ```bash
   python main.py --dataset GBM 2>&1 | tee output.log
   grep -i error output.log
   ```

### 🔴 Plots look incorrect

**Problem**: Visualizations show unexpected results.

**Solutions**:

1. **Verify data ranges**:
   ```python
   print(f"Risk scores range: {risk_df['risk_score'].min():.3f} - {risk_df['risk_score'].max():.3f}")
   print(f"Survival times range: {risk_df['survival_time'].min()} - {risk_df['survival_time'].max()}")
   ```

2. **Check for data issues**:
   ```python
   # Look for outliers
   q1, q3 = risk_df['risk_score'].quantile([0.25, 0.75])
   iqr = q3 - q1
   outliers = risk_df[(risk_df['risk_score'] < q1 - 1.5*iqr) | 
                      (risk_df['risk_score'] > q3 + 1.5*iqr)]
   print(f"Outliers: {len(outliers)}")
   ```

---

## Common Error Messages

### 🔴 "RuntimeError: Expected all tensors to be on the same device"

**Solution**:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Ensure all inputs are on same device
imaging = imaging.to(device)
clinical = clinical.to(device)
text = text.to(device)
```

### 🔴 "ValueError: Target size (torch.Size([32])) must be the same as input size (torch.Size([32, 1]))"

**Solution**:
```python
# Squeeze extra dimensions
predictions = model(inputs).squeeze(-1)  # Remove last dimension
# Or
targets = targets.unsqueeze(-1)  # Add dimension to targets
```

### 🔴 "IndexError: index 256 is out of bounds for dimension 0 with size 256"

**Solution**:
Check data preprocessing and ensure indices are within bounds:
```python
# If using embeddings
assert all(idx < embedding_dim for idx in indices)

# If using categorical encoding
num_categories = df['category'].nunique()
assert all(encoded_val < num_categories for encoded_val in encoded_categories)
```

### 🔴 "TypeError: can't convert cuda:0 device type tensor to numpy"

**Solution**:
```python
# Move to CPU before converting to numpy
tensor_cpu = tensor.cpu()
numpy_array = tensor_cpu.numpy()
```

---

## Getting Further Help

If these solutions don't resolve your issue:

1. **Create a minimal reproducible example**:
   ```python
   # minimal_example.py
   from eagle import UnifiedPipeline, GBM_CONFIG
   
   # Minimal code that reproduces the error
   pipeline = UnifiedPipeline(GBM_CONFIG)
   results = pipeline.run()  # Error occurs here
   ```

2. **Gather system information**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
   python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
   nvidia-smi  # GPU information
   ```

3. **Open a GitHub issue** with:
   - Error message and full traceback
   - Minimal reproducible example
   - System information
   - What you've already tried

4. **Search existing issues**:
   - [GitHub Issues](https://github.com/lab-rasool/EAGLE/issues)
   - Look for similar problems and solutions

---

Remember: Most issues have simple solutions. Start with the basics (data format, file paths, dependencies) before diving into complex debugging!