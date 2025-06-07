#!/bin/bash
# Script to run all baseline models on MedGemma embeddings

# Set MedGemma data paths
GBM_MEDGEMMA_PATH="/proj/rasool_lab_projects/Aakash/EAGLE/results/BASELINES/data/MedGemma-GBM.parquet"
IPMN_MEDGEMMA_PATH="/proj/rasool_lab_projects/Aakash/EAGLE/results/BASELINES/data/MedGemma-IPMN.parquet"
NSCLC_MEDGEMMA_PATH="/proj/rasool_lab_projects/Aakash/EAGLE/results/BASELINES/data/MedGemma-NSCLC.parquet"

# Create output directory
OUTPUT_DIR="medgemma_baseline_results/"
mkdir -p $OUTPUT_DIR

echo "Running MedGemma baseline experiments..."
echo "Output directory: $OUTPUT_DIR"

# Check if MedGemma data files exist
echo "Checking MedGemma data files..."
for path in "$GBM_MEDGEMMA_PATH" "$IPMN_MEDGEMMA_PATH" "$NSCLC_MEDGEMMA_PATH"; do
    if [ ! -f "$path" ]; then
        echo "Warning: $path not found"
    else
        echo "Found: $path"
    fi
done

# Run RSF baselines with MedGemma embeddings
echo -e "\n=== Running Random Survival Forest on MedGemma Embeddings ==="
if [ -f "$GBM_MEDGEMMA_PATH" ]; then
    python medgemma_baseline_rsf.py --dataset GBM --data-path "$GBM_MEDGEMMA_PATH" --output-dir $OUTPUT_DIR
fi

if [ -f "$IPMN_MEDGEMMA_PATH" ]; then
    python medgemma_baseline_rsf.py --dataset IPMN --data-path "$IPMN_MEDGEMMA_PATH" --output-dir $OUTPUT_DIR
fi

if [ -f "$NSCLC_MEDGEMMA_PATH" ]; then
    python medgemma_baseline_rsf.py --dataset NSCLC --data-path "$NSCLC_MEDGEMMA_PATH" --output-dir $OUTPUT_DIR
fi

# Run DeepSurv baselines with MedGemma embeddings
echo -e "\n=== Running DeepSurv on MedGemma Embeddings ==="
if [ -f "$GBM_MEDGEMMA_PATH" ]; then
    python medgemma_baseline_deepsurv.py --dataset GBM --data-path "$GBM_MEDGEMMA_PATH" --output-dir $OUTPUT_DIR
fi

if [ -f "$IPMN_MEDGEMMA_PATH" ]; then
    python medgemma_baseline_deepsurv.py --dataset IPMN --data-path "$IPMN_MEDGEMMA_PATH" --output-dir $OUTPUT_DIR
fi

if [ -f "$NSCLC_MEDGEMMA_PATH" ]; then
    python medgemma_baseline_deepsurv.py --dataset NSCLC --data-path "$NSCLC_MEDGEMMA_PATH" --output-dir $OUTPUT_DIR
fi

# Run CoxPH baselines with MedGemma embeddings
echo -e "\n=== Running Cox Proportional Hazards on MedGemma Embeddings ==="
if [ -f "$GBM_MEDGEMMA_PATH" ]; then
    python medgemma_baseline_coxph.py --dataset GBM --data-path "$GBM_MEDGEMMA_PATH" --output-dir $OUTPUT_DIR
fi

if [ -f "$IPMN_MEDGEMMA_PATH" ]; then
    python medgemma_baseline_coxph.py --dataset IPMN --data-path "$IPMN_MEDGEMMA_PATH" --output-dir $OUTPUT_DIR
fi

if [ -f "$NSCLC_MEDGEMMA_PATH" ]; then
    python medgemma_baseline_coxph.py --dataset NSCLC --data-path "$NSCLC_MEDGEMMA_PATH" --output-dir $OUTPUT_DIR
fi

echo -e "\n=== All MedGemma baseline experiments completed ==="
echo "Results saved in: $OUTPUT_DIR"

# Generate comparison report
echo -e "\n=== Generating Results Comparison ==="
python compare_medgemma_results.py --results-dir $OUTPUT_DIR