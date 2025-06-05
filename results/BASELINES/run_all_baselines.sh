#!/bin/bash
# Script to run all baseline models on all datasets

# Set data paths
GBM_PATH="/proj/rasool_lab_projects/Aakash/ARCHIVE/data/GBM.parquet"
IPMN_PATH="/proj/rasool_lab_projects/Aakash/ARCHIVE/data/IPMN_with_text.parquet"
NSCLC_PATH="/proj/rasool_lab_projects/Aakash/ARCHIVE/data/NSCLC.parquet"

# Create output directory
OUTPUT_DIR="baseline_results/"
mkdir -p $OUTPUT_DIR

echo "Running baseline experiments..."
echo "Output directory: $OUTPUT_DIR"

# Run RSF baselines
echo -e "\n=== Running Random Survival Forest ==="
python baseline_rsf.py --dataset GBM --data-path $GBM_PATH --output-dir $OUTPUT_DIR
python baseline_rsf.py --dataset IPMN --data-path $IPMN_PATH --output-dir $OUTPUT_DIR
python baseline_rsf.py --dataset NSCLC --data-path $NSCLC_PATH --output-dir $OUTPUT_DIR

# Run DeepSurv baselines
echo -e "\n=== Running DeepSurv ==="
python baseline_deepsurv.py --dataset GBM --data-path $GBM_PATH --output-dir $OUTPUT_DIR
python baseline_deepsurv.py --dataset IPMN --data-path $IPMN_PATH --output-dir $OUTPUT_DIR
python baseline_deepsurv.py --dataset NSCLC --data-path $NSCLC_PATH --output-dir $OUTPUT_DIR

# Run CoxPH baselines
echo -e "\n=== Running Cox Proportional Hazards ==="
python baseline_coxph.py --dataset GBM --data-path $GBM_PATH --output-dir $OUTPUT_DIR
python baseline_coxph.py --dataset IPMN --data-path $IPMN_PATH --output-dir $OUTPUT_DIR
python baseline_coxph.py --dataset NSCLC --data-path $NSCLC_PATH --output-dir $OUTPUT_DIR

echo -e "\n=== All experiments completed ==="
echo "Results saved in: $OUTPUT_DIR"