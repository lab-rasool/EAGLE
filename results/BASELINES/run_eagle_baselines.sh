#!/bin/bash
# Script to extract EAGLE embeddings and train baselines for all datasets

# Set your model directories
GBM_MODEL_DIR="/proj/rasool_lab_projects/Aakash/EAGLE/results/GBM/2025-06-05_18-13-59/models"
IPMN_MODEL_DIR="/proj/rasool_lab_projects/Aakash/EAGLE/results/IPMN/2025-06-05_18-22-37/models"
NSCLC_MODEL_DIR="/proj/rasool_lab_projects/Aakash/EAGLE/results/NSCLC/2025-06-05_18-54-14/models"

OUTPUT_DIR="eagle_baseline_results"

echo "Extracting EAGLE embeddings and training baseline models..."

# GBM
echo "Processing GBM..."
python extract_eagle_embeddings.py \
    --dataset GBM \
    --model-dir $GBM_MODEL_DIR \
    --output-dir $OUTPUT_DIR \
    --n-folds 5

# IPMN
echo "Processing IPMN..."
python extract_eagle_embeddings.py \
    --dataset IPMN \
    --model-dir $IPMN_MODEL_DIR \
    --output-dir $OUTPUT_DIR \
    --n-folds 5

# NSCLC
echo "Processing NSCLC..."
python extract_eagle_embeddings.py \
    --dataset NSCLC \
    --model-dir $NSCLC_MODEL_DIR \
    --output-dir $OUTPUT_DIR \
    --n-folds 5

echo "All experiments completed!"