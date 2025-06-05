"""
Utilities for baseline model experiments
Processes embeddings from parquet files similar to EAGLE
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import ast
import re
import logging
from typing import Dict, Tuple, List


def parse_embedding_shape(shape_value):
    """Parse embedding shape from various formats"""
    if isinstance(shape_value, str):
        try:
            return tuple(ast.literal_eval(shape_value))
        except (ValueError, SyntaxError):
            if shape_value.startswith("array("):
                numbers = re.findall(r"\d+", shape_value)
                return tuple(int(n) for n in numbers)
            else:
                raise ValueError(f"Cannot parse shape: {shape_value}")
    elif isinstance(shape_value, (list, tuple)):
        return tuple(shape_value)
    elif isinstance(shape_value, np.ndarray):
        return tuple(shape_value.tolist())
    else:
        raise ValueError(f"Unexpected shape type: {type(shape_value)}")


def process_imaging_embedding(img_bytes, img_shape_data, target_dim=1000, aggregation='mean'):
    """Process a single imaging embedding"""
    if pd.isna(img_bytes):
        return np.zeros(target_dim, dtype=np.float32)
    
    try:
        # Check if shape data exists and is valid
        has_shape = False
        if img_shape_data is not None:
            if isinstance(img_shape_data, np.ndarray):
                has_shape = True
            elif isinstance(img_shape_data, (list, tuple, str)):
                try:
                    # Try to parse it
                    parse_embedding_shape(img_shape_data)
                    has_shape = True
                except:
                    has_shape = False
            elif not pd.isna(img_shape_data):
                has_shape = True
        
        if has_shape:
            img_shape = parse_embedding_shape(img_shape_data)
            img_array = np.frombuffer(img_bytes, dtype=np.float32).copy()
            expected_size = np.prod(img_shape)
            
            if len(img_array) >= expected_size:
                img_array = img_array[:expected_size]
                img_array = img_array.reshape(img_shape)
                
                # Apply aggregation
                if len(img_shape) > 1 and aggregation == 'mean':
                    img_features = img_array.mean(axis=0)
                else:
                    img_features = img_array.flatten()
            else:
                img_features = img_array
        else:
            # No shape info, assume 1D
            img_features = np.frombuffer(img_bytes, dtype=np.float32).copy()
        
        # Ensure correct dimension
        if len(img_features) != target_dim:
            if len(img_features) > target_dim:
                img_features = img_features[:target_dim]
            else:
                img_features = np.pad(img_features, (0, target_dim - len(img_features)))
        
        return img_features
        
    except Exception as e:
        logging.warning(f"Error processing imaging: {str(e)}")
        return np.zeros(target_dim, dtype=np.float32)


def process_text_embedding(emb_bytes, target_dim=1024):
    """Process a single text embedding"""
    if pd.isna(emb_bytes):
        return np.zeros(target_dim, dtype=np.float32)
    
    try:
        emb_array = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        
        # Ensure correct dimension
        if len(emb_array) != target_dim:
            if len(emb_array) > target_dim:
                emb_array = emb_array[:target_dim]
            else:
                emb_array = np.pad(emb_array, (0, target_dim - len(emb_array)))
        
        return emb_array
    except Exception as e:
        logging.warning(f"Error processing text embedding: {str(e)}")
        return np.zeros(target_dim, dtype=np.float32)


def load_gbm_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and process GBM dataset"""
    df = pd.read_parquet(data_path)
    
    # Filter valid samples
    df = df[
        (df["mri_available"] == True) &
        (df["has_pathology_report"] == True) &
        (df["has_radiology_report"] == True) &
        (df["survival_time_in_months"].notna()) &
        (df["vital_status_desc"].notna())
    ].copy()
    
    features_list = []
    
    for idx, row in df.iterrows():
        # Process MRI embeddings
        mri_features = process_imaging_embedding(
            row.get("mri_embeddings"),
            row.get("mri_embedding_shape"),
            target_dim=1000
        )
        
        # Process text embeddings
        radiology_emb = process_text_embedding(row.get("radiology_report_embeddings"))
        pathology_emb = process_text_embedding(row.get("pathology_report_embeddings"))
        treatment_emb = process_text_embedding(row.get("treatment_embeddings"))
        
        # Concatenate all features
        features = np.concatenate([
            mri_features,
            radiology_emb,
            pathology_emb,
            treatment_emb
        ])
        
        features_list.append(features)
    
    X = np.array(features_list, dtype=np.float32)
    y_time = df["survival_time_in_months"].values.astype(np.float32)
    y_event = (df["vital_status_desc"] == "DEAD").astype(np.float32)
    
    return X, y_time, y_event


def load_ipmn_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and process IPMN dataset"""
    df = pd.read_parquet(data_path)
    
    # Filter valid samples
    df = df[
        (df["survival_time_days"] > 0) &
        (df["survival_time_days"].notna())
    ].copy()
    
    features_list = []
    
    for idx, row in df.iterrows():
        # Process CT embeddings
        ct_features = process_imaging_embedding(
            row.get("ct_embeddings"),
            row.get("ct_embedding_shape"),
            target_dim=1000
        )
        
        # Process text embeddings
        radiology_emb = process_text_embedding(row.get("radiology_embeddings"))
        pathology_emb = process_text_embedding(row.get("pathology_embeddings"))
        
        # Concatenate all features
        features = np.concatenate([
            ct_features,
            radiology_emb,
            pathology_emb
        ])
        
        features_list.append(features)
    
    X = np.array(features_list, dtype=np.float32)
    y_time = (df["survival_time_days"].values / 30.44).astype(np.float32)  # Convert to months
    y_event = (df["vital_status"] == "DEAD").astype(np.float32)
    
    return X, y_time, y_event


def load_nsclc_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and process NSCLC dataset"""
    df = pd.read_parquet(data_path)
    
    # Filter valid samples
    df = df[
        ((df["ct_contrast_embeddings"].notna()) | (df["ct_wo_contrast_embeddings"].notna())) &
        (df["clinical_embeddings"].notna()) &
        (df["SURVIVAL_TIME_IN_MONTHS"].notna()) &
        (df["event"].notna())
    ].copy()
    
    features_list = []
    
    for idx, row in df.iterrows():
        # Process CT embeddings (both contrast and non-contrast)
        contrast_features = process_imaging_embedding(
            row.get("ct_contrast_embeddings"),
            row.get("ct_contrast_embedding_shape"),
            target_dim=1024
        )
        
        wo_contrast_features = process_imaging_embedding(
            row.get("ct_wo_contrast_embeddings"),
            row.get("ct_wo_contrast_embedding_shape"),
            target_dim=1024
        )
        
        # Process clinical embeddings
        clinical_emb = process_text_embedding(
            row.get("clinical_embeddings"),
            target_dim=1024
        )
        
        # Concatenate all features
        features = np.concatenate([
            contrast_features,
            wo_contrast_features,
            clinical_emb
        ])
        
        features_list.append(features)
    
    X = np.array(features_list, dtype=np.float32)
    y_time = df["SURVIVAL_TIME_IN_MONTHS"].values.astype(np.float32)
    y_event = df["event"].values.astype(np.float32)
    
    return X, y_time, y_event


def prepare_data(X: np.ndarray, scale: bool = True) -> np.ndarray:
    """Prepare features for modeling"""
    # Remove NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X