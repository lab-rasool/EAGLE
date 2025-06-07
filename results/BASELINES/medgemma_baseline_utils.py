"""
Utilities for MedGemma baseline model experiments
Processes MedGemma embeddings from parquet files
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Tuple, List
import logging


def process_medgemma_embedding(emb_data, emb_shape_data=None):
    """Process a single MedGemma embedding from parquet data"""
    try:
        # Handle None/NaN cases
        if emb_data is None:
            return None
        
        # Handle pandas NaN (scalar)
        if pd.isna(emb_data) and not isinstance(emb_data, (np.ndarray, bytes)):
            return None
        
        # Handle empty arrays or empty bytes
        if isinstance(emb_data, (np.ndarray, list)) and len(emb_data) == 0:
            return None
        
        if isinstance(emb_data, bytes) and len(emb_data) == 0:
            return None
        
        # Case 1: Already a numpy array (from direct storage)
        if isinstance(emb_data, np.ndarray):
            emb_array = emb_data.astype(np.float32)
        
        # Case 2: Bytes data (from .tobytes() storage)
        elif isinstance(emb_data, bytes):
            emb_array = np.frombuffer(emb_data, dtype=np.float32)
            
            # Reshape if shape information is available
            if emb_shape_data is not None:
                try:
                    if isinstance(emb_shape_data, (list, tuple)):
                        shape = tuple(emb_shape_data)
                    elif isinstance(emb_shape_data, str):
                        shape = eval(emb_shape_data)
                    else:
                        shape = emb_shape_data
                    
                    expected_size = np.prod(shape)
                    if len(emb_array) >= expected_size:
                        emb_array = emb_array[:expected_size].reshape(shape)
                    else:
                        # Pad if necessary
                        emb_array = np.pad(emb_array, (0, expected_size - len(emb_array)))
                        emb_array = emb_array.reshape(shape)
                except Exception as e:
                    logging.warning(f"Error reshaping embedding: {str(e)}")
                    # Continue with flattened array
        
        # Case 3: List or other iterable
        elif hasattr(emb_data, '__iter__') and not isinstance(emb_data, str):
            try:
                emb_array = np.array(emb_data, dtype=np.float32)
            except:
                return None
        
        # Case 4: Unexpected type
        else:
            logging.warning(f"Unexpected embedding data type: {type(emb_data)}")
            return None
        
        # Handle multi-dimensional arrays by averaging across non-feature dimensions
        if emb_array.ndim > 2:
            # For 3D+ arrays, flatten all but the last dimension
            original_shape = emb_array.shape
            emb_array = emb_array.reshape(-1, original_shape[-1])
            # Average across the flattened dimension
            emb_array = np.mean(emb_array, axis=0)
        elif emb_array.ndim == 2:
            # For 2D arrays, average across the first dimension (slices/tokens)
            emb_array = np.mean(emb_array, axis=0)
        
        # Ensure 1D output
        if emb_array.ndim == 0:
            emb_array = np.array([float(emb_array)])
        
        return emb_array.astype(np.float32)
        
    except Exception as e:
        logging.warning(f"Error processing MedGemma embedding: {str(e)}")
        return None


def load_medgemma_gbm_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and process GBM MedGemma dataset"""
    df = pd.read_parquet(data_path)
    
    # Filter valid samples with survival data
    df = df[
        (df["survival_time_in_months"].notna()) &
        (df["vital_status_desc"].notna()) &
        (df["survival_time_in_months"] > 0)
    ].copy()
    
    print(f"Found {len(df)} patients with survival data")
    
    features_list = []
    valid_indices = []
    
    # Try different embedding column names from GBM processing
    embedding_columns = [
        'combined_embeddings',  # Main combined embedding
        'FL_embeddings',        # FLAIR embeddings  
        'T1_embeddings',        # T1 embeddings
        'T1CE_embeddings',      # T1 with contrast embeddings
        'T2_embeddings'         # T2 embeddings
    ]
    
    shape_columns = [
        'combined_embedding_shape',
        'FL_embedding_shape',
        'T1_embedding_shape', 
        'T1CE_embedding_shape',
        'T2_embedding_shape'
    ]
    
    for idx, row in df.iterrows():
        patient_features = None
        
        # Try to find any available embedding for this patient
        for emb_col, shape_col in zip(embedding_columns, shape_columns):
            if emb_col in df.columns and pd.notna(row[emb_col]):
                shape_data = row.get(shape_col) if shape_col in df.columns else None
                embedding = process_medgemma_embedding(row[emb_col], shape_data)
                
                if embedding is not None and len(embedding) > 0:
                    patient_features = embedding
                    break
        
        if patient_features is not None:
            features_list.append(patient_features)
            valid_indices.append(idx)
    
    if not features_list:
        raise ValueError("No valid MedGemma embeddings found in GBM data")
    
    # Get corresponding survival data
    valid_df = df.loc[valid_indices]
    
    X = np.array(features_list, dtype=np.float32)
    y_time = valid_df["survival_time_in_months"].values.astype(np.float32)
    y_event = (valid_df["vital_status_desc"] == "DEAD").astype(np.float32)
    
    print(f"Loaded {len(X)} patients with embeddings. Feature dimension: {X.shape[1]}")
    return X, y_time, y_event


def load_medgemma_ipmn_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and process IPMN MedGemma dataset"""
    df = pd.read_parquet(data_path)
    
    # Filter valid samples with survival data
    df = df[
        (df["survival_time_days"].notna()) &
        (df["survival_time_days"] > 0)
    ].copy()
    
    print(f"Found {len(df)} patients with survival data")
    
    features_list = []
    valid_indices = []
    
    # Try different embedding column names from IPMN processing
    embedding_columns = [
        'multimodal_ct_fused_embeddings',  # Main fused embedding
        'multimodal_Art_embeddings',       # Arterial phase
        'multimodal_NonCon_embeddings',    # Non-contrast
        'multimodal_Ven_embeddings'        # Venous phase
    ]
    
    shape_columns = [
        'multimodal_ct_fused_embedding_shape',
        'multimodal_Art_embedding_shape',
        'multimodal_NonCon_embedding_shape', 
        'multimodal_Ven_embedding_shape'
    ]
    
    for idx, row in df.iterrows():
        patient_features = None
        
        # Try to find any available embedding for this patient
        for emb_col, shape_col in zip(embedding_columns, shape_columns):
            if emb_col in df.columns and pd.notna(row[emb_col]):
                shape_data = row.get(shape_col) if shape_col in df.columns else None
                embedding = process_medgemma_embedding(row[emb_col], shape_data)
                
                if embedding is not None and len(embedding) > 0:
                    patient_features = embedding
                    break
        
        if patient_features is not None:
            features_list.append(patient_features)
            valid_indices.append(idx)
    
    if not features_list:
        raise ValueError("No valid MedGemma embeddings found in IPMN data")
    
    # Get corresponding survival data
    valid_df = df.loc[valid_indices]
    
    X = np.array(features_list, dtype=np.float32)
    y_time = (valid_df["survival_time_days"].values / 30.44).astype(np.float32)  # Convert to months
    y_event = (valid_df["vital_status"] == "DEAD").astype(np.float32)
    
    print(f"Loaded {len(X)} patients with embeddings. Feature dimension: {X.shape[1]}")
    return X, y_time, y_event


def load_medgemma_nsclc_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and process NSCLC MedGemma dataset"""
    df = pd.read_parquet(data_path)
    
    # Filter valid samples with survival data
    df = df[
        (df["SURVIVAL_TIME_IN_MONTHS"].notna()) &
        (df["event"].notna()) &
        (df["SURVIVAL_TIME_IN_MONTHS"] > 0)
    ].copy()
    
    print(f"Found {len(df)} patients with survival data")
    
    features_list = []
    valid_indices = []
    
    # Try different embedding column names from NSCLC processing
    embedding_columns = [
        'multimodal_contrast_embeddings',     # CT with contrast
        'multimodal_wo_contrast_embeddings'   # CT without contrast
    ]
    
    shape_columns = [
        'multimodal_contrast_embedding_shape',
        'multimodal_wo_contrast_embedding_shape'
    ]
    
    for idx, row in df.iterrows():
        patient_features = None
        
        # Try to find any available embedding for this patient
        for emb_col, shape_col in zip(embedding_columns, shape_columns):
            if emb_col in df.columns and pd.notna(row[emb_col]):
                shape_data = row.get(shape_col) if shape_col in df.columns else None
                embedding = process_medgemma_embedding(row[emb_col], shape_data)
                
                if embedding is not None and len(embedding) > 0:
                    patient_features = embedding
                    break
        
        if patient_features is not None:
            features_list.append(patient_features)
            valid_indices.append(idx)
    
    if not features_list:
        raise ValueError("No valid MedGemma embeddings found in NSCLC data")
    
    # Get corresponding survival data
    valid_df = df.loc[valid_indices]
    
    X = np.array(features_list, dtype=np.float32)
    y_time = valid_df["SURVIVAL_TIME_IN_MONTHS"].values.astype(np.float32)
    y_event = valid_df["event"].values.astype(np.float32)
    
    print(f"Loaded {len(X)} patients with embeddings. Feature dimension: {X.shape[1]}")
    return X, y_time, y_event


def prepare_medgemma_data(X: np.ndarray, scale: bool = True) -> np.ndarray:
    """Prepare MedGemma features for modeling"""
    # Remove NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X