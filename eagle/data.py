"""
Data processing and dataset classes for EAGLE
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import re
import ast
import logging


# Utility functions
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


# Dataset configuration
@dataclass
class DatasetConfig:
    """Configuration for dataset-specific parameters"""

    name: str
    data_path: str

    # Imaging configuration
    imaging_modality: str
    imaging_embedding_dim: int
    imaging_num_slices: Optional[int] = None
    imaging_aggregation: str = "mean"

    # Text configuration
    text_embedding_dim: int = 1024
    has_treatment_text: bool = False

    # Clinical features
    clinical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)

    # Survival configuration
    survival_time_col: str = "survival_time_in_months"
    event_col: str = "vital_status_desc"
    event_positive_value: Union[str, int] = "DEAD"

    # Column names
    imaging_col: str = "mri_embeddings"
    imaging_shape_col: str = "mri_embedding_shape"
    radiology_embedding_col: str = "radiology_report_embeddings"
    pathology_embedding_col: str = "pathology_report_embeddings"
    treatment_embedding_col: str = "treatment_embeddings"

    # NSCLC-specific fields
    imaging_col_secondary: Optional[str] = None
    imaging_shape_col_secondary: Optional[str] = None


# Preset configurations
GBM_CONFIG = DatasetConfig(
    name="GBM",
    data_path="/mnt/f/Projects/EAGLE/data/GBM/unimodal.parquet",
    imaging_modality="MRI",
    imaging_embedding_dim=1000,
    imaging_num_slices=155,
    has_treatment_text=True,
    clinical_features=[
        "age_at_diagnosis_num",
        "gender_src_desc",
        "survival_time_in_months",
    ],
    survival_time_col="survival_time_in_months",
    event_col="vital_status_desc",
    event_positive_value="DEAD",
    imaging_col="mri_embeddings",
    imaging_shape_col="mri_embedding_shape",
    radiology_embedding_col="radiology_report_embeddings",
    pathology_embedding_col="pathology_report_embeddings",
    treatment_embedding_col="treatment_embeddings",
)

IPMN_CONFIG = DatasetConfig(
    name="IPMN",
    data_path="/mnt/f/Projects/EAGLE/data/IPMN/unimodal.parquet",
    imaging_modality="CT",
    imaging_embedding_dim=1000,
    imaging_num_slices=190,  # Based on the shape we saw
    imaging_aggregation="mean",  # Mean pool across slices
    has_treatment_text=False,
    clinical_features=["bmi", "serum_ca_19.9", "histology", "grade", "smoking_status"],
    categorical_features=["histology", "grade", "smoking_status"],
    numerical_features=["bmi", "serum_ca_19.9"],
    survival_time_col="survival_time_days",
    event_col="vital_status",  # Use vital_status which has DEAD/ALIVE
    event_positive_value="DEAD",  # DEAD indicates event occurred
    imaging_col="ct_embeddings",
    imaging_shape_col="ct_embedding_shape",
    radiology_embedding_col="radiology_embeddings",
    pathology_embedding_col="pathology_embeddings",
)

NSCLC_CONFIG = DatasetConfig(
    name="NSCLC",
    data_path="/mnt/f/Projects/EAGLE/data/NSCLC/unimodal.parquet",
    imaging_modality="CT",
    imaging_embedding_dim=1024,
    text_embedding_dim=1024,
    has_treatment_text=False,
    clinical_features=[
        "AGE_AT_DIAGNOSIS_NUM",
        "RACE_CR_SRC_DESC_1",
        "ETHNICITY_SRC_DESC",
        "HISTOLOGY_DESC",
        "STAGE_CLINICAL_TNM_T_DESC",
        "STAGE_CLINICAL_TNM_N_DESC",
        "STAGE_CLINICAL_TNM_M_DESC",
        "STAGE_PATHOLOGICAL_TNM_T_DESC",
        "STAGE_PATHOLOGICAL_TNM_N_DESC",
        "STAGE_PATHOLOGICAL_TNM_M_DESC",
        "DERIVED_TOBACCO_SMOKING_STATUS_DESC",
    ],
    categorical_features=[
        "RACE_CR_SRC_DESC_1",
        "ETHNICITY_SRC_DESC",
        "HISTOLOGY_DESC",
        "STAGE_CLINICAL_TNM_T_DESC",
        "STAGE_CLINICAL_TNM_N_DESC",
        "STAGE_CLINICAL_TNM_M_DESC",
        "STAGE_PATHOLOGICAL_TNM_T_DESC",
        "STAGE_PATHOLOGICAL_TNM_N_DESC",
        "STAGE_PATHOLOGICAL_TNM_M_DESC",
        "DERIVED_TOBACCO_SMOKING_STATUS_DESC",
    ],
    numerical_features=["AGE_AT_DIAGNOSIS_NUM"],
    survival_time_col="SURVIVAL_TIME_IN_MONTHS",
    event_col="event",
    event_positive_value=1,
    imaging_col="ct_contrast_embeddings",
    imaging_shape_col="ct_contrast_embedding_shape",
    radiology_embedding_col="clinical_embeddings",
    pathology_embedding_col="clinical_embeddings",
    imaging_col_secondary="ct_wo_contrast_embeddings",
    imaging_shape_col_secondary="ct_wo_contrast_embedding_shape",
)

# MedGemma configurations
GBM_MEDGEMMA_CONFIG = DatasetConfig(
    name="GBM",
    data_path="/mnt/f/Projects/EAGLE/data/GBM/medgemma.parquet",
    imaging_modality="MRI",
    imaging_embedding_dim=1000,
    imaging_num_slices=155,
    has_treatment_text=True,
    clinical_features=[
        "age_at_diagnosis_num",
        "gender_src_desc",
        "survival_time_in_months",
    ],
    survival_time_col="survival_time_in_months",
    event_col="vital_status_desc",
    event_positive_value="DEAD",
    imaging_col="mri_embeddings",
    imaging_shape_col="mri_embedding_shape",
    radiology_embedding_col="radiology_report_embeddings",
    pathology_embedding_col="pathology_report_embeddings",
    treatment_embedding_col="treatment_embeddings",
)

IPMN_MEDGEMMA_CONFIG = DatasetConfig(
    name="IPMN",
    data_path="/mnt/f/Projects/EAGLE/data/IPMN/medgemma.parquet",
    imaging_modality="CT",
    imaging_embedding_dim=1000,
    imaging_num_slices=190,
    imaging_aggregation="mean",
    has_treatment_text=False,
    clinical_features=["bmi", "serum_ca_19.9", "histology", "grade", "smoking_status"],
    categorical_features=["histology", "grade", "smoking_status"],
    numerical_features=["bmi", "serum_ca_19.9"],
    survival_time_col="survival_time_days",
    event_col="vital_status",
    event_positive_value="DEAD",
    imaging_col="ct_embeddings",
    imaging_shape_col="ct_embedding_shape",
    radiology_embedding_col="radiology_embeddings",
    pathology_embedding_col="pathology_embeddings",
)

NSCLC_MEDGEMMA_CONFIG = DatasetConfig(
    name="NSCLC",
    data_path="/mnt/f/Projects/EAGLE/data/NSCLC/medgemma.parquet",
    imaging_modality="CT",
    imaging_embedding_dim=1024,
    text_embedding_dim=1024,
    has_treatment_text=False,
    clinical_features=[
        "AGE_AT_DIAGNOSIS_NUM",
        "RACE_CR_SRC_DESC_1",
        "ETHNICITY_SRC_DESC",
        "HISTOLOGY_DESC",
        "STAGE_CLINICAL_TNM_T_DESC",
        "STAGE_CLINICAL_TNM_N_DESC",
        "STAGE_CLINICAL_TNM_M_DESC",
        "STAGE_PATHOLOGICAL_TNM_T_DESC",
        "STAGE_PATHOLOGICAL_TNM_N_DESC",
        "STAGE_PATHOLOGICAL_TNM_M_DESC",
        "DERIVED_TOBACCO_SMOKING_STATUS_DESC",
    ],
    categorical_features=[
        "RACE_CR_SRC_DESC_1",
        "ETHNICITY_SRC_DESC",
        "HISTOLOGY_DESC",
        "STAGE_CLINICAL_TNM_T_DESC",
        "STAGE_CLINICAL_TNM_N_DESC",
        "STAGE_CLINICAL_TNM_M_DESC",
        "STAGE_PATHOLOGICAL_TNM_T_DESC",
        "STAGE_PATHOLOGICAL_TNM_N_DESC",
        "STAGE_PATHOLOGICAL_TNM_M_DESC",
        "DERIVED_TOBACCO_SMOKING_STATUS_DESC",
    ],
    numerical_features=["AGE_AT_DIAGNOSIS_NUM"],
    survival_time_col="SURVIVAL_TIME_IN_MONTHS",
    event_col="event",
    event_positive_value=1,
    imaging_col="ct_contrast_embeddings",
    imaging_shape_col="ct_contrast_embedding_shape",
    radiology_embedding_col="clinical_embeddings",
    pathology_embedding_col="clinical_embeddings",
    imaging_col_secondary="ct_wo_contrast_embeddings",
    imaging_shape_col_secondary="ct_wo_contrast_embedding_shape",
)


# Feature extractors
class BaseFeatureExtractor(ABC):
    """Base class for feature extraction"""

    @abstractmethod
    def extract_features(self, text: str) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        pass


class GBMFeatureExtractor(BaseFeatureExtractor):
    """Extract GBM-specific features from text"""

    def extract_features(
        self, pathology_text: str, radiology_text: str
    ) -> Dict[str, float]:
        combined_text = f"{pathology_text} {radiology_text}".lower()

        features = {}

        # Tumor type classification
        features["is_oligodendroglioma"] = (
            1 if "oligodendroglioma" in combined_text else 0
        )
        features["is_gbm"] = (
            1 if any(term in combined_text for term in ["glioblastoma", "gbm"]) else 0
        )
        features["is_low_grade"] = (
            1 if any(term in combined_text for term in ["grade ii", "grade 2"]) else 0
        )

        # Molecular markers
        features["mgmt_methylated"] = (
            1 if re.search(r"mgmt.{0,20}methylat", combined_text) else 0
        )
        features["idh_mutant"] = (
            1 if re.search(r"idh.{0,20}(mutant|mutation)", combined_text) else 0
        )
        features["1p19q_codeleted"] = (
            1 if re.search(r"1p.{0,10}19q.{0,20}(delet|loss)", combined_text) else 0
        )

        # Treatment and characteristics
        features["gross_total_resection"] = (
            1 if re.search(r"gross total|gtr|complete resection", combined_text) else 0
        )
        features["multifocal"] = 1 if "multifocal" in combined_text else 0
        features["enhancement"] = 1 if "enhanc" in combined_text else 0
        features["necrosis"] = 1 if "necrosis" in combined_text else 0

        return features

    def get_feature_names(self) -> List[str]:
        return [
            "is_oligodendroglioma",
            "is_low_grade",
            "is_gbm",
            "mgmt_methylated",
            "idh_mutant",
            "1p19q_codeleted",
            "gross_total_resection",
            "multifocal",
            "enhancement",
            "necrosis",
        ]


class IPMNFeatureExtractor(BaseFeatureExtractor):
    """Extract IPMN-specific features from text"""

    def __init__(self):
        self.feature_patterns = {
            "main_duct": r"main[\s-]?duct|md[\s-]?ipmn",
            "branch_duct": r"branch[\s-]?duct|bd[\s-]?ipmn",
            "worrisome_features": r"worrisome|mural nodule",
            "high_risk": r"high[\s-]?risk|obstructive jaundice",
            "invasive": r"invasive|invasion|infiltrat",
            "size_large": r"[4-9]\.\d+\s*cm|\d{2,}\s*mm",
        }

    def extract_features(self, text: str) -> Dict[str, float]:
        if pd.isna(text) or text == "":
            return {name: 0.0 for name in self.feature_patterns.keys()}

        text_lower = str(text).lower()
        features = {}

        for pattern_name, pattern in self.feature_patterns.items():
            features[pattern_name] = 1.0 if re.search(pattern, text_lower) else 0.0

        return features

    def get_feature_names(self) -> List[str]:
        return list(self.feature_patterns.keys())


class NSCLCFeatureExtractor(BaseFeatureExtractor):
    """Extract NSCLC-specific features from clinical text"""

    def __init__(self):
        self.stage_patterns = {
            "early_stage": r"stage\s*(i|1|ia|ib|1a|1b)",
            "locally_advanced": r"stage\s*(ii|2|iii|3|iiia|iiib|3a|3b)",
            "metastatic": r"stage\s*(iv|4|iva|ivb|4a|4b)",
            "adenocarcinoma": r"adenocarcinoma",
            "squamous": r"squamous|epidermoid",
            "small_cell": r"small\s*cell",
            "egfr_mutation": r"egfr\s*(mutation|mutant|positive)",
            "alk_rearrangement": r"alk\s*(rearrangement|fusion|positive)",
            "pdl1_expression": r"pd-?l1\s*(positive|expression|high)",
            "smoking_heavy": r"pack\s*year[s]?\s*>\s*30|heavy\s*smoker",
            "pleural_effusion": r"pleural\s*effusion",
            "brain_metastases": r"brain\s*(metastases|mets)",
        }

    def extract_features(self, text: str) -> Dict[str, float]:
        if pd.isna(text) or text == "":
            return {name: 0.0 for name in self.stage_patterns.keys()}

        text_lower = str(text).lower()
        features = {}

        for pattern_name, pattern in self.stage_patterns.items():
            features[pattern_name] = 1.0 if re.search(pattern, text_lower) else 0.0

        return features

    def get_feature_names(self) -> List[str]:
        return list(self.stage_patterns.keys())


def get_text_extractor(dataset_name: str) -> Optional[BaseFeatureExtractor]:
    """Get appropriate text extractor based on dataset"""
    if dataset_name == "GBM":
        return GBMFeatureExtractor()
    elif dataset_name == "IPMN":
        return IPMNFeatureExtractor()
    elif dataset_name == "NSCLC":
        return NSCLCFeatureExtractor()
    else:
        return None


# Clinical feature processor
class UnifiedClinicalProcessor:
    """Unified clinical feature processor"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}

    def fit(self, df: pd.DataFrame):
        """Fit processors on training data"""
        # Fit encoders for categorical features
        for col in self.config.categorical_features:
            if col in df.columns:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col].fillna("Unknown"))

        # Fit imputers and scalers for numerical features
        numerical_data = []
        for col in self.config.numerical_features:
            if col in df.columns:
                self.imputers[col] = SimpleImputer(strategy="median")
                imputed = self.imputers[col].fit_transform(
                    df[col].values.reshape(-1, 1)
                )
                numerical_data.append(imputed)

        if numerical_data:
            numerical_array = np.hstack(numerical_data)
            self.scalers["numerical"] = StandardScaler()
            self.scalers["numerical"].fit(numerical_array)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform clinical features"""
        features = []

        # Encode categorical features
        for col in self.config.categorical_features:
            if col in df.columns and col in self.encoders:
                encoded = self.encoders[col].transform(df[col].fillna("Unknown"))
                features.append(encoded.reshape(-1, 1))

        # Process numerical features
        numerical_features = []
        for col in self.config.numerical_features:
            if col in df.columns and col in self.imputers:
                imputed = self.imputers[col].transform(df[col].values.reshape(-1, 1))
                numerical_features.append(imputed)

        if numerical_features and "numerical" in self.scalers:
            numerical_array = np.hstack(numerical_features)
            scaled = self.scalers["numerical"].transform(numerical_array)
            features.append(scaled)

        if features:
            return np.hstack(features).astype(np.float32)
        else:
            return np.zeros((len(df), 1), dtype=np.float32)


# Main dataset class
class UnifiedSurvivalDataset(Dataset):
    """Unified dataset for GBM, IPMN, and NSCLC"""

    def __init__(
        self,
        df: pd.DataFrame,
        config: DatasetConfig,
        clinical_processor: UnifiedClinicalProcessor,
        text_extractor: Optional[BaseFeatureExtractor] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.clinical_processor = clinical_processor
        self.text_extractor = text_extractor

        # For IPMN, check and update configuration based on actual data
        if self.config.name == "IPMN":
            self._update_ipmn_config()

        # Process clinical features
        self.clinical_features = self.clinical_processor.transform(self.df)

        # Extract text features if available
        self.text_features = None
        if self.text_extractor:
            self._extract_text_features()

        # Log statistics
        self._log_stats()

    def _update_ipmn_config(self):
        """Update IPMN configuration based on actual data columns"""
        logging.info(f"IPMN: Checking data structure...")
        logging.info(f"IPMN: Total columns: {len(self.df.columns)}")

        # The IPMN data has ct_embeddings and ct_embedding_shape
        # No need to create dummy data or update columns
        logging.info(
            f"IPMN: Using configured columns - imaging: {self.config.imaging_col}, shape: {self.config.imaging_shape_col}"
        )

    def _extract_text_features(self):
        """Extract features from text reports"""
        self.text_features = []

        for idx, row in self.df.iterrows():
            if isinstance(self.text_extractor, GBMFeatureExtractor):
                pathology = str(row.get("pathology_report", ""))
                radiology = str(row.get("radiology_report", ""))
                features = self.text_extractor.extract_features(pathology, radiology)
            elif isinstance(self.text_extractor, NSCLCFeatureExtractor):
                clinical_text = str(row.get("clinical_text", ""))
                features = self.text_extractor.extract_features(clinical_text)
            elif isinstance(self.text_extractor, IPMNFeatureExtractor):
                # For IPMN, combine reports
                text = (
                    str(row.get("pathology_report", ""))
                    + " "
                    + str(row.get("radiology_report", ""))
                )
                features = self.text_extractor.extract_features(text)
            else:
                # Default case
                features = {
                    name: 0.0 for name in self.text_extractor.get_feature_names()
                }

            self.text_features.append(list(features.values()))

        self.text_features = np.array(self.text_features, dtype=np.float32)

    def _log_stats(self):
        """Log dataset statistics"""
        logging.info(f"Dataset: {self.config.name}")
        logging.info(f"Size: {len(self.df)}")
        logging.info(f"Clinical features shape: {self.clinical_features.shape}")
        if self.text_features is not None:
            logging.info(f"Text features shape: {self.text_features.shape}")

        # Log all columns for debugging
        logging.info(
            f"Available columns: {list(self.df.columns)[:20]}..."
        )  # Show first 20 columns

        # For IPMN, check for possible imaging columns
        if self.config.name == "IPMN":
            possible_imaging_cols = [
                col
                for col in self.df.columns
                if "ct" in col.lower() or "embedding" in col.lower()
            ]
            logging.info(f"Possible imaging columns: {possible_imaging_cols}")

        # Check imaging availability
        if self.config.imaging_col in self.df.columns:
            imaging_available = self.df[self.config.imaging_col].notna().sum()
            logging.info(
                f"Primary imaging embeddings: {imaging_available}/{len(self.df)}"
            )
        else:
            logging.warning(
                f"Primary imaging column '{self.config.imaging_col}' not found"
            )
            # Try to find alternative imaging columns
            if self.config.name == "IPMN":
                # Check if there's a different naming convention
                for col in ["ct_embeddings", "CT_embeddings", "imaging_embeddings"]:
                    if col in self.df.columns:
                        logging.info(f"Found alternative imaging column: {col}")
                        imaging_available = self.df[col].notna().sum()
                        logging.info(
                            f"Alternative imaging embeddings: {imaging_available}/{len(self.df)}"
                        )
                        break

        # Check text embeddings
        if self.config.name == "NSCLC":
            if "clinical_embeddings" in self.df.columns:
                available = self.df["clinical_embeddings"].notna().sum()
                logging.info(f"Clinical embeddings: {available}/{len(self.df)}")
        else:
            # GBM/IPMN use radiology/pathology/treatment embeddings
            embedding_configs = [
                ("radiology", "radiology_embedding_col"),
                ("pathology", "pathology_embedding_col"),
            ]
            if self.config.name == "GBM" and self.config.has_treatment_text:
                embedding_configs.append(("treatment", "treatment_embedding_col"))

            for col_name, col_attr in embedding_configs:
                if hasattr(self.config, col_attr):
                    col = getattr(self.config, col_attr)
                    if col and col in self.df.columns:
                        available = self.df[col].notna().sum()
                        logging.info(
                            f"{col_name} embeddings: {available}/{len(self.df)}"
                        )
                    elif col:
                        logging.info(
                            f"{col_name} embeddings column '{col}' not found in dataframe"
                        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            # Process imaging
            imaging_features = self._process_imaging(row)

            # Process text embeddings
            text_embeddings = self._process_text_embeddings(row)

            # Debug logging
            # Uncomment for debugging
            # if idx < 5:  # Log first few samples
            #     logging.debug(f"Sample {idx} - Raw text embeddings keys: {list(text_embeddings.keys())}")

            # Ensure all expected keys are present for the dataset type
            if self.config.name == "NSCLC":
                expected_keys = ["clinical"]
            elif self.config.name == "GBM":
                expected_keys = ["radiology", "pathology"]
                if self.config.has_treatment_text:
                    expected_keys.append("treatment")
            else:  # IPMN
                expected_keys = ["radiology", "pathology"]

            # Convert text embeddings to tensors and ensure all keys exist
            text_emb_tensors = {}
            for key in expected_keys:
                if key in text_embeddings:
                    text_emb_tensors[key] = torch.FloatTensor(
                        text_embeddings[key].copy()
                    )
                else:
                    # Create dummy embedding for missing keys
                    text_emb_tensors[key] = torch.zeros(
                        self.config.text_embedding_dim, dtype=torch.float32
                    )

            # Debug logging
            # Uncomment for debugging
            # if idx < 5:  # Log first few samples
            #     logging.debug(f"Sample {idx} - Final text embeddings keys: {list(text_emb_tensors.keys())}")
            #     for k, v in text_emb_tensors.items():
            #         logging.debug(f"  {k}: shape {v.shape}")

            # Handle different time units based on dataset
            if self.config.survival_time_col == "survival_time_days":
                survival_time = (
                    float(row[self.config.survival_time_col]) / 30.44
                )  # Convert days to months
            else:
                survival_time = float(row[self.config.survival_time_col])

            result = {
                "imaging_features": torch.FloatTensor(
                    imaging_features.copy()
                ),  # Copy to make writable
                "text_embeddings": text_emb_tensors,
                "clinical_features": torch.FloatTensor(
                    self.clinical_features[idx].copy()
                ),
                "text_features": torch.FloatTensor(self.text_features[idx].copy())
                if self.text_features is not None
                else torch.zeros(1),
                "survival_time": survival_time,
                "event": float(
                    row[self.config.event_col] == self.config.event_positive_value
                ),
                "patient_id": str(row.get("patient_id", f"patient_{idx}")),
            }

            # Verify text_embeddings structure
            assert all(key in result["text_embeddings"] for key in expected_keys), (
                f"Missing keys in text_embeddings. Expected: {expected_keys}, Got: {list(result['text_embeddings'].keys())}"
            )

            return result

        except Exception as e:
            logging.error(f"Error processing sample {idx}: {str(e)}")
            logging.error(f"Row data: {row.to_dict()}")
            raise

    def _process_imaging(self, row) -> np.ndarray:
        """Process imaging embeddings based on configuration"""
        if self.config.name == "NSCLC":
            return self._process_nsclc_imaging(row)
        else:
            return self._process_single_modality_imaging(row)

    def _process_nsclc_imaging(self, row) -> np.ndarray:
        """Process NSCLC dual CT imaging"""
        contrast_img = None
        wo_contrast_img = None
        embed_dim = self.config.imaging_embedding_dim

        # Process contrast CT if available
        if pd.notna(row[self.config.imaging_col]):
            contrast_img = self._process_single_imaging(
                row[self.config.imaging_col], row[self.config.imaging_shape_col]
            )

        # Process non-contrast CT if available
        if self.config.imaging_col_secondary and pd.notna(
            row[self.config.imaging_col_secondary]
        ):
            wo_contrast_img = self._process_single_imaging(
                row[self.config.imaging_col_secondary],
                row[self.config.imaging_shape_col_secondary],
            )

        # Handle different availability scenarios
        if contrast_img is not None and wo_contrast_img is not None:
            combined = np.concatenate([contrast_img, wo_contrast_img])
        elif contrast_img is not None:
            combined = np.concatenate([contrast_img, np.zeros(embed_dim)])
        elif wo_contrast_img is not None:
            combined = np.concatenate([np.zeros(embed_dim), wo_contrast_img])
        else:
            combined = np.zeros(embed_dim * 2)

        return combined

    def _process_single_modality_imaging(self, row) -> np.ndarray:
        """Process single modality imaging"""
        # Get the value from the row
        img_value = row.get(self.config.imaging_col)

        if pd.notna(img_value):
            # Check if this is dummy imaging data
            if self.config.imaging_col == "dummy_imaging":
                return np.zeros(self.config.imaging_embedding_dim, dtype=np.float32)

            # Check if shape column exists and has data
            has_shape = False
            shape_value = None

            if (
                self.config.imaging_shape_col is not None
                and self.config.imaging_shape_col in self.df.columns
            ):
                shape_value = row.get(self.config.imaging_shape_col)
                # Check if shape_value is valid (not None, not NaN if scalar)
                if shape_value is not None:
                    if isinstance(shape_value, np.ndarray):
                        has_shape = True  # numpy arrays are valid shapes
                    elif not pd.isna(shape_value):
                        has_shape = True  # non-NaN scalar or other type

            if has_shape:
                return self._process_single_imaging(img_value, shape_value)
            else:
                # Handle case where shape column is missing - assume 1D embedding
                logging.debug(f"Shape column missing or empty, assuming 1D embedding")
                img_array = np.frombuffer(img_value, dtype=np.float32).copy()

                # Ensure correct dimension
                if len(img_array) != self.config.imaging_embedding_dim:
                    if len(img_array) > self.config.imaging_embedding_dim:
                        img_array = img_array[: self.config.imaging_embedding_dim]
                    else:
                        img_array = np.pad(
                            img_array,
                            (0, self.config.imaging_embedding_dim - len(img_array)),
                        )

                return img_array
        else:
            return np.zeros(self.config.imaging_embedding_dim, dtype=np.float32)

    def _process_single_imaging(self, img_bytes, img_shape_data) -> np.ndarray:
        """Process a single imaging embedding"""
        try:
            img_shape = parse_embedding_shape(img_shape_data)
            logging.debug(f"Processing imaging with shape: {img_shape}")

            img_array = np.frombuffer(img_bytes, dtype=np.float32).copy()
            expected_size = np.prod(img_shape)

            if len(img_array) != expected_size:
                logging.warning(
                    f"Size mismatch: array has {len(img_array)} elements, expected {expected_size} from shape {img_shape}"
                )
                # Try to reshape anyway if it's close
                if len(img_array) >= expected_size:
                    img_array = img_array[:expected_size]
                else:
                    raise ValueError(
                        f"Array too small: {len(img_array)} < {expected_size}"
                    )

            img_array = img_array.reshape(img_shape)

            # Apply aggregation based on configuration
            if len(img_shape) > 1:
                if self.config.imaging_aggregation == "mean":
                    # Mean pool across slices (first dimension)
                    img_features = img_array.mean(axis=0)
                    logging.debug(
                        f"Applied mean pooling: {img_shape} -> {img_features.shape}"
                    )
                elif self.config.imaging_aggregation == "concat":
                    # Flatten all slices
                    img_features = img_array.flatten()
                else:
                    # Default: flatten
                    img_features = img_array.flatten()
            else:
                img_features = img_array

            # Ensure output matches expected dimension
            if (
                self.config.imaging_aggregation == "mean"
                and len(img_features) != self.config.imaging_embedding_dim
            ):
                # For datasets where the actual embedding dimension differs slightly
                # This is not necessarily an error - just adjust silently
                if (
                    abs(len(img_features) - self.config.imaging_embedding_dim) <= 24
                ):  # Small difference
                    logging.debug(
                        f"Adjusting dimension from {len(img_features)} to {self.config.imaging_embedding_dim}"
                    )
                else:
                    logging.warning(
                        f"Large dimension mismatch after aggregation: got {len(img_features)}, expected {self.config.imaging_embedding_dim}"
                    )

                if len(img_features) > self.config.imaging_embedding_dim:
                    img_features = img_features[: self.config.imaging_embedding_dim]
                else:
                    img_features = np.pad(
                        img_features,
                        (0, self.config.imaging_embedding_dim - len(img_features)),
                    )

            return img_features

        except Exception as e:
            logging.warning(
                f"Error processing imaging with shape {img_shape_data}: {str(e)}"
            )
            # Fallback: try to decode as 1D array
            img_array = np.frombuffer(img_bytes, dtype=np.float32).copy()

            # Ensure correct dimension
            if len(img_array) != self.config.imaging_embedding_dim:
                if len(img_array) > self.config.imaging_embedding_dim:
                    img_array = img_array[: self.config.imaging_embedding_dim]
                else:
                    img_array = np.pad(
                        img_array,
                        (0, self.config.imaging_embedding_dim - len(img_array)),
                    )

            return img_array

    def _process_text_embeddings(self, row) -> Dict[str, np.ndarray]:
        """Process text embeddings based on dataset type"""
        text_embeddings = {}

        if self.config.name == "GBM":
            for key, col_attr in [
                ("radiology", "radiology_embedding_col"),
                ("pathology", "pathology_embedding_col"),
                ("treatment", "treatment_embedding_col"),
            ]:
                if hasattr(self.config, col_attr):
                    col = getattr(self.config, col_attr)
                    if col and col in self.df.columns and pd.notna(row.get(col)):
                        text_embeddings[key] = self._process_single_text_embedding(
                            row[col]
                        )

        elif self.config.name == "IPMN":
            for key, col_attr in [
                ("radiology", "radiology_embedding_col"),
                ("pathology", "pathology_embedding_col"),
            ]:
                if hasattr(self.config, col_attr):
                    col = getattr(self.config, col_attr)
                    if col and col in self.df.columns and pd.notna(row.get(col)):
                        text_embeddings[key] = self._process_single_text_embedding(
                            row[col]
                        )

        elif self.config.name == "NSCLC":
            if hasattr(self.config, "radiology_embedding_col"):
                col = self.config.radiology_embedding_col
                if col in self.df.columns and pd.notna(row.get(col)):
                    text_embeddings["clinical"] = self._process_single_text_embedding(
                        row[col]
                    )

        return text_embeddings

    def _process_single_text_embedding(self, emb_bytes) -> np.ndarray:
        """Process a single text embedding"""
        emb_array = np.frombuffer(
            emb_bytes, dtype=np.float32
        ).copy()  # Copy to ensure writability

        # Ensure correct dimension
        if len(emb_array) != self.config.text_embedding_dim:
            if len(emb_array) > self.config.text_embedding_dim:
                emb_array = emb_array[: self.config.text_embedding_dim]
            else:
                emb_array = np.pad(
                    emb_array, (0, self.config.text_embedding_dim - len(emb_array))
                )

        return emb_array