"""
EAGLE: Multimodal Survival Prediction Framework with Attribution Support
"""

from .data import (
    DatasetConfig,
    UnifiedSurvivalDataset,
    UnifiedClinicalProcessor,
    get_text_extractor,
    GBM_CONFIG,
    IPMN_CONFIG,
    NSCLC_CONFIG,
    GBM_MEDGEMMA_CONFIG,
    IPMN_MEDGEMMA_CONFIG,
    NSCLC_MEDGEMMA_CONFIG,
)
from .models import ModelConfig, UnifiedSurvivalModel
from .train import UnifiedTrainer
from .eval import UnifiedRiskStratification
from .viz import plot_km_curves, create_comprehensive_plots
from .attribution import (
    ModalityAttributionAnalyzer,
    AttributionResult,
    plot_modality_contributions,
    plot_patient_level_attribution,
    create_attribution_report,
)

import os
import logging
import pandas as pd


# Main pipeline class
class UnifiedPipeline:
    """Main pipeline for running the unified framework with attribution support"""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelConfig = None,
        output_dirs: dict = None,
    ):
        self.dataset_config = dataset_config
        self.output_dirs = output_dirs or {
            "models": ".",
            "results": ".",
            "figures": ".",
            "logs": ".",
            "attribution": ".",
        }

        # Create dataset-specific default model config if not provided
        if model_config is None:
            if dataset_config.name == "NSCLC":
                model_config = ModelConfig(
                    imaging_encoder_dims=[512, 256, 128],
                    text_encoder_dims=[256, 128],
                    clinical_encoder_dims=[128, 64, 32],
                    fusion_dims=[256, 128, 64],
                    dropout=0.35,
                    batch_size=24,
                    learning_rate=5e-5,
                )
            elif dataset_config.name == "IPMN":
                model_config = ModelConfig(
                    imaging_encoder_dims=[256, 128],
                    text_encoder_dims=[256, 128],
                    clinical_encoder_dims=[64, 32],
                    fusion_dims=[128, 64],
                    batch_size=32,
                )
            else:  # GBM
                model_config = ModelConfig()

        self.model_config = model_config

    def run(
        self, n_folds: int = 5, n_risk_groups: int = 3, enable_attribution: bool = False,
        enable_comprehensive_attribution: bool = False
    ):
        """Run the complete pipeline with optional attribution analysis"""
        import logging
        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader
        from sklearn.model_selection import StratifiedKFold
        from .data import UnifiedClinicalProcessor, get_text_extractor

        # Setup logging
        log_file = os.path.join(self.output_dirs["logs"], "training.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),  # Also print to console
            ],
        )
        logging.info(f"Running {self.dataset_config.name} analysis")
        if enable_attribution:
            logging.info("Attribution analysis enabled")
        if enable_comprehensive_attribution:
            logging.info("Comprehensive attribution analysis enabled (all three methods)")

        # Load data
        df = pd.read_parquet(self.dataset_config.data_path)
        df = self._filter_data(df)

        # Initialize processors
        clinical_processor = UnifiedClinicalProcessor(self.dataset_config)
        clinical_processor.fit(df)

        # Initialize text extractor
        text_extractor = get_text_extractor(self.dataset_config.name)

        # Create dataset
        dataset = UnifiedSurvivalDataset(
            df, self.dataset_config, clinical_processor, text_extractor
        )

        # Get feature dimensions
        num_clinical = dataset.clinical_features.shape[1]
        num_text_features = (
            len(text_extractor.get_feature_names()) if text_extractor else 0
        )

        # Cross-validation
        all_scores = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Create stratification variable
        if self.dataset_config.event_col in df.columns:
            if isinstance(self.dataset_config.event_positive_value, str):
                stratify_var = (
                    (
                        df[self.dataset_config.event_col]
                        == self.dataset_config.event_positive_value
                    )
                    .astype(int)
                    .values
                )
            else:
                stratify_var = (
                    (
                        df[self.dataset_config.event_col]
                        == self.dataset_config.event_positive_value
                    )
                    .astype(int)
                    .values
                )
        else:
            logging.warning(
                f"Event column '{self.dataset_config.event_col}' not found, using zeros for stratification"
            )
            stratify_var = np.zeros(len(df), dtype=int)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(range(len(dataset)), stratify_var)
        ):
            logging.info(f"Fold {fold + 1}/{n_folds}")

            # Create data loaders
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.model_config.batch_size,
                shuffle=True,
                num_workers=2,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.model_config.batch_size,
                shuffle=False,
                num_workers=2,
            )

            # Create model
            model = UnifiedSurvivalModel(
                dataset_config=self.dataset_config,
                model_config=self.model_config,
                num_clinical_features=num_clinical,
                num_text_features=num_text_features,
            )

            try:
                # Train
                trainer = UnifiedTrainer()
                model, val_cindex = trainer.train_model(
                    model,
                    train_loader,
                    val_loader,
                    self.model_config,
                    save_path=os.path.join(
                        self.output_dirs["models"], f"best_model_fold{fold + 1}.pth"
                    ),
                )

                # Evaluate
                test_cindex = trainer.evaluate(model, val_loader)
                all_scores.append(test_cindex)

                logging.info(f"Fold {fold + 1} C-index: {test_cindex:.4f}")

                # Save fold model
                model_path = os.path.join(
                    self.output_dirs["models"], f"fold{fold + 1}.pth"
                )
                torch.save(model.state_dict(), model_path)
            except Exception as e:
                logging.error(f"Error in fold {fold + 1}: {str(e)}")
                # Try to get a sample from the dataset to debug
                try:
                    sample = dataset[train_idx[0]]
                    logging.error(f"Sample structure: {sample.keys()}")
                    if "text_embeddings" in sample:
                        logging.error(
                            f"Text embeddings keys: {sample['text_embeddings'].keys()}"
                        )
                except:
                    pass
                raise

        # Results
        best_fold = np.argmax(all_scores)
        results = {
            "dataset": self.dataset_config.name,
            "mean_cindex": np.mean(all_scores),
            "std_cindex": np.std(all_scores),
            "all_scores": all_scores,
            "best_fold": best_fold,
        }

        # Risk stratification on best model
        model = UnifiedSurvivalModel(
            dataset_config=self.dataset_config,
            model_config=self.model_config,
            num_clinical_features=num_clinical,
            num_text_features=num_text_features,
        )
        best_model_path = os.path.join(
            self.output_dirs["models"], f"fold{best_fold + 1}.pth"
        )
        model.load_state_dict(torch.load(best_model_path))

        # Run risk stratification with attribution if enabled
        risk_analyzer = UnifiedRiskStratification(
            model, dataset, 
            enable_attribution=enable_attribution,
            enable_comprehensive_attribution=enable_comprehensive_attribution
        )
        risk_df = risk_analyzer.generate_risk_scores(
            compute_attributions=enable_attribution
        )
        risk_df = risk_analyzer.stratify_patients(risk_df, n_groups=n_risk_groups)

        # Calculate statistics
        stats = risk_analyzer.calculate_statistics(risk_df)

        # Analyze top patients if attribution is enabled
        if enable_attribution and "imaging_contribution" in risk_df.columns:
            top_analysis = risk_analyzer.analyze_top_patients(
                risk_df, n_top=10, n_bottom=10
            )
            if top_analysis is not None:
                top_analysis_path = os.path.join(
                    self.output_dirs["attribution"], "top_bottom_patients_analysis.csv"
                )
                top_analysis.to_csv(top_analysis_path, index=False)
                logging.info(
                    f"Top/bottom patient analysis saved to: {top_analysis_path}"
                )

        return results, risk_df, stats

    def _filter_data(self, df):
        """Apply dataset-specific filtering"""
        if self.dataset_config.name == "GBM":
            return df[
                (df["mri_available"] == True)
                & (df["has_pathology_report"] == True)
                & (df["has_radiology_report"] == True)
                & (df["survival_time_in_months"].notna())
                & (df["vital_status_desc"].notna())
            ].copy()
        elif self.dataset_config.name == "IPMN":
            # More flexible filtering for IPMN
            filtered = df[
                (df["survival_time_days"] > 0) & (df["survival_time_days"].notna())
            ].copy()

            # Check event column - IPMN might use different naming
            if self.dataset_config.event_col not in filtered.columns:
                logging.warning(
                    f"Event column '{self.dataset_config.event_col}' not found."
                )
                # Try to find event column
                event_cols = [
                    col
                    for col in filtered.columns
                    if any(
                        term in col.lower()
                        for term in ["event", "vital", "status", "death"]
                    )
                ]
                logging.info(f"Possible event columns: {event_cols}")

                # Priority order for event columns
                for col in [
                    "vital_status",
                    "vital_status_binary",
                    "event_observed",
                    "death",
                    "status",
                ]:
                    if col in filtered.columns:
                        self.dataset_config.event_col = col
                        break

                # Update event positive value based on the column found
                if self.dataset_config.event_col in filtered.columns:
                    unique_values = filtered[self.dataset_config.event_col].unique()
                    logging.info(
                        f"Event column '{self.dataset_config.event_col}' values: {unique_values}"
                    )

                    # Determine positive value
                    if set(unique_values) <= {0, 1, "0", "1", 0.0, 1.0, True, False}:
                        # Binary column
                        self.dataset_config.event_positive_value = 1
                    else:
                        # Text column - look for death-related terms
                        for val in unique_values:
                            if pd.notna(val) and any(
                                term in str(val).upper()
                                for term in ["DEAD", "DECEASED", "DIED", "EXPIRE"]
                            ):
                                self.dataset_config.event_positive_value = val
                                break

                    logging.info(
                        f"Using event_col='{self.dataset_config.event_col}' with positive_value='{self.dataset_config.event_positive_value}'"
                    )

            return filtered
        elif self.dataset_config.name == "NSCLC":
            return df[
                (
                    (df["ct_contrast_embeddings"].notna())
                    | (df["ct_wo_contrast_embeddings"].notna())
                )
                & (df["clinical_embeddings"].notna())
                & (df["SURVIVAL_TIME_IN_MONTHS"].notna())
                & (df["event"].notna())
            ].copy()
        else:
            return df.copy()


__all__ = [
    "UnifiedPipeline",
    "DatasetConfig",
    "ModelConfig",
    "UnifiedSurvivalDataset",
    "UnifiedSurvivalModel",
    "UnifiedTrainer",
    "UnifiedRiskStratification",
    "UnifiedClinicalProcessor",
    "get_text_extractor",
    "GBM_CONFIG",
    "IPMN_CONFIG",
    "NSCLC_CONFIG",
    "GBM_MEDGEMMA_CONFIG",
    "IPMN_MEDGEMMA_CONFIG",
    "NSCLC_MEDGEMMA_CONFIG",
    "plot_km_curves",
    "create_comprehensive_plots",
    "ModalityAttributionAnalyzer",
    "AttributionResult",
    "plot_modality_contributions",
    "plot_patient_level_attribution",
    "create_attribution_report",
]
