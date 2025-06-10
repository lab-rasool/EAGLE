"""
Evaluation and risk stratification utilities for EAGLE with attribution support
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, List
from lifelines.utils import concordance_index
from lifelines.statistics import pairwise_logrank_test
import logging

from .models import UnifiedSurvivalModel
from .data import UnifiedSurvivalDataset


class UnifiedRiskStratification:
    """Unified risk stratification analysis with attribution support"""

    def __init__(
        self,
        model: UnifiedSurvivalModel,
        dataset: UnifiedSurvivalDataset,
        device: str = "cuda",
        enable_attribution: bool = False,
        enable_comprehensive_attribution: bool = False,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.enable_attribution = enable_attribution
        self.enable_comprehensive_attribution = enable_comprehensive_attribution

        # Store attribution information if enabled
        self.attribution_scores = []

    def generate_risk_scores(self, compute_attributions: bool = False) -> pd.DataFrame:
        """Generate risk scores for all patients with optional attribution analysis"""
        
        # If comprehensive attribution is enabled, use the attribution analyzer
        if self.enable_comprehensive_attribution and compute_attributions:
            from .attribution import ModalityAttributionAnalyzer
            
            logging.info("Running comprehensive attribution analysis...")
            analyzer = ModalityAttributionAnalyzer(self.model, self.dataset, self.device)
            attribution_df = analyzer.analyze_cohort_comprehensive()
            
            logging.info(f"Attribution analysis completed for {len(attribution_df)} patients")
            
            # Get risk scores for consistency
            loader = DataLoader(self.dataset, batch_size=32, shuffle=False)
            all_risks = []
            all_times = []
            all_events = []
            all_ids = []
            
            with torch.no_grad():
                for batch in tqdm(loader, desc="Computing risk scores"):
                    imaging = batch["imaging_features"].to(self.device)
                    text_emb = {
                        k: v.to(self.device) for k, v in batch["text_embeddings"].items()
                    }
                    clinical = batch["clinical_features"].to(self.device)
                    text_feat = batch["text_features"].to(self.device)
                    
                    risk_scores, _ = self.model(imaging, text_emb, clinical, text_feat)
                    
                    all_risks.extend(risk_scores.cpu().numpy().flatten())
                    all_times.extend(batch["survival_time"].numpy())
                    all_events.extend(batch["event"].numpy())
                    all_ids.extend(batch["patient_id"])
            
            # Create risk dataframe
            risk_df = pd.DataFrame({
                "patient_id": all_ids,
                "risk_score": all_risks,
                "survival_time": all_times,
                "event": all_events,
            })
            
            # Merge with comprehensive attribution results
            # Drop duplicate columns from attribution_df before merging
            merge_cols = [col for col in attribution_df.columns if col not in ['risk_score', 'survival_time', 'event'] or col == 'patient_id']
            risk_df = risk_df.merge(attribution_df[merge_cols], on="patient_id", how="left")
            
            # Add simple attribution columns for backward compatibility
            if "simple_imaging" in risk_df.columns:
                risk_df["imaging_contribution"] = risk_df["simple_imaging"]
                risk_df["text_contribution"] = risk_df["simple_text"]
                risk_df["clinical_contribution"] = risk_df["simple_clinical"]
                
            # Log summary of comprehensive attribution results
            if all(col in risk_df.columns for col in ["simple_imaging", "gradient_imaging", "ig_imaging"]):
                logging.info("\nComprehensive Attribution Summary:")
                logging.info(f"  Simple:    Imaging={risk_df['simple_imaging'].mean():.1f}%, Text={risk_df['simple_text'].mean():.1f}%, Clinical={risk_df['simple_clinical'].mean():.1f}%")
                logging.info(f"  Gradient:  Imaging={risk_df['gradient_imaging'].mean():.1f}%, Text={risk_df['gradient_text'].mean():.1f}%, Clinical={risk_df['gradient_clinical'].mean():.1f}%")
                logging.info(f"  IG:        Imaging={risk_df['ig_imaging'].mean():.1f}%, Text={risk_df['ig_text'].mean():.1f}%, Clinical={risk_df['ig_clinical'].mean():.1f}%")
            
            # Verify required columns exist
            required_cols = ["patient_id", "risk_score", "survival_time", "event"]
            missing_cols = [col for col in required_cols if col not in risk_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns after merge: {missing_cols}")
            
            return risk_df
        
        # Original simple attribution path
        loader = DataLoader(self.dataset, batch_size=32, shuffle=False)

        all_risks = []
        all_times = []
        all_events = []
        all_ids = []

        # Attribution tracking
        all_imaging_contributions = []
        all_text_contributions = []
        all_clinical_contributions = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing risk scores"):
                imaging = batch["imaging_features"].to(self.device)
                text_emb = {
                    k: v.to(self.device) for k, v in batch["text_embeddings"].items()
                }
                clinical = batch["clinical_features"].to(self.device)
                text_feat = batch["text_features"].to(self.device)

                risk_scores, aux_outputs = self.model(
                    imaging,
                    text_emb,
                    clinical,
                    text_feat,
                    return_attention_weights=compute_attributions,
                )

                all_risks.extend(risk_scores.cpu().numpy().flatten())
                all_times.extend(batch["survival_time"].numpy())
                all_events.extend(batch["event"].numpy())
                all_ids.extend(batch["patient_id"])

                # Compute simple attribution scores based on encoder outputs
                if compute_attributions and self.enable_attribution:
                    # Get modality embeddings
                    modality_embeddings = self.model.get_modality_embeddings(
                        imaging, text_emb, clinical, text_feat
                    )

                    # Compute contribution scores based on embedding magnitudes
                    imaging_contrib = modality_embeddings["imaging"].abs().mean(dim=1)
                    text_contrib = modality_embeddings["text"].abs().mean(dim=1)
                    clinical_contrib = modality_embeddings["clinical"].abs().mean(dim=1)

                    # Normalize to percentages
                    total = imaging_contrib + text_contrib + clinical_contrib
                    imaging_pct = (imaging_contrib / total * 100).cpu().numpy()
                    text_pct = (text_contrib / total * 100).cpu().numpy()
                    clinical_pct = (clinical_contrib / total * 100).cpu().numpy()

                    all_imaging_contributions.extend(imaging_pct)
                    all_text_contributions.extend(text_pct)
                    all_clinical_contributions.extend(clinical_pct)

        # Create base dataframe
        risk_df = pd.DataFrame(
            {
                "patient_id": all_ids,
                "risk_score": all_risks,
                "survival_time": all_times,
                "event": all_events,
            }
        )

        # Add attribution scores if computed
        if compute_attributions and all_imaging_contributions:
            risk_df["imaging_contribution"] = all_imaging_contributions
            risk_df["text_contribution"] = all_text_contributions
            risk_df["clinical_contribution"] = all_clinical_contributions

            # Log average contributions
            logging.info("\nAverage Modality Contributions:")
            logging.info(f"  Imaging: {np.mean(all_imaging_contributions):.1f}%")
            logging.info(f"  Text: {np.mean(all_text_contributions):.1f}%")
            logging.info(f"  Clinical: {np.mean(all_clinical_contributions):.1f}%")

        return risk_df

    def stratify_patients(
        self, risk_df: pd.DataFrame, n_groups: int = 3
    ) -> pd.DataFrame:
        """Stratify patients into risk groups"""
        if n_groups == 2:
            labels = ["Low Risk", "High Risk"]
            quantiles = [0, 0.5, 1.0]
        elif n_groups == 3:
            labels = ["Low Risk", "Medium Risk", "High Risk"]
            quantiles = [0, 0.33, 0.67, 1.0]
        elif n_groups == 4:
            labels = ["Low Risk", "Low-Medium Risk", "High-Medium Risk", "High Risk"]
            quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        else:
            raise ValueError(f"Unsupported number of groups: {n_groups}")

        risk_df["risk_group"] = pd.qcut(
            risk_df["risk_score"], q=quantiles, labels=labels
        )

        # Log statistics
        logging.info(f"\nRisk stratification into {n_groups} groups:")
        for group in labels:
            group_data = risk_df[risk_df["risk_group"] == group]
            event_rate = group_data["event"].mean()
            median_surv = group_data[group_data["event"] == 1]["survival_time"].median()
            logging.info(
                f"{group}: n={len(group_data)}, event_rate={event_rate:.1%}, "
                f"median_survival={median_surv:.1f}"
            )

            # If attribution scores are available, show by group
            if "imaging_contribution" in risk_df.columns:
                logging.info(f"  Average modality contributions:")
                logging.info(
                    f"    Imaging: {group_data['imaging_contribution'].mean():.1f}%"
                )
                logging.info(f"    Text: {group_data['text_contribution'].mean():.1f}%")
                logging.info(
                    f"    Clinical: {group_data['clinical_contribution'].mean():.1f}%"
                )

        return risk_df

    def calculate_statistics(self, risk_df: pd.DataFrame) -> Dict:
        """Calculate statistical tests"""
        results = {}

        # Pairwise log-rank tests
        if len(risk_df["risk_group"].unique()) > 1:
            pairwise_results = pairwise_logrank_test(
                risk_df["survival_time"], risk_df["risk_group"], risk_df["event"]
            )
            results["pairwise_logrank"] = pairwise_results
            logging.info("\nPairwise log-rank test results:")
            logging.info(pairwise_results.summary)

        # C-index
        c_index = concordance_index(
            risk_df["survival_time"], -risk_df["risk_score"], risk_df["event"]
        )
        results["c_index"] = c_index
        logging.info(f"\nModel C-index: {c_index:.4f}")

        # If attribution scores are available, compute correlations
        if "imaging_contribution" in risk_df.columns:
            results["attribution_stats"] = {
                "imaging_risk_corr": risk_df["imaging_contribution"].corr(
                    risk_df["risk_score"]
                ),
                "text_risk_corr": risk_df["text_contribution"].corr(
                    risk_df["risk_score"]
                ),
                "clinical_risk_corr": risk_df["clinical_contribution"].corr(
                    risk_df["risk_score"]
                ),
                "imaging_event_diff": (
                    risk_df[risk_df["event"] == 1]["imaging_contribution"].mean()
                    - risk_df[risk_df["event"] == 0]["imaging_contribution"].mean()
                ),
                "text_event_diff": (
                    risk_df[risk_df["event"] == 1]["text_contribution"].mean()
                    - risk_df[risk_df["event"] == 0]["text_contribution"].mean()
                ),
                "clinical_event_diff": (
                    risk_df[risk_df["event"] == 1]["clinical_contribution"].mean()
                    - risk_df[risk_df["event"] == 0]["clinical_contribution"].mean()
                ),
            }

            logging.info("\nModality Contribution Statistics:")
            logging.info("Correlation with risk score:")
            logging.info(
                f"  Imaging: {results['attribution_stats']['imaging_risk_corr']:.3f}"
            )
            logging.info(
                f"  Text: {results['attribution_stats']['text_risk_corr']:.3f}"
            )
            logging.info(
                f"  Clinical: {results['attribution_stats']['clinical_risk_corr']:.3f}"
            )

        return results

    def analyze_top_patients(
        self, risk_df: pd.DataFrame, n_top: int = 10, n_bottom: int = 10
    ) -> pd.DataFrame:
        """Analyze attribution patterns for highest and lowest risk patients"""
        if "imaging_contribution" not in risk_df.columns:
            logging.warning("Attribution scores not available for top patient analysis")
            return None

        # Get top and bottom risk patients
        sorted_df = risk_df.sort_values("risk_score", ascending=False)
        top_patients = sorted_df.head(n_top)
        bottom_patients = sorted_df.tail(n_bottom)

        # Create analysis summary
        analysis = pd.DataFrame(
            {
                "Group": ["High Risk"] * n_top + ["Low Risk"] * n_bottom,
                "patient_id": list(top_patients["patient_id"])
                + list(bottom_patients["patient_id"]),
                "risk_score": list(top_patients["risk_score"])
                + list(bottom_patients["risk_score"]),
                "imaging_contribution": list(top_patients["imaging_contribution"])
                + list(bottom_patients["imaging_contribution"]),
                "text_contribution": list(top_patients["text_contribution"])
                + list(bottom_patients["text_contribution"]),
                "clinical_contribution": list(top_patients["clinical_contribution"])
                + list(bottom_patients["clinical_contribution"]),
                "event": list(top_patients["event"]) + list(bottom_patients["event"]),
                "survival_time": list(top_patients["survival_time"])
                + list(bottom_patients["survival_time"]),
            }
        )

        # Log summary statistics
        logging.info("\nModality contributions by risk level:")
        for group in ["High Risk", "Low Risk"]:
            group_data = analysis[analysis["Group"] == group]
            logging.info(f"\n{group} patients (n={len(group_data)}):")
            logging.info(
                f"  Average imaging contribution: {group_data['imaging_contribution'].mean():.1f}%"
            )
            logging.info(
                f"  Average text contribution: {group_data['text_contribution'].mean():.1f}%"
            )
            logging.info(
                f"  Average clinical contribution: {group_data['clinical_contribution'].mean():.1f}%"
            )
            logging.info(f"  Event rate: {group_data['event'].mean():.1%}")

        return analysis
