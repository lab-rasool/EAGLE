"""
Attribution and interpretability module for EAGLE
Tracks and analyzes modality contributions to risk predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from dataclasses import dataclass
from collections import defaultdict
import os


@dataclass
class AttributionResult:
    """Container for attribution analysis results"""
    patient_id: str
    risk_score: float
    modality_scores: Dict[str, float]
    attention_weights: Optional[Dict[str, np.ndarray]] = None
    gradient_attributions: Optional[Dict[str, float]] = None
    feature_level_attributions: Optional[Dict[str, np.ndarray]] = None


class ModalityAttributionAnalyzer:
    """Analyze modality contributions using multiple attribution methods"""
    
    def __init__(self, model, dataset, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        
        # Store intermediate activations and gradients
        self.activations = {}
        self.gradients = {}
        self.attention_weights = {}
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        
        # Hook for imaging encoder output
        def imaging_hook(module, input, output):
            self.activations['imaging'] = output.detach()
        
        # Hook for text encoder output  
        def text_hook(module, input, output):
            self.activations['text'] = output.detach()
            
        # Hook for clinical encoder output
        def clinical_hook(module, input, output):
            self.activations['clinical'] = output.detach()
        
        # Register forward hooks
        self.model.imaging_encoder.register_forward_hook(imaging_hook)
        self.model.text_encoder.register_forward_hook(text_hook)
        self.model.clinical_encoder.register_forward_hook(clinical_hook)
        
        # Hook for attention weights if using cross-attention
        if hasattr(self.model, 'imaging_text_attention'):
            def attention_hook(module, input, output):
                # MultiheadAttention returns (output, attention_weights)
                if hasattr(module, 'attention'):
                    self.attention_weights['imaging_text'] = output.detach()
            
            self.model.imaging_text_attention.register_forward_hook(attention_hook)
            
            if hasattr(self.model, 'imaging_clinical_attention'):
                def clinical_attention_hook(module, input, output):
                    if hasattr(module, 'attention'):
                        self.attention_weights['imaging_clinical'] = output.detach()
                
                self.model.imaging_clinical_attention.register_forward_hook(clinical_attention_hook)
    
    def compute_gradient_attribution(self, sample_idx: int) -> Dict[str, float]:
        """Compute gradient-based attribution for each modality"""
        
        # Get sample
        sample = self.dataset[sample_idx]
        
        # Prepare inputs
        imaging = sample['imaging_features'].unsqueeze(0).to(self.device).requires_grad_(True)
        text_emb = {k: v.unsqueeze(0).to(self.device).requires_grad_(True) 
                   for k, v in sample['text_embeddings'].items()}
        clinical = sample['clinical_features'].unsqueeze(0).to(self.device).requires_grad_(True)
        text_feat = sample['text_features'].unsqueeze(0).to(self.device).requires_grad_(True)
        
        # Forward pass
        risk_score, _ = self.model(imaging, text_emb, clinical, text_feat)
        
        # Compute gradients
        self.model.zero_grad()
        risk_score.backward()
        
        # Compute integrated gradients approximation
        attributions = {}
        
        # Imaging attribution
        imaging_grad = imaging.grad.abs().mean().item()
        imaging_activation = self.activations['imaging'].abs().mean().item()
        attributions['imaging'] = imaging_grad * imaging_activation
        
        # Text attribution
        text_grad = sum(v.grad.abs().mean().item() for v in text_emb.values())
        text_activation = self.activations['text'].abs().mean().item()
        attributions['text'] = text_grad * text_activation
        
        # Clinical attribution
        clinical_grad = clinical.grad.abs().mean().item()
        clinical_activation = self.activations['clinical'].abs().mean().item()
        attributions['clinical'] = clinical_grad * clinical_activation
        
        # Normalize to percentages
        total = sum(attributions.values())
        if total > 0:
            attributions = {k: v/total * 100 for k, v in attributions.items()}
        
        return attributions
    
    def compute_integrated_gradients(self, sample_idx: int, n_steps: int = 50) -> Dict[str, float]:
        """Compute integrated gradients for more accurate attribution"""
        
        sample = self.dataset[sample_idx]
        
        # Prepare inputs
        imaging = sample['imaging_features'].unsqueeze(0).to(self.device)
        text_emb = {k: v.unsqueeze(0).to(self.device) for k, v in sample['text_embeddings'].items()}
        clinical = sample['clinical_features'].unsqueeze(0).to(self.device)
        text_feat = sample['text_features'].unsqueeze(0).to(self.device)
        
        # Create baselines (zeros)
        imaging_baseline = torch.zeros_like(imaging)
        text_emb_baseline = {k: torch.zeros_like(v) for k, v in text_emb.items()}
        clinical_baseline = torch.zeros_like(clinical)
        text_feat_baseline = torch.zeros_like(text_feat)
        
        # Accumulate gradients
        imaging_grads = []
        text_grads = []
        clinical_grads = []
        
        for step in range(n_steps):
            # Interpolate inputs
            alpha = step / n_steps
            
            imaging_interp = imaging_baseline + alpha * (imaging - imaging_baseline)
            imaging_interp.requires_grad_(True)
            
            text_emb_interp = {}
            for k in text_emb:
                text_emb_interp[k] = text_emb_baseline[k] + alpha * (text_emb[k] - text_emb_baseline[k])
                text_emb_interp[k].requires_grad_(True)
            
            clinical_interp = clinical_baseline + alpha * (clinical - clinical_baseline)
            clinical_interp.requires_grad_(True)
            
            text_feat_interp = text_feat_baseline + alpha * (text_feat - text_feat_baseline)
            text_feat_interp.requires_grad_(True)
            
            # Forward pass
            risk_score, _ = self.model(imaging_interp, text_emb_interp, clinical_interp, text_feat_interp)
            
            # Compute gradients
            self.model.zero_grad()
            risk_score.backward()
            
            # Collect gradients
            imaging_grads.append(imaging_interp.grad.detach())
            text_grads.append([v.grad.detach() for v in text_emb_interp.values()])
            clinical_grads.append(clinical_interp.grad.detach())
        
        # Compute integrated gradients
        imaging_ig = torch.stack(imaging_grads).mean(0) * (imaging - imaging_baseline)
        text_ig = sum(torch.stack([g[i] for g in text_grads]).mean(0) * (text_emb[k] - text_emb_baseline[k])
                     for i, k in enumerate(text_emb))
        clinical_ig = torch.stack(clinical_grads).mean(0) * (clinical - clinical_baseline)
        
        # Compute attribution scores
        attributions = {
            'imaging': imaging_ig.abs().sum().item(),
            'text': text_ig.abs().sum().item(),
            'clinical': clinical_ig.abs().sum().item()
        }
        
        # Normalize
        total = sum(attributions.values())
        if total > 0:
            attributions = {k: v/total * 100 for k, v in attributions.items()}
        
        return attributions
    
    def analyze_patient(self, sample_idx: int) -> AttributionResult:
        """Perform complete attribution analysis for a single patient"""
        
        sample = self.dataset[sample_idx]
        patient_id = sample['patient_id']
        
        # Get risk score
        with torch.no_grad():
            imaging = sample['imaging_features'].unsqueeze(0).to(self.device)
            text_emb = {k: v.unsqueeze(0).to(self.device) for k, v in sample['text_embeddings'].items()}
            clinical = sample['clinical_features'].unsqueeze(0).to(self.device)
            text_feat = sample['text_features'].unsqueeze(0).to(self.device)
            
            risk_score, _ = self.model(imaging, text_emb, clinical, text_feat)
            risk_score = risk_score.item()
        
        # Compute attributions
        gradient_attrs = self.compute_gradient_attribution(sample_idx)
        integrated_grads = self.compute_integrated_gradients(sample_idx)
        
        # Average the two methods
        modality_scores = {}
        for modality in ['imaging', 'text', 'clinical']:
            modality_scores[modality] = (gradient_attrs[modality] + integrated_grads[modality]) / 2
        
        # Get attention weights if available
        attention_weights = None
        if self.attention_weights:
            attention_weights = {k: v.cpu().numpy() for k, v in self.attention_weights.items()}
        
        return AttributionResult(
            patient_id=patient_id,
            risk_score=risk_score,
            modality_scores=modality_scores,
            attention_weights=attention_weights,
            gradient_attributions=gradient_attrs
        )
    
    def analyze_cohort(self, sample_indices: Optional[List[int]] = None, 
                      max_samples: int = None) -> pd.DataFrame:
        """Analyze attribution for multiple patients"""
        
        if sample_indices is None:
            sample_indices = list(range(len(self.dataset)))
            if max_samples:
                sample_indices = sample_indices[:max_samples]
        
        results = []
        
        for idx in tqdm(sample_indices, desc="Analyzing patients"):
            try:
                result = self.analyze_patient(idx)
                
                # Create row for dataframe
                row = {
                    'patient_id': result.patient_id,
                    'risk_score': result.risk_score,
                    'imaging_contribution': result.modality_scores['imaging'],
                    'text_contribution': result.modality_scores['text'],
                    'clinical_contribution': result.modality_scores['clinical']
                }
                
                # Add survival info
                sample = self.dataset[idx]
                row['survival_time'] = sample['survival_time'].item()
                row['event'] = sample['event'].item()
                
                results.append(row)
                
            except Exception as e:
                logging.warning(f"Error analyzing patient {idx}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def compute_feature_importance(self, sample_idx: int) -> Dict[str, np.ndarray]:
        """Compute feature-level importance within each modality"""
        
        sample = self.dataset[sample_idx]
        feature_importance = {}
        
        # For clinical features, use permutation importance
        clinical_features = sample['clinical_features'].numpy()
        clinical_importance = np.zeros_like(clinical_features)
        
        # Get baseline risk score
        with torch.no_grad():
            imaging = sample['imaging_features'].unsqueeze(0).to(self.device)
            text_emb = {k: v.unsqueeze(0).to(self.device) for k, v in sample['text_embeddings'].items()}
            clinical = sample['clinical_features'].unsqueeze(0).to(self.device)
            text_feat = sample['text_features'].unsqueeze(0).to(self.device)
            
            baseline_risk, _ = self.model(imaging, text_emb, clinical, text_feat)
            baseline_risk = baseline_risk.item()
        
        # Permute each clinical feature
        for i in range(len(clinical_features)):
            clinical_perm = clinical.clone()
            # Shuffle this feature across the batch dimension (here just adding noise)
            clinical_perm[0, i] = clinical_perm[0, i] + torch.randn(1).to(self.device) * 0.5
            
            with torch.no_grad():
                risk_perm, _ = self.model(imaging, text_emb, clinical_perm, text_feat)
                clinical_importance[i] = abs(baseline_risk - risk_perm.item())
        
        feature_importance['clinical'] = clinical_importance / (clinical_importance.sum() + 1e-8)
        
        # For text features if available
        if self.dataset.text_features is not None:
            text_features = sample['text_features'].numpy()
            text_importance = np.zeros_like(text_features)
            
            for i in range(len(text_features)):
                text_feat_perm = text_feat.clone()
                text_feat_perm[0, i] = 1 - text_feat_perm[0, i]  # Flip binary feature
                
                with torch.no_grad():
                    risk_perm, _ = self.model(imaging, text_emb, clinical, text_feat_perm)
                    text_importance[i] = abs(baseline_risk - risk_perm.item())
            
            feature_importance['text_features'] = text_importance / (text_importance.sum() + 1e-8)
        
        return feature_importance


def plot_modality_contributions(attribution_df: pd.DataFrame, 
                               save_path: str = 'modality_contributions.png',
                               dataset_name: str = None):
    """Plot modality contribution analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Average contribution pie chart
    ax = axes[0, 0]
    avg_contributions = {
        'Imaging': attribution_df['imaging_contribution'].mean(),
        'Text': attribution_df['text_contribution'].mean(),
        'Clinical': attribution_df['clinical_contribution'].mean()
    }
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    wedges, texts, autotexts = ax.pie(avg_contributions.values(), 
                                      labels=avg_contributions.keys(),
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      explode=(0.05, 0.05, 0.05))
    ax.set_title(f'Average Modality Contributions\n({dataset_name if dataset_name else "All Patients"})',
                fontsize=14, fontweight='bold')
    
    # 2. Contribution distribution boxplots
    ax = axes[0, 1]
    contribution_data = pd.DataFrame({
        'Imaging': attribution_df['imaging_contribution'],
        'Text': attribution_df['text_contribution'], 
        'Clinical': attribution_df['clinical_contribution']
    })
    
    box = ax.boxplot([contribution_data[col] for col in contribution_data.columns],
                     labels=contribution_data.columns,
                     patch_artist=True,
                     showmeans=True)
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Contribution (%)', fontsize=12)
    ax.set_title('Distribution of Modality Contributions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Contribution vs Risk Score scatter
    ax = axes[1, 0]
    
    # Create scatter plot with different colors for each modality
    ax.scatter(attribution_df['risk_score'], attribution_df['imaging_contribution'],
              alpha=0.6, label='Imaging', color=colors[0], s=30)
    ax.scatter(attribution_df['risk_score'], attribution_df['text_contribution'],
              alpha=0.6, label='Text', color=colors[1], s=30)
    ax.scatter(attribution_df['risk_score'], attribution_df['clinical_contribution'],
              alpha=0.6, label='Clinical', color=colors[2], s=30)
    
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_ylabel('Contribution (%)', fontsize=12)
    ax.set_title('Modality Contribution vs Risk Score', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Contribution by outcome
    ax = axes[1, 1]
    
    # Group by event status
    event_groups = attribution_df.groupby('event')
    
    x = np.arange(3)
    width = 0.35
    
    for i, (event, group) in enumerate(event_groups):
        means = [group['imaging_contribution'].mean(),
                group['text_contribution'].mean(),
                group['clinical_contribution'].mean()]
        
        label = 'Death' if event == 1 else 'Censored'
        offset = width/2 if i == 0 else -width/2
        bars = ax.bar(x + offset, means, width, label=label, alpha=0.8)
    
    ax.set_xlabel('Modality', fontsize=12)
    ax.set_ylabel('Average Contribution (%)', fontsize=12)
    ax.set_title('Modality Contributions by Outcome', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Imaging', 'Text', 'Clinical'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_patient_level_attribution(result: AttributionResult, 
                                 feature_names: Dict[str, List[str]] = None,
                                 save_path: str = None):
    """Plot attribution analysis for a single patient"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Modality contributions
    ax = axes[0]
    modalities = list(result.modality_scores.keys())
    contributions = list(result.modality_scores.values())
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(modalities, contributions, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, contributions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    ax.set_ylabel('Contribution (%)', fontsize=12)
    ax.set_title(f'Modality Contributions for Patient {result.patient_id}\nRisk Score: {result.risk_score:.3f}',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(contributions) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Feature importance within modalities (if available)
    ax = axes[1]
    
    if result.feature_level_attributions:
        # Show top features from each modality
        top_features = []
        feature_values = []
        feature_colors = []
        
        for modality, importance in result.feature_level_attributions.items():
            if feature_names and modality in feature_names:
                names = feature_names[modality]
                # Get top 5 features
                top_indices = np.argsort(importance)[-5:][::-1]
                
                for idx in top_indices:
                    if idx < len(names):
                        top_features.append(f"{modality}: {names[idx]}")
                        feature_values.append(importance[idx] * 100)
                        
                        # Color by modality
                        if modality == 'clinical':
                            feature_colors.append(colors[2])
                        elif modality == 'text_features':
                            feature_colors.append(colors[1])
                        else:
                            feature_colors.append(colors[0])
        
        if top_features:
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, feature_values, color=feature_colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features, fontsize=10)
            ax.set_xlabel('Feature Importance (%)', fontsize=12)
            ax.set_title('Top Contributing Features', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'Feature-level attribution not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.text(0.5, 0.5, 'Feature-level attribution not available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_attribution_report(attribution_df: pd.DataFrame, 
                            output_dir: str,
                            dataset_name: str = None):
    """Create a comprehensive attribution analysis report"""
    
    # Summary statistics
    summary_stats = pd.DataFrame({
        'Mean': attribution_df[['imaging_contribution', 'text_contribution', 'clinical_contribution']].mean(),
        'Std': attribution_df[['imaging_contribution', 'text_contribution', 'clinical_contribution']].std(),
        'Min': attribution_df[['imaging_contribution', 'text_contribution', 'clinical_contribution']].min(),
        'Max': attribution_df[['imaging_contribution', 'text_contribution', 'clinical_contribution']].max()
    }).T
    
    # Save summary
    summary_path = os.path.join(output_dir, 'modality_contribution_summary.csv')
    summary_stats.to_csv(summary_path)
    
    # Create text report
    report_path = os.path.join(output_dir, 'attribution_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Modality Attribution Analysis Report\n")
        f.write(f"{'='*50}\n\n")
        
        if dataset_name:
            f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of patients analyzed: {len(attribution_df)}\n\n")
        
        f.write("Average Modality Contributions:\n")
        f.write(f"  - Imaging: {attribution_df['imaging_contribution'].mean():.1f}%\n")
        f.write(f"  - Text: {attribution_df['text_contribution'].mean():.1f}%\n")
        f.write(f"  - Clinical: {attribution_df['clinical_contribution'].mean():.1f}%\n\n")
        
        # Correlation with outcomes
        f.write("Correlation with Risk Score:\n")
        for col in ['imaging_contribution', 'text_contribution', 'clinical_contribution']:
            corr = attribution_df[col].corr(attribution_df['risk_score'])
            f.write(f"  - {col.replace('_contribution', '').title()}: {corr:.3f}\n")
        
        f.write("\n")
        
        # By outcome group
        f.write("Average Contributions by Outcome:\n")
        for event in [0, 1]:
            event_data = attribution_df[attribution_df['event'] == event]
            event_name = "Death" if event == 1 else "Censored"
            f.write(f"\n{event_name} (n={len(event_data)}):\n")
            f.write(f"  - Imaging: {event_data['imaging_contribution'].mean():.1f}%\n")
            f.write(f"  - Text: {event_data['text_contribution'].mean():.1f}%\n")
            f.write(f"  - Clinical: {event_data['clinical_contribution'].mean():.1f}%\n")
    
    logging.info(f"Attribution analysis report saved to {output_dir}")
    
    return summary_stats