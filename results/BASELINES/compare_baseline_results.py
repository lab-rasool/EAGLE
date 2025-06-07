#!/usr/bin/env python
"""
Comprehensive comparison of baseline survival models across different embedding approaches
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")


class BaselineResultsComparator:
    """Compare baseline survival model results across different embedding approaches"""
    
    def __init__(self, results_base_dir: str):
        self.results_base_dir = Path(results_base_dir)
        self.datasets = ["GBM", "IPMN", "NSCLC"]
        self.models = ["coxph", "rsf", "deepsurv"]
        self.embedding_types = {
            "baseline_results": "Unimodal",
            "medgemma_baseline_results": "MedGemma",
            "eagle_baseline_results": "EAGLE"
        }
        self.results_data = {}
        
    def load_all_results(self):
        """Load all baseline results from different embedding approaches"""
        print("Loading results...")
        
        for embedding_dir, embedding_name in self.embedding_types.items():
            embedding_path = self.results_base_dir / embedding_dir
            
            if not embedding_path.exists():
                print(f"Warning: Directory not found: {embedding_path}")
                continue
                
            self.results_data[embedding_name] = {}
            
            # Load results for each dataset and model
            for dataset in self.datasets:
                self.results_data[embedding_name][dataset] = {}
                
                for model in self.models:
                    # Find result files for this combination
                    if embedding_name == "EAGLE":
                        # EAGLE results have different naming pattern
                        pattern = f"eagle_embeddings_{dataset}_*_results.json"
                    elif embedding_name == "MedGemma":
                        pattern = f"medgemma_{model}_{dataset}_*_results.json"
                    else:
                        # Baseline results
                        pattern = f"{model}_{dataset}_*_results.json"
                    
                    result_files = list(embedding_path.glob(pattern))
                    
                    if result_files:
                        # Use the most recent file
                        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                        
                        try:
                            with open(latest_file, 'r') as f:
                                data = json.load(f)
                            
                            # Extract results based on file structure
                            if embedding_name == "EAGLE":
                                # EAGLE results have nested structure
                                if model in data.get('results', {}):
                                    model_data = data['results'][model]
                                    self.results_data[embedding_name][dataset][model] = {
                                        'mean_cindex': model_data.get('mean', 0),
                                        'std_cindex': model_data.get('std', 0),
                                        'cv_scores': model_data.get('scores', []),
                                        'n_samples': data.get('n_samples', 0),
                                        'n_features': data.get('n_features', 0)
                                    }
                            else:
                                # Standard baseline results
                                self.results_data[embedding_name][dataset][model] = {
                                    'mean_cindex': data.get('mean_cindex', 0),
                                    'std_cindex': data.get('std_cindex', 0),
                                    'cv_scores': data.get('cv_scores', []),
                                    'n_samples': data.get('n_samples', 0),
                                    'n_features': data.get('n_features_reduced', data.get('n_features', 0))
                                }
                                
                        except Exception as e:
                            print(f"Error loading {latest_file}: {e}")
                            self.results_data[embedding_name][dataset][model] = None
                    else:
                        print(f"No results found for {embedding_name} - {dataset} - {model}")
                        self.results_data[embedding_name][dataset][model] = None
        
        print("Results loaded successfully!")
        return self.results_data
    
    def create_results_dataframe(self):
        """Create a comprehensive DataFrame with all results"""
        rows = []
        
        for embedding_type, datasets_data in self.results_data.items():
            for dataset, models_data in datasets_data.items():
                for model, result in models_data.items():
                    if result is not None:
                        rows.append({
                            'Embedding': embedding_type,
                            'Dataset': dataset,
                            'Model': model.upper(),
                            'Mean_C_Index': result['mean_cindex'],
                            'Std_C_Index': result['std_cindex'],
                            'N_Samples': result['n_samples'],
                            'N_Features': result['n_features'],
                            'CV_Scores': result['cv_scores']
                        })
                    else:
                        rows.append({
                            'Embedding': embedding_type,
                            'Dataset': dataset,
                            'Model': model.upper(),
                            'Mean_C_Index': np.nan,
                            'Std_C_Index': np.nan,
                            'N_Samples': np.nan,
                            'N_Features': np.nan,
                            'CV_Scores': []
                        })
        
        return pd.DataFrame(rows)
    
    def create_summary_table(self, df):
        """Create summary table with best results highlighted"""
        # Pivot table for better visualization
        summary = df.pivot_table(
            index=['Dataset', 'Model'], 
            columns='Embedding', 
            values='Mean_C_Index',
            aggfunc='first'
        ).round(4)
        
        return summary
    
    def plot_performance_comparison(self, df, output_dir):
        """Create comprehensive performance comparison plots"""
        
        # 1. Heatmap of C-index scores
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Baseline Model Performance Comparison Across Embedding Types', fontsize=16)
        
        # Overall heatmap
        pivot_data = df.pivot_table(
            index=['Dataset', 'Model'], 
            columns='Embedding', 
            values='Mean_C_Index'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=axes[0,0], cbar_kws={'label': 'C-index'})
        axes[0,0].set_title('C-index Heatmap (All Methods)')
        axes[0,0].set_xlabel('Embedding Type')
        axes[0,0].set_ylabel('Dataset - Model')
        
        # 2. Bar plot by dataset
        df_melted = df.melt(id_vars=['Embedding', 'Dataset', 'Model'], 
                           value_vars=['Mean_C_Index'], 
                           var_name='Metric', value_name='Score')
        
        sns.barplot(data=df_melted, x='Dataset', y='Score', hue='Embedding', ax=axes[0,1])
        axes[0,1].set_title('Mean C-index by Dataset')
        axes[0,1].set_ylabel('C-index')
        axes[0,1].legend(title='Embedding Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Box plot of CV scores
        cv_data = []
        for _, row in df.iterrows():
            if row['CV_Scores'] and len(row['CV_Scores']) > 0:
                for score in row['CV_Scores']:
                    cv_data.append({
                        'Embedding': row['Embedding'],
                        'Dataset': row['Dataset'],
                        'Model': row['Model'],
                        'CV_Score': score
                    })
        
        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            sns.boxplot(data=cv_df, x='Dataset', y='CV_Score', hue='Embedding', ax=axes[1,0])
            axes[1,0].set_title('Cross-Validation Score Distribution')
            axes[1,0].set_ylabel('C-index')
            axes[1,0].legend(title='Embedding Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Model comparison
        sns.barplot(data=df_melted, x='Model', y='Score', hue='Embedding', ax=axes[1,1])
        axes[1,1].set_title('Mean C-index by Model Type')
        axes[1,1].set_ylabel('C-index')
        axes[1,1].legend(title='Embedding Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'baseline_comparison_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Detailed per-dataset comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance by Dataset and Embedding Type', fontsize=16)
        
        for i, dataset in enumerate(self.datasets):
            dataset_data = df[df['Dataset'] == dataset]
            
            if not dataset_data.empty:
                # Create pivot table for this dataset
                dataset_pivot = dataset_data.pivot_table(
                    index='Model', 
                    columns='Embedding', 
                    values='Mean_C_Index'
                )
                
                # Bar plot
                dataset_pivot.plot(kind='bar', ax=axes[i], rot=45)
                axes[i].set_title(f'{dataset} Dataset')
                axes[i].set_ylabel('C-index')
                axes[i].legend(title='Embedding Type')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'baseline_comparison_by_dataset.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_statistical_comparison(self, df):
        """Create statistical comparison of methods"""
        stats_results = {}
        
        for dataset in self.datasets:
            stats_results[dataset] = {}
            dataset_data = df[df['Dataset'] == dataset]
            
            if dataset_data.empty:
                continue
                
            # Compare embedding types for each model
            for model in self.models:
                model_data = dataset_data[dataset_data['Model'] == model.upper()]
                
                if len(model_data) > 1:  # Need at least 2 embedding types to compare
                    # Get CV scores for statistical testing
                    embedding_scores = {}
                    for _, row in model_data.iterrows():
                        if row['CV_Scores'] and len(row['CV_Scores']) > 0:
                            embedding_scores[row['Embedding']] = row['CV_Scores']
                    
                    if len(embedding_scores) >= 2:
                        # Find best performing embedding type
                        best_embedding = model_data.loc[model_data['Mean_C_Index'].idxmax(), 'Embedding']
                        best_score = model_data['Mean_C_Index'].max()
                        
                        stats_results[dataset][model] = {
                            'best_embedding': best_embedding,
                            'best_score': best_score,
                            'all_scores': dict(zip(model_data['Embedding'], model_data['Mean_C_Index']))
                        }
        
        return stats_results
    
    def generate_latex_table(self, summary_df, output_dir):
        """Generate LaTeX table for publication"""
        latex_str = summary_df.to_latex(
            float_format=lambda x: f'{x:.3f}' if not pd.isna(x) else '-',
            escape=False,
            caption='Comparison of survival model performance (C-index) across different embedding approaches',
            label='tab:baseline_comparison'
        )
        
        # Save to file
        with open(Path(output_dir) / 'baseline_comparison_table.tex', 'w') as f:
            f.write(latex_str)
        
        return latex_str
    
    def create_comprehensive_report(self, output_dir):
        """Create comprehensive analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load all results
        self.load_all_results()
        
        # Create DataFrame
        df = self.create_results_dataframe()
        
        # Save raw results
        df.to_csv(output_path / 'all_baseline_results.csv', index=False)
        
        # Create summary table
        summary_df = self.create_summary_table(df)
        summary_df.to_csv(output_path / 'baseline_summary_table.csv')
        
        # Create plots
        self.plot_performance_comparison(df, output_path)
        
        # Statistical comparison
        stats_results = self.create_statistical_comparison(df)
        
        # Generate LaTeX table
        latex_table = self.generate_latex_table(summary_df, output_path)
        
        # Create comprehensive text report
        report_path = output_path / 'baseline_comparison_report.txt'
        with open(report_path, 'w') as f:
            f.write("BASELINE SURVIVAL MODEL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SUMMARY OF RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(str(summary_df))
            f.write("\n\n")
            
            f.write("BEST PERFORMING METHODS BY DATASET:\n")
            f.write("-" * 35 + "\n")
            for dataset in self.datasets:
                dataset_data = df[df['Dataset'] == dataset]
                if not dataset_data.empty:
                    best_overall = dataset_data.loc[dataset_data['Mean_C_Index'].idxmax()]
                    f.write(f"{dataset}: {best_overall['Model']} with {best_overall['Embedding']} "
                           f"(C-index: {best_overall['Mean_C_Index']:.4f} ± {best_overall['Std_C_Index']:.4f})\n")
            
            f.write("\n\nDETAILED ANALYSIS:\n")
            f.write("-" * 18 + "\n")
            
            for embedding_type in self.embedding_types.values():
                f.write(f"\n{embedding_type} Embeddings:\n")
                embedding_data = df[df['Embedding'] == embedding_type]
                if not embedding_data.empty:
                    mean_performance = embedding_data['Mean_C_Index'].mean()
                    f.write(f"  Average C-index: {mean_performance:.4f}\n")
                    f.write(f"  Best performance: {embedding_data['Mean_C_Index'].max():.4f}\n")
                    f.write(f"  Worst performance: {embedding_data['Mean_C_Index'].min():.4f}\n")
                else:
                    f.write("  No results available\n")
        
        print(f"\nComprehensive report generated in: {output_path}")
        print("\nGenerated files:")
        print(f"- all_baseline_results.csv: Raw results data")
        print(f"- baseline_summary_table.csv: Summary table")
        print(f"- baseline_comparison_overview.png: Overview plots")
        print(f"- baseline_comparison_by_dataset.png: Dataset-specific plots")
        print(f"- baseline_comparison_table.tex: LaTeX table")
        print(f"- baseline_comparison_report.txt: Text report")
        
        return df, summary_df, stats_results


def main():
    parser = argparse.ArgumentParser(description="Compare baseline survival model results")
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default="/proj/rasool_lab_projects/Aakash/EAGLE/results/BASELINES",
        help="Base directory containing all baseline results"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="baseline_comparison_analysis",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    # Create comparator and run analysis
    comparator = BaselineResultsComparator(args.results_dir)
    df, summary_df, stats_results = comparator.create_comprehensive_report(args.output_dir)
    
    # Print quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print("\nSummary Table (C-index):")
    print(summary_df)
    
    print("\nBest performing method for each dataset:")
    for dataset in comparator.datasets:
        dataset_data = df[df['Dataset'] == dataset]
        if not dataset_data.empty:
            best = dataset_data.loc[dataset_data['Mean_C_Index'].idxmax()]
            print(f"{dataset}: {best['Model']} + {best['Embedding']} "
                  f"({best['Mean_C_Index']:.4f} ± {best['Std_C_Index']:.4f})")
    
    print(f"\nFull analysis saved to: {args.output_dir}")


if __name__ == "__main__":
    main()