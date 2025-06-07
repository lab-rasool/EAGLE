"""
Compare MedGemma baseline results
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from tabulate import tabulate


def load_medgemma_results(results_dir):
    """Load all MedGemma results from directory"""
    results = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    for json_file in results_path.glob("medgemma_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return pd.DataFrame(results)


def create_comparison_plot(df, output_dir):
    """Create comparison plots"""
    if df.empty:
        print("No data available for plotting")
        return
    
    # Convert Mean C-index to numeric
    df['Mean C-index Numeric'] = df['mean_cindex'].astype(float)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Grouped bar plot
    datasets = df['dataset'].unique()
    models = df['model'].unique()
    
    x = np.arange(len(datasets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        c_indices = []
        
        for dataset in datasets:
            dataset_model_data = model_data[model_data['dataset'] == dataset]
            if not dataset_model_data.empty:
                c_indices.append(dataset_model_data['Mean C-index Numeric'].iloc[0])
            else:
                c_indices.append(0)
        
        ax.bar(x + i * width, c_indices, width, label=model)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('C-index')
    ax.set_title('MedGemma Embedding Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'medgemma_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {Path(output_dir) / 'medgemma_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Compare MedGemma baseline results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, help="Save comparison table to file")
    
    args = parser.parse_args()
    
    # Load results
    df = load_medgemma_results(args.results_dir)
    
    if df.empty:
        print("No MedGemma results found!")
        return
    
    print("\n=== MEDGEMMA EMBEDDING SURVIVAL PREDICTION RESULTS ===\n")
    
    # Prepare display data
    display_data = []
    for _, row in df.iterrows():
        display_data.append({
            'Model': row['model'],
            'Dataset': row['dataset'],
            'C-index': f"{row['mean_cindex']:.4f} ± {row['std_cindex']:.4f}",
            'N Samples': row['n_samples'],
            'N Features': row.get('n_features_reduced', row.get('n_features', 0))
        })
    
    display_df = pd.DataFrame(display_data)
    
    # Show results by dataset
    for dataset in display_df['Dataset'].unique():
        print(f"\n{dataset} Dataset:")
        dataset_df = display_df[display_df['Dataset'] == dataset][['Model', 'C-index', 'N Samples', 'N Features']]
        print(tabulate(dataset_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Create pivot table for summary
    try:
        pivot_data = {}
        for _, row in df.iterrows():
            model = row['model']
            dataset = row['dataset']
            c_index = row['mean_cindex']
            
            if model not in pivot_data:
                pivot_data[model] = {}
            pivot_data[model][dataset] = c_index
        
        pivot_df = pd.DataFrame(pivot_data).T
        print("\n\nSummary Table (Mean C-index):")
        print(tabulate(pivot_df, headers='keys', tablefmt='grid'))
    except Exception as e:
        print(f"Could not create pivot table: {e}")
    
    # Create comparison plot
    try:
        create_comparison_plot(df, args.results_dir)
    except Exception as e:
        print(f"Could not create comparison plot: {e}")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Total experiments: {len(df)}")
    print(f"Datasets: {', '.join(df['dataset'].unique())}")
    print(f"Models: {', '.join(df['model'].unique())}")
    
    print(f"\nBest performing models:")
    for dataset in df['dataset'].unique():
        dataset_results = df[df['dataset'] == dataset]
        if not dataset_results.empty:
            best_idx = dataset_results['mean_cindex'].idxmax()
            best_result = dataset_results.loc[best_idx]
            print(f"  {dataset}: {best_result['model']} (C-index: {best_result['mean_cindex']:.4f} ± {best_result['std_cindex']:.4f})")
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write("=== MEDGEMMA BASELINE RESULTS ===\n\n")
            for dataset in display_df['Dataset'].unique():
                f.write(f"\n{dataset} Dataset:\n")
                dataset_df = display_df[display_df['Dataset'] == dataset][['Model', 'C-index', 'N Samples', 'N Features']]
                f.write(tabulate(dataset_df, headers='keys', tablefmt='grid', showindex=False))
                f.write("\n")
        
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()