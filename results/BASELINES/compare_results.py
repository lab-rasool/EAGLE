"""
Compare baseline results with EAGLE performance
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from tabulate import tabulate


def load_results(results_dir):
    """Load all results from directory"""
    results = []
    
    for json_file in Path(results_dir).glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append({
                'Model': data['model'],
                'Dataset': data['dataset'],
                'Mean C-index': f"{data['mean_cindex']:.4f}",
                'Std C-index': f"{data['std_cindex']:.4f}",
                'C-index': f"{data['mean_cindex']:.4f} ± {data['std_cindex']:.4f}"
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Compare baseline results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--eagle-results", type=str, help="Path to EAGLE results JSON")
    
    args = parser.parse_args()
    
    # Load baseline results
    df = load_results(args.results_dir)
    
    # Add EAGLE results if provided
    if args.eagle_results and Path(args.eagle_results).exists():
        with open(args.eagle_results, 'r') as f:
            eagle_data = json.load(f)
            for dataset in ['GBM', 'IPMN', 'NSCLC']:
                if dataset in eagle_data:
                    df = df.append({
                        'Model': 'EAGLE',
                        'Dataset': dataset,
                        'Mean C-index': f"{eagle_data[dataset]['mean_cindex']:.4f}",
                        'Std C-index': f"{eagle_data[dataset]['std_cindex']:.4f}",
                        'C-index': f"{eagle_data[dataset]['mean_cindex']:.4f} ± {eagle_data[dataset]['std_cindex']:.4f}"
                    }, ignore_index=True)
    
    # Sort and display results
    df = df.sort_values(['Dataset', 'Model'])
    
    print("\n=== SURVIVAL PREDICTION RESULTS ===\n")
    
    # Show results by dataset
    for dataset in df['Dataset'].unique():
        print(f"\n{dataset} Dataset:")
        dataset_df = df[df['Dataset'] == dataset][['Model', 'C-index']]
        print(tabulate(dataset_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Create pivot table
    pivot = df.pivot(index='Model', columns='Dataset', values='Mean C-index')
    print("\n\nSummary Table (Mean C-index):")
    print(tabulate(pivot, headers='keys', tablefmt='grid'))


if __name__ == "__main__":
    main()