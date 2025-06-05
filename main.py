#!/usr/bin/env python
"""
Example script showing how to use the EAGLE library with attribution analysis
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from eagle import (
    UnifiedPipeline,
    ModelConfig,
    GBM_CONFIG,
    IPMN_CONFIG,
    NSCLC_CONFIG,
    plot_km_curves,
    create_comprehensive_plots,
    ModalityAttributionAnalyzer,
    plot_modality_contributions,
    plot_patient_level_attribution,
    create_attribution_report,
)


def create_output_directory(dataset_name: str, base_dir: str = "results") -> dict:
    """Create organized output directory structure"""
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create directory structure
    output_dir = Path(base_dir) / dataset_name / timestamp
    
    # Create subdirectories
    dirs = {
        "base": output_dir,
        "models": output_dir / "models",
        "results": output_dir / "results",
        "figures": output_dir / "figures",
        "logs": output_dir / "logs",
        "attribution": output_dir / "attribution"  # Add attribution directory
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create a run info file
    run_info_path = output_dir / "run_info.txt"
    with open(run_info_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Output directory: {output_dir}\n")
    
    return dirs


def main():
    parser = argparse.ArgumentParser(description="EAGLE Survival Analysis with Attribution")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["GBM", "IPMN", "NSCLC"],
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="Custom path to dataset"
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--risk-groups",
        type=int,
        default=3,
        help="Number of risk stratification groups",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--output-base-dir", 
        type=str, 
        default="results", 
        help="Base directory for outputs"
    )
    parser.add_argument(
        "--analyze-attribution",
        action="store_true",
        help="Perform modality attribution analysis"
    )
    parser.add_argument(
        "--attribution-samples",
        type=int,
        default=None,
        help="Number of samples for detailed attribution analysis (default: all)"
    )
    parser.add_argument(
        "--top-patients",
        type=int,
        default=20,
        help="Number of top/bottom risk patients to analyze in detail"
    )

    args = parser.parse_args()

    # Create output directory structure
    output_dirs = create_output_directory(args.dataset, args.output_base_dir)
    
    print(f"\nOutput directory created: {output_dirs['base']}")

    # Select dataset configuration
    if args.dataset == "GBM":
        config = GBM_CONFIG
    elif args.dataset == "IPMN":
        config = IPMN_CONFIG
    else:  # NSCLC
        config = NSCLC_CONFIG

    # Override data path if provided
    if args.data_path:
        config.data_path = args.data_path

    # Create model configuration
    model_config = ModelConfig(
        num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr
    )

    # Create and run pipeline with output directory
    print(f"\nRunning {args.dataset} analysis...")
    print(f"Data path: {config.data_path}")
    print(f"Folds: {args.folds}, Risk groups: {args.risk_groups}")
    print(f"Attribution analysis: {'Enabled' if args.analyze_attribution else 'Disabled'}")
    print("-" * 60)

    pipeline = UnifiedPipeline(config, model_config, output_dirs=output_dirs)
    results, risk_df, stats = pipeline.run(
        n_folds=args.folds, 
        n_risk_groups=args.risk_groups,
        enable_attribution=args.analyze_attribution
    )

    # Print results
    print("\nResults:")
    print(f"Mean C-index: {results['mean_cindex']:.4f} ± {results['std_cindex']:.4f}")
    print(f"Per-fold C-indices: {[f'{s:.4f}' for s in results['all_scores']]}")

    # Save results to organized structure
    risk_scores_path = output_dirs["results"] / "risk_scores.csv"
    risk_df.to_csv(risk_scores_path, index=False)
    print(f"\nRisk scores saved to: {risk_scores_path}")

    # Save summary results
    summary_path = output_dirs["results"] / "summary_results.txt"
    with open(summary_path, "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Mean C-index: {results['mean_cindex']:.4f} ± {results['std_cindex']:.4f}\n")
        f.write(f"Per-fold C-indices: {results['all_scores']}\n")
        f.write(f"Number of patients: {len(risk_df)}\n")
        f.write(f"Number of events: {risk_df['event'].sum()}\n")
        f.write(f"Event rate: {risk_df['event'].mean():.2%}\n")

    # Create standard visualizations
    print("\nCreating visualizations...")
    km_curves_path = output_dirs["figures"] / "km_curves.png"
    plot_km_curves(
        risk_df,
        title=f"{args.dataset} Risk-Stratified Survival",
        save_path=str(km_curves_path),
    )
    
    create_comprehensive_plots(risk_df, output_dir=str(output_dirs["figures"]))

    # Perform detailed attribution analysis if requested
    if args.analyze_attribution:
        print("\nPerforming detailed attribution analysis...")
        
        # Check if attribution scores are in risk_df (from simple analysis)
        if "imaging_contribution" in risk_df.columns:
            # Create attribution visualizations
            print("Creating attribution visualizations...")
            
            # Overall modality contributions
            contrib_path = output_dirs["attribution"] / "modality_contributions.png"
            plot_modality_contributions(
                risk_df, 
                save_path=str(contrib_path),
                dataset_name=args.dataset
            )
            
            # Create comprehensive attribution report
            attribution_summary = create_attribution_report(
                risk_df,
                output_dir=str(output_dirs["attribution"]),
                dataset_name=args.dataset
            )
            
            print(f"\nAttribution analysis saved to: {output_dirs['attribution']}")
            
            # Analyze top and bottom risk patients
            if args.top_patients > 0:
                print(f"\nAnalyzing top {args.top_patients} highest and lowest risk patients...")
                from eagle import UnifiedRiskStratification
                
                # Create analyzer with the best model
                best_fold = results.get('best_fold', 0)
                model_path = output_dirs["models"] / f"fold{best_fold + 1}.pth"
                
                if model_path.exists():
                    # Load model and dataset for detailed analysis
                    from eagle import UnifiedSurvivalModel, UnifiedSurvivalDataset, UnifiedClinicalProcessor, get_text_extractor
                    import pandas as pd
                    import torch
                    
                    # Load data
                    df = pd.read_parquet(config.data_path)
                    df = pipeline._filter_data(df)
                    
                    # Initialize processors
                    clinical_processor = UnifiedClinicalProcessor(config)
                    clinical_processor.fit(df)
                    text_extractor = get_text_extractor(config.name)
                    
                    # Create dataset
                    dataset = UnifiedSurvivalDataset(
                        df, config, clinical_processor, text_extractor
                    )
                    
                    # Load model
                    num_clinical = dataset.clinical_features.shape[1]
                    num_text_features = len(text_extractor.get_feature_names()) if text_extractor else 0
                    
                    model = UnifiedSurvivalModel(
                        dataset_config=config,
                        model_config=model_config,
                        num_clinical_features=num_clinical,
                        num_text_features=num_text_features,
                    )
                    model.load_state_dict(torch.load(model_path))
                    
                    # Create attribution analyzer
                    analyzer = ModalityAttributionAnalyzer(model, dataset)
                    
                    # Analyze specific patients
                    top_patients_df = risk_df.nlargest(args.top_patients, 'risk_score')
                    bottom_patients_df = risk_df.nsmallest(args.top_patients, 'risk_score')
                    
                    # Create detailed patient-level plots for a few examples
                    n_examples = min(3, args.top_patients)
                    
                    print(f"Creating detailed attribution plots for {n_examples} high-risk patients...")
                    for i, (idx, patient) in enumerate(top_patients_df.head(n_examples).iterrows()):
                        # Find dataset index
                        dataset_idx = df[df.index == idx].index[0] if idx in df.index else i
                        
                        try:
                            result = analyzer.analyze_patient(dataset_idx)
                            patient_path = output_dirs["attribution"] / f"high_risk_patient_{i+1}.png"
                            
                            # Get feature names for the dataset
                            feature_names = {}
                            if config.name == "GBM":
                                feature_names['text_features'] = text_extractor.get_feature_names() if text_extractor else []
                                feature_names['clinical'] = config.clinical_features
                            elif config.name == "IPMN":
                                feature_names['text_features'] = text_extractor.get_feature_names() if text_extractor else []
                                feature_names['clinical'] = config.clinical_features
                            else:  # NSCLC
                                feature_names['text_features'] = text_extractor.get_feature_names() if text_extractor else []
                                feature_names['clinical'] = config.clinical_features
                            
                            plot_patient_level_attribution(
                                result,
                                feature_names=feature_names,
                                save_path=str(patient_path)
                            )
                        except Exception as e:
                            print(f"Could not analyze patient {i+1}: {str(e)}")
                
        else:
            print("Basic attribution scores not found in results. Run with UnifiedRiskStratification.generate_risk_scores(compute_attributions=True)")

    print(f"\nAll outputs saved to: {output_dirs['base']}")
    print("\nAnalysis complete!")
    
    return results, risk_df, output_dirs


if __name__ == "__main__":
    results, risk_df, output_dirs = main()