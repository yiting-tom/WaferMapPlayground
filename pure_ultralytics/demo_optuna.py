#!/usr/bin/env python3
"""
Optuna Optimization Demo for Wafer Defect Classification
This script demonstrates hyperparameter optimization using Optuna
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent))
from main import WaferDefectClassifier
from optuna_optimizer import OptunaWaferOptimizer


def quick_optuna_demo(n_trials: int = 10):
    """Run a quick Optuna optimization demo"""
    print("=== Quick Optuna Optimization Demo ===\n")

    # Initialize optimizer
    print("1. Initializing Optuna optimizer...")
    optimizer = OptunaWaferOptimizer(
        study_name=f"wafer_demo_{n_trials}trials",
        storage_path="sqlite:///wafer_demo.db",
    )

    # Set smaller subset for quick demo
    optimizer.use_subset_size = 1000
    print(f"Using subset of {optimizer.use_subset_size} samples for fast optimization")

    # Run optimization
    print(f"\n2. Running optimization with {n_trials} trials...")
    print("This may take a few minutes...")

    study = optimizer.optimize(n_trials=n_trials)

    # Show results
    print("\n3. Optimization Results:")
    print(f"   Best accuracy: {study.best_value:.4f}")
    print(f"   Best trial: #{study.best_trial.number}")
    print("\n   Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")

    # Create visualizations
    print("\n4. Creating optimization plots...")
    optimizer.visualize_optimization(study, "quick_demo_plots")
    print("   Plots saved to: quick_demo_plots/")

    # Train final model with best parameters
    print("\n5. Training final model with best parameters...")
    final_results = optimizer.train_best_model(
        use_full_data=False,  # Use subset for demo
        save_path="quick_demo_best",
    )

    print("\n=== Demo Complete ===")
    print(f"Optimization accuracy: {study.best_value:.4f}")
    print(
        f"Final model accuracy: {final_results['evaluation_results']['accuracy']:.4f}"
    )
    print(f"Model saved in: {final_results['model_path']}")

    return study, final_results


def full_optuna_optimization(n_trials: int = 50, use_full_data: bool = True):
    """Run comprehensive Optuna optimization"""
    print("=== Full Optuna Optimization ===\n")

    # Initialize optimizer
    optimizer = OptunaWaferOptimizer(
        study_name=f"wafer_full_{n_trials}trials",
        storage_path="sqlite:///wafer_full.db",
    )

    if not use_full_data:
        optimizer.use_subset_size = 5000
        print(f"Using subset of {optimizer.use_subset_size} samples")
    else:
        print("Using full dataset (38,015 samples)")

    # Run optimization
    print(f"\nRunning optimization with {n_trials} trials...")
    print("This will take significant time (30min - 2hours depending on hardware)...")

    study = optimizer.optimize(n_trials=n_trials)

    # Show detailed results
    trials_df, summary = optimizer.get_study_summary(study)

    print("\nDetailed Results:")
    print(f"Total trials: {summary['total_trials']}")
    print(f"Completed trials: {summary['completed_trials']}")
    print(f"Best accuracy: {summary['best_value']:.4f}")

    # Save detailed results
    trials_df.to_csv("optimization_trials.csv", index=False)
    print("Detailed results saved to: optimization_trials.csv")

    # Create comprehensive visualizations
    optimizer.visualize_optimization(study, "full_optimization_plots")

    # Train final model
    print("\nTraining final model with best parameters...")
    final_results = optimizer.train_best_model(
        use_full_data=use_full_data, save_path="production_model"
    )

    print("\n=== Optimization Complete ===")
    print(f"Best validation accuracy: {summary['best_value']:.4f}")
    print(f"Final test accuracy: {final_results['evaluation_results']['accuracy']:.4f}")
    print(
        f"Improvement: {final_results['evaluation_results']['accuracy'] - 0.431:.4f}"
    )  # vs baseline

    return study, final_results


def compare_with_baseline():
    """Compare optimized model with baseline"""
    print("=== Baseline vs Optimized Comparison ===\n")

    # Run baseline (default parameters)
    print("1. Training baseline model...")
    baseline_classifier = WaferDefectClassifier()
    baseline_results = baseline_classifier.run_complete_pipeline(
        epochs=50, img_size=224
    )
    baseline_accuracy = baseline_results["evaluation_results"]["accuracy"]

    # Run quick optimization
    print("\n2. Running optimization...")
    optimizer = OptunaWaferOptimizer()
    optimizer.use_subset_size = 2000  # Balanced subset

    study = optimizer.optimize(n_trials=15)
    optimized_results = optimizer.train_best_model(
        use_full_data=False, save_path="comparison_optimized"
    )
    optimized_accuracy = optimized_results["evaluation_results"]["accuracy"]

    # Show comparison
    print("\n=== Comparison Results ===")
    print(f"Baseline accuracy:  {baseline_accuracy:.4f}")
    print(f"Optimized accuracy: {optimized_accuracy:.4f}")
    print(f"Improvement:        {optimized_accuracy - baseline_accuracy:.4f}")
    print(
        f"Relative improvement: {((optimized_accuracy - baseline_accuracy) / baseline_accuracy * 100):.1f}%"
    )

    # Plot comparison
    categories = ["Baseline\n(Default params)", "Optimized\n(Optuna params)"]
    accuracies = [baseline_accuracy, optimized_accuracy]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, accuracies, color=["lightblue", "lightgreen"], alpha=0.8)
    plt.ylabel("Accuracy")
    plt.title("Baseline vs Optuna-Optimized Model Performance")
    plt.ylim(0, max(accuracies) * 1.1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("baseline_vs_optimized.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nComparison plot saved as: baseline_vs_optimized.png")

    return baseline_results, optimized_results


def load_and_use_optimized_model(params_file: str):
    """Load and use a previously optimized model"""

    if not Path(params_file).exists():
        print(f"Parameters file not found: {params_file}")
        return

    # Load best parameters
    with open(params_file, "r") as f:
        data = json.load(f)

    best_params = data["best_params"]
    print("Loaded optimized parameters:")
    print(f"Optimization date: {data.get('optimization_date', 'Unknown')}")
    print(f"Best validation accuracy: {data['best_score']:.4f}")

    # Create classifier with optimized parameters
    classifier = WaferDefectClassifier()

    # Apply parameters (this would require modifying the main classifier)
    print("\nOptimized parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    print("\nTo use these parameters, modify your training call:")
    print("classifier.train_model(")
    print(f"    epochs={best_params['epochs']},")
    print(f"    img_size={best_params['img_size']},")
    print(f"    batch_size={best_params['batch_size']},")
    print(f"    model_name='yolov8{best_params['model_size']}-cls.pt'")
    print(")")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Optuna Optimization Demo for Wafer Defect Classification"
    )

    parser.add_argument(
        "--quick-demo",
        action="store_true",
        help="Run quick optimization demo (10 trials)",
    )
    parser.add_argument(
        "--full-optimization",
        action="store_true",
        help="Run full optimization (50+ trials)",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare optimized vs baseline model",
    )
    parser.add_argument(
        "--load-params",
        type=str,
        metavar="FILE",
        help="Load and display optimized parameters from file",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        metavar="N",
        help="Number of optimization trials (default: 10)",
    )
    parser.add_argument(
        "--subset-only",
        action="store_true",
        help="Use data subset even for full optimization",
    )

    args = parser.parse_args()

    if args.quick_demo:
        quick_optuna_demo(args.trials)
    elif args.full_optimization:
        full_optuna_optimization(args.trials, not args.subset_only)
    elif args.compare_baseline:
        compare_with_baseline()
    elif args.load_params:
        load_and_use_optimized_model(args.load_params)
    else:
        # Default: quick demo
        print("No specific option selected. Running quick demo...")
        print("Use --help to see all available options.\n")
        quick_optuna_demo(args.trials)


if __name__ == "__main__":
    main()
