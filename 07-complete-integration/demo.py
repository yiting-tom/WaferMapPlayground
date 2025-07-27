"""
Simple Demo: Complete ML Tools Integration

A quick 5-minute demonstration of all ML tools working together:
- PyTorch Lightning: Clean training framework
- Optuna: Smart hyperparameter optimization
- MLflow: Experiment tracking
- Torchvision: Pre-trained models
- YOLO-inspired: Modern architectures

Run with: python demo.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent / "shared"))

from integrated_pipeline import ExperimentConfig, IntegratedMLPipeline


def run_simple_demo():
    """Run a simple 5-minute demo of all tools integrated."""

    print("🚀 SIMPLE DEMO: Complete ML Tools Integration")
    print("=" * 60)
    print("This demo showcases ALL tools working together in ~5 minutes:")
    print("⚡ PyTorch Lightning: Professional training framework")
    print("🔍 Optuna: Intelligent hyperparameter optimization")
    print("📊 MLflow: Experiment tracking and model registry")
    print("🖼️ Torchvision: Pre-trained ResNet with transfer learning")
    print("🎯 YOLO-inspired: Modern CNN architecture")
    print("📈 Comprehensive: Metrics, visualization, comparison")
    print("=" * 60)

    # Simple configuration for quick demo
    config = ExperimentConfig(
        experiment_name="simple_integration_demo",
        model_type="all",  # Test all model types
        max_epochs=10,  # Quick training
        num_optuna_trials=2,  # Fast optimization (just 2 trials)
        batch_size=32,
        enable_mlflow=True,  # Track experiments
        enable_optuna=True,  # Optimize hyperparameters
        data_augmentation=False,  # Skip augmentation for speed
        class_balancing=True,  # Handle imbalanced data
        use_pretrained=True,  # Use pre-trained models
    )

    print("⚙️  Demo Configuration:")
    print(f"   • Training epochs: {config.max_epochs} (quick)")
    print(f"   • Optuna trials: {config.num_optuna_trials} per model")
    print("   • Models to test: 4 architectures")
    print("   • Expected runtime: ~5 minutes")
    print(f"   • MLflow tracking: {config.enable_mlflow}")

    # Create and run pipeline
    print("\n🏗️  Creating integrated ML pipeline...")
    pipeline = IntegratedMLPipeline(config)

    print("\n🎯 Starting complete integration demo...")
    print("-" * 60)

    # Run the complete pipeline
    results = pipeline.run_complete_pipeline()

    # Display results
    print("\n🏆 DEMO RESULTS SUMMARY")
    print("=" * 60)

    if results["model_comparison"]:
        print(f"✅ Successfully trained {len(results['model_comparison'])} models!")
        print("\n📊 Model Performance Comparison:")
        print("-" * 50)
        print(f"{'Model':<25} {'Accuracy':<10} {'F1-Score'}")
        print("-" * 50)

        for model_name, metrics in results["model_comparison"].items():
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1_weighted", 0)
            print(f"{model_name:<25} {acc:<10.4f} {f1:.4f}")

        # Highlight best model
        best_model = results["best_model"]
        best_accuracy = results["experiment_summary"]["best_accuracy"]
        print("-" * 50)
        print(f"🥇 Best Model: {best_model}")
        print(f"🎯 Best Accuracy: {best_accuracy:.4f}")

        # Show what was demonstrated
        print("\n✨ Integration Features Demonstrated:")
        print("   ⚡ Lightning: Automatic training loops, callbacks, logging")
        print("   🔍 Optuna: Smart hyperparameter search with pruning")
        print("   📊 MLflow: Complete experiment tracking and model registry")
        print("   🖼️ Torchvision: ResNet18 transfer learning (grayscale adaptation)")
        print("   🎯 Modern Arch: YOLO-inspired CNN with CSP blocks")
        print("   📈 Evaluation: Comprehensive metrics and visualizations")

        # Output locations
        print("\n📁 Generated Outputs:")
        print("   📊 Visualizations: results/07-complete-integration/")
        print("   💾 Model checkpoints: checkpoints/07-complete-integration/")
        print("   📋 MLflow experiments: mlruns/ (run 'mlflow ui' to view)")

        print("\n🎉 Demo completed successfully!")
        print("This showcased a complete production ML pipeline in minutes!")

    else:
        print("❌ Demo encountered issues. Check the logs above for details.")

    return results


def run_single_model_demo():
    """Run an even simpler demo with just one model for ultra-quick testing."""

    print("⚡ ULTRA-QUICK DEMO: Single Model Integration")
    print("=" * 50)
    print("Testing just ResNet18 + Optuna + MLflow (~2 minutes)")
    print("=" * 50)

    # Minimal configuration
    config = ExperimentConfig(
        experiment_name="ultra_quick_demo",
        model_type="torchvision_resnet",  # Just one model
        max_epochs=5,  # Very quick
        num_optuna_trials=1,  # Single optimization trial
        batch_size=32,
        enable_mlflow=True,
        enable_optuna=True,
        data_augmentation=False,
        class_balancing=True,
        use_pretrained=True,
    )

    pipeline = IntegratedMLPipeline(config)

    # Override the complete pipeline to train just one model
    print("📁 Loading dataset...")
    from data_utils import get_tutorial_dataset

    dataset = get_tutorial_dataset(max_samples_per_class=100, balance_classes=True)

    print("🔍 Running Optuna optimization...")
    opt_results = pipeline.run_optuna_optimization("torchvision_resnet", dataset)

    print("🎯 Training final model with best parameters...")
    final_results = pipeline.train_final_model(
        "torchvision_resnet",
        dataset,
        opt_results["best_params"],
        run_name="ultra_quick_resnet",
    )

    # Show results
    metrics = final_results["detailed_metrics"]
    print("\n🏆 ULTRA-QUICK RESULTS:")
    print("   Model: ResNet18 (Torchvision + Transfer Learning)")
    print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"   F1-Score: {metrics.get('f1_weighted', 0):.4f}")
    print(f"   Best Optuna F1: {opt_results['best_value']:.4f}")
    print(f"   Best Parameters: {opt_results['best_params']}")

    print("\n✅ Ultra-quick demo completed!")
    print("🔗 View MLflow dashboard: http://localhost:5000")

    return final_results


def main():
    """Main demo function with options."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Integration Demo")
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "ultra"],
        default="quick",
        help="Demo mode: full (~5min), quick (~3min), ultra (~2min)",
    )
    parser.add_argument(
        "--start-mlflow",
        action="store_true",
        help="Start MLflow UI server automatically",
    )

    args = parser.parse_args()

    # Start MLflow if requested
    if args.start_mlflow:
        import subprocess
        import time

        print("🚀 Starting MLflow server...")
        try:
            subprocess.Popen(
                ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)
            print("📊 MLflow server started at http://localhost:5000")
        except FileNotFoundError:
            print("⚠️  MLflow not found. Install with: pip install mlflow")
        print("-" * 50)

    # Run selected demo
    if args.mode == "full":
        # Use the full experiment runner
        from run_experiment import run_quick_demo

        results = run_quick_demo()
    elif args.mode == "quick":
        results = run_simple_demo()
    else:  # ultra
        results = run_single_model_demo()

    # Final message
    print("\n🎓 What You Just Saw:")
    print("✓ Production ML pipeline with multiple tools integrated")
    print("✓ Automatic hyperparameter optimization (Optuna)")
    print("✓ Professional experiment tracking (MLflow)")
    print("✓ Transfer learning with pre-trained models (Torchvision)")
    print("✓ Modern architectures (YOLO-inspired)")
    print("✓ Clean training framework (PyTorch Lightning)")

    print("\n🚀 Ready to build your own integrated ML systems!")

    return results


if __name__ == "__main__":
    main()
