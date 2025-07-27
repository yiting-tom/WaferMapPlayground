"""
Quick Start: ML Tools Integration

The simplest possible demonstration of all tools working together.
Just run: python quick_start.py

This minimal example shows:
- Lightning: Clean training
- Optuna: Smart optimization
- MLflow: Experiment tracking
- Torchvision: Pre-trained models
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add shared utilities
sys.path.append(str(Path(__file__).parent.parent / "shared"))


def main():
    """Minimal integration example - just the essentials!"""

    print("🚀 QUICK START: ML Tools Integration")
    print("=" * 40)
    print("Showing all tools in ~30 lines of code!")
    print("=" * 40)

    # 1. Import the integrated pipeline
    from integrated_pipeline import ExperimentConfig, IntegratedMLPipeline

    # 2. Simple configuration
    config = ExperimentConfig(
        experiment_name="quick_start_demo",
        model_type="torchvision_resnet",  # Just ResNet for speed
        max_epochs=5,  # Very quick training
        num_optuna_trials=2,  # Minimal optimization
        enable_mlflow=True,  # Track everything
        enable_optuna=True,  # Optimize hyperparameters
        use_pretrained=True,  # Transfer learning
    )

    # 3. Create pipeline and run
    print("🏗️  Creating integrated pipeline...")
    pipeline = IntegratedMLPipeline(config)

    print("🎯 Running complete integration...")
    results = pipeline.run_complete_pipeline()

    # 4. Show results
    if results["model_comparison"]:
        best_model = results["best_model"]
        best_acc = results["experiment_summary"]["best_accuracy"]

        print("\n✅ SUCCESS!")
        print(f"🏆 Best Model: {best_model}")
        print(f"📊 Accuracy: {best_acc:.4f}")
        print("🔗 MLflow: http://localhost:5000")

        print("\n🎉 You just saw:")
        print("  ⚡ Lightning: Automatic training")
        print("  🔍 Optuna: Smart optimization")
        print("  📊 MLflow: Experiment tracking")
        print("  🖼️ Torchvision: Transfer learning")
        print("  📈 Complete: Professional pipeline")

    else:
        print("❌ Something went wrong - check the logs above")

    print("\n🚀 Ready to build your own ML systems!")


if __name__ == "__main__":
    main()
