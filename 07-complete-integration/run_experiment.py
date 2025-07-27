"""
Complete Integration Experiment Runner

This script runs comprehensive experiments demonstrating the integration of:
- PyTorch Lightning (training framework)
- Optuna (hyperparameter optimization)
- MLflow (experiment tracking)
- Torchvision (pre-trained models)
- Ultralytics-inspired architectures

Run with: python run_experiment.py
"""

import argparse
import sys
from pathlib import Path

# Add the parent directory to sys.path to import shared modules
sys.path.append(str(Path(__file__).parent.parent))

from integrated_pipeline import ExperimentConfig, IntegratedMLPipeline


def run_quick_demo():
    """Run a quick demonstration of all tools working together."""
    print("🚀 Quick Demo: Complete ML Tools Integration")
    print("=" * 60)
    print("This demo shows all tools working together in ~10 minutes")
    print("✓ Lightning: Automatic training loops")
    print("✓ Optuna: Smart hyperparameter search")
    print("✓ MLflow: Experiment tracking")
    print("✓ Torchvision: Pre-trained models")
    print("✓ YOLO-inspired: Modern architectures")
    print("=" * 60)

    config = ExperimentConfig(
        experiment_name="complete_integration_demo",
        model_type="all",  # Test all model types
        max_epochs=15,  # Quick training
        num_optuna_trials=3,  # Fast optimization
        batch_size=32,
        enable_mlflow=True,
        enable_optuna=True,
        data_augmentation=True,
        class_balancing=True,
    )

    pipeline = IntegratedMLPipeline(config)
    results = pipeline.run_complete_pipeline()

    print("\n🎊 Quick Demo Results:")
    if results["model_comparison"]:
        best_model = results["best_model"]
        best_acc = results["experiment_summary"]["best_accuracy"]
        print(f"🏆 Best Model: {best_model}")
        print(f"📊 Best Accuracy: {best_acc:.4f}")
        print(
            f"📈 Models Compared: {results['experiment_summary']['total_models_trained']}"
        )

    print("\n✨ Demo completed! Check results/07-complete-integration/ for outputs")
    return results


def run_production_experiment():
    """Run a full production-quality experiment."""
    print("🏭 Production Experiment: Comprehensive Model Comparison")
    print("=" * 60)
    print("This experiment runs thorough optimization (~30-60 minutes)")
    print("⚡ Full Lightning training with callbacks")
    print("🔍 Extensive Optuna hyperparameter optimization")
    print("📊 Complete MLflow experiment tracking")
    print("🖼️ Torchvision transfer learning evaluation")
    print("🎯 YOLO-inspired architecture testing")
    print("=" * 60)

    config = ExperimentConfig(
        experiment_name="wafer_defect_production",
        model_type="all",
        max_epochs=50,  # Full training
        num_optuna_trials=20,  # Thorough optimization
        batch_size=32,
        enable_mlflow=True,
        enable_optuna=True,
        data_augmentation=True,
        class_balancing=True,
    )

    pipeline = IntegratedMLPipeline(config)
    results = pipeline.run_complete_pipeline()

    print("\n🏆 Production Experiment Results:")
    print(
        f"Total Models Trained: {results['experiment_summary']['total_models_trained']}"
    )
    print(f"Best Model: {results['best_model']}")
    print(f"Best Accuracy: {results['experiment_summary']['best_accuracy']:.4f}")

    print("\n📊 Model Performance Summary:")
    for model_name, metrics in results["model_comparison"].items():
        print(
            f"{model_name:25}: {metrics['accuracy']:.4f} acc, {metrics['f1_weighted']:.4f} f1"
        )

    print("\n📁 All results saved to: results/07-complete-integration/")
    print("🔗 MLflow UI: http://localhost:5000")

    return results


def run_architecture_comparison():
    """Compare different architectures without optimization."""
    print("🏗️ Architecture Comparison: Model Performance Analysis")
    print("=" * 60)
    print("Compare base performance of different architectures")
    print("⚡ Lightning CNN (from scratch)")
    print("🖼️ ResNet18 (Torchvision)")
    print("🚀 EfficientNet-B0 (Torchvision)")
    print("🎯 YOLO-inspired CNN")
    print("=" * 60)

    config = ExperimentConfig(
        experiment_name="architecture_comparison",
        model_type="all",
        max_epochs=30,
        num_optuna_trials=0,  # No optimization
        enable_mlflow=True,
        enable_optuna=False,  # Skip optimization for pure comparison
        data_augmentation=True,
        use_pretrained=True,
    )

    pipeline = IntegratedMLPipeline(config)
    results = pipeline.run_complete_pipeline()

    print("\n📊 Architecture Comparison Results:")
    print("-" * 60)
    print(f"{'Architecture':<25} {'Accuracy':<10} {'F1-Score':<10} {'Params'}")
    print("-" * 60)

    # Note: Parameter counting would need to be added to the pipeline
    for model_name, metrics in results["model_comparison"].items():
        acc = metrics["accuracy"]
        f1 = metrics["f1_weighted"]
        print(f"{model_name:<25} {acc:<10.4f} {f1:<10.4f} {'TBD'}")

    return results


def run_ablation_study():
    """Run ablation study on different components."""
    print("🔬 Ablation Study: Component Impact Analysis")
    print("=" * 60)
    print("Test impact of different components:")
    print("1. No pre-training vs Pre-trained")
    print("2. No augmentation vs Data augmentation")
    print("3. No class balancing vs Balanced classes")
    print("4. No optimization vs Optuna optimization")
    print("=" * 60)

    # Define ablation configurations
    ablation_configs = [
        (
            "baseline",
            {
                "use_pretrained": False,
                "data_augmentation": False,
                "class_balancing": False,
                "enable_optuna": False,
            },
        ),
        (
            "pretrained",
            {
                "use_pretrained": True,
                "data_augmentation": False,
                "class_balancing": False,
                "enable_optuna": False,
            },
        ),
        (
            "augmentation",
            {
                "use_pretrained": False,
                "data_augmentation": True,
                "class_balancing": False,
                "enable_optuna": False,
            },
        ),
        (
            "balanced",
            {
                "use_pretrained": False,
                "data_augmentation": False,
                "class_balancing": True,
                "enable_optuna": False,
            },
        ),
        (
            "optimized",
            {
                "use_pretrained": False,
                "data_augmentation": False,
                "class_balancing": False,
                "enable_optuna": True,
            },
        ),
        (
            "full",
            {
                "use_pretrained": True,
                "data_augmentation": True,
                "class_balancing": True,
                "enable_optuna": True,
            },
        ),
    ]

    all_results = {}

    for config_name, config_updates in ablation_configs:
        print(f"\n🧪 Running {config_name} configuration...")

        config = ExperimentConfig(
            experiment_name=f"ablation_{config_name}",
            model_type="torchvision_resnet",  # Use consistent architecture
            max_epochs=25,
            num_optuna_trials=5 if config_updates.get("enable_optuna", False) else 0,
            **config_updates,
        )

        pipeline = IntegratedMLPipeline(config)
        results = pipeline.run_complete_pipeline()
        all_results[config_name] = results

        if results["model_comparison"]:
            best_acc = results["experiment_summary"]["best_accuracy"]
            print(f"✅ {config_name}: {best_acc:.4f} accuracy")

    # Print ablation results
    print("\n📊 Ablation Study Results:")
    print("-" * 50)
    print(f"{'Configuration':<15} {'Accuracy':<10} {'Improvement'}")
    print("-" * 50)

    baseline_acc = (
        all_results.get("baseline", {})
        .get("experiment_summary", {})
        .get("best_accuracy", 0)
    )

    for config_name, results in all_results.items():
        acc = results["experiment_summary"]["best_accuracy"]
        improvement = acc - baseline_acc if config_name != "baseline" else 0
        improvement_str = (
            f"+{improvement:.4f}" if improvement > 0 else f"{improvement:.4f}"
        )
        print(f"{config_name:<15} {acc:<10.4f} {improvement_str}")

    return all_results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Complete ML Integration Experiments")
    parser.add_argument(
        "--mode",
        choices=["demo", "production", "comparison", "ablation"],
        default="demo",
        help="Experiment mode to run",
    )
    parser.add_argument(
        "--mlflow-server",
        action="store_true",
        help="Start MLflow server before running experiments",
    )

    args = parser.parse_args()

    # Start MLflow server if requested
    if args.mlflow_server:
        import subprocess
        import time

        print("🚀 Starting MLflow server...")
        subprocess.Popen(["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"])
        time.sleep(3)  # Give server time to start
        print("📊 MLflow server started at http://localhost:5000")

    # Run selected experiment mode
    print(f"\n{'=' * 60}")
    print("COMPLETE ML INTEGRATION TUTORIAL")
    print(f"Mode: {args.mode.upper()}")
    print(f"{'=' * 60}")

    if args.mode == "demo":
        results = run_quick_demo()
    elif args.mode == "production":
        results = run_production_experiment()
    elif args.mode == "comparison":
        results = run_architecture_comparison()
    elif args.mode == "ablation":
        results = run_ablation_study()

    print("\n🎯 Key Integration Benefits Demonstrated:")
    print("✅ Lightning: Simplified training with automatic features")
    print("✅ Optuna: Intelligent hyperparameter optimization")
    print("✅ MLflow: Professional experiment tracking")
    print("✅ Torchvision: Easy pre-trained model integration")
    print("✅ Modern Architectures: YOLO-inspired designs")
    print("✅ Production Patterns: Scalable ML pipeline")

    print("\n📁 Check these locations for results:")
    print("   📊 Visualizations: results/07-complete-integration/")
    print("   💾 Model checkpoints: checkpoints/07-complete-integration/")
    print("   📋 Experiment logs: lightning_logs/")
    if args.mlflow_server:
        print("   🔗 MLflow dashboard: http://localhost:5000")

    print("\n🎉 Tutorial completed successfully!")

    return results


if __name__ == "__main__":
    main()
