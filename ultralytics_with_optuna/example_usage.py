#!/usr/bin/env python3
"""
Example Usage of the Clean Wafer Defect Classification System

This script demonstrates how simple it is to use the refactored,
modular architecture for wafer defect classification.
"""

from pathlib import Path

from classifier import WaferDefectClassifier
from config import DataConfig, ModelConfig
from optuna_optimizer import OptunaWaferOptimizer


def basic_example():
    """Basic usage example - simple and clean"""
    print("🚀 Basic Training Example")
    print("=" * 50)

    # Initialize classifier - that's it!
    classifier = WaferDefectClassifier()

    # Run complete pipeline with sensible defaults
    results = classifier.run_complete_pipeline(
        epochs=10,  # Quick training for demo
        img_size=128,  # Smaller for speed
        use_subset=True,  # Use subset for speed
        subset_size=500,  # Small subset
        visualize=True,  # Create plots
        save_results=True,  # Save reports
    )

    # Results are clean and well-structured
    accuracy = results["evaluation_results"]["accuracy"]
    num_classes = results["label_analysis"]["num_classes"]

    print("✅ Training completed!")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"🎯 Classes: {num_classes}")
    print("📁 Results saved in: plots/")


def configuration_example():
    """Demonstrate easy configuration management"""
    print("\n⚙️  Configuration Example")
    print("=" * 50)

    # Configuration is centralized and type-safe
    print(f"📂 Default data path: {DataConfig.DEFAULT_DATA_PATH}")
    print(f"🏗️  Default model: YOLOv8{ModelConfig.DEFAULT_MODEL_SIZE}")
    print(f"📐 Default image size: {ModelConfig.DEFAULT_IMG_SIZE}")
    print(f"📦 Default batch size: {ModelConfig.DEFAULT_BATCH_SIZE}")

    # Easy to override
    classifier = WaferDefectClassifier()

    # Custom parameters - no magic numbers!
    results = classifier.run_complete_pipeline(
        epochs=ModelConfig.DEFAULT_EPOCHS,  # From config
        img_size=256,  # Override
        batch_size=16,  # Override
        use_subset=True,
        subset_size=200,
        visualize=False,  # Skip plots for speed
    )

    print("✅ Custom training completed!")
    print(f"📊 Final accuracy: {results['evaluation_results']['accuracy']:.4f}")


def optimization_example():
    """Demonstrate Optuna optimization"""
    print("\n⚡ Optimization Example")
    print("=" * 50)

    # Initialize optimizer
    optimizer = OptunaWaferOptimizer(study_name="demo_optimization")

    # Run optimization - simple!
    print("🔍 Running hyperparameter optimization...")
    study = optimizer.optimize(n_trials=3)  # Just 3 trials for demo

    print(f"🏆 Best accuracy: {study.best_value:.4f}")
    print("🎯 Best parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")

    # Train final model with best parameters
    print("\n🚀 Training final model with optimized parameters...")
    final_results = optimizer.train_best_model(
        use_full_data=False,  # Use subset for demo
        save_path="demo_optimized",
    )

    print(
        f"✅ Optimized model accuracy: {final_results['evaluation_results']['accuracy']:.4f}"
    )


def prediction_example():
    """Demonstrate prediction on new samples"""
    print("\n🎯 Prediction Example")
    print("=" * 50)

    # Quick training first
    classifier = WaferDefectClassifier()
    print("📚 Quick training for prediction demo...")

    classifier.run_complete_pipeline(
        epochs=5, use_subset=True, subset_size=300, visualize=False
    )

    # Find a test image
    test_dir = Path("dataset/test")
    if test_dir.exists():
        # Get first available test image
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                test_images = list(class_dir.glob("*.png"))
                if test_images:
                    sample_image = test_images[0]

                    # Make prediction - simple!
                    prediction = classifier.predict_sample(str(sample_image))

                    print(f"🖼️  Sample image: {sample_image.name}")
                    print(f"🏷️  Predicted class: {prediction['class_name']}")
                    print(f"📊 Confidence: {prediction['confidence']:.4f}")
                    print(f"🔍 Defect pattern: {prediction['defect_pattern']}")
                    break
        else:
            print("ℹ️  No test images available yet - run training first!")
    else:
        print("ℹ️  No dataset directory found - run training first!")


def main():
    """Run all examples"""
    print("🧪 Wafer Defect Classification - Clean Architecture Examples")
    print("=" * 70)
    print("This demonstrates how simple the refactored code is to use!")
    print()

    try:
        # Run examples
        basic_example()
        configuration_example()
        optimization_example()
        prediction_example()

        print("\n" + "=" * 70)
        print("🎉 All examples completed successfully!")
        print("✨ Notice how clean and simple the new architecture is!")
        print("📚 Check out the README.md for comprehensive documentation")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("💡 Make sure you have the dataset available or run a demo first")


if __name__ == "__main__":
    main()
