#!/usr/bin/env python3
"""
Quick demo script for Wafer Defect Classification using Ultralytics YOLOv8
This script runs a quick test with reduced parameters for faster execution
"""

import sys
from pathlib import Path

# Add parent directory to path to import main module
sys.path.append(str(Path(__file__).parent))

from classifier import WaferDefectClassifier
from config import DataConfig, OptunaConfig
from visualization import WaferVisualization


def quick_demo() -> dict:
    """Run a quick demo with reduced dataset and training parameters"""
    print("=== Wafer Defect Classification Quick Demo ===\n")

    # Initialize classifier with clean architecture
    classifier = WaferDefectClassifier(data_path=DataConfig.DEFAULT_DATA_PATH)

    print("Running quick demo with reduced parameters for faster execution...")

    # Run pipeline with demo-friendly settings
    results = classifier.run_complete_pipeline(
        epochs=5,  # Very few epochs for quick demo
        img_size=128,  # Smaller image size for faster training
        use_subset=True,  # Use subset for faster processing
        subset_size=OptunaConfig.DEMO_SUBSET_SIZE,  # Small subset
        balance_classes=False,  # Keep original distribution
        visualize=True,
        save_results=True,
    )

    # Show sample images
    print("\n7. Visualizing sample wafer maps...")
    visualizer = WaferVisualization()

    # Load a small sample of images for visualization
    images, labels = classifier.data_processor.load_data()
    sample_images = images[:9]  # First 9 images
    sample_labels = labels[:9]

    visualizer.plot_sample_wafer_maps(
        sample_images,
        sample_labels,
        classifier.data_processor.class_names,
        classifier.data_processor.label_to_class,
        n_samples=9,
        save_path="sample_wafer_maps_demo.png",
    )

    print("\n=== Demo Complete ===")
    print(f"Demo Accuracy: {results['evaluation_results']['accuracy']:.4f}")
    print(f"Number of classes: {results['label_analysis']['num_classes']}")
    print("Model saved in: runs/classify/wafer_defect_classifier/")

    # Example prediction
    test_dir = Path("dataset/test")
    if test_dir.exists():
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                test_images = list(class_dir.glob("*.png"))
                if test_images:
                    sample_image = test_images[0]
                    try:
                        prediction = classifier.predict_sample(str(sample_image))
                        print("\nSample Prediction:")
                        print(f"Predicted: {prediction['class_name']}")
                        print(f"Confidence: {prediction['confidence']:.4f}")
                    except Exception as e:
                        print(f"Sample prediction failed: {e}")
                    break

    return results


def analyze_dataset_only() -> dict:
    """Just analyze the dataset without training"""
    print("=== Dataset Analysis Only ===\n")

    # Initialize data processor
    from data_processor import WaferDataProcessor

    data_processor = WaferDataProcessor(DataConfig.DEFAULT_DATA_PATH)
    images, labels = data_processor.load_data()
    label_analysis = data_processor.analyze_labels(labels)

    # Show some statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Number of unique defect patterns: {label_analysis['num_classes']}")

    # Plot sample images
    visualizer = WaferVisualization()
    visualizer.plot_sample_wafer_maps(
        images[:9],
        labels[:9],
        data_processor.class_names,
        data_processor.label_to_class,
        save_path="dataset_analysis_samples.png",
    )

    # Plot class distribution
    class_distribution = data_processor.get_class_distribution(labels)
    visualizer.plot_class_distribution(
        class_distribution,
        save_path="dataset_class_distribution.png",
        title="Dataset Class Distribution",
    )

    return label_analysis


def performance_demo() -> dict:
    """Demo focused on performance evaluation"""
    print("=== Performance Evaluation Demo ===\n")

    classifier = WaferDefectClassifier()

    # Run with more reasonable parameters for performance evaluation
    results = classifier.run_complete_pipeline(
        epochs=20,  # More epochs for better performance
        img_size=224,  # Standard image size
        use_subset=True,
        subset_size=3000,  # Larger subset for better evaluation
        balance_classes=True,  # Balance for fair evaluation
        visualize=True,
        save_results=True,
    )

    # Additional performance metrics
    eval_results = results["evaluation_results"]

    print("\n=== Performance Summary ===")
    print(f"Accuracy: {eval_results['accuracy']:.4f}")

    if "classification_report" in eval_results:
        report = eval_results["classification_report"]
        if "weighted avg" in report:
            weighted = report["weighted avg"]
            print(f"Weighted Precision: {weighted.get('precision', 0):.4f}")
            print(f"Weighted Recall: {weighted.get('recall', 0):.4f}")
            print(f"Weighted F1-Score: {weighted.get('f1-score', 0):.4f}")

    return results


def main():
    """Main function with command line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Wafer Defect Classification Demo")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze dataset without training",
    )
    parser.add_argument(
        "--quick-demo",
        action="store_true",
        help="Run quick demo with reduced parameters",
    )
    parser.add_argument(
        "--performance-demo",
        action="store_true",
        help="Run performance-focused demo with better parameters",
    )

    args = parser.parse_args()

    if args.analyze_only:
        analyze_dataset_only()
    elif args.performance_demo:
        performance_demo()
    elif args.quick_demo or len(sys.argv) == 1:
        # Default to quick demo
        quick_demo()
    else:
        print("Use --analyze-only, --quick-demo, or --performance-demo")
        print("Run with --help for more information")


if __name__ == "__main__":
    main()
