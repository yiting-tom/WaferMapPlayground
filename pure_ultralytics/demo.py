#!/usr/bin/env python3
"""
Quick demo script for Wafer Defect Classification using Ultralytics YOLOv8
This script runs a quick test with reduced parameters for faster execution
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add parent directory to path to import main module
sys.path.append(str(Path(__file__).parent))
from main import WaferDefectClassifier


def quick_demo():
    """Run a quick demo with reduced dataset and training parameters"""
    print("=== Wafer Defect Classification Quick Demo ===\n")

    # Initialize classifier
    classifier = WaferDefectClassifier()

    # Load and analyze data
    print("1. Loading dataset...")
    images, labels = classifier.load_data()

    print("\n2. Analyzing labels...")
    label_analysis = classifier.analyze_labels(labels)

    # Use a stratified subset of data for quick demo to ensure all classes are represented
    print("\n3. Using stratified subset of data for quick demo...")
    from sklearn.model_selection import train_test_split

    subset_size = min(2000, len(images))  # Increased size for better representation

    # Convert labels to class indices for stratification
    class_indices = []
    for label in labels:
        label_tuple = tuple(label)
        class_idx = classifier.label_to_class[label_tuple]
        class_indices.append(class_idx)

    # Stratified sampling to ensure representation from all classes
    indices = list(range(len(images)))
    subset_indices, _ = train_test_split(
        indices,
        test_size=1 - subset_size / len(images),
        stratify=class_indices,
        random_state=42,
    )

    images_subset = images[subset_indices]
    labels_subset = labels[subset_indices]

    print(f"Using {subset_size} samples for demo")

    # Create dataset
    print("\n4. Creating YOLO dataset structure...")
    classifier.create_yolo_dataset(images_subset, labels_subset)

    # Train model with reduced parameters
    print("\n5. Training model (quick demo - 5 epochs)...")
    training_results = classifier.train_model(
        epochs=5,  # Very few epochs for quick demo
        img_size=128,  # Smaller image size for faster training
        batch_size=16,  # Smaller batch size
    )

    # Evaluate model
    print("\n6. Evaluating model...")
    eval_results = classifier.evaluate_model()

    # Show sample images
    print("\n7. Visualizing sample wafer maps...")
    plot_sample_images(images_subset[:9], labels_subset[:9], classifier.class_names)

    print("\n=== Demo Complete ===")
    print(f"Demo Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Number of classes: {label_analysis['num_classes']}")
    print("Model saved in: runs/classify/wafer_defect_classifier/")

    # Example prediction
    test_dir = Path("dataset/test")
    if test_dir.exists():
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                test_images = list(class_dir.glob("*.png"))
                if test_images:
                    sample_image = test_images[0]
                    prediction = classifier.predict_sample(str(sample_image))
                    print("\nSample Prediction:")
                    print(f"Predicted: {prediction['class_name']}")
                    print(f"Confidence: {prediction['confidence']:.4f}")
                    break

    return {
        "label_analysis": label_analysis,
        "training_results": training_results,
        "evaluation_results": eval_results,
    }


def plot_sample_images(images, labels, class_names):
    """Plot sample wafer map images with their labels"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle("Sample Wafer Maps with Defect Classifications", fontsize=16)

    for i, (ax, image, label) in enumerate(zip(axes.flat, images, labels)):
        # Display image
        ax.imshow(image, cmap="viridis")

        # Find class name
        label_tuple = tuple(label)
        active_defects = [j for j, val in enumerate(label) if val == 1]
        if not active_defects:
            class_name = "Normal"
        else:
            class_name = f"Defect_{'-'.join(map(str, active_defects))}"

        ax.set_title(f"Sample {i + 1}: {class_name}", fontsize=10)
        ax.set_xlabel(f"Pattern: {label}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("sample_wafer_maps.png", dpi=300, bbox_inches="tight")
    plt.show()


def analyze_dataset_only():
    """Just analyze the dataset without training"""
    print("=== Dataset Analysis Only ===\n")

    classifier = WaferDefectClassifier()
    images, labels = classifier.load_data()
    label_analysis = classifier.analyze_labels(labels)

    # Show some statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(images)}")
    print(f"Image shape: {images[0].shape}")
    print(f"Number of unique defect patterns: {label_analysis['num_classes']}")

    # Plot sample images
    plot_sample_images(images[:9], labels[:9], label_analysis["class_names"])

    return label_analysis


if __name__ == "__main__":
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

    args = parser.parse_args()

    if args.analyze_only:
        analyze_dataset_only()
    elif args.quick_demo or len(sys.argv) == 1:
        # Default to quick demo
        quick_demo()
    else:
        print(
            "Use --analyze-only to just analyze data or --quick-demo for quick training test"
        )
