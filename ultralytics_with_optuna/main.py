#!/usr/bin/env python3
"""
Main entry point for Wafer Defect Classification using Ultralytics YOLOv8
Clean, modular implementation with proper separation of concerns
"""

import sys
from pathlib import Path

# Ensure local imports work
sys.path.append(str(Path(__file__).parent))

from classifier import WaferDefectClassifier
from config import DataConfig, ModelConfig


def main():
    """Main function to run wafer defect classification"""
    print("=== Wafer Defect Classification - Clean Implementation ===\n")

    # Initialize classifier with clean architecture
    classifier = WaferDefectClassifier(data_path=DataConfig.DEFAULT_DATA_PATH)

    # Run complete pipeline with sensible defaults
    results = classifier.run_complete_pipeline(
        epochs=ModelConfig.DEFAULT_EPOCHS,
        img_size=ModelConfig.DEFAULT_IMG_SIZE,
        use_subset=False,  # Use full dataset
        balance_classes=False,  # Keep original distribution
        visualize=True,
        save_results=True,
    )

    # Display summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Final Accuracy: {results['evaluation_results']['accuracy']:.4f}")
    print(f"Number of Classes: {results['label_analysis']['num_classes']}")
    print("Model saved in: runs/classify/wafer_defect_classifier/")
    print(f"Dataset created at: {classifier.dataset_dir}")

    # Example prediction on a test image
    test_dir = Path("dataset/test")
    if test_dir.exists():
        # Find first test image
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                test_images = list(class_dir.glob("*.png"))
                if test_images:
                    sample_image = test_images[0]
                    try:
                        prediction = classifier.predict_sample(str(sample_image))
                        print("\n" + "-" * 40)
                        print("SAMPLE PREDICTION")
                        print("-" * 40)
                        print(f"Image: {sample_image.name}")
                        print(f"Predicted Class: {prediction['class_name']}")
                        print(f"Confidence: {prediction['confidence']:.4f}")
                        print(f"Defect Pattern: {prediction['defect_pattern']}")
                    except Exception as e:
                        print(f"Sample prediction failed: {e}")
                    break

    print("\n=== Classification Complete ===")
    return results


if __name__ == "__main__":
    main()
