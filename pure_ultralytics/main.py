#!/usr/bin/env python3
"""
Wafer Defect Classification using Ultralytics YOLOv8
Dataset: Mixed Wafer Map Dataset with 38,015 samples of 52x52 wafer maps
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


class WaferDefectClassifier:
    """Wafer defect classification using YOLOv8"""

    def __init__(
        self,
        data_path: str = "../data/MixedWM38/Wafer_Map_Datasets.npz",
        optimized_params: Optional[Dict[str, Any]] = None,
    ):
        self.data_path = data_path
        self.dataset_dir = Path("dataset")
        self.model = None
        self.class_names = []
        self.label_to_class = {}
        self.class_to_label = {}
        self.optimized_params = optimized_params

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load wafer map data from NPZ file"""
        print("Loading dataset...")
        data = np.load(self.data_path)
        images = data["arr_0"]  # (38015, 52, 52)
        labels = data["arr_1"]  # (38015, 8) - multi-label binary

        print(f"Loaded {len(images)} samples")
        print(f"Image shape: {images[0].shape}")
        print(f"Label shape: {labels[0].shape}")
        print(f"Image value range: {images.min()} to {images.max()}")

        return images, labels

    def analyze_labels(self, labels: np.ndarray) -> Dict:
        """Analyze label distribution and create class mappings"""
        print("\nAnalyzing labels...")

        # Find unique label combinations
        unique_labels = np.unique(labels, axis=0)
        print(f"Number of unique defect patterns: {len(unique_labels)}")

        # Create class mappings
        self.label_to_class = {}
        self.class_to_label = {}
        self.class_names = []

        for i, label in enumerate(unique_labels):
            label_tuple = tuple(label)
            self.label_to_class[label_tuple] = i
            self.class_to_label[i] = label

            # Create descriptive class names
            active_defects = [j for j, val in enumerate(label) if val == 1]
            if not active_defects:
                class_name = "Normal"
            else:
                class_name = f"Defect_{'-'.join(map(str, active_defects))}"
            self.class_names.append(class_name)

        # Print class distribution
        print("\nClass distribution:")
        for i, label in enumerate(unique_labels):
            count = np.sum(np.all(labels == label, axis=1))
            print(
                f"  Class {i} ({self.class_names[i]}): {count} samples ({count / len(labels) * 100:.1f}%)"
            )

        return {
            "num_classes": len(unique_labels),
            "class_names": self.class_names,
            "label_distribution": {
                i: np.sum(np.all(labels == unique_labels[i], axis=1))
                for i in range(len(unique_labels))
            },
        }

    def prepare_images(self, images: np.ndarray) -> np.ndarray:
        """Prepare images for YOLO classification"""
        print("Preparing images...")

        # Normalize images to 0-255 range
        if images.max() <= 3:
            # Scale from 0-3 to 0-255
            images = (images * 85).astype(np.uint8)  # 3 * 85 = 255

        # Convert to RGB (3 channels) for YOLO
        rgb_images = np.zeros((len(images), 52, 52, 3), dtype=np.uint8)
        for i in range(len(images)):
            # Convert single channel to RGB by duplicating channels
            rgb_images[i] = np.stack([images[i]] * 3, axis=-1)

        return rgb_images

    def convert_labels_to_classes(self, labels: np.ndarray) -> np.ndarray:
        """Convert multi-label format to single class indices"""
        class_indices = []
        for label in labels:
            label_tuple = tuple(label)
            class_idx = self.label_to_class[label_tuple]
            class_indices.append(class_idx)
        return np.array(class_indices)

    def create_yolo_dataset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> None:
        """Create YOLO dataset structure"""
        print("\nCreating YOLO dataset structure...")

        # Prepare images and convert labels
        processed_images = self.prepare_images(images)
        class_indices = self.convert_labels_to_classes(labels)

        # Create train/val/test splits
        X_temp, X_test, y_temp, y_test = train_test_split(
            processed_images,
            class_indices,
            test_size=test_size,
            random_state=42,
            stratify=class_indices,
        )

        val_split = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split, random_state=42, stratify=y_temp
        )

        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # Create directory structure
        for split in ["train", "val", "test"]:
            split_dir = self.dataset_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories for each class
            for class_name in self.class_names:
                (split_dir / class_name).mkdir(exist_ok=True)

        # Save images to appropriate directories
        datasets = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

        for split, (X, y) in datasets.items():
            print(f"Saving {split} images...")
            for i, (image, class_idx) in enumerate(zip(X, y)):
                class_name = self.class_names[class_idx]
                filename = f"{split}_{i:05d}.png"
                filepath = self.dataset_dir / split / class_name / filename
                cv2.imwrite(str(filepath), image)

        # Create data.yaml for YOLO
        data_config = {
            "path": str(self.dataset_dir.absolute()),
            "train": "train",
            "val": "val",
            "test": "test",
            "nc": len(self.class_names),
            "names": {i: name for i, name in enumerate(self.class_names)},
        }

        with open(self.dataset_dir / "data.yaml", "w") as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"Dataset created successfully at {self.dataset_dir}")

    def train_model(
        self,
        epochs: int = 100,
        img_size: int = 224,
        batch_size: int = 32,
        model_name: str = "yolov8n-cls.pt",
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train YOLOv8 classification model with optional optimized parameters"""
        print(f"\nTraining YOLOv8 model ({model_name})...")

        # Use optimized parameters if available
        if self.optimized_params:
            print("Using Optuna-optimized parameters...")
            epochs = self.optimized_params.get("epochs", epochs)
            img_size = self.optimized_params.get("img_size", img_size)
            batch_size = self.optimized_params.get("batch_size", batch_size)
            model_name = f"yolov8{self.optimized_params.get('model_size', 'n')}-cls.pt"

        # Override with custom parameters if provided
        if custom_params:
            print("Applying custom parameters...")
            epochs = custom_params.get("epochs", epochs)
            img_size = custom_params.get("img_size", img_size)
            batch_size = custom_params.get("batch_size", batch_size)

        # Initialize YOLO model
        self.model = YOLO(model_name)

        # Prepare training arguments
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        train_args = {
            "data": str(self.dataset_dir),
            "epochs": epochs,
            "imgsz": img_size,
            "batch": batch_size,
            "name": "wafer_defect_classifier",
            "save": True,
            "plots": True,
            "device": device,
        }

        # Apply optimized hyperparameters if available
        if self.optimized_params:
            train_args.update(
                {
                    "lr0": self.optimized_params.get("lr0", 0.01),
                    "lrf": self.optimized_params.get("lrf", 0.01),
                    "momentum": self.optimized_params.get("momentum", 0.937),
                    "weight_decay": self.optimized_params.get("weight_decay", 0.0005),
                    "warmup_epochs": self.optimized_params.get("warmup_epochs", 3.0),
                    "warmup_momentum": self.optimized_params.get(
                        "warmup_momentum", 0.8
                    ),
                    "warmup_bias_lr": self.optimized_params.get("warmup_bias_lr", 0.1),
                    "hsv_h": self.optimized_params.get("hsv_h", 0.015),
                    "hsv_s": self.optimized_params.get("hsv_s", 0.7),
                    "hsv_v": self.optimized_params.get("hsv_v", 0.4),
                    "degrees": self.optimized_params.get("degrees", 0.0),
                    "translate": self.optimized_params.get("translate", 0.1),
                    "scale": self.optimized_params.get("scale", 0.5),
                    "shear": self.optimized_params.get("shear", 0.0),
                    "perspective": self.optimized_params.get("perspective", 0.0),
                    "fliplr": self.optimized_params.get("fliplr", 0.5),
                    "flipud": self.optimized_params.get("flipud", 0.0),
                    "mixup": self.optimized_params.get("mixup", 0.0),
                    "copy_paste": self.optimized_params.get("copy_paste", 0.0),
                    "dropout": self.optimized_params.get("dropout", 0.0),
                    "optimizer": self.optimized_params.get("optimizer", "auto"),
                }
            )
            print("Applied Optuna-optimized hyperparameters")

        # Apply custom parameters override
        if custom_params:
            train_args.update(custom_params)

        # Train the model
        results = self.model.train(**train_args)

        print("Training completed!")
        return results

    @classmethod
    def from_optimized_params(
        cls,
        params_file: str,
        data_path: str = "../data/MixedWM38/Wafer_Map_Datasets.npz",
    ):
        """Create classifier instance with pre-optimized parameters"""
        import json

        with open(params_file, "r") as f:
            data = json.load(f)

        optimized_params = data["best_params"]
        print(f"Loaded optimized parameters from: {params_file}")
        print(f"Optimization score: {data['best_score']:.4f}")

        return cls(data_path=data_path, optimized_params=optimized_params)

    def evaluate_model(self) -> Dict:
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        print("\nEvaluating model...")

        # Validate on validation set
        val_results = self.model.val(data=str(self.dataset_dir / "data.yaml"))

        # Test on test set - get all test images
        test_dir = self.dataset_dir / "test"
        test_images = []
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                test_images.extend(list(class_dir.glob("*.png")))

        if not test_images:
            print("No test images found. Using validation results.")
            return {
                "accuracy": val_results.top1,
                "classification_report": {},
                "confusion_matrix": [],
            }

        test_results = self.model.predict(source=test_images, save=False, verbose=False)

        # Calculate detailed metrics
        y_true = []
        y_pred = []

        for result in test_results:
            # Get predicted class
            pred_class = result.probs.top1
            y_pred.append(pred_class)

            # Get true class from filename/path
            image_path = result.path
            true_class_name = Path(image_path).parent.name
            true_class = self.class_names.index(true_class_name)
            y_true.append(true_class)

        # Generate classification report
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        return {
            "classification_report": report,
            "confusion_matrix": cm,
            "accuracy": report["accuracy"],
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def plot_results(self, eval_results: Dict) -> None:
        """Plot training results and evaluation metrics"""
        print("\nGenerating plots...")

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        cm = eval_results["confusion_matrix"]

        # Only show classes that have samples in test set
        present_classes = np.unique(eval_results["y_true"] + eval_results["y_pred"])
        present_class_names = [self.class_names[i] for i in present_classes]

        cm_subset = cm[np.ix_(present_classes, present_classes)]

        sns.heatmap(
            cm_subset,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=present_class_names,
            yticklabels=present_class_names,
        )
        plt.title("Confusion Matrix - Wafer Defect Classification")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.show()

        # Plot class distribution
        plt.figure(figsize=(15, 6))
        class_counts = [
            eval_results["classification_report"][name]["support"]
            for name in present_class_names
        ]

        plt.bar(range(len(present_class_names)), class_counts)
        plt.title("Test Set Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.xticks(
            range(len(present_class_names)),
            present_class_names,
            rotation=45,
            ha="right",
        )
        plt.tight_layout()
        plt.savefig("class_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()

    def predict_sample(self, image_path: str) -> Dict:
        """Predict defect class for a single wafer map"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        results = self.model.predict(source=image_path, verbose=False)[0]

        pred_class = results.probs.top1
        confidence = results.probs.top1conf.item()

        return {
            "predicted_class": pred_class,
            "class_name": self.class_names[pred_class],
            "confidence": confidence,
            "defect_pattern": self.class_to_label[pred_class],
        }

    def run_complete_pipeline(self, epochs: int = 50, img_size: int = 224) -> Dict:
        """Run the complete classification pipeline"""
        print("=== Wafer Defect Classification Pipeline ===\n")

        # Load and analyze data
        images, labels = self.load_data()
        label_analysis = self.analyze_labels(labels)

        # Create dataset
        self.create_yolo_dataset(images, labels)

        # Train model
        training_results = self.train_model(epochs=epochs, img_size=img_size)

        # Evaluate model
        eval_results = self.evaluate_model()

        # Plot results
        self.plot_results(eval_results)

        print("\n=== Pipeline Complete ===")
        print(f"Final Accuracy: {eval_results['accuracy']:.4f}")
        print("Model saved in: runs/classify/wafer_defect_classifier/")
        print(f"Dataset created at: {self.dataset_dir}")

        return {
            "label_analysis": label_analysis,
            "training_results": training_results,
            "evaluation_results": eval_results,
        }


def main():
    """Main function to run wafer defect classification"""
    # Initialize classifier
    classifier = WaferDefectClassifier()

    # Run complete pipeline
    results = classifier.run_complete_pipeline(
        epochs=50,  # Adjust as needed
        img_size=224,  # Standard size for classification
    )

    # Example prediction on a test image
    test_dir = Path("dataset/test")
    if test_dir.exists():
        # Find first test image
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                test_images = list(class_dir.glob("*.png"))
                if test_images:
                    sample_image = test_images[0]
                    prediction = classifier.predict_sample(str(sample_image))
                    print("\nSample Prediction:")
                    print(f"Image: {sample_image}")
                    print(f"Predicted: {prediction['class_name']}")
                    print(f"Confidence: {prediction['confidence']:.4f}")
                    print(f"Defect Pattern: {prediction['defect_pattern']}")
                    break


if __name__ == "__main__":
    main()
