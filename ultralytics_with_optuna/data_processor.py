#!/usr/bin/env python3
"""
Data processing utilities for Wafer Defect Classification
Handles data loading, preprocessing, and dataset creation
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml
from config import DataConfig, ValidationConfig
from sklearn.model_selection import train_test_split


class WaferDataProcessor:
    """Handles wafer map data processing and dataset creation"""

    def __init__(self, data_path: str = DataConfig.DEFAULT_DATA_PATH):
        self.data_path = data_path
        self.class_names: List[str] = []
        self.label_to_class: Dict[Tuple, int] = {}
        self.class_to_label: Dict[int, Tuple] = {}

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

    def analyze_labels(self, labels: np.ndarray) -> Dict[str, Any]:
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
            percentage = count / len(labels) * 100
            print(
                f"  Class {i} ({self.class_names[i]}): {count} samples ({percentage:.1f}%)"
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
        if images.max() <= DataConfig.VALUE_RANGE[1]:
            # Scale from 0-3 to 0-255
            images = (images * DataConfig.RGB_SCALE_FACTOR).astype(np.uint8)

        # Convert to RGB (3 channels) for YOLO
        rgb_images = np.zeros(
            (len(images), DataConfig.IMAGE_SHAPE[0], DataConfig.IMAGE_SHAPE[1], 3),
            dtype=np.uint8,
        )
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
        dataset_dir: Path,
        test_size: float = DataConfig.TEST_SPLIT,
        val_size: float = DataConfig.VAL_SPLIT,
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
            random_state=ValidationConfig.RANDOM_SEED,
            stratify=class_indices,
        )

        val_split = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_split,
            random_state=ValidationConfig.RANDOM_SEED,
            stratify=y_temp,
        )

        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # Create directory structure
        for split in ["train", "val", "test"]:
            split_dir = dataset_dir / split
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
                filepath = dataset_dir / split / class_name / filename
                cv2.imwrite(str(filepath), image)

        # Create data.yaml for YOLO
        data_config = {
            "path": str(dataset_dir.absolute()),
            "train": "train",
            "val": "val",
            "test": "test",
            "nc": len(self.class_names),
            "names": {i: name for i, name in enumerate(self.class_names)},
        }

        with open(dataset_dir / "data.yaml", "w") as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"Dataset created successfully at {dataset_dir}")

    def create_stratified_subset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        subset_size: int,
        random_state: int = ValidationConfig.RANDOM_SEED,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a stratified subset of the data for faster processing"""
        if len(images) <= subset_size:
            return images, labels

        # Convert labels to class indices for stratification
        class_indices = self.convert_labels_to_classes(labels)

        # Stratified sampling
        indices = list(range(len(images)))
        subset_indices, _ = train_test_split(
            indices,
            test_size=1 - subset_size / len(images),
            stratify=class_indices,
            random_state=random_state,
        )

        return images[subset_indices], labels[subset_indices]

    def get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """Get distribution of classes in the dataset"""
        class_indices = self.convert_labels_to_classes(labels)
        distribution = {}

        for class_idx, class_name in enumerate(self.class_names):
            count = np.sum(class_indices == class_idx)
            distribution[class_name] = count

        return distribution

    def validate_dataset_structure(self, dataset_dir: Path) -> bool:
        """Validate that the dataset directory structure is correct"""
        required_splits = ["train", "val", "test"]

        # Check if dataset directory exists
        if not dataset_dir.exists():
            print(f"Dataset directory {dataset_dir} does not exist")
            return False

        # Check if all splits exist
        for split in required_splits:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                print(f"Split directory {split_dir} does not exist")
                return False

        # Check if data.yaml exists
        yaml_path = dataset_dir / "data.yaml"
        if not yaml_path.exists():
            print(f"Data configuration file {yaml_path} does not exist")
            return False

        print("Dataset structure validation passed")
        return True

    def get_dataset_stats(self, dataset_dir: Path) -> Dict[str, Any]:
        """Get statistics about the created dataset"""
        if not self.validate_dataset_structure(dataset_dir):
            return {}

        stats = {}

        for split in ["train", "val", "test"]:
            split_dir = dataset_dir / split
            split_stats = {}
            total_images = 0

            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    image_count = len(list(class_dir.glob("*.png")))
                    split_stats[class_dir.name] = image_count
                    total_images += image_count

            split_stats["total"] = total_images
            stats[split] = split_stats

        return stats


def create_balanced_subset(
    data_processor: WaferDataProcessor,
    images: np.ndarray,
    labels: np.ndarray,
    min_samples_per_class: int = ValidationConfig.MIN_SAMPLES_PER_CLASS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a balanced subset by undersampling majority classes"""
    class_indices = data_processor.convert_labels_to_classes(labels)
    unique_classes, class_counts = np.unique(class_indices, return_counts=True)

    # Determine target count (minimum of min_samples and smallest class size)
    target_count = min(min_samples_per_class, min(class_counts))

    balanced_indices = []

    for class_idx in unique_classes:
        class_mask = class_indices == class_idx
        class_indices_list = np.where(class_mask)[0]

        if len(class_indices_list) > 0:
            n_samples = min(target_count, len(class_indices_list))
            selected_indices = np.random.choice(
                class_indices_list, n_samples, replace=False
            )
            balanced_indices.extend(selected_indices)

    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)

    return images[balanced_indices], labels[balanced_indices]
