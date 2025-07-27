"""
Data utilities for wafer defect classification tutorials.

This module provides consistent data loading and preprocessing across all tutorials,
ensuring reproducible results and clean code patterns.
"""

import pickle
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split

warnings.filterwarnings("ignore")


class WaferMapDataset(Dataset):
    """
    Dataset class for wafer map defect classification.

    Supports both the original WM-811K dataset and simplified tutorial datasets.
    Provides consistent preprocessing and class mapping across all tutorials.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        target_size: Tuple[int, int] = (64, 64),
        task_type: str = "multiclass",
        balance_classes: bool = False,
        max_samples_per_class: Optional[int] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize wafer map dataset.

        Args:
            data_path: Path to dataset (.pkl or .npz file)
            target_size: Target image size (height, width)
            task_type: 'binary' or 'multiclass' classification
            balance_classes: Whether to balance class distribution
            max_samples_per_class: Maximum samples per class (for quick demos)
            transform: Optional data transforms
        """
        self.data_path = Path(data_path)
        self.target_size = target_size
        self.task_type = task_type
        self.transform = transform

        # Class mapping for wafer defect types
        self.class_names = [
            "Center",
            "Donut",
            "Edge-Loc",
            "Edge-Ring",
            "Loc",
            "Random",
            "Scratch",
            "Near-full",
            "None",
        ]

        # Load and process data
        self.wafer_maps, self.labels = self._load_data()
        self._print_class_distribution()

        # Apply class balancing if requested
        if balance_classes or max_samples_per_class:
            self.wafer_maps, self.labels = self._balance_classes(max_samples_per_class)
            print(f"After balancing: {len(self.wafer_maps)} samples")

        # Convert to tensors
        self.wafer_maps = torch.FloatTensor(self.wafer_maps)
        self.labels = torch.LongTensor(self.labels)

        # Set up task-specific properties
        if task_type == "binary":
            # Convert to binary: defect (0) vs no-defect (1)
            self.binary_labels = (self.labels != 8).long()
            self.num_classes = 2
            self.task_class_names = ["Defect", "No-Defect"]
        else:
            self.num_classes = len(self.class_names)
            self.task_class_names = self.class_names

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from file based on extension."""
        if self.data_path.suffix == ".pkl":
            return self._load_from_pickle()
        elif self.data_path.suffix == ".npz":
            return self._load_from_npz()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _load_from_pickle(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from WM-811K pickle format."""
        print(f"Loading WM-811K dataset from {self.data_path}...")

        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        wafer_maps, labels = [], []

        for i, item in enumerate(data):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(data)} samples")

            die_map = item.get("dieMap")
            failure_type = item.get("failureType")

            if die_map is not None and failure_type is not None:
                wafer_map = np.array(die_map, dtype=np.float32)
                if wafer_map.size > 0:
                    wafer_maps.append(wafer_map)
                    labels.append(failure_type)

        return self._process_wafer_maps(wafer_maps), np.array(labels)

    def _load_from_npz(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from simplified NPZ format."""
        print(f"Loading dataset from {self.data_path}...")

        data = np.load(self.data_path)
        wafer_maps = data["wafer_maps"]
        labels = data["labels"]

        return self._process_wafer_maps(wafer_maps), labels

    def _process_wafer_maps(self, wafer_maps: List[np.ndarray]) -> np.ndarray:
        """Normalize and resize wafer maps to consistent format."""
        processed_maps = []

        for wafer_map in wafer_maps:
            # Normalize to [0, 1]
            if wafer_map.max() > 1:
                wafer_map = wafer_map / wafer_map.max()

            # Resize if needed
            if wafer_map.shape != self.target_size:
                wafer_map = cv2.resize(
                    wafer_map, self.target_size, interpolation=cv2.INTER_NEAREST
                )

            processed_maps.append(wafer_map)

        return np.array(processed_maps, dtype=np.float32)

    def _print_class_distribution(self) -> None:
        """Print current class distribution."""
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\nDataset: {len(self.labels)} samples, {len(unique)} classes")
        print("Class distribution:")

        for class_id, count in zip(unique, counts):
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"Class_{class_id}"
            )
            percentage = count / len(self.labels) * 100
            print(f"  {class_name:12}: {count:5} ({percentage:5.1f}%)")

    def _balance_classes(
        self, max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance classes by undersampling or limiting samples."""
        from collections import Counter

        class_counts = Counter(self.labels)

        if max_samples is None:
            # Use the minimum class count
            target_count = min(class_counts.values())
        else:
            target_count = max_samples

        balanced_maps, balanced_labels = [], []

        for class_id in range(len(self.class_names)):
            class_indices = np.where(self.labels == class_id)[0]

            if len(class_indices) > 0:
                n_samples = min(target_count, len(class_indices))
                if n_samples > 0:
                    selected_indices = np.random.choice(
                        class_indices, n_samples, replace=False
                    )

                    for idx in selected_indices:
                        balanced_maps.append(self.wafer_maps[idx])
                        balanced_labels.append(self.labels[idx])

        return np.array(balanced_maps), np.array(balanced_labels)

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced data."""
        if self.task_type == "binary":
            labels_for_weights = self.binary_labels.numpy()
        else:
            labels_for_weights = self.labels.numpy()

        class_counts = np.bincount(labels_for_weights)
        total_samples = len(labels_for_weights)
        n_classes = self.num_classes

        # Calculate inverse frequency weights
        weights = total_samples / (n_classes * class_counts)
        return torch.FloatTensor(weights)

    def __len__(self) -> int:
        return len(self.wafer_maps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        # Get wafer map and add channel dimension
        wafer_map = self.wafer_maps[idx].unsqueeze(0)  # Shape: (1, H, W)

        # Get appropriate label based on task type
        if self.task_type == "binary":
            label = self.binary_labels[idx]
        else:
            label = self.labels[idx]

        # Apply transforms if provided
        if self.transform:
            wafer_map = self.transform(wafer_map)

        return wafer_map, label


def create_data_loaders(
    dataset: WaferMapDataset,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.2,
    num_workers: int = 4,
    use_weighted_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test data loaders with proper splits.

    Args:
        dataset: WaferMapDataset instance
        batch_size: Batch size for all loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        num_workers: Number of worker processes
        use_weighted_sampling: Whether to use weighted sampling for training

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create samplers
    train_sampler = None
    if use_weighted_sampling:
        # Get labels for training set
        if dataset.task_type == "binary":
            train_labels = [
                dataset.binary_labels[i].item() for i in train_dataset.indices
            ]
        else:
            train_labels = [dataset.labels[i].item() for i in train_dataset.indices]

        # Calculate sample weights
        class_counts = np.bincount(train_labels)
        weights = 1.0 / class_counts
        sample_weights = [weights[label] for label in train_labels]

        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader


def create_sample_dataset(
    output_path: Union[str, Path],
    n_samples: int = 1000,
    image_size: Tuple[int, int] = (64, 64),
    n_classes: int = 9,
) -> None:
    """
    Create a simplified sample dataset for tutorial purposes.

    This generates synthetic wafer maps that mimic real defect patterns
    but can be used without downloading the full WM-811K dataset.

    Args:
        output_path: Where to save the dataset (.npz)
        n_samples: Total number of samples to generate
        image_size: Size of each wafer map
        n_classes: Number of defect classes
    """
    print(f"Generating sample dataset with {n_samples} samples...")

    wafer_maps = []
    labels = []

    # Generate samples for each class
    samples_per_class = n_samples // n_classes

    for class_id in range(n_classes):
        for _ in range(samples_per_class):
            # Generate synthetic wafer map based on class
            wafer_map = _generate_synthetic_wafer_map(class_id, image_size)
            wafer_maps.append(wafer_map)
            labels.append(class_id)

    # Convert to numpy arrays
    wafer_maps = np.array(wafer_maps, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Save dataset
    np.savez_compressed(
        output_path,
        wafer_maps=wafer_maps,
        labels=labels,
        class_names=[
            "Center",
            "Donut",
            "Edge-Loc",
            "Edge-Ring",
            "Loc",
            "Random",
            "Scratch",
            "Near-full",
            "None",
        ],
    )

    print(f"Sample dataset saved to {output_path}")
    print(f"Shape: {wafer_maps.shape}, Labels: {len(np.unique(labels))} classes")


def _generate_synthetic_wafer_map(
    class_id: int, image_size: Tuple[int, int]
) -> np.ndarray:
    """Generate synthetic wafer map for a specific defect class."""
    h, w = image_size
    wafer_map = np.zeros((h, w), dtype=np.float32)

    # Create circular wafer boundary
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 2 - 2

    y, x = np.ogrid[:h, :w]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

    # Generate defect patterns based on class
    if class_id == 0:  # Center
        center_radius = radius // 4
        center_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= center_radius**2
        wafer_map[center_mask & mask] = 1.0

    elif class_id == 1:  # Donut
        outer_radius = radius // 2
        inner_radius = radius // 4
        donut_mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= outer_radius**2) & (
            (x - center_x) ** 2 + (y - center_y) ** 2 >= inner_radius**2
        )
        wafer_map[donut_mask & mask] = 1.0

    elif class_id == 2:  # Edge-Loc
        edge_width = 5
        edge_mask = (x - center_x) ** 2 + (y - center_y) ** 2 >= (
            radius - edge_width
        ) ** 2
        # Add local defect on edge
        angle = np.random.uniform(0, 2 * np.pi)
        edge_x = int(center_x + (radius - edge_width // 2) * np.cos(angle))
        edge_y = int(center_y + (radius - edge_width // 2) * np.sin(angle))
        if 0 <= edge_x < w and 0 <= edge_y < h:
            wafer_map[
                max(0, edge_y - 2) : min(h, edge_y + 3),
                max(0, edge_x - 2) : min(w, edge_x + 3),
            ] = 1.0

    elif class_id == 3:  # Edge-Ring
        edge_width = 3
        edge_mask = (x - center_x) ** 2 + (y - center_y) ** 2 >= (
            radius - edge_width
        ) ** 2
        wafer_map[edge_mask & mask] = 1.0

    elif class_id == 4:  # Loc (Local defect)
        # Random local defect
        defect_x = np.random.randint(radius // 2, w - radius // 2)
        defect_y = np.random.randint(radius // 2, h - radius // 2)
        defect_size = np.random.randint(3, 8)
        wafer_map[
            defect_y : defect_y + defect_size, defect_x : defect_x + defect_size
        ] = 1.0

    elif class_id == 5:  # Random
        # Random scattered defects
        n_defects = np.random.randint(10, 30)
        for _ in range(n_defects):
            defect_x = np.random.randint(0, w)
            defect_y = np.random.randint(0, h)
            if mask[defect_y, defect_x]:
                wafer_map[defect_y, defect_x] = 1.0

    elif class_id == 6:  # Scratch
        # Linear scratch across wafer
        start_x, start_y = np.random.randint(0, w), np.random.randint(0, h)
        end_x, end_y = np.random.randint(0, w), np.random.randint(0, h)
        scratch_points = np.linspace([start_y, start_x], [end_y, end_x], 50).astype(int)
        for point in scratch_points:
            if 0 <= point[0] < h and 0 <= point[1] < w and mask[point[0], point[1]]:
                wafer_map[point[0], point[1]] = 1.0

    elif class_id == 7:  # Near-full
        # Most of wafer is defective
        wafer_map[mask] = 1.0
        # Leave some good areas
        good_areas = np.random.randint(5, 15)
        for _ in range(good_areas):
            good_x = np.random.randint(radius // 4, w - radius // 4)
            good_y = np.random.randint(radius // 4, h - radius // 4)
            good_size = np.random.randint(3, 8)
            wafer_map[good_y : good_y + good_size, good_x : good_x + good_size] = 0.0

    # Class 8 (None) remains zeros

    # Apply wafer boundary
    wafer_map = wafer_map * mask.astype(np.float32)

    # Add some noise
    noise = np.random.normal(0, 0.05, (h, w))
    wafer_map = np.clip(wafer_map + noise, 0, 1)

    return wafer_map


# Utility functions for quick setup
def get_tutorial_dataset(
    data_dir: Union[str, Path] = "data/sample_wafer_maps",
    create_if_missing: bool = True,
    **kwargs,
) -> WaferMapDataset:
    """
    Get tutorial dataset, creating sample data if needed.

    Args:
        data_dir: Directory containing dataset
        create_if_missing: Create sample dataset if not found
        **kwargs: Additional arguments for WaferMapDataset

    Returns:
        WaferMapDataset instance ready for use
    """
    data_dir = Path(data_dir)
    dataset_path = data_dir / "tutorial_dataset.npz"

    if not dataset_path.exists() and create_if_missing:
        data_dir.mkdir(parents=True, exist_ok=True)
        create_sample_dataset(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Set create_if_missing=True to generate sample data."
        )

    return WaferMapDataset(dataset_path, **kwargs)


if __name__ == "__main__":
    # Demo: Create sample dataset and test loading
    print("Creating sample dataset for tutorials...")

    # Create sample dataset
    sample_path = Path("../data/sample_wafer_maps/tutorial_dataset.npz")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    create_sample_dataset(sample_path, n_samples=1000)

    # Test loading
    dataset = WaferMapDataset(
        sample_path, balance_classes=True, max_samples_per_class=50
    )
    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size=16)

    print("\nDataset ready!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test a batch
    for batch_x, batch_y in train_loader:
        print(f"Batch shape: {batch_x.shape}, Labels: {batch_y.shape}")
        break
