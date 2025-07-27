#!/usr/bin/env python3
"""
Configuration constants and parameters for Wafer Defect Classification
"""

from typing import Any, Dict, List, Tuple


# Data configuration
class DataConfig:
    """Data-related configuration constants"""

    DEFAULT_DATA_PATH: str = "../data/MixedWM38/Wafer_Map_Datasets.npz"
    IMAGE_SHAPE: Tuple[int, int] = (52, 52)
    TARGET_SIZE: Tuple[int, int] = (64, 64)
    NUM_DEFECT_TYPES: int = 8
    VALUE_RANGE: Tuple[int, int] = (0, 3)
    RGB_SCALE_FACTOR: int = 85  # 3 * 85 = 255

    # Dataset splits
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.2
    TEST_SPLIT: float = 0.1

    # Class names for defect types
    DEFECT_TYPES: List[str] = [
        "Center",
        "Donut",
        "Edge-Loc",
        "Edge-Ring",
        "Loc",
        "Random",
        "Scratch",
        "Near-full",
    ]


class ModelConfig:
    """Model-related configuration constants"""

    DEFAULT_MODEL_SIZE: str = "n"  # YOLOv8 nano
    AVAILABLE_MODEL_SIZES: List[str] = ["n", "s", "m", "l", "x"]
    DEFAULT_IMG_SIZE: int = 224
    AVAILABLE_IMG_SIZES: List[int] = [128, 160, 192, 224, 256, 320, 416, 512]
    DEFAULT_BATCH_SIZE: int = 32
    AVAILABLE_BATCH_SIZES: List[int] = [8, 16, 32, 64, 128, 256]
    DEFAULT_EPOCHS: int = 50

    # Training device
    DEFAULT_DEVICE: str = "auto"  # YOLO will auto-detect GPU/CPU


class TrainingConfig:
    """Training-related configuration constants"""

    DEFAULT_LEARNING_RATE: float = 0.01
    DEFAULT_MOMENTUM: float = 0.937
    DEFAULT_WEIGHT_DECAY: float = 0.0005
    DEFAULT_WARMUP_EPOCHS: float = 3.0
    DEFAULT_WARMUP_MOMENTUM: float = 0.8
    DEFAULT_WARMUP_BIAS_LR: float = 0.1

    # Augmentation defaults
    DEFAULT_HSV_H: float = 0.015
    DEFAULT_HSV_S: float = 0.7
    DEFAULT_HSV_V: float = 0.4
    DEFAULT_DEGREES: float = 0.0
    DEFAULT_TRANSLATE: float = 0.1
    DEFAULT_SCALE: float = 0.5
    DEFAULT_SHEAR: float = 0.0
    DEFAULT_PERSPECTIVE: float = 0.0
    DEFAULT_FLIPLR: float = 0.5
    DEFAULT_FLIPUD: float = 0.0
    DEFAULT_MIXUP: float = 0.0
    DEFAULT_COPY_PASTE: float = 0.0
    DEFAULT_DROPOUT: float = 0.0

    # Optimizer
    DEFAULT_OPTIMIZER: str = "auto"
    AVAILABLE_OPTIMIZERS: List[str] = ["SGD", "Adam", "AdamW", "auto"]


class OptunaConfig:
    """Optuna optimization configuration constants"""

    DEFAULT_N_TRIALS: int = 50
    DEFAULT_STUDY_NAME: str = "wafer_defect_optimization"
    DEFAULT_STORAGE: str = "sqlite:///optuna_studies.db"

    # Pruning settings
    PRUNER_N_STARTUP_TRIALS: int = 5
    PRUNER_N_WARMUP_STEPS: int = 10

    # Subset size for optimization
    OPTIMIZATION_SUBSET_SIZE: int = 5000
    DEMO_SUBSET_SIZE: int = 1000

    # Search space ranges
    SEARCH_SPACE: Dict[str, Any] = {
        "epochs": {"type": "int", "low": 20, "high": 100},
        "batch_size": {"type": "categorical", "choices": [8, 16, 32, 64]},
        "img_size": {"type": "categorical", "choices": [128, 160, 192, 224, 256]},
        "lr0": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
        "lrf": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
        "momentum": {"type": "float", "low": 0.8, "high": 0.99},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        "warmup_epochs": {"type": "float", "low": 1.0, "high": 5.0},
        "warmup_momentum": {"type": "float", "low": 0.5, "high": 0.95},
        "warmup_bias_lr": {"type": "float", "low": 0.01, "high": 0.2},
        "hsv_h": {"type": "float", "low": 0.0, "high": 0.1},
        "hsv_s": {"type": "float", "low": 0.0, "high": 0.9},
        "hsv_v": {"type": "float", "low": 0.0, "high": 0.9},
        "degrees": {"type": "float", "low": 0.0, "high": 45.0},
        "translate": {"type": "float", "low": 0.0, "high": 0.2},
        "scale": {"type": "float", "low": 0.1, "high": 0.9},
        "shear": {"type": "float", "low": 0.0, "high": 10.0},
        "perspective": {"type": "float", "low": 0.0, "high": 0.001},
        "fliplr": {"type": "float", "low": 0.0, "high": 0.8},
        "flipud": {"type": "float", "low": 0.0, "high": 0.5},
        "mixup": {"type": "float", "low": 0.0, "high": 0.3},
        "copy_paste": {"type": "float", "low": 0.0, "high": 0.3},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "model_size": {"type": "categorical", "choices": ["n", "s", "m"]},
        "optimizer": {"type": "categorical", "choices": ["SGD", "Adam", "AdamW"]},
    }


class PathConfig:
    """Path-related configuration constants"""

    DATASET_DIR_NAME: str = "dataset"
    TEMP_DIR_PREFIX: str = "temp_trial_"
    RESULTS_DIR: str = "runs"
    PLOTS_DIR: str = "plots"
    MODELS_DIR: str = "models"

    # File extensions
    IMAGE_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg"]
    MODEL_EXTENSION: str = ".pt"
    CONFIG_EXTENSION: str = ".yaml"
    DATA_EXTENSION: str = ".npz"


class LoggingConfig:
    """Logging configuration constants"""

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


class ValidationConfig:
    """Validation and evaluation configuration"""

    RANDOM_SEED: int = 42
    MIN_SAMPLES_PER_CLASS: int = 1000
    CLASSIFICATION_THRESHOLD: float = 0.5

    # Metrics
    AVERAGE_METHOD: str = "weighted"  # For multi-class F1-score

    # Visualization
    FIGURE_SIZE: Tuple[int, int] = (12, 10)
    DPI: int = 300
    CMAP: str = "viridis"


# Helper functions for configuration
def get_model_path(model_size: str) -> str:
    """Get the full model path for a given size"""
    return f"yolov8{model_size}-cls.pt"


def get_dataset_splits() -> Tuple[float, float, float]:
    """Get the dataset split ratios"""
    return DataConfig.TRAIN_SPLIT, DataConfig.VAL_SPLIT, DataConfig.TEST_SPLIT


def validate_config() -> None:
    """Validate configuration consistency"""
    # Check that splits sum to 1.0
    total_split = DataConfig.TRAIN_SPLIT + DataConfig.VAL_SPLIT + DataConfig.TEST_SPLIT
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Dataset splits must sum to 1.0, got {total_split}")

    # Check model sizes are valid
    for size in ModelConfig.AVAILABLE_MODEL_SIZES:
        if size not in ["n", "s", "m", "l", "x"]:
            raise ValueError(f"Invalid model size: {size}")


# Validate configuration on import
validate_config()
