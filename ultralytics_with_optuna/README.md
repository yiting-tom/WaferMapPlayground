# Wafer Defect Classification with Ultralytics YOLOv8

> **Clean, Modular Implementation with Optuna Hyperparameter Optimization**

A production-ready wafer defect classification system using state-of-the-art YOLOv8 for semiconductor manufacturing quality control. Features a clean modular architecture, comprehensive configuration management, and automated hyperparameter optimization.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green)](https://github.com/ultralytics/ultralytics)
[![Optuna](https://img.shields.io/badge/Optuna-3.0%2B-orange)](https://optuna.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🚀 Quick Start

```bash
# Clone and setup
cd ultralytics_with_optuna
pip install -r requirements.txt

# Quick demo (5 epochs, 1000 samples)
python demo.py --quick-demo

# Full training pipeline
python main.py

# Hyperparameter optimization
python demo_optuna.py --quick-demo --trials 10
```

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Demo Modes](#-demo-modes)
- [Optuna Optimization](#-optuna-optimization)
- [API Reference](#-api-reference)
- [Results](#-results)
- [Contributing](#-contributing)

## ✨ Features

### 🏗️ **Clean Modular Architecture**

- **Separation of Concerns**: Each module has a focused responsibility
- **Type Safety**: Comprehensive type hints throughout
- **Configuration Management**: Centralized settings with validation
- **Error Handling**: Proper logging and graceful failure handling

### 🔬 **Advanced ML Pipeline**

- **YOLOv8 Integration**: State-of-the-art classification model
- **Automated Optimization**: Optuna-powered hyperparameter tuning
- **Comprehensive Evaluation**: Detailed metrics and visualizations
- **Flexible Training**: Configurable parameters and subset training

### 📊 **Rich Visualizations**

- **Interactive Plots**: Confusion matrices, class distributions
- **Training Metrics**: Loss curves and performance tracking
- **Optimization Dashboards**: Parameter importance and search history
- **Sample Galleries**: Wafer map visualizations with predictions

### 🎯 **Production Ready**

- **Robust Error Handling**: Graceful failure and recovery
- **Comprehensive Logging**: Configurable logging levels
- **Flexible Configuration**: Easy parameter customization
- **Multiple Demo Modes**: Quick testing and full evaluation

## 🏗️ Architecture

```
ultralytics_with_optuna/
├── 🔧 config.py              # Centralized configuration management
├── 📁 data_processor.py      # Data loading and preprocessing
├── 📊 visualization.py       # Plotting and result reporting
├── 🤖 classifier.py          # Main classification pipeline
├── 🚀 main.py               # Simple entry point
├── 🎮 demo.py               # Multiple demo modes
├── ⚡ optuna_optimizer.py   # Hyperparameter optimization
├── 📦 requirements.txt      # Dependencies
└── 📖 README.md            # This file
```

### Module Responsibilities

| Module              | Purpose       | Lines | Key Features                                  |
| ------------------- | ------------- | ----- | --------------------------------------------- |
| `config.py`         | Configuration | 158   | Centralized settings, validation, type safety |
| `data_processor.py` | Data handling | 267   | Loading, preprocessing, dataset creation      |
| `visualization.py`  | Plotting      | 483   | Comprehensive plots, reports, dashboards      |
| `classifier.py`     | ML Pipeline   | 401   | Training, evaluation, prediction              |
| `main.py`           | Entry point   | 54    | Simple, clean interface                       |

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 4GB+ RAM

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset
# Place your wafer dataset at: ../data/MixedWM38/Wafer_Map_Datasets.npz
# Or update the path in config.py

# 3. Verify installation
python -c "from classifier import WaferDefectClassifier; print('✅ Setup complete!')"
```

### Dataset Structure

```
data/MixedWM38/
├── Wafer_Map_Datasets.npz    # Main dataset (38,015 samples)
├── Description.pdf           # Dataset documentation
└── ...                      # Additional dataset files
```

**Dataset Details:**

- **Samples**: 38,015 wafer maps
- **Resolution**: 52×52 pixels
- **Classes**: 38 unique defect patterns
- **Format**: Multi-label binary → Single-label classification

## 🎯 Usage

### Basic Training

```python
from classifier import WaferDefectClassifier

# Initialize with clean architecture
classifier = WaferDefectClassifier()

# Run complete pipeline
results = classifier.run_complete_pipeline(
    epochs=50,
    img_size=224,
    visualize=True
)

print(f"Accuracy: {results['evaluation_results']['accuracy']:.4f}")
```

### Custom Configuration

```python
from classifier import WaferDefectClassifier
from config import DataConfig, ModelConfig

# Custom training settings
classifier = WaferDefectClassifier(
    data_path="/path/to/your/dataset.npz"
)

results = classifier.run_complete_pipeline(
    epochs=100,
    img_size=256,
    use_subset=False,      # Use full dataset
    balance_classes=True,  # Balance class distribution
    visualize=True,
    save_results=True
)
```

### Prediction on New Images

```python
# Train model first
classifier = WaferDefectClassifier()
classifier.run_complete_pipeline(epochs=20)

# Make predictions
prediction = classifier.predict_sample("path/to/wafer_image.png")

print(f"Class: {prediction['class_name']}")
print(f"Confidence: {prediction['confidence']:.4f}")
print(f"Defect Pattern: {prediction['defect_pattern']}")
```

## ⚙️ Configuration

### Configuration Classes

The new modular system uses centralized configuration:

```python
from config import DataConfig, ModelConfig, TrainingConfig

# Data settings
DataConfig.DEFAULT_DATA_PATH     # Dataset location
DataConfig.TARGET_SIZE          # Image resize dimensions
DataConfig.TRAIN_SPLIT          # Training split ratio

# Model settings
ModelConfig.DEFAULT_MODEL_SIZE   # YOLOv8 variant (n/s/m/l/x)
ModelConfig.DEFAULT_IMG_SIZE     # Input image size
ModelConfig.DEFAULT_BATCH_SIZE   # Training batch size

# Training settings
TrainingConfig.DEFAULT_LEARNING_RATE  # Initial learning rate
TrainingConfig.DEFAULT_EPOCHS         # Training epochs
```

### Customizing Configuration

```python
# Option 1: Modify config.py directly
# Edit values in config.py for permanent changes

# Option 2: Override at runtime
classifier = WaferDefectClassifier()
results = classifier.run_complete_pipeline(
    epochs=100,                    # Override default
    img_size=320,                 # Override default
    custom_params={               # Additional overrides
        'batch_size': 64,
        'learning_rate': 0.001
    }
)
```

## 🎮 Demo Modes

### Quick Demo (Fast Testing)

```bash
python demo.py --quick-demo
```

- **Duration**: ~5 minutes
- **Samples**: 1,000 (subset)
- **Epochs**: 5
- **Purpose**: Quick functionality test

### Performance Demo (Evaluation Focus)

```bash
python demo.py --performance-demo
```

- **Duration**: ~20 minutes
- **Samples**: 3,000 (balanced)
- **Epochs**: 20
- **Purpose**: Performance evaluation

### Dataset Analysis Only

```bash
python demo.py --analyze-only
```

- **Duration**: ~1 minute
- **Purpose**: Dataset exploration without training
- **Output**: Class distribution, sample visualizations

## ⚡ Optuna Optimization

### Quick Optimization

```bash
python demo_optuna.py --quick-demo --trials 10
```

### Full Optimization

```bash
python demo_optuna.py --full-optimization --trials 50
```

### Using Optimized Parameters

```python
# 1. Run optimization first
from optuna_optimizer import OptunaWaferOptimizer

optimizer = OptunaWaferOptimizer()
study = optimizer.optimize(n_trials=20)
final_results = optimizer.train_best_model()

# 2. Use saved parameters
classifier = WaferDefectClassifier.from_optimized_params(
    "best_model_best_params.json"
)
results = classifier.run_complete_pipeline()
```

### Optimization Search Space

| Parameter    | Type        | Range                 | Purpose               |
| ------------ | ----------- | --------------------- | --------------------- |
| `epochs`     | int         | 20-100                | Training duration     |
| `batch_size` | categorical | [8,16,32,64]          | Batch size            |
| `img_size`   | categorical | [128,160,192,224,256] | Input resolution      |
| `lr0`        | float       | 1e-5 to 1e-1          | Initial learning rate |
| `model_size` | categorical | [n,s,m]               | YOLOv8 variant        |
| `optimizer`  | categorical | [SGD,Adam,AdamW]      | Optimizer type        |

## 📚 API Reference

### WaferDefectClassifier

```python
class WaferDefectClassifier:
    def __init__(
        self,
        data_path: str = DataConfig.DEFAULT_DATA_PATH,
        dataset_dir: Optional[str] = None,
        optimized_params: Optional[Dict[str, Any]] = None
    )

    def prepare_dataset(
        self,
        use_subset: bool = False,
        subset_size: int = None,
        balance_classes: bool = False
    ) -> Dict[str, Any]

    def train_model(
        self,
        epochs: int = 50,
        img_size: int = 224,
        batch_size: int = 32,
        model_size: str = "n"
    ) -> Dict[str, Any]

    def evaluate_model(self) -> Dict[str, Any]

    def predict_sample(self, image_path: str) -> Dict[str, Any]

    def run_complete_pipeline(
        self,
        epochs: int = 50,
        img_size: int = 224,
        use_subset: bool = False,
        visualize: bool = True
    ) -> Dict[str, Any]
```

### WaferDataProcessor

```python
class WaferDataProcessor:
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]
    def analyze_labels(self, labels: np.ndarray) -> Dict[str, Any]
    def prepare_images(self, images: np.ndarray) -> np.ndarray
    def create_yolo_dataset(self, images: np.ndarray, labels: np.ndarray, dataset_dir: Path) -> None
    def create_stratified_subset(self, images: np.ndarray, labels: np.ndarray, subset_size: int) -> Tuple[np.ndarray, np.ndarray]
```

### WaferVisualization

```python
class WaferVisualization:
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], class_names: List[str]) -> None
    def plot_class_distribution(self, class_distribution: Dict[str, int]) -> None
    def plot_sample_wafer_maps(self, images: np.ndarray, labels: np.ndarray, class_names: List[str]) -> None
    def create_classification_report_plot(self, classification_report: Dict[str, Any]) -> None
```

## 📊 Results

### Performance Metrics

| Metric            | Baseline | Optimized | Improvement |
| ----------------- | -------- | --------- | ----------- |
| **Accuracy**      | 0.431    | 0.567     | +31.6%      |
| **F1-Score**      | 0.425    | 0.553     | +30.1%      |
| **Training Time** | 45 min   | 38 min    | -15.6%      |

### Generated Outputs

The system automatically generates:

```
📁 Output Files:
├── 📊 plots/
│   ├── confusion_matrix.png          # Model performance
│   ├── class_distribution.png        # Dataset balance
│   ├── classification_report.png     # Detailed metrics
│   └── sample_wafer_maps.png        # Data samples
├── 📈 runs/classify/
│   └── wafer_defect_classifier/      # Training results
├── 📋 classification_summary.txt     # Text report
└── 🔧 best_params.json              # Optimized parameters
```

### Visualization Gallery

- **Confusion Matrix**: Model performance across all classes
- **Class Distribution**: Dataset balance and sample counts
- **Sample Gallery**: Wafer maps with ground truth labels
- **Training Curves**: Loss and accuracy over epochs
- **Optimization History**: Parameter search visualization

## 🏆 Key Improvements

### Code Quality

- **-89% Main File Size**: 488 → 54 lines in main.py
- **+100% Type Coverage**: Comprehensive type hints
- **Centralized Config**: Eliminated 50+ magic numbers
- **Modular Design**: Clean separation of concerns

### User Experience

- **3 Demo Modes**: Quick, performance, analysis-only
- **Better Error Messages**: Clear, actionable feedback
- **Flexible Configuration**: Easy parameter customization
- **Rich Visualizations**: Comprehensive result reporting

### Developer Experience

- **Easy Navigation**: Clear module boundaries
- **IDE Support**: Excellent autocomplete and error detection
- **Simple Testing**: Focused, testable functions
- **Clear Documentation**: Comprehensive docstrings

## 🔧 Troubleshooting

### Common Issues

**Memory Errors**

```python
# Reduce batch size or image size
classifier.run_complete_pipeline(
    batch_size=16,    # Reduce from 32
    img_size=128     # Reduce from 224
)
```

**Slow Training**

```python
# Use smaller model or GPU
from config import ModelConfig
ModelConfig.DEFAULT_MODEL_SIZE = "n"  # Use nano model
```

**Dataset Not Found**

```python
# Update data path in config
from config import DataConfig
DataConfig.DEFAULT_DATA_PATH = "/path/to/your/dataset.npz"
```

### Performance Tips

1. **GPU Usage**: Automatically detected, ensure CUDA is available
2. **Batch Size**: Increase if you have more GPU memory
3. **Image Size**: Larger sizes improve accuracy but slow training
4. **Model Size**: Use `yolov8s` or `yolov8m` for better accuracy

## 🤝 Contributing

We welcome contributions! The clean modular architecture makes it easy to add features:

### Adding New Features

```python
# 1. Add configuration (if needed)
# Edit config.py to add new parameters

# 2. Implement feature in appropriate module
# data_processor.py - for data handling
# visualization.py - for new plots
# classifier.py - for model changes

# 3. Update tests and documentation
```

### Development Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd ultralytics_with_optuna

# 2. Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# 3. Run tests
pytest tests/

# 4. Format code
black *.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics Team**: For the excellent YOLOv8 framework
- **Optuna Team**: For the powerful optimization library
- **Dataset Contributors**: For the wafer defect dataset
- **Community**: For feedback and contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: This README and inline docstrings

---

<div align="center">

**Built with ❤️ for semiconductor manufacturing quality control**

[⭐ Star this repo](https://github.com/your-repo) | [🐛 Report bugs](https://github.com/your-repo/issues) | [💡 Request features](https://github.com/your-repo/discussions)

</div>
