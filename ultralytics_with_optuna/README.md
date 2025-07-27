# Wafer Defect Classification using Ultralytics YOLOv8

This project implements a wafer defect classification system using Ultralytics YOLOv8 for semiconductor manufacturing quality control. The solution classifies 52x52 pixel wafer maps into 38 different defect patterns.

## Dataset Overview

- **Dataset**: Mixed Wafer Map Dataset (MixedWM38)
- **Samples**: 38,015 wafer maps
- **Image Size**: 52x52 pixels
- **Classes**: 38 unique defect patterns (multi-label combinations)
- **Value Range**: 0-3 (representing different defect intensities)

The dataset contains wafer maps with various defect patterns that can occur individually or in combination, making this a multi-label classification problem converted to single-label by treating each unique combination as a separate class.

## Features

- **Complete Pipeline**: Data loading, preprocessing, training, and evaluation
- **YOLOv8 Integration**: Uses state-of-the-art YOLOv8 classification model
- **Optuna Optimization**: Automated hyperparameter tuning for optimal performance
- **Comprehensive Analysis**: Detailed label analysis and class distribution
- **Visualization**: Confusion matrices, class distributions, and optimization plots
- **Flexible Training**: Configurable epochs, image size, and batch size
- **Model Evaluation**: Classification reports, accuracy metrics, and confusion matrices
- **Interactive Plots**: Optuna visualization dashboard with parameter importance

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

The dataset should be placed in the following structure:

```
data/MixedWM38/
├── Wafer_Map_Datasets.npz  # Main dataset file
├── Description.pdf         # Dataset description
└── mixedtype-wafer-defect-datasets.zip  # Backup archive
```

## Usage

### Quick Demo

Run a quick demo with reduced parameters for testing:

```bash
cd pure_ultralytics
python demo.py --quick-demo
```

This will:

- Use first 1000 samples
- Train for 5 epochs
- Use 128x128 image size
- Display sample results

### Dataset Analysis Only

To just analyze the dataset without training:

```bash
python demo.py --analyze-only
```

### Full Training Pipeline

For complete training with default parameters:

```bash
python main.py
```

### Custom Training

Modify parameters in `main.py` or create a custom script:

```python
from main import WaferDefectClassifier

classifier = WaferDefectClassifier()
results = classifier.run_complete_pipeline(
    epochs=100,      # Number of training epochs
    img_size=224     # Input image size for training
)
```

### Optuna Hyperparameter Optimization

Automatically find the best hyperparameters using Optuna:

```bash
# Quick optimization demo (10 trials)
python demo_optuna.py --quick-demo --trials 10

# Full optimization (50+ trials)
python demo_optuna.py --full-optimization --trials 50

# Compare baseline vs optimized performance
python demo_optuna.py --compare-baseline

# Load and use pre-optimized parameters
python demo_optuna.py --load-params best_params.json
```

**Optuna optimizes:**

- Learning rate (lr0, lrf)
- Model architecture (YOLOv8n/s/m)
- Batch size and image size
- Data augmentation parameters
- Regularization (dropout, weight decay)
- Training schedule (epochs, warmup)

## Class Structure

The dataset contains 38 unique defect patterns:

- **Class 0**: Normal (no defects)
- **Class 1-37**: Various defect combinations (e.g., Defect_0-2-6, Defect_1-3-4-6)

Each class represents a unique combination of 8 possible defect types that can occur simultaneously on a wafer.

## Model Architecture

- **Base Model**: YOLOv8n-cls.pt (nano classification model)
- **Input Size**: 224x224 (configurable)
- **Output**: 38 classes (unique defect patterns)
- **Training**: Transfer learning from pre-trained YOLOv8 weights

## Results and Evaluation

The system provides:

1. **Classification Report**: Precision, recall, F1-score for each class
2. **Confusion Matrix**: Visual representation of classification performance
3. **Accuracy Metrics**: Overall and per-class accuracy
4. **Class Distribution**: Analysis of dataset balance
5. **Sample Predictions**: Example classifications with confidence scores

## File Structure

```
pure_ultralytics/
├── main.py              # Main classification pipeline
├── demo.py              # Quick demo script
├── demo_optuna.py       # Optuna optimization demo
├── optuna_optimizer.py  # Optuna hyperparameter optimization
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── dataset/            # Generated YOLO dataset structure
│   ├── train/         # Training images (by class)
│   ├── val/           # Validation images (by class)
│   ├── test/          # Test images (by class)
│   └── data.yaml      # YOLO dataset configuration
├── runs/               # Training results and models
│   └── classify/
│       └── wafer_defect_classifier/
├── *.db               # Optuna study databases
└── *_plots/           # Optimization visualization plots
```

## Configuration

Key parameters can be adjusted:

- **Epochs**: Number of training iterations (default: 50)
- **Image Size**: Input resolution (default: 224x224)
- **Batch Size**: Training batch size (default: 32)
- **Model**: YOLOv8 variant (n/s/m/l/x)
- **Train/Val/Test Split**: Data distribution (70/10/20 by default)

## Performance Optimization

For better performance:

1. **GPU Training**: Automatically uses GPU if available
2. **Batch Size**: Increase if you have more memory
3. **Image Size**: Larger sizes may improve accuracy
4. **Model Size**: Use yolov8s/m/l/x for better accuracy (slower training)
5. **Epochs**: More epochs for better convergence

## Output Files

The system generates:

- **Trained Model**: Best model weights saved automatically
- **Training Plots**: Loss curves, metrics plots
- **Confusion Matrix**: `confusion_matrix.png`
- **Class Distribution**: `class_distribution.png`
- **Sample Images**: `sample_wafer_maps.png`

## Troubleshooting

1. **Memory Issues**: Reduce batch size or image size
2. **Slow Training**: Use smaller model (yolov8n) or GPU
3. **Poor Accuracy**: Increase epochs, try larger model, or adjust learning rate
4. **Dataset Issues**: Ensure NPZ file is in correct location

## Example Usage

### Basic Usage

```python
# Initialize classifier
classifier = WaferDefectClassifier()

# Run complete pipeline
results = classifier.run_complete_pipeline(epochs=50)

# Make predictions on new images
prediction = classifier.predict_sample('path/to/wafer_image.png')
print(f"Predicted class: {prediction['class_name']}")
print(f"Confidence: {prediction['confidence']:.4f}")
```

### Using Optuna-Optimized Parameters

```python
# Load classifier with pre-optimized parameters
classifier = WaferDefectClassifier.from_optimized_params('best_params.json')

# Train with optimized hyperparameters automatically applied
results = classifier.run_complete_pipeline()

# Or run Optuna optimization programmatically
from optuna_optimizer import OptunaWaferOptimizer

optimizer = OptunaWaferOptimizer()
study = optimizer.optimize(n_trials=20)
final_results = optimizer.train_best_model(use_full_data=True)
```

## Dependencies

- ultralytics >= 8.0.0
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- PyYAML >= 6.0
- torch >= 1.9.0
- torchvision >= 0.10.0

## License

This project is for educational/research purposes. Please check the original dataset license for commercial use restrictions.
