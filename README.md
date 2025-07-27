# 🚀 ML Tools Integration Tutorial with PyTorch Lightning

> **Learn to integrate Optuna, MLflow, Ultralytics, and Torchvision with PyTorch Lightning through hands-on wafer defect classification**

A comprehensive, step-by-step tutorial series that teaches you how to integrate popular ML tools with PyTorch Lightning. Each tutorial builds on the previous one, using a consistent wafer defect classification problem to demonstrate real-world applications.

## 🎯 What You'll Learn

- **PyTorch Lightning**: Clean, scalable training loops and model organization
- **Torchvision**: Leveraging pre-trained models for transfer learning
- **Optuna**: Automated hyperparameter optimization and pruning
- **MLflow**: Experiment tracking, model registry, and reproducibility
- **Ultralytics**: Modern YOLO models for classification tasks
- **Integration Patterns**: Best practices for combining these tools

## 📚 Tutorial Progression

| Tutorial                                                              | Focus                  | Tools                   | Complexity | Duration |
| --------------------------------------------------------------------- | ---------------------- | ----------------------- | ---------- | -------- |
| **[01-basic-lightning](./01-basic-lightning/)**                       | Lightning Fundamentals | PyTorch Lightning       | ⭐         | 30 min   |
| **[02-lightning-with-torchvision](./02-lightning-with-torchvision/)** | Pre-trained Models     | Lightning + Torchvision | ⭐⭐       | 45 min   |
| **[03-lightning-with-optuna](./03-lightning-with-optuna/)**           | Hyperparameter Tuning  | Lightning + Optuna      | ⭐⭐⭐     | 60 min   |
| **[04-lightning-with-mlflow](./04-lightning-with-mlflow/)**           | Experiment Tracking    | Lightning + MLflow      | ⭐⭐⭐     | 60 min   |
| **[05-ultralytics-basics](./05-ultralytics-basics/)**                 | YOLO Classification    | Ultralytics YOLO        | ⭐⭐       | 45 min   |
| **[06-ultralytics-with-optuna](./06-ultralytics-with-optuna/)**       | YOLO Optimization      | Ultralytics + Optuna    | ⭐⭐⭐     | 60 min   |
| **[07-complete-integration](./07-complete-integration/)**             | Full Pipeline          | All Tools Together      | ⭐⭐⭐⭐   | 90 min   |

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# CUDA GPU recommended (optional)
nvidia-smi
```

### Installation

```bash
# Clone or download this tutorial
git clone <repository-url>
cd ml-integration-tutorial

# Install base requirements
pip install torch torchvision pytorch-lightning

# Additional tools (install as needed per tutorial)
pip install optuna mlflow ultralytics
```

### Running Your First Tutorial

```bash
cd 01-basic-lightning
python train.py
```

## 📖 Tutorial Details

### 🔰 [01-basic-lightning](./01-basic-lightning/)

**Learn PyTorch Lightning fundamentals**

- Clean model architecture with LightningModule
- Automatic training/validation loops
- Built-in logging and checkpointing
- **Output**: Trained CNN model for wafer classification

### 🖼️ [02-lightning-with-torchvision](./02-lightning-with-torchvision/)

**Add pre-trained models and transfer learning**

- Using torchvision pre-trained models (ResNet, EfficientNet)
- Transfer learning techniques
- Feature extraction vs fine-tuning
- **Output**: Improved accuracy with pre-trained features

### ⚡ [03-lightning-with-optuna](./03-lightning-with-optuna/)

**Automated hyperparameter optimization**

- Optuna study setup and objective functions
- Pruning strategies for efficient search
- Lightning callbacks for trial management
- **Output**: Optimized hyperparameters and performance

### 📊 [04-lightning-with-mlflow](./04-lightning-with-mlflow/)

**Experiment tracking and model management**

- MLflow experiment logging
- Parameter and metric tracking
- Model registry and versioning
- **Output**: Organized experiment runs and model artifacts

### 🎯 [05-ultralytics-basics](./05-ultralytics-basics/)

**Modern YOLO for classification**

- YOLO model setup for classification tasks
- Ultralytics API and configuration
- Training and evaluation workflows
- **Output**: YOLO-based classifier with modern architecture

### 🔧 [06-ultralytics-with-optuna](./06-ultralytics-with-optuna/)

**Optimizing YOLO models**

- Hyperparameter optimization for YOLO
- Custom objective functions
- Model size vs accuracy trade-offs
- **Output**: Optimized YOLO model with best parameters

### 🏆 [07-complete-integration](./07-complete-integration/)

**Bringing it all together**

- Complete ML pipeline with all tools
- Experiment comparison and model selection
- Production-ready implementation patterns
- **Output**: End-to-end ML system with full observability

## 🛠️ Shared Components

The tutorial uses common utilities located in the [`shared/`](./shared/) directory:

- **`data_utils.py`**: Dataset loading and preprocessing
- **`metrics.py`**: Common evaluation functions
- **`visualizations.py`**: Standardized plotting utilities

These components ensure consistency across tutorials while demonstrating reusable code patterns.

## 📊 Dataset

All tutorials use the same **wafer defect classification** dataset:

- **Problem**: Classify semiconductor wafer defect patterns
- **Data**: Grayscale wafer maps (64x64 pixels)
- **Classes**: 9 defect types (Center, Donut, Edge-Ring, etc.)
- **Size**: ~10K samples for quick experimentation

## 🎓 Learning Path Recommendations

### **Beginner**: New to PyTorch Lightning

```
01 → 02 → 05 → 07
Focus on core concepts and simple integrations
```

### **Intermediate**: Familiar with PyTorch

```
01 → 03 → 04 → 06 → 07
Emphasize optimization and experiment tracking
```

### **Advanced**: Production ML focus

```
03 → 04 → 06 → 07
Skip basics, focus on advanced tooling
```

## 💡 Key Concepts Demonstrated

### **Clean Architecture**

- Separation of concerns between data, model, and training
- Reusable components and configurations
- Type hints and documentation

### **Experiment Management**

- Reproducible training runs
- Hyperparameter tracking
- Model versioning and comparison

### **Optimization Strategies**

- Automated hyperparameter search
- Early stopping and pruning
- Multi-objective optimization

### **Production Patterns**

- Model packaging and deployment preparation
- Monitoring and logging
- Error handling and validation

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**

```python
# Reduce batch size in any tutorial
batch_size = 16  # Instead of 32
```

**Package Conflicts**

```bash
# Create fresh environment
conda create -n ml-tutorial python=3.9
conda activate ml-tutorial
```

**Slow Training**

```python
# Use CPU for testing
device = "cpu"  # Instead of "cuda"
```

### Getting Help

- **Tutorial-specific issues**: Check individual README files
- **General questions**: Create GitHub issues
- **Quick fixes**: Review troubleshooting sections in each tutorial

## 🤝 Contributing

We welcome contributions to improve the tutorials:

1. **Bug fixes**: Correct errors or improve clarity
2. **New examples**: Add variations or extended use cases
3. **Documentation**: Enhance explanations or add diagrams
4. **Performance**: Optimize code or add new techniques

See individual tutorial READMEs for specific contribution guidelines.

## 📄 License

This tutorial is open source under the MIT license. Feel free to use, modify, and share.

## 🙏 Acknowledgments

- **PyTorch Lightning Team**: For the excellent framework
- **Optuna Developers**: For making hyperparameter optimization accessible
- **MLflow Community**: For comprehensive experiment tracking
- **Ultralytics**: For modern, easy-to-use YOLO implementations
- **Torchvision Team**: For pre-trained models and utilities

---

<div align="center">

**Ready to start learning? Begin with [01-basic-lightning](./01-basic-lightning/)! 🚀**

[⭐ Star this repo](https://github.com/your-repo) | [🐛 Report issues](https://github.com/your-repo/issues) | [💡 Suggest improvements](https://github.com/your-repo/discussions)

</div>
