# Tutorial 01: PyTorch Lightning Fundamentals

> **Learn the basics of PyTorch Lightning with wafer defect classification**

This tutorial introduces PyTorch Lightning through a practical wafer defect classification example. You'll learn how Lightning simplifies training loops, handles device placement, and provides professional ML engineering patterns.

## 🎯 Learning Objectives

By completing this tutorial, you will understand:

- **LightningModule structure**: How to organize your PyTorch code with Lightning
- **Automatic training loops**: Let Lightning handle training/validation/testing
- **Built-in logging**: Automatic metrics tracking and TensorBoard integration
- **Callbacks system**: Early stopping, checkpointing, and monitoring
- **Device handling**: Automatic GPU/CPU placement and distributed training
- **Clean code organization**: Professional ML code structure

## 📋 Prerequisites

- Basic PyTorch knowledge (tensors, nn.Module, training loops)
- Python 3.8+
- Familiarity with CNNs for image classification

## 🚀 Quick Start

```bash
# Navigate to tutorial directory
cd 01-basic-lightning

# Install requirements
pip install torch torchvision pytorch-lightning tensorboard matplotlib seaborn scikit-learn

# Run the tutorial
python train.py
```

**Expected runtime**: 5-10 minutes on CPU, 2-3 minutes on GPU

## 📖 Tutorial Structure

### Files Overview

```
01-basic-lightning/
├── simple_model.py    # Lightning model definition
├── train.py          # Main training script
├── README.md         # This file
└── requirements.txt  # Dependencies
```

### Core Components

#### 1. `simple_model.py` - Lightning Model

```python
class WaferLightningModel(pl.LightningModule):
    def __init__(self, num_classes=9, learning_rate=1e-3):
        super().__init__()
        self.model = WaferCNN(num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        # Lightning calls this for each training batch
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # Lightning uses this to set up optimization
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
```

#### 2. `train.py` - Training Script

```python
# Create model, data, and trainer
model = WaferLightningModel()
trainer = pl.Trainer(max_epochs=50, callbacks=[EarlyStopping()])

# Train (Lightning handles the loops!)
trainer.fit(model, train_loader, val_loader)

# Test
trainer.test(model, test_loader)
```

## 🔧 Key Lightning Concepts

### 1. LightningModule Structure

Lightning modules organize your code into logical sections:

```python
class MyLightningModel(pl.LightningModule):
    def __init__(self):
        # Model architecture

    def forward(self, x):
        # Inference logic

    def training_step(self, batch, batch_idx):
        # What happens in one training step

    def validation_step(self, batch, batch_idx):
        # What happens in one validation step

    def configure_optimizers(self):
        # How to optimize the model
```

**Benefits**:

- Clear separation of concerns
- Reusable across different trainers
- Easy to test individual components

### 2. Automatic Training Loops

**Without Lightning** (traditional PyTorch):

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

    # Validation loop
    with torch.no_grad():
        for batch in val_loader:
            # validation logic...
```

**With Lightning**:

```python
trainer = pl.Trainer(max_epochs=num_epochs)
trainer.fit(model, train_loader, val_loader)
```

**Benefits**:

- No boilerplate code
- Automatic gradient handling
- Built-in validation
- Error handling

### 3. Automatic Logging

Lightning automatically logs your metrics:

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)

    # This automatically goes to TensorBoard/logger
    self.log('train_loss', loss, prog_bar=True)

    return loss
```

**Benefits**:

- TensorBoard integration
- Progress bar updates
- Metric aggregation

### 4. Callbacks System

Add functionality without modifying your model:

```python
trainer = pl.Trainer(
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(monitor='val_loss', save_top_k=3),
        LearningRateMonitor()
    ]
)
```

**Benefits**:

- Modular functionality
- Reusable across projects
- Professional ML patterns

## 📊 Expected Results

After running the tutorial, you should see:

### Console Output

```
🚀 Starting PyTorch Lightning Tutorial - Basic Training
===============================================================
📁 Loading wafer dataset...
Dataset loaded: 900 samples, 9 classes

🧠 Initializing Lightning model...
Model created:
  total_parameters: 2,435,081
  trainable_parameters: 2,435,081
  architecture: WaferCNN

🎯 Starting training for up to 30 epochs...
===============================================================

Epoch 1/30: 100%|████████| 20/20 [00:05<00:00,  3.8it/s, loss=2.21, v_num=0]
Epoch 2/30: 100%|████████| 20/20 [00:04<00:00,  4.2it/s, loss=1.89, v_num=0]
...

✅ Training completed!
Final Test Results:
Accuracy: 0.7833
Precision (macro): 0.7642
Recall (macro): 0.7615
F1-Score (macro): 0.7591
```

### Generated Files

```
├── lightning_logs/           # TensorBoard logs
├── checkpoints/             # Model checkpoints
├── results/                 # Visualizations
│   ├── confusion_matrix.png
│   └── sample_predictions.png
```

### TensorBoard Visualization

View training progress:

```bash
tensorboard --logdir lightning_logs
```

## 🎛️ Configuration Options

### Model Architecture

```python
model = WaferLightningModel(
    num_classes=9,           # Number of defect classes
    base_channels=32,        # CNN base channels
    dropout_rate=0.3,        # Dropout probability
    learning_rate=1e-3       # Learning rate
)
```

### Training Settings

```python
trainer = pl.Trainer(
    max_epochs=50,           # Maximum epochs
    accelerator='auto',      # GPU/CPU automatically
    callbacks=[...],         # Add callbacks
    logger=TensorBoardLogger(...)  # Configure logging
)
```

### Data Options

```python
dataset = get_tutorial_dataset(
    max_samples_per_class=100,  # Limit dataset size
    balance_classes=True        # Balance class distribution
)
```

## 🔍 Understanding the Output

### Training Progress

- **train_loss**: Decreasing indicates learning
- **val_loss**: Should track train_loss (if not, possible overfitting)
- **train_acc/val_acc**: Classification accuracy

### Final Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Visualizations

- **Confusion Matrix**: Shows per-class performance
- **Sample Predictions**: Visual verification of model performance

## 🚨 Common Issues

### CUDA Out of Memory

```python
# Reduce batch size
train_loader = DataLoader(..., batch_size=16)  # Instead of 32

# Or reduce model size
model = WaferLightningModel(base_channels=16)  # Instead of 32
```

### Slow Training

```python
# Use GPU if available
trainer = pl.Trainer(accelerator='gpu')

# Reduce dataset size
dataset = get_tutorial_dataset(max_samples_per_class=50)
```

### Poor Performance

```python
# Try different learning rates
model = WaferLightningModel(learning_rate=1e-2)  # Higher LR

# Use class weights for imbalanced data
model = WaferLightningModel(class_weights=dataset.get_class_weights())
```

## 🎓 Key Takeaways

### What You Learned

1. **Lightning Structure**: Clean, organized ML code
2. **Automatic Features**: Training loops, logging, device handling
3. **Professional Patterns**: Callbacks, checkpointing, monitoring
4. **Rapid Experimentation**: Easy to modify and test

### Lightning vs Pure PyTorch

| Aspect              | Pure PyTorch         | Lightning |
| ------------------- | -------------------- | --------- |
| **Code Lines**      | ~200+                | ~50       |
| **Training Loop**   | Manual               | Automatic |
| **Device Handling** | Manual `.to(device)` | Automatic |
| **Logging**         | Manual setup         | Built-in  |
| **Validation**      | Manual loop          | Automatic |
| **Checkpointing**   | Manual save/load     | Built-in  |

### When to Use Lightning

✅ **Good for**:

- Rapid prototyping
- Standard training patterns
- Multi-GPU/distributed training
- Experiment tracking
- Production ML pipelines

❌ **Consider alternatives for**:

- Very custom training loops
- Non-standard architectures
- Research requiring fine control

## 🔄 Next Steps

You're now ready for more advanced integrations:

1. **[Tutorial 02](../02-lightning-with-torchvision/)**: Add pre-trained models with Torchvision
2. **[Tutorial 03](../03-lightning-with-optuna/)**: Hyperparameter optimization with Optuna
3. **[Tutorial 04](../04-lightning-with-mlflow/)**: Experiment tracking with MLflow

## 📚 Additional Resources

### Lightning Documentation

- [Official PyTorch Lightning Docs](https://pytorch-lightning.readthedocs.io/)
- [Lightning Examples](https://github.com/PyTorchLightning/pytorch-lightning/tree/master/pl_examples)
- [Best Practices Guide](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)

### Related Tutorials

- [Lightning + Weights & Biases](https://wandb.ai/site/articles/pytorch-lightning)
- [Lightning + Hydra Configuration](https://hydra.cc/docs/tutorials/structured_config/intro/)
- [Multi-GPU Training](https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu.html)

---

<div align="center">

**Congratulations! 🎉 You've mastered PyTorch Lightning fundamentals!**

Ready for the next challenge? Try [Tutorial 02: Lightning + Torchvision](../02-lightning-with-torchvision/) →

</div>
