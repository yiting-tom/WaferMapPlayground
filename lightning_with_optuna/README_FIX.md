# PyTorch Lightning + Optuna Compatibility Fix

## The Problem

The original Optuna example (`official_example.py`) fails with the error:

```
ValueError: Expected a parent
```

This occurs because the `PyTorchLightningPruningCallback` from Optuna's integration is incompatible with newer versions of PyTorch Lightning (2.0+). The callback validation system in PyTorch Lightning changed, causing this compatibility issue.

## The Solution

I've created two fixed versions that work with current PyTorch Lightning versions. Both versions include **comprehensive comments** explaining:

- How each part of the code works
- Why design decisions were made
- How to customize the hyperparameters
- Tips for better optimization results

### 1. `fixed_example.py` - Complete Solution with Pruning

This version implements a custom `OptunaPruningCallback` that's compatible with newer PyTorch Lightning versions:

```python
class OptunaPruningCallback(pl.Callback):
    """Custom Optuna pruning callback compatible with newer PyTorch Lightning versions."""

    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val_acc"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the val epoch ends."""
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        # Report the score to Optuna
        self.trial.report(current_score.item(), trainer.current_epoch)

        # Check if the trial should be pruned
        if self.trial.should_prune():
            raise optuna.TrialPruned()
```

**Features:**

- Full pruning support for faster optimization
- Detailed comments explaining every component
- Educational docstrings for learning PyTorch Lightning patterns
- Advanced hyperparameter suggestions

**Usage:**

```bash
python fixed_example.py --pruning
```

### 2. `simple_fixed_example.py` - Simplified Solution (Recommended for Beginners)

This version removes the pruning functionality entirely, focusing on basic hyperparameter optimization. It's more stable and easier to understand:

**Features:**

- No pruning callback complications
- Clean, simple optimization loop
- Better error handling and user feedback
- Cleaner output with progress indicators
- Step-by-step explanations in comments
- Beginner-friendly explanations of PyTorch Lightning concepts
- Practical tips for improving optimization results

**Usage:**

```bash
python simple_fixed_example.py
```

## Key Learning Features

Both examples now include extensive educational content:

### 📚 **Comprehensive Documentation**

- Every class and function has detailed docstrings
- Inline comments explain complex operations
- Architecture decisions are explained with reasoning

### 🎯 **Practical Examples**

- Shows how to structure PyTorch Lightning projects
- Demonstrates proper DataModule usage
- Explains hyperparameter selection strategies

### 💡 **Tips and Best Practices**

- How to adjust hyperparameter ranges
- When to use pruning vs. simple optimization
- Performance optimization suggestions
- How to apply results to your own projects

### 🔧 **Customization Guidance**

- Easy-to-find configuration constants
- Clear instructions for modifying architectures
- Examples of how to use the optimized hyperparameters

## Key Differences from Original

1. **Custom Callback**: Replaced `PyTorchLightningPruningCallback` with a custom implementation
2. **Proper Error Handling**: Added try-catch blocks for pruned trials
3. **Modern PyTorch Lightning**: Uses current API patterns and best practices
4. **Reduced Complexity**: Simplified version removes pruning entirely for better stability
5. **Educational Focus**: Extensive comments and documentation for learning
6. **Better User Experience**: Clear output, progress indicators, and helpful tips

## Version Compatibility

These fixes work with:

- PyTorch Lightning 2.0+
- Optuna 3.0+
- PyTorch 2.0+

## Which Version to Use?

### For Beginners: `simple_fixed_example.py`

- ✅ Easier to understand and debug
- ✅ More stable (no pruning-related errors)
- ✅ Better for learning Optuna and PyTorch Lightning basics
- ✅ Still provides effective hyperparameter optimization
- ✅ Comprehensive educational comments

### For Advanced Users: `fixed_example.py`

- ✅ Full pruning functionality for faster optimization
- ✅ More sophisticated hyperparameter search
- ✅ Better for production use cases
- ✅ Advanced PyTorch Lightning patterns
- ✅ Detailed technical documentation

Both versions will give you working hyperparameter optimization with PyTorch Lightning and Optuna, while teaching you best practices along the way!
