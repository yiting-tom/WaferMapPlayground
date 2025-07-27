"""
Fixed Optuna example that optimizes multi-layer perceptrons using PyTorch Lightning.

This version fixes the compatibility issues with newer PyTorch Lightning versions
by implementing a custom pruning callback instead of using the deprecated
PyTorchLightningPruningCallback.

WHAT THIS EXAMPLE DOES:
- Optimizes neural network hyperparameters using Optuna
- Uses PyTorch Lightning for clean model training
- Implements pruning to stop unpromising trials early
- Trains on FashionMNIST dataset for image classification

HYPERPARAMETERS OPTIMIZED:
- Number of hidden layers (1-3)
- Dropout rate (0.2-0.5)
- Number of units per layer (4-128, log scale)

You can run this example as follows:
    $ python fixed_example.py [--pruning]

    --pruning: Enable pruning to stop unpromising trials early (recommended)
"""

import argparse
import os
from typing import List, Optional

import optuna
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from packaging import version
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Ensure we have a compatible PyTorch Lightning version
if version.parse(pl.__version__) < version.parse("1.6.0"):
    raise RuntimeError("PyTorch Lightning>=1.6.0 is required for this example.")

# Configuration constants - adjust these based on your needs
PERCENT_VALID_EXAMPLES = 0.1  # Use 10% of validation data for faster training
BATCHSIZE = 128  # Batch size for training
CLASSES = 10  # Number of classes in FashionMNIST
EPOCHS = 10  # Number of training epochs per trial
DIR = os.getcwd()  # Directory for dataset download


class OptunaPruningCallback(pl.Callback):
    """
    Custom Optuna pruning callback compatible with newer PyTorch Lightning versions.

    This callback:
    1. Reports intermediate values to Optuna after each validation epoch
    2. Checks if the trial should be pruned based on the pruner's decision
    3. Raises TrialPruned exception to stop unpromising trials early

    Args:
        trial: Optuna trial object for reporting values and checking pruning
        monitor: Metric name to monitor (should match what's logged in validation_step)
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val_acc"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Called when the validation epoch ends.

        This is where we:
        1. Get the current metric value
        2. Report it to Optuna for pruning decisions
        3. Check if we should prune (stop) this trial
        """
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        # Report the score to Optuna for this epoch
        self.trial.report(current_score.item(), trainer.current_epoch)

        # Check if the trial should be pruned based on intermediate results
        if self.trial.should_prune():
            raise optuna.TrialPruned()


class Net(nn.Module):
    """
    Simple feed-forward neural network for image classification.

    Architecture:
    - Flatten 28x28 images to 784 features
    - Variable number of hidden layers with ReLU activation and dropout
    - Final linear layer for classification

    Args:
        dropout: Dropout probability for regularization
        output_dims: List of hidden layer sizes (e.g., [64, 32] for 2 layers)
    """

    def __init__(self, dropout: float, output_dims: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        # Start with flattened image size (28*28 = 784)
        input_dim: int = 28 * 28

        # Add hidden layers with ReLU activation and dropout
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        # Final classification layer (no activation, will use log_softmax in forward)
        layers.append(nn.Linear(input_dim, CLASSES))

        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass with log-softmax output for NLL loss."""
        logits = self.layers(data)
        return F.log_softmax(logits, dim=1)


class LightningNet(pl.LightningModule):
    """
    PyTorch Lightning wrapper for our neural network.

    This handles:
    - Training and validation steps
    - Metric logging
    - Optimizer configuration

    The Lightning module automatically handles:
    - Moving data to GPU
    - Gradient computation and backpropagation
    - Metric aggregation across batches
    """

    def __init__(self, dropout: float, output_dims: List[int]) -> None:
        super().__init__()
        self.model = Net(dropout, output_dims)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Flatten the input and pass through the network."""
        return self.model(data.view(-1, 28 * 28))

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step - called for each training batch.

        Returns:
            loss: Training loss (Lightning will automatically call backward())
        """
        data, target = batch
        output = self(data)
        return F.nll_loss(output, target)

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """
        Validation step - called for each validation batch.

        We log both accuracy and a hyperparameter metric for Optuna to monitor.
        The logged values are automatically aggregated across batches.
        """
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()

        # Log metrics - these will be available in trainer.callback_metrics
        self.log("val_acc", accuracy)  # This is what Optuna monitors
        self.log(
            "hp_metric", accuracy, on_step=False, on_epoch=True
        )  # For hyperparameter tracking

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure the optimizer - using Adam with default learning rate."""
        return optim.Adam(self.model.parameters())


class FashionMNISTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FashionMNIST dataset.

    This handles:
    - Dataset downloading and preprocessing
    - Train/validation/test splits
    - DataLoader creation with proper settings

    Benefits of using DataModule:
    - Clean separation of data logic
    - Reproducible data splits
    - Easy to swap datasets
    """

    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training and validation.
        Called automatically by Lightning trainer.
        """
        # Download and prepare test set
        self.mnist_test = datasets.FashionMNIST(
            self.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )

        # Download training set and split into train/validation
        mnist_full = datasets.FashionMNIST(
            self.data_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        # Split: 55k for training, 5k for validation
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self) -> DataLoader:
        """Training data loader with shuffling enabled."""
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Validation data loader - no shuffling needed."""
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Test data loader for final evaluation."""
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )


def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function for Optuna optimization.

    This function:
    1. Suggests hyperparameters for the current trial
    2. Creates and trains a model with those hyperparameters
    3. Returns the validation accuracy for Optuna to optimize

    Args:
        trial: Optuna trial object for suggesting hyperparameters

    Returns:
        float: Validation accuracy (higher is better)
    """

    # Suggest hyperparameters for this trial
    # Optuna will intelligently choose values to optimize the objective
    n_layers = trial.suggest_int("n_layers", 1, 3)  # Number of hidden layers
    dropout = trial.suggest_float(
        "dropout", 0.2, 0.5
    )  # Dropout rate for regularization

    # Suggest number of units for each layer (log scale for better exploration)
    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        for i in range(n_layers)
    ]

    # Create model and data module with suggested hyperparameters
    model = LightningNet(dropout, output_dims)
    datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)

    # Create the custom pruning callback for this trial
    pruning_callback = OptunaPruningCallback(trial, monitor="val_acc")

    # Configure trainer
    trainer = pl.Trainer(
        logger=True,  # Enable logging
        limit_val_batches=PERCENT_VALID_EXAMPLES,  # Use subset of validation data
        enable_checkpointing=False,  # No need to save checkpoints for optimization
        max_epochs=EPOCHS,
        accelerator="auto",  # Use GPU if available
        devices=1,  # Use single device
        callbacks=[pruning_callback],  # Add our custom pruning callback
    )

    # Log hyperparameters for this trial
    hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyperparameters)

    try:
        # Train the model
        trainer.fit(model, datamodule=datamodule)
        # Return the final validation accuracy for Optuna to optimize
        return trainer.callback_metrics["val_acc"].item()
    except optuna.TrialPruned:
        # Handle pruned trials gracefully
        raise optuna.TrialPruned()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning example with Optuna."
    )
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    # Choose pruner based on command line argument
    pruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    print("🚀 Starting Optuna hyperparameter optimization...")
    print(f"📊 Pruning: {'Enabled' if args.pruning else 'Disabled'}")
    print("=" * 60)

    # Create Optuna study
    study = optuna.create_study(direction="maximize", pruner=pruner)

    # Run optimization
    # Adjust n_trials and timeout based on your computational budget
    study.optimize(objective, n_trials=10, timeout=300)  # 10 trials, 5 min timeout

    # Print results
    print("\n" + "=" * 60)
    print("📊 Optimization completed!")
    print("=" * 60)
    print(f"Number of finished trials: {len(study.trials)}")

    print("\n🏆 Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.4f}")

    print("\n🔧 Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("\n💡 To use these hyperparameters in your own code:")
    print(
        f"    model = LightningNet(dropout={trial.params['dropout']}, output_dims={trial.params.get('output_dims', [])})"
    )
