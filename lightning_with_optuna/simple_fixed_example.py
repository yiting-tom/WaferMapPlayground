"""
Simplified Optuna example with PyTorch Lightning (without pruning).

This version demonstrates basic hyperparameter optimization without pruning,
making it easier to understand and more stable for beginners.

WHAT THIS EXAMPLE DOES:
- Optimizes neural network hyperparameters using Optuna
- Uses PyTorch Lightning for clean, organized model training
- Trains on FashionMNIST dataset for image classification
- No pruning complexity - each trial runs to completion

HYPERPARAMETERS OPTIMIZED:
- Number of hidden layers (1-3)
- Dropout rate (0.2-0.5)
- Learning rate (1e-5 to 1e-1, log scale)
- Number of units per layer (4-128, log scale)

WHY USE THIS VERSION:
- Simpler to understand and debug
- More stable (no pruning-related errors)
- Better for learning Optuna basics
- Still provides effective hyperparameter optimization

You can run this example as follows:
    $ python simple_fixed_example.py
"""

import os
from typing import List, Optional

import optuna
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Configuration constants - easily adjustable for your needs
PERCENT_VALID_EXAMPLES = 0.1  # Use 10% of validation data for faster optimization
BATCHSIZE = 128  # Batch size for training
CLASSES = 10  # Number of classes in FashionMNIST (T-shirt, Trouser, etc.)
EPOCHS = 5  # Number of training epochs per trial (reduced for faster testing)
DIR = os.getcwd()  # Directory where FashionMNIST dataset will be downloaded


class Net(nn.Module):
    """
    Simple feed-forward neural network for image classification.

    This is a flexible architecture where you can specify:
    - Number of hidden layers
    - Size of each hidden layer
    - Dropout rate for regularization

    Architecture flow:
    Input (28x28 image) → Flatten → Hidden Layers → Output (10 classes)

    Args:
        dropout: Probability of dropping neurons during training (prevents overfitting)
        output_dims: List specifying the size of each hidden layer
                    Example: [64, 32] creates 2 hidden layers with 64 and 32 neurons
    """

    def __init__(self, dropout: float, output_dims: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []

        # Start with flattened image: 28x28 = 784 input features
        input_dim: int = 28 * 28

        # Build hidden layers dynamically based on output_dims
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))  # Linear transformation
            layers.append(nn.ReLU())  # Non-linear activation function
            layers.append(nn.Dropout(dropout))  # Regularization to prevent overfitting
            input_dim = output_dim  # Update input size for next layer

        # Final classification layer - outputs logits for 10 classes
        layers.append(nn.Linear(input_dim, CLASSES))
        self.layers = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            data: Input tensor of shape (batch_size, 28, 28)

        Returns:
            Log probabilities for each class (for use with NLL loss)
        """
        logits = self.layers(data)
        return F.log_softmax(logits, dim=1)


class LightningNet(pl.LightningModule):
    """
    PyTorch Lightning wrapper that handles the training lifecycle.

    Lightning automatically handles:
    - Moving tensors to GPU/CPU
    - Gradient computation and backpropagation
    - Metric aggregation across batches
    - Distributed training (if needed)
    - Logging and checkpointing

    You just need to define:
    - training_step: What happens in each training batch
    - validation_step: What happens in each validation batch
    - configure_optimizers: Which optimizer to use
    """

    def __init__(self, dropout: float, output_dims: List[int]) -> None:
        super().__init__()
        self.model = Net(dropout, output_dims)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - flatten input and pass through network.

        Note: We flatten here rather than in the dataset because
        PyTorch Lightning handles batching automatically.
        """
        return self.model(data.view(-1, 28 * 28))

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Called for each training batch.

        Lightning will automatically:
        1. Call this function
        2. Compute gradients from the returned loss
        3. Update model parameters
        4. Zero gradients for next batch

        Args:
            batch: List containing [images, labels]
            batch_idx: Index of current batch (not used here)

        Returns:
            Loss tensor that Lightning will use for backpropagation
        """
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)  # Negative log-likelihood loss

        # Log training loss for monitoring (optional)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> None:
        """
        Called for each validation batch.

        We compute and log metrics here. Lightning automatically:
        - Aggregates metrics across all validation batches
        - Makes them available in trainer.callback_metrics
        - Handles device placement

        Args:
            batch: List containing [images, labels]
            batch_idx: Index of current batch (not used here)
        """
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)  # Get predicted class
        accuracy = (
            pred.eq(target.view_as(pred)).float().mean()
        )  # Compare with true labels

        # Log metrics - these will be available for Optuna to read
        self.log("val_loss", loss, prog_bar=True)  # Show in progress bar
        self.log("val_acc", accuracy, prog_bar=True)  # This is what Optuna optimizes

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Define which optimizer to use.

        Note: The learning rate will be overridden in the objective function
        to allow Optuna to optimize it.
        """
        return optim.Adam(self.model.parameters())


class FashionMNISTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FashionMNIST dataset.

    DataModules provide a clean way to handle:
    - Dataset downloading and preprocessing
    - Train/validation/test splits
    - DataLoader configuration

    Benefits:
    - Reusable across different experiments
    - Ensures consistent data splits
    - Handles data on multiple devices automatically
    - Makes code more organized and testable
    """

    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Download and prepare datasets.

        This is called automatically by the Lightning trainer before training starts.
        The 'stage' parameter can be used to set up different data for different phases
        (fit, validate, test, predict), but we'll prepare everything at once.
        """
        # Download test set (10,000 samples)
        self.mnist_test = datasets.FashionMNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),  # Convert PIL images to tensors [0,1]
        )

        # Download full training set (60,000 samples)
        mnist_full = datasets.FashionMNIST(
            self.data_dir, train=True, download=True, transform=transforms.ToTensor()
        )

        # Split training set into train (55k) and validation (5k)
        # This ensures we have held-out data for hyperparameter validation
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self) -> DataLoader:
        """
        Create training data loader.

        Key settings:
        - shuffle=True: Randomize order each epoch (helps training)
        - num_workers=2: Use 2 processes for data loading (faster)
        """
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation data loader.

        Key settings:
        - shuffle=False: No need to randomize validation data
        - num_workers=2: Parallel data loading for speed
        """
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=2
        )


def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function that Optuna will optimize.

    This function:
    1. Asks Optuna to suggest hyperparameter values for this trial
    2. Creates and trains a model with those hyperparameters
    3. Returns a score that Optuna tries to maximize (validation accuracy)

    Optuna uses the results from multiple trials to intelligently suggest
    better hyperparameters for future trials.

    Args:
        trial: Optuna trial object that provides hyperparameter suggestions

    Returns:
        float: Validation accuracy (0.0 to 1.0) - higher is better
    """

    # Ask Optuna to suggest hyperparameters for this trial
    # Optuna will use previous trial results to make intelligent suggestions

    n_layers = trial.suggest_int("n_layers", 1, 3)
    # Number of hidden layers: 1, 2, or 3

    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    # Dropout rate: between 20% and 50% of neurons dropped during training

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    # Learning rate: from 0.00001 to 0.1, using log scale for better exploration

    # For each hidden layer, suggest the number of neurons
    output_dims = [
        trial.suggest_int(f"n_units_l{i}", 4, 128, log=True) for i in range(n_layers)
    ]
    # Example: if n_layers=2, this creates a list like [64, 32]
    # log=True means Optuna explores smaller and larger values more evenly

    # Create model and data module with the suggested hyperparameters
    model = LightningNet(dropout, output_dims)
    datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)

    # Override the default learning rate with Optuna's suggestion
    # We do this by replacing the configure_optimizers method
    model.configure_optimizers = lambda: optim.Adam(
        model.model.parameters(), lr=learning_rate
    )

    # Create Lightning trainer with settings optimized for hyperparameter search
    trainer = pl.Trainer(
        limit_val_batches=PERCENT_VALID_EXAMPLES,  # Use only 10% of validation data (faster)
        enable_checkpointing=False,  # Don't save model checkpoints (not needed for optimization)
        max_epochs=EPOCHS,  # Train for specified number of epochs
        accelerator="auto",  # Use GPU if available, otherwise CPU
        devices=1,  # Use single device (prevents device conflicts)
        logger=False,  # Disable logging for cleaner output during optimization
        enable_progress_bar=False,  # Disable progress bar for cleaner output
    )

    # Train the model with the current hyperparameters
    trainer.fit(model, datamodule=datamodule)

    # Return the final validation accuracy for Optuna to optimize
    # Higher accuracy = better hyperparameters
    return trainer.callback_metrics["val_acc"].item()


def main():
    """
    Main function that runs the hyperparameter optimization.

    This function:
    1. Creates an Optuna study (optimization session)
    2. Runs multiple trials with different hyperparameters
    3. Reports the best results found
    """
    print("🚀 Starting Optuna hyperparameter optimization...")
    print("📚 This is the SIMPLIFIED version (no pruning)")
    print("💡 Each trial will train a complete model and report final accuracy")
    print("=" * 60)

    # Create Optuna study
    # direction="maximize" means we want to maximize validation accuracy
    study = optuna.create_study(direction="maximize")

    # Run the optimization
    # n_trials: Number of different hyperparameter combinations to try
    # timeout: Maximum time to spend optimizing (in seconds)
    print("⏳ Running optimization trials...")
    study.optimize(objective, n_trials=5, timeout=300)  # 5 trials, max 5 minutes

    # Print detailed results
    print("\n" + "=" * 60)
    print("📊 Optimization Results")
    print("=" * 60)
    print(f"Number of completed trials: {len(study.trials)}")
    print(
        f"Best validation accuracy: {study.best_value:.4f} ({study.best_value * 100:.2f}%)"
    )

    print("\n🏆 Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Show performance of all trials for comparison
    print("\n📈 All trial results:")
    for i, trial in enumerate(study.trials):
        status = "✅" if trial.value is not None else "❌"
        if trial.value is not None:
            accuracy_percent = trial.value * 100
            print(
                f"  Trial {i + 1}: {status} {trial.value:.4f} ({accuracy_percent:.2f}%)"
            )
        else:
            print(f"  Trial {i + 1}: {status} Failed")

    print("\n💡 To use the best hyperparameters in your own code:")
    print("model = LightningNet(")
    print(f"    dropout={study.best_params['dropout']:.3f},")

    # Reconstruct the output_dims list from the best parameters
    n_layers = study.best_params["n_layers"]
    output_dims = [study.best_params[f"n_units_l{i}"] for i in range(n_layers)]
    print(f"    output_dims={output_dims}")
    print(")")
    print(
        f"optimizer = torch.optim.Adam(model.parameters(), lr={study.best_params['learning_rate']:.6f})"
    )

    print("\n🎯 Tips for better optimization:")
    print(f"  - Increase n_trials (currently {5}) for more thorough search")
    print(f"  - Increase timeout (currently {300}s) if trials are slow")
    print("  - Try different hyperparameter ranges in the objective function")
    print(
        "  - Use the full version (fixed_example.py) with pruning for faster optimization"
    )


if __name__ == "__main__":
    main()
