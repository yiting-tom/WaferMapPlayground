"""
PyTorch Lightning Training Script - Tutorial 01

This script demonstrates the basics of training with PyTorch Lightning:
- Automatic training loops
- Built-in validation and testing
- Easy logging and checkpointing
- Clean, organized code structure

Run with: python train.py
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from data_utils import create_data_loaders, get_tutorial_dataset
from metrics import MetricsCalculator, ModelEvaluator

# Import our Lightning model
from simple_model import WaferLightningModel
from visualizations import create_results_dashboard


def create_lightning_trainer(
    max_epochs: int = 50,
    enable_early_stopping: bool = True,
    enable_checkpointing: bool = True,
    log_dir: str = "lightning_logs",
) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer with standard callbacks.

    Args:
        max_epochs: Maximum number of training epochs
        enable_early_stopping: Whether to use early stopping
        enable_checkpointing: Whether to save model checkpoints
        log_dir: Directory for logs

    Returns:
        Configured Lightning trainer
    """
    callbacks = []

    # Early stopping callback
    if enable_early_stopping:
        early_stop = EarlyStopping(
            monitor="val_loss", patience=15, mode="min", verbose=True, min_delta=0.001
        )
        callbacks.append(early_stop)

    # Model checkpointing
    if enable_checkpointing:
        checkpoint = ModelCheckpoint(
            dirpath="checkpoints/01-basic-lightning",
            filename="wafer-cnn-{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Logger
    logger = TensorBoardLogger(log_dir, name="basic_lightning_tutorial")

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",  # Automatically use GPU if available
        devices="auto",  # Use all available devices
        log_every_n_steps=10,  # Log every 10 batches
        check_val_every_n_epoch=1,  # Validate every epoch
        enable_progress_bar=True,  # Show progress bar
        enable_model_summary=True,  # Show model summary
        deterministic=True,  # For reproducibility
    )

    return trainer


def train_basic_lightning_model(
    batch_size: int = 32,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    optimizer_name: str = "Adam",
    use_class_weights: bool = True,
    create_visualizations: bool = True,
) -> Dict[str, Any]:
    """
    Train a basic PyTorch Lightning model for wafer defect classification.

    Args:
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Learning rate for optimizer
        optimizer_name: Name of optimizer to use
        use_class_weights: Whether to use class weights for imbalanced data
        create_visualizations: Whether to create result visualizations

    Returns:
        Dictionary containing training results and metrics
    """
    print("🚀 Starting PyTorch Lightning Tutorial - Basic Training")
    print("=" * 60)

    # Set seed for reproducibility
    pl.seed_everything(42)

    # 1. Load and prepare data
    print("📁 Loading wafer dataset...")
    dataset = get_tutorial_dataset(
        max_samples_per_class=100,  # Limit for quick tutorial
        balance_classes=True,
    )

    print(f"Dataset loaded: {len(dataset)} samples, {dataset.num_classes} classes")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=batch_size,
        use_weighted_sampling=not use_class_weights,  # Use one or the other
    )

    print("Data loaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # 2. Initialize model
    print("\n🧠 Initializing Lightning model...")

    # Get class weights if requested
    class_weights = dataset.get_class_weights() if use_class_weights else None
    if class_weights is not None:
        print("Using class weights to handle imbalanced data")

    model = WaferLightningModel(
        num_classes=dataset.num_classes,
        class_names=dataset.task_class_names,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        class_weights=class_weights,
    )

    # Print model info
    model_info = model.get_model_info()
    print("Model created:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # 3. Create trainer
    print("\n⚡ Setting up Lightning trainer...")
    trainer = create_lightning_trainer(
        max_epochs=max_epochs, enable_early_stopping=True, enable_checkpointing=True
    )

    # 4. Train model
    print(f"\n🎯 Starting training for up to {max_epochs} epochs...")
    print("=" * 60)

    trainer.fit(model, train_loader, val_loader)

    print("✅ Training completed!")

    # 5. Test model
    print("\n📊 Evaluating on test set...")
    test_results = trainer.test(model, test_loader)

    # 6. Detailed evaluation
    print("\n📈 Computing detailed metrics...")
    evaluator = ModelEvaluator(model)
    detailed_metrics = evaluator.evaluate_on_loader(
        test_loader,
        dataset.task_class_names,
        task_type=dataset.task_type,
        return_predictions=True,
    )

    # Print detailed results
    calculator = MetricsCalculator(dataset.task_class_names, dataset.task_type)
    calculator.print_metrics_summary(detailed_metrics)

    # 7. Create visualizations
    if create_visualizations:
        print("\n🎨 Creating visualizations...")

        # Get sample data for visualization
        sample_batch = next(iter(test_loader))
        with torch.no_grad():
            sample_predictions = model(sample_batch[0]).argmax(dim=1)

        # Create dashboard
        create_results_dashboard(
            metrics=detailed_metrics,
            confusion_matrix=detailed_metrics["confusion_matrix"],
            class_names=dataset.task_class_names,
            sample_predictions=(
                sample_batch[0]
                .squeeze(1)
                .numpy(),  # Remove channel dim for visualization
                sample_batch[1].numpy(),
                sample_predictions.numpy(),
            ),
            save_dir="results/01-basic-lightning",
        )

    # 8. Save final results
    results = {
        "model_info": model_info,
        "training_metrics": test_results,
        "detailed_metrics": detailed_metrics,
        "hyperparameters": {
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "use_class_weights": use_class_weights,
        },
        "dataset_info": {
            "num_samples": len(dataset),
            "num_classes": dataset.num_classes,
            "class_names": dataset.task_class_names,
        },
    }

    print("\n✨ Tutorial completed! Check 'lightning_logs' for TensorBoard logs")
    print("📁 Model checkpoints saved in 'checkpoints/01-basic-lightning'")
    print("🎨 Visualizations saved in 'results/01-basic-lightning'")

    return results


def main():
    """Main function to run the tutorial."""
    print("PyTorch Lightning Tutorial 01: Basic Training")
    print("=" * 50)
    print("This tutorial demonstrates:")
    print("✓ PyTorch Lightning model structure")
    print("✓ Automatic training and validation loops")
    print("✓ Built-in logging and metrics")
    print("✓ Easy callbacks and checkpointing")
    print("✓ Clean code organization")
    print("=" * 50)

    # Run training with one configuration for quick demo
    configurations = [
        {
            "name": "Adam Optimizer",
            "optimizer_name": "Adam",
            "learning_rate": 1e-3,
            "batch_size": 32,
        },
    ]

    all_results = {}

    for config in configurations:
        print(f"\n🔧 Running configuration: {config['name']}")
        print("-" * 40)

        results = train_basic_lightning_model(
            batch_size=config["batch_size"],
            max_epochs=2,  # Very short for quick demo
            learning_rate=config["learning_rate"],
            optimizer_name=config["optimizer_name"],
            use_class_weights=True,
            create_visualizations=(config == configurations[0]),  # Only first config
        )

        all_results[config["name"]] = results

    # Compare results
    print("\n📊 COMPARISON RESULTS")
    print("=" * 50)

    for name, results in all_results.items():
        metrics = results["detailed_metrics"]
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {metrics.get('f1_weighted', 0):.4f}")
        print(f"  Parameters: {results['model_info']['total_parameters']:,}")

    print("\n🎉 All experiments completed!")
    print("Key Lightning benefits demonstrated:")
    print("✓ Automatic device handling (GPU/CPU)")
    print("✓ Built-in training loop with validation")
    print("✓ Easy logging and visualization")
    print("✓ Callbacks for early stopping and checkpointing")
    print("✓ Clean, organized code structure")


if __name__ == "__main__":
    main()
