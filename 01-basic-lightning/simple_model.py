"""
Simple PyTorch Lightning Model for Wafer Defect Classification

This tutorial demonstrates the basics of PyTorch Lightning:
- LightningModule structure
- Automatic training/validation loops
- Built-in logging and metrics
- Clean model organization
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from metrics import TorchMetrics


class WaferCNN(nn.Module):
    """
    Simple CNN architecture for wafer defect classification.

    This is a clean, straightforward CNN that demonstrates good practices
    without being overly complex.
    """

    def __init__(
        self,
        num_classes: int = 9,
        input_channels: int = 1,
        base_channels: int = 32,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize CNN architecture.

        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (1 for grayscale)
            base_channels: Base number of channels (doubles each layer)
            dropout_rate: Dropout probability
        """
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(input_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            # Second block
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            # Third block
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            # Fourth block
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(base_channels * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.features(x)
        output = self.classifier(features)
        return output


class WaferLightningModel(pl.LightningModule):
    """
    PyTorch Lightning wrapper for wafer defect classification.

    This class demonstrates the key Lightning patterns:
    - Clean separation of model definition and training logic
    - Automatic handling of device placement
    - Built-in logging and metrics
    - Organized step functions
    """

    def __init__(
        self,
        num_classes: int = 9,
        class_names: Optional[List[str]] = None,
        learning_rate: float = 1e-3,
        optimizer_name: str = "Adam",
        base_channels: int = 32,
        dropout_rate: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Lightning model.

        Args:
            num_classes: Number of output classes
            class_names: List of class names for logging
            learning_rate: Learning rate for optimizer
            optimizer_name: Name of optimizer ('Adam', 'SGD', 'AdamW')
            base_channels: Base channels for CNN
            dropout_rate: Dropout rate
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()

        # Save hyperparameters (enables easy checkpointing)
        self.save_hyperparameters(ignore=["class_weights"])

        # Store configuration
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        # Initialize model
        self.model = WaferCNN(
            num_classes=num_classes,
            base_channels=base_channels,
            dropout_rate=dropout_rate,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics tracker
        self.metrics = TorchMetrics()

        # Store outputs for epoch-end processing
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - Lightning automatically handles device placement."""
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        """
        Shared step logic for train/val/test.

        This demonstrates good Lightning practice: shared logic
        between different training stages.
        """
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate metrics
        preds = logits.argmax(dim=1)
        acc = self.metrics.accuracy(logits, y)

        # Log everything (Lightning handles this automatically)
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)

        return {
            "loss": loss,
            "preds": preds.detach(),
            "targets": y.detach(),
            "logits": logits.detach(),
        }

    def training_step(self, batch, batch_idx):
        """Training step - Lightning calls this automatically."""
        outputs = self._shared_step(batch, "train")
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        """Validation step - Lightning calls this automatically."""
        outputs = self._shared_step(batch, "val")
        self.validation_outputs.append(outputs)
        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """Test step - Lightning calls this automatically."""
        outputs = self._shared_step(batch, "test")
        self.test_outputs.append(outputs)
        return outputs["loss"]

    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch.

        This is where we can compute more complex metrics
        that require all predictions.
        """
        if not self.validation_outputs:
            return

        # Gather all predictions and targets
        all_preds = torch.cat([x["preds"] for x in self.validation_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_outputs])

        # Calculate additional metrics
        precision, recall, f1 = self.metrics.precision_recall_f1(
            all_preds, all_targets, self.num_classes, average="macro"
        )

        # Log additional metrics
        self.log("val_precision", precision, sync_dist=True)
        self.log("val_recall", recall, sync_dist=True)
        self.log("val_f1", f1, sync_dist=True)

        # Clear outputs
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        if not self.test_outputs:
            return

        # Gather all predictions and targets
        all_preds = torch.cat([x["preds"] for x in self.test_outputs])
        all_targets = torch.cat([x["targets"] for x in self.test_outputs])

        # Calculate final test metrics
        precision, recall, f1 = self.metrics.precision_recall_f1(
            all_preds, all_targets, self.num_classes, average="macro"
        )

        # Log final metrics
        self.log("test_precision_final", precision, sync_dist=True)
        self.log("test_recall_final", recall, sync_dist=True)
        self.log("test_f1_final", f1, sync_dist=True)

        # Print detailed results
        print("\nFinal Test Results:")
        print(f"Accuracy: {self.metrics.accuracy(all_preds, all_targets):.4f}")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"F1-Score (macro): {f1:.4f}")

        # Clear outputs
        self.test_outputs.clear()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Lightning automatically handles the optimization loop,
        we just need to return the optimizer configuration.
        """
        # Choose optimizer
        if self.hparams.optimizer_name == "SGD":
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )
        elif self.hparams.optimizer_name == "AdamW":
            optimizer = AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-2
            )
        else:  # Default to Adam
            optimizer = Adam(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
            )

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to monitor
                "frequency": 1,  # How often to check
                "strict": True,  # Crash if metric not found
            },
        }

    def get_model_info(self) -> Dict:
        """Get model information for logging/debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "architecture": "WaferCNN",
            "num_classes": self.num_classes,
            "learning_rate": self.hparams.learning_rate,
            "optimizer": self.hparams.optimizer_name,
        }


if __name__ == "__main__":
    # Quick test of the model
    print("Testing WaferLightningModel...")

    # Create sample data
    batch_size = 8
    sample_input = torch.randn(batch_size, 1, 64, 64)
    sample_targets = torch.randint(0, 9, (batch_size,))

    # Initialize model
    model = WaferLightningModel(num_classes=9, learning_rate=1e-3)

    # Test forward pass
    with torch.no_grad():
        outputs = model(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Model info: {model.get_model_info()}")

    # Test training step
    batch = (sample_input, sample_targets)
    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss:.4f}")

    print("✅ Model test completed successfully!")
