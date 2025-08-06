from typing import Literal

import torch
from lightning import LightningModule, seed_everything
from torch import nn
from torchmetrics import Accuracy, F1Score, Precision, Recall

seed_everything(42)


class Classifier(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_labels: int,
        task_type: Literal["binary", "multiclass", "multilabel"] = "multilabel",
        lr: float = 1e-5,
        fully_finetune: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.task_type = task_type
        self.num_labels = num_labels

        # Select appropriate loss function based on task type
        if task_type == "binary":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif task_type == "multiclass":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif task_type == "multilabel":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        if fully_finetune:
            for param in self.model.parameters():
                param.requires_grad = True

        # Initialize metrics based on task type
        self.metrics = self._initialize_metrics()

    def _initialize_metrics(self) -> nn.ModuleList:
        """Initialize appropriate metrics based on task type."""
        if self.task_type == "binary":
            return nn.ModuleList(
                [
                    Accuracy(task="binary"),
                    F1Score(task="binary"),
                    Precision(task="binary"),
                    Recall(task="binary"),
                ]
            )
        elif self.task_type == "multiclass":
            return nn.ModuleList(
                [
                    Accuracy(task="multiclass", num_classes=self.num_labels),
                    F1Score(task="multiclass", num_classes=self.num_labels),
                    Precision(task="multiclass", num_classes=self.num_labels),
                    Recall(task="multiclass", num_classes=self.num_labels),
                ]
            )
        elif self.task_type == "multilabel":
            return nn.ModuleList(
                [
                    Accuracy(task="multilabel", num_labels=self.num_labels),
                    F1Score(task="multilabel", num_labels=self.num_labels),
                    Precision(task="multilabel", num_labels=self.num_labels),
                    Recall(task="multilabel", num_labels=self.num_labels),
                ]
            )
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _process_outputs_and_labels(
        self, outputs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process outputs and labels based on task type."""
        if self.task_type == "binary":
            # For binary, ensure outputs and labels are 1D
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            # Convert labels to float for BCE loss
            labels = labels.float()
        elif self.task_type == "multiclass":
            # For multiclass, ensure labels are long for CrossEntropy
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            labels = labels.long()
        elif self.task_type == "multilabel":
            # For multilabel, ensure labels are float
            labels = labels.float()

        return outputs, labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        outputs, labels = self._process_outputs_and_labels(outputs, labels)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        outputs, labels = self._process_outputs_and_labels(outputs, labels)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        for metric in self.metrics:
            metric(outputs, labels)
        return loss

    def on_validation_epoch_end(self):
        for metric in self.metrics:
            self.log(
                f"val_{metric.__class__.__name__.lower()}",
                metric.compute(),
                on_step=False,
                on_epoch=True,
            )
        for metric in self.metrics:
            metric.reset()

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        outputs, labels = self._process_outputs_and_labels(outputs, labels)
        loss = self.loss_fn(outputs, labels)
        self.log("test_loss", loss)
        for metric in self.metrics:
            metric(outputs, labels)
        return loss

    def on_test_epoch_end(self):
        for metric in self.metrics:
            self.log(f"test_{metric.__class__.__name__.lower()}", metric.compute())
        for metric in self.metrics:
            metric.reset()

    def predict_step(
        self, batch: torch.Tensor | tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Prediction step that handles different input formats and task types."""
        # Handle both single tensor and tuple inputs
        if isinstance(batch, tuple):
            images, labels = batch
            has_labels = True
        else:
            images = batch
            labels = None
            has_labels = False

        # Get raw model outputs
        outputs = self(images)

        # Process outputs based on task type
        if self.task_type == "binary":
            # Apply sigmoid for binary classification
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)
                probabilities = probabilities.squeeze(-1)
                predictions = predictions.squeeze(-1)

        elif self.task_type == "multiclass":
            # Apply softmax for multiclass classification
            probabilities = torch.softmax(outputs, dim=-1)
            predictions = torch.argmax(outputs, dim=-1)

        elif self.task_type == "multilabel":
            # Apply sigmoid for multilabel classification
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        # Prepare return dictionary
        result = {
            "predictions": predictions,
            "probabilities": probabilities,
            "logits": outputs,
        }

        # Add labels if available
        if has_labels:
            processed_outputs, processed_labels = self._process_outputs_and_labels(
                outputs, labels
            )
            result["labels"] = processed_labels

            # Calculate loss if labels are provided
            loss = self.loss_fn(processed_outputs, processed_labels)
            result["loss"] = loss

        return result

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # Add warmup scheduler before CosineAnnealingWarmRestarts
        from torch.optim.lr_scheduler import (
            CosineAnnealingWarmRestarts,
            LinearLR,
            SequentialLR,
        )

        warmup_epochs = 5
        cosine_t0 = 10
        cosine_tmult = 2

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-2,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_t0,
            T_mult=cosine_tmult,
            eta_min=1e-6,
        )
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
