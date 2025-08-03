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
        lr: float = 1e-5,
        fully_finetune: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model = model
        self.lr = lr

        if fully_finetune:
            for param in self.model.parameters():
                param.requires_grad = True

        # Initialize metrics as nn.ModuleList to ensure they move to the correct device
        self.metrics = nn.ModuleList(
            [
                Accuracy(task="multilabel", num_labels=num_labels),
                F1Score(task="multilabel", num_labels=num_labels),
                Precision(task="multilabel", num_labels=num_labels),
                Recall(task="multilabel", num_labels=num_labels),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
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
