# %%
import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

seed_everything(42)


class MobileNetModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = F.binary_cross_entropy_with_logits

        # Load pretrained backbone
        self.backbone = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        )

        # Modify the first convolutional layer to accept 1 channel instead of 3
        first_conv = self.backbone.features[0][0]  # First conv layer

        # Create new conv layer with 1 input channel
        new_first_conv = nn.Conv2d(
            in_channels=1,  # Changed from 3 to 1
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        # Initialize the new layer weights
        # Option 1: Average the RGB weights to create grayscale weights
        with torch.no_grad():
            new_first_conv.weight = nn.Parameter(
                first_conv.weight.mean(dim=1, keepdim=True)
            )
            if first_conv.bias is not None:
                new_first_conv.bias = nn.Parameter(first_conv.bias.clone())

        # Replace the first conv layer
        self.backbone.features[0][0] = new_first_conv

        # Freeze feature extractor
        self.backbone.features.requires_grad_(False)

        # Modify classifier
        input_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 8),
        )

        self.metrics = [
            Accuracy(task="multilabel", num_labels=8),
            F1Score(task="multilabel", num_labels=8),
            Precision(task="multilabel", num_labels=8),
            Recall(task="multilabel", num_labels=8),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        self.log("val_loss", loss)
        for metric in self.metrics:
            metric(outputs, labels)
        self.log_dict(metric.compute())
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        self.log("test_loss", loss)
        for metric in self.metrics:
            metric(outputs, labels)
        self.log_dict(metric.compute())
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# %%
def main():
    from datamodule import WM38DataModule

    module = MobileNetModule()
    datamodule = WM38DataModule(
        npz_file="sparse_wm38.npz",
        batch_size=128,
        num_workers=4,
        pin_memory=True,
    )
    trainer = Trainer(
        max_epochs=300,
        gpus=1,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=2,
                save_last=True,
            )
        ],
    )
    trainer.fit(module, datamodule)
    trainer.test(module, datamodule)


if __name__ == "__main__":
    main()
