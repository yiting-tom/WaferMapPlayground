# %%
import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision import transforms
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
        return loss

    def on_validation_epoch_end(self):
        for metric in self.metrics:
            self.log(f"val_{metric.__class__.__name__.lower()}", metric.compute())
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
        train_transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
            ]
        ),
        val_transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
            ]
        ),
        test_transform=transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
            ]
        ),
    )
    trainer = Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices=1,
        strategy="ddp",
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=2,
                save_last=True,
            )
        ],
        logger=TensorBoardLogger(
            save_dir="lightning_logs",
            name="mobile_net",
        ),
    )
    trainer.fit(module, datamodule)
    trainer.test(module, datamodule)


if __name__ == "__main__":
    main()

# %%
