from pathlib import Path
from time import time
from typing import Literal

import click
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms as T

from datamodule import WM38DataModule
from models import MobileNetV3Large, TinyViT
from modules.classifier import Classifier


def categorical_to_rgb(tensor):
    """Convert categorical tensor to RGB"""
    if isinstance(tensor, torch.Tensor):
        categorical = tensor
        device = tensor.device
    else:
        categorical = torch.from_numpy(tensor)
        device = categorical.device

    # Create RGB tensor directly in torch (no numpy conversion)
    rgb = torch.zeros(*categorical.shape, 3, dtype=torch.float32, device=device)

    rgb[categorical == 0] = torch.tensor([0.0, 0.0, 0.0], device=device)  # Background
    rgb[categorical == 1] = torch.tensor([0.0, 0.5, 0.5], device=device)  # Wafer
    rgb[categorical == 2] = torch.tensor([1.0, 0.0, 0.0], device=device)  # Defects

    return rgb.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)


def get_transforms(resize: int = 224):
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = T.Compose(
        [
            T.Lambda(categorical_to_rgb),
            T.Resize(resize),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=180),
            norm,
        ]
    )

    val_transform = T.Compose(
        [
            T.Lambda(categorical_to_rgb),
            T.Resize(resize),
            norm,
        ]
    )

    return train_transform, val_transform, val_transform  # test same as val


def get_model(
    backbone: Literal["mobilenet_v3_large", "tiny_vit_21m_512"],
    num_labels: int,
):
    if backbone == "mobilenet_v3_large":
        return MobileNetV3Large(num_labels=num_labels)
    elif backbone == "tiny_vit_21m_512":
        return TinyViT(num_labels=num_labels)
    else:
        raise ValueError(f"Invalid backbone: {backbone}")


@click.group()
def cli():
    """A CLI tool for training and testing machine learning models with MobileNetV3 and TinyViT backbones."""
    pass


@cli.command()
@click.option(
    "--backbone",
    type=click.Choice(["mobilenet_v3_large", "tiny_vit_21m_512"]),
    required=True,
    help="Choose the backbone architecture (MobileNetV3Large or TinyViT)",
)
@click.option("--lr", type=float, required=True, help="Learning rate for training")
@click.option(
    "--datasize",
    type=int,
    required=True,
    help="Size of the dataset to use for training",
)
@click.option("--epochs", type=int, required=True, help="Number of training epochs")
@click.option(
    "--npz_file",
    type=str,
    required=True,
    help="Path to the NPZ data file containing the dataset",
)
@click.option(
    "--resize",
    type=int,
    default=224,
    help="Image resize dimension in pixels (default: 224)",
)
@click.option(
    "--fully_finetune",
    is_flag=False,
    help="Whether to fully finetune the model",
)
def train(
    backbone: Literal["mobilenet_v3_large", "tiny_vit_21m_512"],
    lr: float,
    datasize: int,
    epochs: int,
    npz_file: str,
    resize: int = 224,
    fully_finetune: bool = False,
):
    """Train a machine learning model with the specified parameters and save checkpoints."""
    npz_file = Path(npz_file)
    exp_name = f"{backbone}_{lr}-{npz_file.stem}_{resize}_{datasize}-e{epochs}"
    model = get_model(backbone, num_labels=8)
    classifier = Classifier(model, num_labels=8, lr=lr, fully_finetune=fully_finetune)
    train_transform, val_transform, test_transform = get_transforms(resize=resize)
    datamodule = WM38DataModule(
        npz_file=npz_file,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        datasize=datasize,
    )
    trainer = Trainer(
        enable_progress_bar=False,
        enable_model_summary=True,
        max_epochs=epochs,
        logger=TensorBoardLogger(
            save_dir="logs",
            name=exp_name,
        ),
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                filename="{epoch:02d}-{val_loss:.4f}-{multilabel}",
                mode="min",
                save_top_k=2,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    t_start = time()
    trainer.fit(classifier, datamodule)
    t_end = time()
    print(f"Training time: {t_end - t_start} seconds")

    t_start = time()
    trainer.test(classifier, datamodule)
    t_end = time()
    print(f"Testing time: {t_end - t_start} seconds")


@cli.command()
@click.option(
    "--backbone",
    type=click.Choice(["mobilenet_v3_large", "tiny_vit_21m_512"]),
    help="Backbone architecture (usually inferred from checkpoint)",
)
@click.option(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the trained model checkpoint file",
)
@click.option(
    "--npz_file", type=str, required=True, help="Path to the NPZ data file for testing"
)
@click.option("--datasize", type=int, required=True, help="Size of the test dataset")
@click.option(
    "--resize",
    type=int,
    default=224,
    help="Image resize dimension in pixels (default: 224)",
)
def test(
    checkpoint: str,
    npz_file: str,
    datasize: int,
    resize: int = 224,
):
    """Test a trained model using a checkpoint file and report performance metrics."""
    classifier = Classifier.load_from_checkpoint(checkpoint)
    _, _, test_transform = get_transforms(resize=resize)
    datamodule = WM38DataModule(
        npz_file=npz_file,
        train_transform=test_transform,
        val_transform=test_transform,
        test_transform=test_transform,
        datasize=datasize,
    )
    trainer = Trainer(
        logger=False,
    )
    t_start = time()
    trainer.test(classifier, datamodule)
    t_end = time()
    print(f"Testing time: {t_end - t_start} seconds")


if __name__ == "__main__":
    cli()
