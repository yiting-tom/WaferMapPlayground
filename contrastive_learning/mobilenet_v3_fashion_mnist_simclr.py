import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset


class ContrastiveTransform:
    """Data augmentation for contrastive learning"""

    def __init__(self, size=224):
        # Strong augmentations for contrastive learning
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, x):
        # Apply transform twice to get two different augmented views
        view1 = self.transform(x)
        view2 = self.transform(x)
        return view1, view2


class FashionMNISTContrastiveDataset(Dataset):
    """Fashion-MNIST dataset wrapper for contrastive learning"""

    def __init__(self, root="./data", train=True, download=True):
        # Convert grayscale to RGB for MobileNet
        base_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            ]
        )

        self.dataset = torchvision.datasets.FashionMNIST(
            root=root, train=train, download=download, transform=base_transform
        )
        self.contrastive_transform = ContrastiveTransform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Get two augmented views
        aug1, aug2 = self.contrastive_transform(image)
        return aug1, aug2, label


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""

    def __init__(self, input_dim=960, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.projection(x)


class MobileNetV3Contrastive(pl.LightningModule):
    """MobileNetV3-Large with contrastive learning using SimCLR"""

    def __init__(
        self, temperature=0.07, learning_rate=1e-3, weight_decay=1e-4, max_epochs=100
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained MobileNetV3-Large
        self.backbone = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V1")

        # Remove the classifier head and get feature dimension
        self.feature_dim = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()

        # Add projection head
        self.projection_head = ProjectionHead(
            input_dim=self.feature_dim, hidden_dim=512, output_dim=128
        )

        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)

    def contrastive_loss(self, z1, z2):
        """NT-Xent (InfoNCE) loss for contrastive learning"""
        batch_size = z1.shape[0]
        device = z1.device

        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # 2B x D

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # 2B x 2B

        # Create masks for positive and negative pairs
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -float("inf"))

        # Create labels for contrastive loss
        # For samples 0 to B-1, positives are at B to 2B-1
        # For samples B to 2B-1, positives are at 0 to B-1
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(0, batch_size, device=device),
            ]
        )

        # Compute cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def training_step(self, batch, batch_idx):
        aug1, aug2, _ = batch

        # Get representations
        z1 = self(aug1)
        z2 = self(aug2)

        # Compute contrastive loss
        loss = self.contrastive_loss(z1, z2)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        aug1, aug2, _ = batch

        z1 = self(aug1)
        z2 = self(aug2)

        loss = self.contrastive_loss(z1, z2)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class LinearEvaluator(pl.LightningModule):
    """Linear evaluation protocol for contrastive representations"""

    def __init__(self, backbone_model, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone_model"])

        # Freeze the backbone
        self.backbone = backbone_model.backbone
        self.projection_head = backbone_model.projection_head

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.projection_head.parameters():
            param.requires_grad = False

        # Linear classifier
        self.classifier = nn.Linear(backbone_model.feature_dim, num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("test_acc", acc, on_step=False, on_epoch=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)


def create_data_loaders(batch_size=128, num_workers=4):
    """Create data loaders for contrastive learning and linear evaluation"""

    # Determine actual num_workers based on system
    actual_workers = min(num_workers, 4) if num_workers > 0 else 0
    persistent = actual_workers > 0

    # Contrastive learning data
    train_contrastive = FashionMNISTContrastiveDataset(train=True, download=True)
    val_contrastive = FashionMNISTContrastiveDataset(train=False, download=True)

    contrastive_train_loader = DataLoader(
        train_contrastive,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_workers,
        persistent_workers=persistent,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    contrastive_val_loader = DataLoader(
        val_contrastive,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        persistent_workers=persistent,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Linear evaluation data
    eval_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_eval = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=eval_transform
    )
    test_eval = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=eval_transform
    )

    eval_train_loader = DataLoader(
        train_eval,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_workers,
        persistent_workers=persistent,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    eval_test_loader = DataLoader(
        test_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        persistent_workers=persistent,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return (
        contrastive_train_loader,
        contrastive_val_loader,
        eval_train_loader,
        eval_test_loader,
    )


def train_contrastive_model():
    """Train the contrastive learning model"""

    # Create data loaders
    (
        contrastive_train_loader,
        contrastive_val_loader,
        eval_train_loader,
        eval_test_loader,
    ) = create_data_loaders(batch_size=64, num_workers=2)

    # Initialize model
    model = MobileNetV3Contrastive(
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=50,  # Reduced for faster testing
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="mobilenetv3-contrastive-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Logger
    logger = TensorBoardLogger("logs/", name="mobilenetv3_contrastive")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,  # Reduced for faster testing
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    print("Starting contrastive learning training...")
    trainer.fit(model, contrastive_train_loader, contrastive_val_loader)

    return model, trainer


def evaluate_linear_probe(contrastive_model):
    """Evaluate learned representations with linear probing"""

    # Create data loaders
    (_, _, eval_train_loader, eval_test_loader) = create_data_loaders(
        batch_size=64, num_workers=2
    )

    # Initialize linear evaluator
    evaluator = LinearEvaluator(
        backbone_model=contrastive_model, num_classes=10, learning_rate=1e-3
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="checkpoints/",
        filename="linear-probe-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1,
        mode="max",
        save_last=True,
    )

    # Logger
    logger = TensorBoardLogger("logs/", name="linear_evaluation")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=30,  # Reduced for faster testing
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    # Train linear probe
    print("Starting linear evaluation...")
    trainer.fit(evaluator, eval_train_loader, eval_test_loader)

    # Test
    test_results = trainer.test(evaluator, eval_test_loader)
    print(f"Final test accuracy: {test_results[0]['test_acc']:.4f}")

    return evaluator, test_results


def test_pipeline():
    """Test the pipeline with minimal configuration"""
    import os

    print("Testing pipeline with minimal configuration...")

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Test data loading
    print("✓ Testing data loading...")
    (train_loader, val_loader, eval_train_loader, eval_test_loader) = (
        create_data_loaders(
            batch_size=8,
            num_workers=0,  # Small batch, no workers for testing
        )
    )

    # Test one batch
    batch = next(iter(train_loader))
    aug1, aug2, labels = batch
    print(
        f"✓ Batch shapes: aug1={aug1.shape}, aug2={aug2.shape}, labels={labels.shape}"
    )

    # Test model
    print("✓ Testing model...")
    model = MobileNetV3Contrastive(temperature=0.07, learning_rate=1e-3, max_epochs=2)

    # Test forward pass
    z1 = model(aug1)
    z2 = model(aug2)
    print(f"✓ Embeddings shape: z1={z1.shape}, z2={z2.shape}")

    # Test loss
    loss = model.contrastive_loss(z1, z2)
    print(f"✓ Contrastive loss: {loss.item():.4f}")

    # Test linear evaluator
    print("✓ Testing linear evaluator...")
    evaluator = LinearEvaluator(model, num_classes=10)

    # Test evaluation batch
    eval_batch = next(iter(eval_train_loader))
    eval_x, eval_y = eval_batch
    logits = evaluator(eval_x)
    print(f"✓ Evaluation logits shape: {logits.shape}")

    print("✓ All tests passed! Pipeline is ready to run.")
    return True


if __name__ == "__main__":
    # Set random seeds for reproducibility
    pl.seed_everything(42)

    # Test the pipeline first
    if test_pipeline():
        print("\n" + "=" * 70)
        print("Training MobileNetV3-Large with Contrastive Learning on Fashion-MNIST")
        print("=" * 70)

        # Phase 1: Contrastive learning
        contrastive_model, contrastive_trainer = train_contrastive_model()

        print("\nContrastive learning completed!")
        print("=" * 70)

        # Phase 2: Linear evaluation
        evaluator, test_results = evaluate_linear_probe(contrastive_model)

        print("\nTraining completed!")
        print("Best contrastive model saved in: checkpoints/")
        print("Logs available in: logs/")
        print(f"Final test accuracy: {test_results[0]['test_acc']:.4f}")
    else:
        print("❌ Pipeline test failed. Please check the issues above.")
