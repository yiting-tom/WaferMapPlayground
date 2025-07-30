import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_large

# Set seed for reproducibility
seed_everything(42)


# Custom Dataset for Contrastive Learning
class ContrastiveFashionMNIST(Dataset):
    def __init__(self, root="./data", train=True):
        self.dataset = torchvision.datasets.FashionMNIST(
            root=root, train=train, download=True
        )
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5)),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )

    def __getitem__(self, index):
        img, label = self.dataset[index]
        xi = self.transform(img)
        xj = self.transform(img)
        return xi, xj

    def __len__(self):
        return len(self.dataset)


# Lightning Module for SimCLR
class SimCLR(LightningModule):
    def __init__(self, hidden_dim=512, projection_dim=128, temperature=0.5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Load MobileNetV3 Large
        self.encoder = mobilenet_v3_large(weights="IMAGENET1K_V2")
        self.encoder.classifier = nn.Identity()  # Remove classification head

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(960, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, projection_dim)
        )

        self.temperature = temperature
        self.lr = lr

    def forward(self, x):
        features = self.encoder(x)
        return F.normalize(self.projection(features), features)

    def nt_xent_loss(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)
        n = z.shape[0]

        # Similarity matrix
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        sim = sim / self.temperature

        # Create positive mask (diagonals represent same image pairs)
        mask = torch.eye(n, dtype=torch.bool, device=self.device)
        mask = mask ^ torch.roll(
            mask, shifts=n // 2, dims=0
        )  # Shift to match positive pairs

        # Remove diagonals
        sim = sim[~torch.eye(n, dtype=bool, device=self.device)].view(n, n - 1)
        positives = sim[mask].view(n, 1)
        negatives = sim[~mask].view(n, n - 2)

        # Compute logits
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(n, dtype=torch.long, device=self.device)

        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        x_i, x_j = batch

        # Convert grayscale to 3 channels
        x_i = x_i.repeat(1, 3, 1, 1)
        x_j = x_j.repeat(1, 3, 1, 1)

        z_i, _ = self(x_i)
        z_j, _ = self(x_j)

        loss = self.nt_xent_loss(z_i, z_j)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-6
        )
        return [optimizer], [scheduler]


# Data Module
class FashionDataModule(LightningModule):
    def __init__(self, batch_size=256, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = ContrastiveFashionMNIST(train=True)
        self.val_dataset = ContrastiveFashionMNIST(train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# Training Configuration
def train_simclr():
    datamodule = FashionDataModule(batch_size=256)
    model = SimCLR(lr=1e-3)

    logger = TensorBoardLogger("logs/", name="simclr_mobilenetv3")
    checkpoint = ModelCheckpoint(
        monitor="train_loss", mode="min", save_top_k=1, filename="best-checkpoint"
    )

    trainer = Trainer(
        max_epochs=50,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train_simclr()
