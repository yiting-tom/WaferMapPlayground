from collections import Counter

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.models import mobilenet_v3_large


# Contrastive augmentations
class ContrastiveTransform:
    def __init__(self, base_transforms):
        self.base = base_transforms

    def __call__(self, x):
        return self.base(x), self.base(x)


# Dataset Module
class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "./", batch_size: int = 256, num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.class_counts = None
        aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.transform = ContrastiveTransform(aug)

    def setup(self, stage=None):
        self.train_set = FashionMNIST(
            self.data_dir, train=True, download=True, transform=self.transform
        )
        self.val_set = FashionMNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )

        labels = [label for _, label in self.train_set]
        self.class_counts = Counter(labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# Focal Loss for multiclass classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        focal_term = (1 - probs) ** self.gamma
        loss = -focal_term * log_probs

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_factor = alpha[targets]
            loss = loss * alpha_factor.unsqueeze(1)

        loss = (loss * targets_one_hot).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# LightningModule for contrastive learning
class ContrastiveModel(pl.LightningModule):
    def __init__(self, class_counts, lr: float = 1e-3, temperature: float = 0.5):
        super().__init__()
        self.save_hyperparameters()

        backbone = mobilenet_v3_large(pretrained=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(backbone.classifier[0].in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        total = sum(class_counts[i] for i in range(len(class_counts)))
        alpha = [total / class_counts[i] for i in range(len(class_counts))]
        alpha = torch.tensor(alpha, dtype=torch.float)
        alpha = alpha / alpha.sum()
        self.loss_fn = FocalLoss(alpha=alpha, gamma=2.0)

    def forward(self, x):
        h = self.encoder(x).squeeze(-1).squeeze(-1)
        z = self.projector(h)
        return F.normalize(z, dim=1)

    def info_nce_loss(self, z1, z2):
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        pos = torch.diag(sim_matrix, batch_size) + torch.diag(sim_matrix, -batch_size)
        mask = (
            ~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool, device=self.device)
        ).float()
        exp_sim = torch.exp(sim_matrix / self.hparams.temperature) * mask
        log_prob = pos.div(self.hparams.temperature).exp() / exp_sim.sum(dim=1)
        loss = -torch.log(log_prob).mean()
        return loss

    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        z1, z2 = self(x1), self(x2)
        loss = self.info_nce_loss(z1, z2)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    dm = FashionMNISTDataModule(batch_size=256)
    dm.setup()
    model = ContrastiveModel(class_counts=dm.class_counts, lr=1e-3, temperature=0.5)
    trainer = pl.Trainer(gpus=1, max_epochs=20, precision=16)
    trainer.fit(model, dm)
