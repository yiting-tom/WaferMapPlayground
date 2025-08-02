# %%
import sys

import numpy as np
import torch
from lightning import LightningDataModule, seed_everything
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

seed_everything(42)


class NumpyDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        task: str,
        transform: transforms.Compose = None,
    ):
        self.images = images
        self.labels = labels
        self.task = task
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.task == "multilabel":
            label = torch.from_numpy(label).float()
        else:
            label = torch.from_numpy(label).long()

        return image, label


class WM38DataModule(LightningDataModule):
    task = "multilabel"
    raw_data_url = (
        "https://drive.google.com/file/d/1M59pX-lPqL9APBIbp2AKQRTvngeUK8Va/view"
    )

    def __init__(
        self,
        npz_file: str,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transform: transforms.Compose = None,
        val_transform: transforms.Compose = None,
        test_transform: transforms.Compose = None,
    ):
        super().__init__()
        self.npz_file = npz_file
        self.batch_size = batch_size
        # Set num_workers to 0 in interactive environments to avoid multiprocessing issues
        if hasattr(sys, "ps1") or "ipykernel" in sys.modules:
            self.num_workers = 0
        else:
            self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self):
        from pathlib import Path

        if not Path(self.npz_file).exists():
            import requests

            print("Downloading data from Google Drive:", self.raw_data_url)
            response = requests.get(
                f"https://drive.google.com/uc?export=download&id={self.raw_data_url.split('/')[-2]}"
            )
            with open(self.npz_file, "wb") as f:
                f.write(response.content)

        npz = np.load(self.npz_file, allow_pickle=True)
        # images: (38015, 256, 256)
        self.images = npz["images"]
        # labels: (38015, 8)
        self.labels = npz["labels"]

    def setup(self):
        train_images, val_images, train_labels, val_labels = train_test_split(
            self.images,
            self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels,
        )
        val_images, test_images, val_labels, test_labels = train_test_split(
            val_images, val_labels, test_size=0.5, random_state=42, stratify=val_labels
        )
        self.train_dataset = NumpyDataset(
            train_images, train_labels, self.task, self.train_transform
        )
        self.val_dataset = NumpyDataset(
            val_images, val_labels, self.task, self.val_transform
        )
        self.test_dataset = NumpyDataset(
            test_images, test_labels, self.task, self.test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def draw_transform() -> np.ndarray:
    sparse_dm = WM38DataModule(
        npz_file="sparse_wm38.npz",
        batch_size=4,
        train_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
        val_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
        test_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )
    sparse_dm.prepare_data()
    sparse_dm.setup()
    # Test single sample
    sparse_sample = sparse_dm.val_dataset[0]
    print(
        f"Single sample shape: {sparse_sample[0].shape}, label shape: {sparse_sample[1].shape}"
    )
    # Test dataloader
    sparse_dl = sparse_dm.val_dataloader()
    sparse_data = next(iter(sparse_dl))
    print(
        f"Batch shape: {sparse_data[0].shape}, batch label shape: {sparse_data[1].shape}"
    )
    # %%
    restored_dm = WM38DataModule(
        npz_file="restored_wm38.npz",
        batch_size=4,
        train_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
        val_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
        test_transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        ),
    )
    restored_dm.prepare_data()
    restored_dm.setup()
    # Test single sample
    restored_sample = restored_dm.val_dataset[0]
    print(
        f"Single sample shape: {restored_sample[0].shape}, label shape: {restored_sample[1].shape}"
    )
    # Test dataloader
    restored_dl = restored_dm.val_dataloader()
    restored_data = next(iter(restored_dl))
    print(
        f"Batch shape: {restored_data[0].shape}, batch label shape: {restored_data[1].shape}"
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sparse_data[0][0].permute(1, 2, 0).numpy())
    plt.colorbar()
    plt.title("Sparse WM38")
    plt.subplot(1, 2, 2)
    plt.imshow(restored_data[0][0].permute(1, 2, 0).numpy())
    plt.colorbar()
    plt.title("Restored Sparsed WM39")

    plt.imsave("datamodule_inputs.png", sparse_data[0][0].permute(1, 2, 0).numpy())


# %%
datamodule = WM38DataModule(
    npz_file="sparse_wm38.npz",
    batch_size=4,
)
# %%
datamodule.prepare_data()
