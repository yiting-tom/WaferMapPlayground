import argparse
import pickle
import warnings
from collections import Counter
from typing import List, Optional, Tuple

import cv2
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split

warnings.filterwarnings("ignore")


class WM811KDataset(Dataset):
    """Dataset for WM-811K wafer map data"""

    def __init__(
        self,
        pkl_path: str,
        task_type: str = "multiclass",
        target_size: tuple = (64, 64),
        balance_classes: bool = False,
        min_samples_per_class: int = 1000,
    ):
        self.task_type = task_type
        self.target_size = target_size

        # Load and process data
        print("Loading WM-811K dataset...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        wafer_maps, labels = self._extract_data(data)
        self.labels = np.array(labels)

        # Process labels by task type
        if task_type == "binary":
            self.labels = (self.labels != 8).astype(int)
            self.num_classes = 2
            self.class_names = ["Pattern", "No Pattern"]
        else:  # multiclass
            valid_mask = (self.labels >= 0) & (self.labels <= 8)
            wafer_maps = [
                wafer_maps[i] for i in range(len(wafer_maps)) if valid_mask[i]
            ]
            self.labels = self.labels[valid_mask]
            self.num_classes = 9
            self.class_names = [
                "Center",
                "Donut",
                "Edge-Loc",
                "Edge-Ring",
                "Loc",
                "Random",
                "Scratch",
                "Near-full",
                "None",
            ]

        self._print_class_distribution()

        # Balance classes if requested
        if balance_classes:
            wafer_maps, self.labels = self._balance_classes(
                wafer_maps, self.labels, min_samples_per_class
            )
            print(f"After balancing: {len(wafer_maps)} samples")

        # Process wafer maps
        self.wafer_maps = self._process_wafer_maps(wafer_maps)
        print(
            f"Final dataset: {len(self.wafer_maps)} samples, {self.num_classes} classes"
        )

    def _extract_data(self, data):
        """Extract wafer maps and labels from raw data"""
        wafer_maps, labels = [], []

        for i, item in enumerate(data):
            if i % 50000 == 0:
                print(f"Processed {i}/{len(data)} samples")

            die_map = item.get("dieMap")
            failure_type = item.get("failureType")

            if die_map is not None and failure_type is not None:
                wafer_map = np.array(die_map, dtype=np.float32)
                if wafer_map.size > 0:
                    wafer_maps.append(wafer_map)
                    labels.append(failure_type)

        return wafer_maps, labels

    def _print_class_distribution(self):
        """Print class distribution"""
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nClass distribution:")
        for class_id, count in zip(unique, counts):
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"Class_{class_id}"
            )
            print(f"  {class_name}: {count} ({count / len(self.labels) * 100:.1f}%)")

    def _balance_classes(self, wafer_maps: List, labels: np.ndarray, min_samples: int):
        """Balance classes by undersampling"""
        class_counts = Counter(labels)
        target_count = min(min_samples, min(class_counts.values()))

        balanced_maps, balanced_labels = [], []
        for class_id in range(self.num_classes):
            class_indices = np.where(labels == class_id)[0]
            if len(class_indices) > 0:
                n_samples = min(target_count, len(class_indices))
                selected_indices = np.random.choice(
                    class_indices, n_samples, replace=False
                )
                for idx in selected_indices:
                    balanced_maps.append(wafer_maps[idx])
                    balanced_labels.append(labels[idx])

        return balanced_maps, np.array(balanced_labels)

    def _process_wafer_maps(self, wafer_maps: List) -> np.ndarray:
        """Normalize and resize wafer maps"""
        processed_maps = []
        for wafer_map in wafer_maps:
            # Normalize
            if wafer_map.max() > 1:
                wafer_map = wafer_map / wafer_map.max()

            # Resize
            if wafer_map.shape != self.target_size:
                wafer_map = cv2.resize(
                    wafer_map, self.target_size, interpolation=cv2.INTER_NEAREST
                )

            processed_maps.append(wafer_map)

        return np.array(processed_maps, dtype=np.float32)

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        class_counts = np.bincount(self.labels)
        weights = len(self.labels) / (self.num_classes * class_counts)
        return torch.FloatTensor(weights)

    def __len__(self):
        return len(self.wafer_maps)

    def __getitem__(self, idx):
        wafer_map = self.wafer_maps[idx][np.newaxis, ...]  # Add channel dim
        label = self.labels[idx]
        return torch.tensor(wafer_map, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


class WaferCNN(nn.Module):
    """CNN for wafer map classification"""

    def __init__(
        self,
        num_classes: int,
        num_conv_layers: int = 4,
        base_channels: int = 32,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Build conv layers
        channels = [1] + [base_channels * (2**i) for i in range(num_conv_layers)]
        conv_layers = []

        for i in range(num_conv_layers):
            conv_layers.extend(
                [
                    nn.Conv2d(channels[i], channels[i + 1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        channels[i + 1], channels[i + 1], 3, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(dropout_rate),
                ]
            )

        self.features = nn.Sequential(*conv_layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)


class OptunaPruningCallback(Callback):
    """Lightning callback for Optuna pruning"""

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        self.trial.report(current_score.item(), trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


class WaferClassificationModel(pl.LightningModule):
    """Lightning module for wafer classification"""

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        task_type: str = "multiclass",
        learning_rate: float = 1e-3,
        optimizer_name: str = "AdamW",
        num_conv_layers: int = 4,
        base_channels: int = 32,
        dropout_rate: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["class_weights"])
        self.num_classes = num_classes
        self.class_names = class_names
        self.task_type = task_type

        # Model
        self.model = WaferCNN(num_classes, num_conv_layers, base_channels, dropout_rate)

        # Loss function
        if task_type == "binary":
            pos_weight = (
                class_weights[1] / class_weights[0]
                if class_weights is not None
                else None
            )
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing
            )

        self.test_outputs = []

    @classmethod
    def from_trial(cls, trial: optuna.Trial, dataset, **fixed_params):
        """Create model with Optuna-suggested hyperparameters"""
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "optimizer_name": trial.suggest_categorical(
                "optimizer", ["Adam", "AdamW", "SGD"]
            ),
            "num_conv_layers": trial.suggest_int("num_conv_layers", 3, 6),
            "base_channels": trial.suggest_categorical("base_channels", [16, 32, 64]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.6),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2)
            if dataset.task_type == "multiclass"
            else 0.0,
        }
        return cls(**{**params, **fixed_params})

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)

        if self.task_type == "binary" and self.num_classes == 1:
            logits = logits.squeeze()
            y = y.float()

        loss = self.criterion(logits, y)

        # Calculate accuracy
        if self.task_type == "binary":
            preds = (torch.sigmoid(logits) > 0.5).long()
        else:
            preds = torch.argmax(logits, dim=1)

        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": y}

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")["loss"]

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")["loss"]

    def test_step(self, batch, batch_idx):
        outputs = self._step(batch, "test")
        self.test_outputs.append(outputs)
        return outputs["loss"]

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return

        # Aggregate predictions
        all_preds = torch.cat([x["preds"] for x in self.test_outputs]).cpu().numpy()
        all_targets = torch.cat([x["targets"] for x in self.test_outputs]).cpu().numpy()

        # Calculate metrics
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(
            all_targets,
            all_preds,
            average="weighted" if self.task_type == "multiclass" else "binary",
        )

        self.log("test_acc_final", acc)
        self.log("test_f1_final", f1)

        print("\nFinal Test Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(all_targets, all_preds, target_names=self.class_names)
        )

        self.test_outputs.clear()

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )
        else:  # AdamW
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-2
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def create_data_loaders(
    dataset: WM811KDataset,
    batch_size: int = 64,
    train_split: float = 0.7,
    val_split: float = 0.2,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders"""

    # Split dataset
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Weighted sampling for training
    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    weights = 1.0 / class_counts
    sample_weights = [weights[label] for label in train_labels]
    train_sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader


class OptunaTrainer:
    """Trainer with Optuna integration"""

    def __init__(self, dataset: WM811KDataset, study_name: str = "wm811k_optimization"):
        self.dataset = dataset
        self.study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        # Create data loaders
        train_loader, val_loader, _ = create_data_loaders(
            self.dataset, batch_size=batch_size
        )

        # Create model
        model = WaferClassificationModel.from_trial(
            trial=trial,
            dataset=self.dataset,
            num_classes=self.dataset.num_classes,
            class_names=self.dataset.class_names,
            task_type=self.dataset.task_type,
            class_weights=self.dataset.get_class_weights(),
        )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=30,
            callbacks=[
                OptunaPruningCallback(trial, monitor="val_loss"),
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            accelerator="auto",
            devices="auto",
        )

        # Train
        try:
            trainer.fit(model, train_loader, val_loader)
            return trainer.callback_metrics["val_loss"].item()
        except optuna.TrialPruned:
            raise optuna.TrialPruned()

    def optimize(self, n_trials: int = 50) -> optuna.Study:
        """Run hyperparameter optimization"""
        print(f"Starting optimization with {n_trials} trials...")
        self.study.optimize(self.objective, n_trials=n_trials)

        print("Optimization completed!")
        print(f"Best value: {self.study.best_value:.4f}")
        print("Best parameters:")
        for key, value in self.study.best_params.items():
            print(f"  {key}: {value}")

        return self.study

    def train_best_model(
        self, max_epochs: int = 100
    ) -> Tuple[WaferClassificationModel, pl.Trainer]:
        """Train final model with best hyperparameters"""
        best_params = self.study.best_params.copy()
        batch_size = best_params.pop("batch_size", 64)

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            self.dataset, batch_size=batch_size
        )

        # Create model
        model = WaferClassificationModel(
            num_classes=self.dataset.num_classes,
            class_names=self.dataset.class_names,
            task_type=self.dataset.task_type,
            class_weights=self.dataset.get_class_weights(),
            **best_params,
        )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=20, mode="min"),
                ModelCheckpoint(
                    dirpath="checkpoints/",
                    filename="wm811k-best-{epoch:02d}-{val_loss:.3f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=3,
                ),
            ],
            logger=TensorBoardLogger("tb_logs", name="wm811k_final"),
            accelerator="auto",
            devices="auto",
        )

        # Train and test
        print("Training final model...")
        trainer.fit(model, train_loader, val_loader)
        print("Testing final model...")
        trainer.test(model, test_loader)

        return model, trainer


def main():
    parser = argparse.ArgumentParser(description="WM-811K Wafer Map Classification")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to LSWMD.pkl")
    parser.add_argument(
        "--task_type", choices=["binary", "multiclass"], default="multiclass"
    )
    parser.add_argument("--target_size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--balance_classes", action="store_true")
    parser.add_argument("--min_samples_per_class", type=int, default=1000)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--final_epochs", type=int, default=100)
    parser.add_argument("--study_name", type=str, default="wm811k_optimization")

    args = parser.parse_args()

    # Set seed for reproducibility
    pl.seed_everything(42)

    # Create dataset
    print("Creating dataset...")
    dataset = WM811KDataset(
        pkl_path=args.pkl_path,
        task_type=args.task_type,
        target_size=tuple(args.target_size),
        balance_classes=args.balance_classes,
        min_samples_per_class=args.min_samples_per_class,
    )

    # Optimize hyperparameters
    trainer = OptunaTrainer(dataset, args.study_name)
    study = trainer.optimize(args.n_trials)

    # Train final model
    final_model, final_trainer = trainer.train_best_model(args.final_epochs)

    # Save results
    study_df = study.trials_dataframe()
    study_df.to_csv(f"{args.study_name}_results.csv", index=False)
    print(f"Results saved to {args.study_name}_results.csv")


if __name__ == "__main__":
    main()
