"""
Complete ML Integration Pipeline

This module demonstrates the integration of all major ML tools:
- PyTorch Lightning: Training framework
- Optuna: Hyperparameter optimization
- MLflow: Experiment tracking and model registry
- Ultralytics: Modern YOLO architecture
- Torchvision: Pre-trained models and transforms

A production-ready pipeline for wafer defect classification.
"""

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from data_utils import WaferMapDataset, create_data_loaders, get_tutorial_dataset
from metrics import ModelEvaluator
from visualizations import WaferVisualization, create_results_dashboard

warnings.filterwarnings("ignore")


@dataclass
class ExperimentConfig:
    """Configuration for complete integration experiments."""

    experiment_name: str = "wafer_defect_complete_integration"
    model_type: str = "lightning_cnn"  # 'lightning_cnn', 'torchvision_resnet', 'yolo'
    use_pretrained: bool = True
    max_epochs: int = 50
    batch_size: int = 32
    num_optuna_trials: int = 20
    enable_mlflow: bool = True
    enable_optuna: bool = True
    data_augmentation: bool = True
    class_balancing: bool = True


class TorchvisionLightningModel(pl.LightningModule):
    """
    Lightning model using Torchvision pre-trained networks.

    Demonstrates transfer learning with popular architectures like ResNet, EfficientNet.
    """

    def __init__(
        self,
        num_classes: int = 9,
        architecture: str = "resnet18",
        pretrained: bool = True,
        learning_rate: float = 1e-3,
        optimizer_name: str = "AdamW",
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Create backbone
        if architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif architecture == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Adapt input for grayscale
        if hasattr(self.backbone, "conv1"):
            # ResNet-style
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False,
            )
        elif hasattr(self.backbone, "features") and hasattr(
            self.backbone.features[0], "0"
        ):
            # EfficientNet-style
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                1,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False,
            )

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics storage
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x):
        # Handle grayscale to RGB conversion if needed
        if x.shape[1] == 1 and hasattr(self.backbone, "conv1"):
            x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB

        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate metrics
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": preds.detach(), "targets": y.detach()}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, "val")
        self.validation_outputs.append(outputs)
        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, "test")
        self.test_outputs.append(outputs)
        return outputs["loss"]

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return

        all_preds = torch.cat([x["preds"] for x in self.validation_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_outputs])

        # Calculate F1 score
        from sklearn.metrics import f1_score

        f1 = f1_score(
            all_targets.cpu().numpy(), all_preds.cpu().numpy(), average="macro"
        )
        self.log("val_f1", f1, sync_dist=True)

        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return

        all_preds = torch.cat([x["preds"] for x in self.test_outputs])
        all_targets = torch.cat([x["targets"] for x in self.test_outputs])

        from sklearn.metrics import accuracy_score, f1_score

        f1 = f1_score(
            all_targets.cpu().numpy(), all_preds.cpu().numpy(), average="macro"
        )
        acc = accuracy_score(all_targets.cpu().numpy(), all_preds.cpu().numpy())

        self.log("test_f1_final", f1, sync_dist=True)
        self.log("test_acc_final", acc, sync_dist=True)

        print("\nFinal Test Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score (macro): {f1:.4f}")

        self.test_outputs.clear()

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )
        elif self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4
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
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class YOLOLightningAdapter(pl.LightningModule):
    """
    Lightning adapter for Ultralytics YOLO models.

    Wraps YOLO for classification in Lightning framework.
    """

    def __init__(
        self,
        num_classes: int = 9,
        model_size: str = "n",  # n, s, m, l, x
        learning_rate: float = 1e-3,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Note: This is a simplified adapter. In practice, you'd need to
        # properly integrate YOLO's architecture for classification
        # For this demo, we'll use a CNN similar to YOLO's backbone

        # YOLO-inspired CNN architecture
        base_channels = 16 if model_size == "n" else 32 if model_size == "s" else 64

        self.backbone = nn.Sequential(
            # Focus layer (YOLO-style)
            nn.Conv2d(1, base_channels, 6, 2, 2),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True),
            # CSP blocks (simplified)
            self._make_csp_block(base_channels, base_channels * 2),
            self._make_csp_block(base_channels * 2, base_channels * 4),
            self._make_csp_block(base_channels * 4, base_channels * 8),
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 8, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.validation_outputs = []
        self.test_outputs = []

    def _make_csp_block(self, in_channels, out_channels):
        """Create a simplified CSP block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def _shared_step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)

        return {"loss": loss, "preds": preds.detach(), "targets": y.detach()}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, "val")
        self.validation_outputs.append(outputs)
        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        outputs = self._shared_step(batch, "test")
        self.test_outputs.append(outputs)
        return outputs["loss"]

    def on_validation_epoch_end(self):
        if self.validation_outputs:
            all_preds = torch.cat([x["preds"] for x in self.validation_outputs])
            all_targets = torch.cat([x["targets"] for x in self.validation_outputs])

            from sklearn.metrics import f1_score

            f1 = f1_score(
                all_targets.cpu().numpy(), all_preds.cpu().numpy(), average="macro"
            )
            self.log("val_f1", f1, sync_dist=True)

            self.validation_outputs.clear()

    def on_test_epoch_end(self):
        if self.test_outputs:
            all_preds = torch.cat([x["preds"] for x in self.test_outputs])
            all_targets = torch.cat([x["targets"] for x in self.test_outputs])

            from sklearn.metrics import accuracy_score, f1_score

            f1 = f1_score(
                all_targets.cpu().numpy(), all_preds.cpu().numpy(), average="macro"
            )
            acc = accuracy_score(all_targets.cpu().numpy(), all_preds.cpu().numpy())

            self.log("test_f1_final", f1, sync_dist=True)
            self.log("test_acc_final", acc, sync_dist=True)

            print(f"\nYOLO Test Results - Accuracy: {acc:.4f}, F1: {f1:.4f}")
            self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.0005
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )

        return [optimizer], [scheduler]


class IntegratedMLPipeline:
    """
    Complete ML Pipeline integrating all tools.

    Orchestrates Lightning training, Optuna optimization, MLflow tracking,
    and model comparison across different architectures.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.mlflow_experiment_id = None
        self.best_models = {}
        self.all_results = {}

        # Setup MLflow
        if config.enable_mlflow:
            self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow experiment tracking."""
        mlflow.set_experiment(self.config.experiment_name)
        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        self.mlflow_experiment_id = experiment.experiment_id
        print(f"📊 MLflow experiment: {self.config.experiment_name}")
        print(
            f"🔗 View at: http://localhost:5000/#/experiments/{self.mlflow_experiment_id}"
        )

    def create_model(
        self,
        model_type: str,
        trial_params: Optional[Dict] = None,
        dataset: Optional[WaferMapDataset] = None,
    ) -> pl.LightningModule:
        """Create model based on type and parameters."""

        # Default parameters
        params = {
            "num_classes": 9,
            "learning_rate": 1e-3,
            "class_weights": dataset.get_class_weights() if dataset else None,
        }

        # Update with trial parameters
        if trial_params:
            params.update(trial_params)

        if model_type == "torchvision_resnet":
            return TorchvisionLightningModel(
                architecture="resnet18", pretrained=self.config.use_pretrained, **params
            )
        elif model_type == "torchvision_efficientnet":
            return TorchvisionLightningModel(
                architecture="efficientnet_b0",
                pretrained=self.config.use_pretrained,
                **params,
            )
        elif model_type == "yolo_inspired":
            return YOLOLightningAdapter(**params)
        else:
            # Default to simple CNN from Tutorial 01
            from simple_model import WaferLightningModel

            return WaferLightningModel(**params)

    def create_data_transforms(self, stage: str = "train"):
        """Create data augmentation transforms."""
        if not self.config.data_augmentation:
            return None

        if stage == "train":
            return transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomRotation(15),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

    def optuna_objective(
        self, trial: optuna.Trial, model_type: str, dataset: WaferMapDataset
    ):
        """Optuna objective function for hyperparameter optimization."""

        # Suggest hyperparameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }

        if model_type in ["torchvision_resnet", "torchvision_efficientnet"]:
            params.update(
                {
                    "freeze_backbone": trial.suggest_boolean("freeze_backbone"),
                    "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.6),
                    "optimizer_name": trial.suggest_categorical(
                        "optimizer_name", ["Adam", "AdamW", "SGD"]
                    ),
                }
            )

        # Create model and data loaders
        model = self.create_model(model_type, params, dataset)
        train_loader, val_loader, _ = create_data_loaders(
            dataset,
            batch_size=params["batch_size"],
            use_weighted_sampling=self.config.class_balancing,
        )

        # Setup trainer with pruning
        callbacks = [
            PyTorchLightningPruningCallback(trial, monitor="val_f1"),
            EarlyStopping(monitor="val_loss", patience=7, mode="min"),
        ]

        trainer = pl.Trainer(
            max_epochs=min(30, self.config.max_epochs),
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="auto",
        )

        # Train and return metric
        try:
            trainer.fit(model, train_loader, val_loader)

            # Return validation F1 score (to maximize)
            val_f1 = trainer.callback_metrics.get("val_f1", 0)
            return val_f1.item() if hasattr(val_f1, "item") else val_f1

        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0

    def run_optuna_optimization(
        self, model_type: str, dataset: WaferMapDataset
    ) -> Dict:
        """Run Optuna hyperparameter optimization."""

        print(f"🔍 Starting Optuna optimization for {model_type}...")

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
            study_name=f"{model_type}_optimization",
        )

        # Run optimization
        study.optimize(
            lambda trial: self.optuna_objective(trial, model_type, dataset),
            n_trials=self.config.num_optuna_trials,
            timeout=None,
        )

        print("✅ Optimization completed!")
        print(f"Best F1 score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study,
        }

    def train_final_model(
        self,
        model_type: str,
        dataset: WaferMapDataset,
        best_params: Optional[Dict] = None,
        run_name: Optional[str] = None,
    ) -> Dict:
        """Train final model with best parameters."""

        print(f"🎯 Training final {model_type} model...")

        # Create model with best parameters
        model = self.create_model(model_type, best_params, dataset)

        # Create data loaders
        batch_size = (
            best_params.get("batch_size", self.config.batch_size)
            if best_params
            else self.config.batch_size
        )
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset,
            batch_size=batch_size,
            use_weighted_sampling=self.config.class_balancing,
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
            ModelCheckpoint(
                dirpath=f"checkpoints/07-complete-integration/{model_type}",
                filename=f"{model_type}-best-{{epoch:02d}}-{{val_f1:.3f}}",
                monitor="val_f1",
                mode="max",
                save_top_k=1,
            ),
        ]

        # Setup logger
        logger = None
        if self.config.enable_mlflow:
            logger = MLFlowLogger(
                experiment_name=self.config.experiment_name,
                run_name=run_name or f"{model_type}_final",
            )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator="auto",
            log_every_n_steps=10,
        )

        # Start MLflow run
        if self.config.enable_mlflow:
            with mlflow.start_run(run_name=run_name or f"{model_type}_final"):
                # Log parameters
                mlflow.log_params(
                    {
                        "model_type": model_type,
                        "batch_size": batch_size,
                        "max_epochs": self.config.max_epochs,
                        "use_pretrained": self.config.use_pretrained,
                        "data_augmentation": self.config.data_augmentation,
                        **(best_params or {}),
                    }
                )

                # Train model
                trainer.fit(model, train_loader, val_loader)

                # Test model
                test_results = trainer.test(model, test_loader)

                # Log model
                mlflow.pytorch.log_model(model, f"{model_type}_model")

                # Detailed evaluation
                evaluator = ModelEvaluator(model)
                detailed_metrics = evaluator.evaluate_on_loader(
                    test_loader, dataset.task_class_names, return_predictions=True
                )

                # Log metrics
                for key, value in detailed_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
        else:
            # Train without MLflow
            trainer.fit(model, train_loader, val_loader)
            test_results = trainer.test(model, test_loader)

            evaluator = ModelEvaluator(model)
            detailed_metrics = evaluator.evaluate_on_loader(
                test_loader, dataset.task_class_names, return_predictions=True
            )

        return {
            "model": model,
            "trainer": trainer,
            "test_results": test_results,
            "detailed_metrics": detailed_metrics,
            "best_params": best_params,
        }

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete integrated ML pipeline."""

        print("🚀 Starting Complete ML Integration Pipeline")
        print("=" * 60)
        print("Configuration:")
        print(f"  Experiment: {self.config.experiment_name}")
        print("  Models: All architectures")
        print(f"  Optuna trials: {self.config.num_optuna_trials}")
        print(f"  Max epochs: {self.config.max_epochs}")
        print(f"  MLflow tracking: {self.config.enable_mlflow}")
        print("=" * 60)

        # 1. Load dataset
        print("📁 Loading and preparing dataset...")
        dataset = get_tutorial_dataset(max_samples_per_class=150, balance_classes=True)
        print(f"Dataset ready: {len(dataset)} samples, {dataset.num_classes} classes")

        # 2. Define models to compare
        model_types = [
            "lightning_cnn",
            "torchvision_resnet",
            "torchvision_efficientnet",
            "yolo_inspired",
        ]

        # 3. Run optimization and training for each model
        for model_type in model_types:
            print(f"\n{'=' * 20} {model_type.upper()} {'=' * 20}")

            try:
                # Run Optuna optimization if enabled
                best_params = None
                if self.config.enable_optuna:
                    opt_results = self.run_optuna_optimization(model_type, dataset)
                    best_params = opt_results["best_params"]

                # Train final model
                results = self.train_final_model(
                    model_type,
                    dataset,
                    best_params,
                    run_name=f"{model_type}_optimized"
                    if best_params
                    else f"{model_type}_baseline",
                )

                self.best_models[model_type] = results["model"]
                self.all_results[model_type] = results

                # Print results
                metrics = results["detailed_metrics"]
                print(f"✅ {model_type} Results:")
                print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"   F1-Score: {metrics.get('f1_weighted', 0):.4f}")

            except Exception as e:
                print(f"❌ Error training {model_type}: {e}")
                continue

        # 4. Model comparison and visualization
        print("\n🏆 FINAL COMPARISON")
        print("=" * 60)

        comparison_metrics = {}
        for model_name, results in self.all_results.items():
            metrics = results["detailed_metrics"]
            comparison_metrics[model_name] = {
                "accuracy": metrics.get("accuracy", 0),
                "f1_weighted": metrics.get("f1_weighted", 0),
                "f1_macro": metrics.get("f1_macro", 0),
            }

        # Print comparison
        for model_name, metrics in comparison_metrics.items():
            print(
                f"{model_name:20}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_weighted']:.4f}"
            )

        # 5. Create comprehensive visualizations
        if self.all_results:
            best_model_name = max(
                comparison_metrics.keys(),
                key=lambda x: comparison_metrics[x]["accuracy"],
            )
            best_results = self.all_results[best_model_name]

            print(f"\n🎨 Creating visualizations for best model: {best_model_name}")

            # Get sample predictions for visualization
            test_dataset = get_tutorial_dataset(max_samples_per_class=20)
            _, _, test_loader = create_data_loaders(test_dataset, batch_size=16)

            sample_batch = next(iter(test_loader))
            with torch.no_grad():
                best_model = best_results["model"]
                best_model.eval()
                sample_predictions = best_model(sample_batch[0]).argmax(dim=1)

            # Create comprehensive dashboard
            create_results_dashboard(
                metrics=best_results["detailed_metrics"],
                confusion_matrix=best_results["detailed_metrics"]["confusion_matrix"],
                class_names=dataset.task_class_names,
                sample_predictions=(
                    sample_batch[0].squeeze(1).numpy(),
                    sample_batch[1].numpy(),
                    sample_predictions.numpy(),
                ),
                save_dir="results/07-complete-integration",
            )

            # Model comparison visualization
            viz = WaferVisualization()
            viz.plot_model_comparison(
                comparison_metrics,
                metrics=["accuracy", "f1_weighted"],
                title="Model Architecture Comparison",
                save_path="results/07-complete-integration/model_comparison.png",
            )

        # 6. Save final results
        final_results = {
            "config": self.config.__dict__,
            "model_comparison": comparison_metrics,
            "best_model": best_model_name if self.all_results else None,
            "experiment_summary": {
                "total_models_trained": len(self.all_results),
                "mlflow_experiment_id": self.mlflow_experiment_id,
                "best_accuracy": max(m["accuracy"] for m in comparison_metrics.values())
                if comparison_metrics
                else 0,
            },
        }

        # Save to JSON
        results_file = Path("results/07-complete-integration/experiment_summary.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        print("\n✨ Complete integration pipeline finished!")
        print(f"📊 Results saved to: {results_file}")
        if self.config.enable_mlflow:
            print("🔗 View MLflow dashboard: http://localhost:5000")

        return final_results


if __name__ == "__main__":
    # Demo: Run complete integration
    config = ExperimentConfig(
        experiment_name="wafer_defect_complete_demo",
        max_epochs=20,  # Reduced for demo
        num_optuna_trials=5,  # Reduced for demo
        enable_mlflow=True,
        enable_optuna=True,
    )

    pipeline = IntegratedMLPipeline(config)
    results = pipeline.run_complete_pipeline()

    print("\n🎉 Complete integration demo finished!")
    print("This demonstrated:")
    print("✓ PyTorch Lightning: Clean training framework")
    print("✓ Optuna: Automated hyperparameter optimization")
    print("✓ MLflow: Experiment tracking and model registry")
    print("✓ Torchvision: Pre-trained model integration")
    print("✓ YOLO-inspired: Modern architecture patterns")
    print("✓ Comprehensive evaluation and visualization")
