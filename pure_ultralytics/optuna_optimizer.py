#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization for Wafer Defect Classification
Automatically finds optimal hyperparameters for YOLOv8 training
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
from main import WaferDefectClassifier
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from ultralytics import YOLO


class OptunaWaferOptimizer:
    """Optuna-based hyperparameter optimization for wafer defect classification"""

    def __init__(
        self,
        data_path: str = "../data/MixedWM38/Wafer_Map_Datasets.npz",
        study_name: str = "wafer_defect_optimization",
        storage_path: str = "sqlite:///optuna_studies.db",
    ):
        self.data_path = data_path
        self.study_name = study_name
        self.storage_path = storage_path
        self.best_params = None
        self.best_score = 0.0

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load data once to avoid repeated loading
        self.classifier = WaferDefectClassifier(data_path)
        self.images, self.labels = self.classifier.load_data()
        self.label_analysis = self.classifier.analyze_labels(self.labels)

        # Use subset for optimization (faster iterations)
        self.use_subset_size = 5000  # Configurable subset size

    def create_study(
        self,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        """Create or load Optuna study"""

        if sampler is None:
            sampler = TPESampler(seed=42)

        if pruner is None:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_path,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        return study

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space"""

        # Model architecture
        model_size = trial.suggest_categorical("model_size", ["n", "s", "m"])

        # Training parameters
        epochs = trial.suggest_int("epochs", 20, 100)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        img_size = trial.suggest_categorical("img_size", [128, 160, 192, 224, 256])

        # Learning rate parameters
        lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)
        lrf = trial.suggest_float("lrf", 0.001, 0.1, log=True)
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # Warmup parameters
        warmup_epochs = trial.suggest_float("warmup_epochs", 1.0, 5.0)
        warmup_momentum = trial.suggest_float("warmup_momentum", 0.5, 0.95)
        warmup_bias_lr = trial.suggest_float("warmup_bias_lr", 0.01, 0.2)

        # Augmentation parameters
        hsv_h = trial.suggest_float("hsv_h", 0.0, 0.1)
        hsv_s = trial.suggest_float("hsv_s", 0.0, 0.9)
        hsv_v = trial.suggest_float("hsv_v", 0.0, 0.9)
        degrees = trial.suggest_float("degrees", 0.0, 45.0)
        translate = trial.suggest_float("translate", 0.0, 0.2)
        scale = trial.suggest_float("scale", 0.1, 0.9)
        shear = trial.suggest_float("shear", 0.0, 10.0)
        perspective = trial.suggest_float("perspective", 0.0, 0.001)
        fliplr = trial.suggest_float("fliplr", 0.0, 0.8)
        flipud = trial.suggest_float("flipud", 0.0, 0.5)
        mixup = trial.suggest_float("mixup", 0.0, 0.3)
        copy_paste = trial.suggest_float("copy_paste", 0.0, 0.3)

        # Regularization
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        # Optimizer
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])

        return {
            "model_size": model_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "lr0": lr0,
            "lrf": lrf,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "warmup_momentum": warmup_momentum,
            "warmup_bias_lr": warmup_bias_lr,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "degrees": degrees,
            "translate": translate,
            "scale": scale,
            "shear": shear,
            "perspective": perspective,
            "fliplr": fliplr,
            "flipud": flipud,
            "mixup": mixup,
            "copy_paste": copy_paste,
            "dropout": dropout,
            "optimizer": optimizer,
        }

    def prepare_subset_data(self, trial_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare subset of data for faster optimization"""
        from sklearn.model_selection import train_test_split

        # Convert labels to class indices for stratification
        class_indices = []
        for label in self.labels:
            label_tuple = tuple(label)
            class_idx = self.classifier.label_to_class[label_tuple]
            class_indices.append(class_idx)

        # Use different random state for each trial to avoid overfitting
        random_state = 42 + trial_num

        # Stratified sampling
        if len(self.images) > self.use_subset_size:
            indices = list(range(len(self.images)))
            subset_indices, _ = train_test_split(
                indices,
                test_size=1 - self.use_subset_size / len(self.images),
                stratify=class_indices,
                random_state=random_state,
            )
            return self.images[subset_indices], self.labels[subset_indices]
        else:
            return self.images, self.labels

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""
        try:
            # Get hyperparameters
            params = self.suggest_hyperparameters(trial)

            # Prepare data subset
            images_subset, labels_subset = self.prepare_subset_data(trial.number)

            # Create temporary directory for this trial
            temp_dir = Path(f"temp_trial_{trial.number}")
            temp_dir.mkdir(exist_ok=True)

            # Create classifier with temporary dataset directory
            trial_classifier = WaferDefectClassifier(self.data_path)
            trial_classifier.dataset_dir = temp_dir / "dataset"
            trial_classifier.label_to_class = self.classifier.label_to_class
            trial_classifier.class_to_label = self.classifier.class_to_label
            trial_classifier.class_names = self.classifier.class_names

            # Create dataset
            trial_classifier.create_yolo_dataset(images_subset, labels_subset)

            # Train model with trial parameters
            model_name = f"yolov8{params['model_size']}-cls.pt"

            # Custom training arguments
            train_args = {
                "data": str(trial_classifier.dataset_dir),
                "epochs": params["epochs"],
                "imgsz": params["img_size"],
                "batch": params["batch_size"],
                "name": f"trial_{trial.number}",
                "save": False,  # Don't save intermediate models
                "plots": False,  # Disable plots for speed
                "verbose": False,  # Reduce logging
                "device": "cuda" if trial_classifier.model else "cpu",
                # Learning rate parameters
                "lr0": params["lr0"],
                "lrf": params["lrf"],
                "momentum": params["momentum"],
                "weight_decay": params["weight_decay"],
                # Warmup parameters
                "warmup_epochs": params["warmup_epochs"],
                "warmup_momentum": params["warmup_momentum"],
                "warmup_bias_lr": params["warmup_bias_lr"],
                # Augmentation parameters
                "hsv_h": params["hsv_h"],
                "hsv_s": params["hsv_s"],
                "hsv_v": params["hsv_v"],
                "degrees": params["degrees"],
                "translate": params["translate"],
                "scale": params["scale"],
                "shear": params["shear"],
                "perspective": params["perspective"],
                "fliplr": params["fliplr"],
                "flipud": params["flipud"],
                "mixup": params["mixup"],
                "copy_paste": params["copy_paste"],
                # Regularization
                "dropout": params["dropout"],
                # Optimizer
                "optimizer": params["optimizer"],
            }

            # Initialize and train model
            import torch
            from ultralytics import YOLO

            device = "cuda" if torch.cuda.is_available() else "cpu"
            train_args["device"] = device

            model = YOLO(model_name)
            results = model.train(**train_args)

            # Get validation accuracy
            val_accuracy = results.results_dict.get("metrics/accuracy_top1", 0.0)

            # Report intermediate values for pruning
            trial.report(val_accuracy, step=params["epochs"])

            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)

            self.logger.info(f"Trial {trial.number}: accuracy = {val_accuracy:.4f}")

            return val_accuracy

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            # Cleanup on error
            temp_dir = Path(f"temp_trial_{trial.number}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise optuna.TrialPruned()

    def optimize(
        self, n_trials: int = 50, timeout: Optional[int] = None, n_jobs: int = 1
    ) -> optuna.Study:
        """Run hyperparameter optimization"""

        study = self.create_study()

        self.logger.info(f"Starting optimization with {n_trials} trials")
        self.logger.info(f"Using subset size: {self.use_subset_size}")

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )

        self.best_params = study.best_params
        self.best_score = study.best_value

        self.logger.info("Optimization completed!")
        self.logger.info(f"Best accuracy: {self.best_score:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")

        return study

    def train_best_model(
        self, use_full_data: bool = True, save_path: str = "best_model"
    ) -> Dict[str, Any]:
        """Train final model with best hyperparameters"""

        if self.best_params is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")

        self.logger.info("Training final model with best hyperparameters...")

        # Use full dataset or subset
        if use_full_data:
            images_data, labels_data = self.images, self.labels
            self.logger.info("Using full dataset for final training")
        else:
            images_data, labels_data = self.prepare_subset_data(0)
            self.logger.info(f"Using subset of {len(images_data)} samples")

        # Create final classifier
        final_classifier = WaferDefectClassifier(self.data_path)
        final_classifier.dataset_dir = Path(f"{save_path}_dataset")
        final_classifier.label_to_class = self.classifier.label_to_class
        final_classifier.class_to_label = self.classifier.class_to_label
        final_classifier.class_names = self.classifier.class_names

        # Create dataset
        final_classifier.create_yolo_dataset(images_data, labels_data)

        # Train with best parameters
        model_name = f"yolov8{self.best_params['model_size']}-cls.pt"
        final_classifier.model = YOLO(model_name)

        # Apply best hyperparameters
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        train_args = {
            "data": str(final_classifier.dataset_dir),
            "epochs": self.best_params["epochs"],
            "imgsz": self.best_params["img_size"],
            "batch": self.best_params["batch_size"],
            "name": save_path,
            "save": True,
            "plots": True,
            "device": device,
            "lr0": self.best_params["lr0"],
            "lrf": self.best_params["lrf"],
            "momentum": self.best_params["momentum"],
            "weight_decay": self.best_params["weight_decay"],
            "warmup_epochs": self.best_params["warmup_epochs"],
            "warmup_momentum": self.best_params["warmup_momentum"],
            "warmup_bias_lr": self.best_params["warmup_bias_lr"],
            "hsv_h": self.best_params["hsv_h"],
            "hsv_s": self.best_params["hsv_s"],
            "hsv_v": self.best_params["hsv_v"],
            "degrees": self.best_params["degrees"],
            "translate": self.best_params["translate"],
            "scale": self.best_params["scale"],
            "shear": self.best_params["shear"],
            "perspective": self.best_params["perspective"],
            "fliplr": self.best_params["fliplr"],
            "flipud": self.best_params["flipud"],
            "mixup": self.best_params["mixup"],
            "copy_paste": self.best_params["copy_paste"],
            "dropout": self.best_params["dropout"],
            "optimizer": self.best_params["optimizer"],
        }

        results = final_classifier.model.train(**train_args)

        # Evaluate final model
        eval_results = final_classifier.evaluate_model()

        # Save best parameters
        best_params_file = f"{save_path}_best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(
                {
                    "best_params": self.best_params,
                    "best_score": self.best_score,
                    "final_accuracy": eval_results["accuracy"],
                    "optimization_date": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        self.logger.info(f"Final model accuracy: {eval_results['accuracy']:.4f}")
        self.logger.info(f"Best parameters saved to: {best_params_file}")

        return {
            "training_results": results,
            "evaluation_results": eval_results,
            "best_params": self.best_params,
            "model_path": f"runs/classify/{save_path}",
        }

    def visualize_optimization(
        self, study: optuna.Study, save_path: str = "optimization_plots"
    ):
        """Create optimization visualization plots"""

        # Create plots directory
        plots_dir = Path(save_path)
        plots_dir.mkdir(exist_ok=True)

        try:
            # Optimization history
            fig1 = vis.plot_optimization_history(study)
            fig1.write_html(plots_dir / "optimization_history.html")

            # Parameter importances
            fig2 = vis.plot_param_importances(study)
            fig2.write_html(plots_dir / "param_importances.html")

            # Parameter relationships
            fig3 = vis.plot_parallel_coordinate(study)
            fig3.write_html(plots_dir / "parallel_coordinate.html")

            # Slice plot
            fig4 = vis.plot_slice(study)
            fig4.write_html(plots_dir / "slice_plot.html")

            self.logger.info(f"Visualization plots saved to: {plots_dir}")

        except Exception as e:
            self.logger.warning(f"Could not create all plots: {str(e)}")

    def get_study_summary(self, study: optuna.Study) -> pd.DataFrame:
        """Get summary of optimization results"""

        trials_df = study.trials_dataframe()

        # Summary statistics
        summary = {
            "total_trials": len(study.trials),
            "completed_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
        }

        return trials_df, summary


def main():
    """Example usage of Optuna optimization"""

    # Initialize optimizer
    optimizer = OptunaWaferOptimizer(
        study_name="wafer_defect_optuna", storage_path="sqlite:///wafer_optuna.db"
    )

    # Run optimization
    study = optimizer.optimize(n_trials=20)  # Adjust number of trials as needed

    # Visualize results
    optimizer.visualize_optimization(study)

    # Get summary
    trials_df, summary = optimizer.get_study_summary(study)
    print("\nOptimization Summary:")
    print(f"Total trials: {summary['total_trials']}")
    print(f"Best accuracy: {summary['best_value']:.4f}")
    print(f"Best parameters: {summary['best_params']}")

    # Train best model
    final_results = optimizer.train_best_model(
        use_full_data=False
    )  # Use subset for demo
    print(
        f"\nFinal model accuracy: {final_results['evaluation_results']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
