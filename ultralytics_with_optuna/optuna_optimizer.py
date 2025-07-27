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
from classifier import WaferDefectClassifier
from config import DataConfig, OptunaConfig, PathConfig, ValidationConfig
from data_processor import WaferDataProcessor
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


class OptunaWaferOptimizer:
    """Optuna-based hyperparameter optimization for wafer defect classification"""

    def __init__(
        self,
        data_path: str = DataConfig.DEFAULT_DATA_PATH,
        study_name: str = OptunaConfig.DEFAULT_STUDY_NAME,
        storage_path: str = OptunaConfig.DEFAULT_STORAGE,
    ):
        self.data_path = data_path
        self.study_name = study_name
        self.storage_path = storage_path
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = 0.0

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load data once to avoid repeated loading
        self.data_processor = WaferDataProcessor(data_path)
        self.images, self.labels = self.data_processor.load_data()
        self.label_analysis = self.data_processor.analyze_labels(self.labels)

        # Use subset for optimization (faster iterations)
        self.use_subset_size = OptunaConfig.OPTIMIZATION_SUBSET_SIZE

    def create_study(
        self,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> optuna.Study:
        """Create or load Optuna study"""

        if sampler is None:
            sampler = TPESampler(seed=ValidationConfig.RANDOM_SEED)

        if pruner is None:
            pruner = MedianPruner(
                n_startup_trials=OptunaConfig.PRUNER_N_STARTUP_TRIALS,
                n_warmup_steps=OptunaConfig.PRUNER_N_WARMUP_STEPS,
            )

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
        """Define hyperparameter search space using configuration"""

        params = {}
        search_space = OptunaConfig.SEARCH_SPACE

        for param_name, param_config in search_space.items():
            if param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )

        return params

    def prepare_subset_data(self, trial_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare subset of data for faster optimization"""
        # Use different random state for each trial to avoid overfitting
        random_state = ValidationConfig.RANDOM_SEED + trial_num

        # Create stratified subset
        if len(self.images) > self.use_subset_size:
            return self.data_processor.create_stratified_subset(
                self.images, self.labels, self.use_subset_size, random_state
            )
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
            temp_dir = Path(f"{PathConfig.TEMP_DIR_PREFIX}{trial.number}")
            temp_dir.mkdir(exist_ok=True)

            # Create classifier with temporary dataset directory
            trial_classifier = WaferDefectClassifier(
                data_path=self.data_path,
                dataset_dir=str(temp_dir / PathConfig.DATASET_DIR_NAME),
                optimized_params=params,
            )

            # Prepare dataset
            trial_classifier.data_processor.label_to_class = (
                self.data_processor.label_to_class
            )
            trial_classifier.data_processor.class_to_label = (
                self.data_processor.class_to_label
            )
            trial_classifier.data_processor.class_names = (
                self.data_processor.class_names
            )

            # Create dataset
            trial_classifier.data_processor.create_yolo_dataset(
                images_subset, labels_subset, trial_classifier.dataset_dir
            )

            # Train model with trial parameters
            training_results = trial_classifier.train_model(
                epochs=params["epochs"],
                img_size=params["img_size"],
                batch_size=params["batch_size"],
                model_size=params["model_size"],
                experiment_name=f"trial_{trial.number}",
            )

            # Get validation accuracy from training results
            val_accuracy = training_results["training_results"].results_dict.get(
                "metrics/accuracy_top1", 0.0
            )

            # Report intermediate values for pruning
            trial.report(val_accuracy, step=params["epochs"])

            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)

            self.logger.info(f"Trial {trial.number}: accuracy = {val_accuracy:.4f}")

            return val_accuracy

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            # Cleanup on error
            temp_dir = Path(f"{PathConfig.TEMP_DIR_PREFIX}{trial.number}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise optuna.TrialPruned()

    def optimize(
        self,
        n_trials: int = OptunaConfig.DEFAULT_N_TRIALS,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
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
        final_classifier = WaferDefectClassifier(
            data_path=self.data_path,
            dataset_dir=f"{save_path}_dataset",
            optimized_params=self.best_params,
        )

        # Set up data processor mappings
        final_classifier.data_processor.label_to_class = (
            self.data_processor.label_to_class
        )
        final_classifier.data_processor.class_to_label = (
            self.data_processor.class_to_label
        )
        final_classifier.data_processor.class_names = self.data_processor.class_names

        # Prepare dataset and train
        final_classifier.data_processor.create_yolo_dataset(
            images_data, labels_data, final_classifier.dataset_dir
        )

        # Train with best parameters
        training_results = final_classifier.train_model(
            epochs=self.best_params["epochs"],
            img_size=self.best_params["img_size"],
            batch_size=self.best_params["batch_size"],
            model_size=self.best_params["model_size"],
            experiment_name=save_path,
        )

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
            "training_results": training_results,
            "evaluation_results": eval_results,
            "best_params": self.best_params,
            "model_path": f"runs/classify/{save_path}",
        }

    def visualize_optimization(
        self, study: optuna.Study, save_path: str = PathConfig.PLOTS_DIR
    ) -> None:
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

    def get_study_summary(
        self, study: optuna.Study
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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


def main() -> None:
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
