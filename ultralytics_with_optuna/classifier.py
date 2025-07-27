#!/usr/bin/env python3
"""
Refactored Wafer Defect Classifier using Ultralytics YOLOv8
Clean, modular implementation with separation of concerns
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from config import (
    DataConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
    ValidationConfig,
    get_model_path,
)
from data_processor import WaferDataProcessor
from ultralytics import YOLO
from visualization import WaferVisualization, create_summary_report


class WaferDefectClassifier:
    """Clean, modular wafer defect classification using YOLOv8"""

    def __init__(
        self,
        data_path: str = DataConfig.DEFAULT_DATA_PATH,
        dataset_dir: Optional[str] = None,
        optimized_params: Optional[Dict[str, Any]] = None,
    ):
        self.data_path = data_path
        self.dataset_dir = (
            Path(dataset_dir) if dataset_dir else Path(PathConfig.DATASET_DIR_NAME)
        )
        self.optimized_params = optimized_params

        # Initialize components
        self.data_processor = WaferDataProcessor(data_path)
        self.visualizer = WaferVisualization()

        # Model and training state
        self.model: Optional[YOLO] = None
        self.is_trained = False

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, ValidationConfig.LOG_LEVEL, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def prepare_dataset(
        self,
        use_subset: bool = False,
        subset_size: int = None,
        balance_classes: bool = False,
    ) -> Dict[str, Any]:
        """Prepare dataset for training"""
        self.logger.info("Starting dataset preparation...")

        # Load and analyze data
        images, labels = self.data_processor.load_data()
        label_analysis = self.data_processor.analyze_labels(labels)

        # Create subset if requested
        if use_subset:
            subset_size = subset_size or ValidationConfig.MIN_SAMPLES_PER_CLASS
            images, labels = self.data_processor.create_stratified_subset(
                images, labels, subset_size
            )
            self.logger.info(f"Using subset of {len(images)} samples")

        # Balance classes if requested
        if balance_classes:
            from data_processor import create_balanced_subset

            images, labels = create_balanced_subset(self.data_processor, images, labels)
            self.logger.info(f"Balanced dataset to {len(images)} samples")

        # Create YOLO dataset
        self.data_processor.create_yolo_dataset(images, labels, self.dataset_dir)

        # Validate dataset structure
        if not self.data_processor.validate_dataset_structure(self.dataset_dir):
            raise RuntimeError("Dataset creation failed validation")

        self.logger.info("Dataset preparation completed successfully")
        return label_analysis

    def train_model(
        self,
        epochs: int = ModelConfig.DEFAULT_EPOCHS,
        img_size: int = ModelConfig.DEFAULT_IMG_SIZE,
        batch_size: int = ModelConfig.DEFAULT_BATCH_SIZE,
        model_size: str = ModelConfig.DEFAULT_MODEL_SIZE,
        custom_params: Optional[Dict[str, Any]] = None,
        experiment_name: str = "wafer_defect_classifier",
    ) -> Dict[str, Any]:
        """Train YOLOv8 classification model"""
        self.logger.info(f"Starting model training with {epochs} epochs...")

        # Use optimized parameters if available
        training_params = self._prepare_training_params(
            epochs, img_size, batch_size, model_size, custom_params
        )

        # Initialize model
        model_path = get_model_path(training_params["model_size"])
        self.model = YOLO(model_path)

        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {device}")

        # Prepare training arguments
        train_args = {
            "data": str(self.dataset_dir),
            "epochs": training_params["epochs"],
            "imgsz": training_params["img_size"],
            "batch": training_params["batch_size"],
            "name": experiment_name,
            "save": True,
            "plots": True,
            "device": device,
            **self._get_hyperparameters(training_params),
        }

        # Train the model
        self.logger.info("Starting training...")
        results = self.model.train(**train_args)

        self.is_trained = True
        self.logger.info("Training completed successfully!")

        return {"training_results": results, "training_params": training_params}

    def _prepare_training_params(
        self,
        epochs: int,
        img_size: int,
        batch_size: int,
        model_size: str,
        custom_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Prepare training parameters from various sources"""
        params = {
            "epochs": epochs,
            "img_size": img_size,
            "batch_size": batch_size,
            "model_size": model_size,
        }

        # Apply optimized parameters if available
        if self.optimized_params:
            self.logger.info("Applying Optuna-optimized parameters...")
            params.update(
                {
                    "epochs": self.optimized_params.get("epochs", epochs),
                    "img_size": self.optimized_params.get("img_size", img_size),
                    "batch_size": self.optimized_params.get("batch_size", batch_size),
                    "model_size": self.optimized_params.get("model_size", model_size),
                }
            )

        # Apply custom parameters override
        if custom_params:
            self.logger.info("Applying custom parameters...")
            params.update(custom_params)

        return params

    def _get_hyperparameters(self, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get hyperparameters for training"""
        hyperparams = {}

        if self.optimized_params:
            # Use optimized hyperparameters
            hyperparams.update(
                {
                    "lr0": self.optimized_params.get(
                        "lr0", TrainingConfig.DEFAULT_LEARNING_RATE
                    ),
                    "lrf": self.optimized_params.get("lrf", 0.01),
                    "momentum": self.optimized_params.get(
                        "momentum", TrainingConfig.DEFAULT_MOMENTUM
                    ),
                    "weight_decay": self.optimized_params.get(
                        "weight_decay", TrainingConfig.DEFAULT_WEIGHT_DECAY
                    ),
                    "warmup_epochs": self.optimized_params.get(
                        "warmup_epochs", TrainingConfig.DEFAULT_WARMUP_EPOCHS
                    ),
                    "warmup_momentum": self.optimized_params.get(
                        "warmup_momentum", TrainingConfig.DEFAULT_WARMUP_MOMENTUM
                    ),
                    "warmup_bias_lr": self.optimized_params.get(
                        "warmup_bias_lr", TrainingConfig.DEFAULT_WARMUP_BIAS_LR
                    ),
                    "optimizer": self.optimized_params.get(
                        "optimizer", TrainingConfig.DEFAULT_OPTIMIZER
                    ),
                }
            )

            # Add augmentation parameters
            hyperparams.update(
                {
                    "hsv_h": self.optimized_params.get(
                        "hsv_h", TrainingConfig.DEFAULT_HSV_H
                    ),
                    "hsv_s": self.optimized_params.get(
                        "hsv_s", TrainingConfig.DEFAULT_HSV_S
                    ),
                    "hsv_v": self.optimized_params.get(
                        "hsv_v", TrainingConfig.DEFAULT_HSV_V
                    ),
                    "degrees": self.optimized_params.get(
                        "degrees", TrainingConfig.DEFAULT_DEGREES
                    ),
                    "translate": self.optimized_params.get(
                        "translate", TrainingConfig.DEFAULT_TRANSLATE
                    ),
                    "scale": self.optimized_params.get(
                        "scale", TrainingConfig.DEFAULT_SCALE
                    ),
                    "shear": self.optimized_params.get(
                        "shear", TrainingConfig.DEFAULT_SHEAR
                    ),
                    "perspective": self.optimized_params.get(
                        "perspective", TrainingConfig.DEFAULT_PERSPECTIVE
                    ),
                    "fliplr": self.optimized_params.get(
                        "fliplr", TrainingConfig.DEFAULT_FLIPLR
                    ),
                    "flipud": self.optimized_params.get(
                        "flipud", TrainingConfig.DEFAULT_FLIPUD
                    ),
                    "mixup": self.optimized_params.get(
                        "mixup", TrainingConfig.DEFAULT_MIXUP
                    ),
                    "copy_paste": self.optimized_params.get(
                        "copy_paste", TrainingConfig.DEFAULT_COPY_PASTE
                    ),
                    "dropout": self.optimized_params.get(
                        "dropout", TrainingConfig.DEFAULT_DROPOUT
                    ),
                }
            )

        return hyperparams

    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the trained model"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        self.logger.info("Starting model evaluation...")

        # Validate on validation set
        val_results = self.model.val(data=str(self.dataset_dir / "data.yaml"))

        # Test on test set
        test_dir = self.dataset_dir / "test"
        test_images = []
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                test_images.extend(list(class_dir.glob("*.png")))

        if not test_images:
            self.logger.warning("No test images found. Using validation results.")
            return {
                "accuracy": val_results.top1,
                "classification_report": {},
                "confusion_matrix": [],
            }

        # Get predictions
        test_results = self.model.predict(source=test_images, save=False, verbose=False)

        # Calculate detailed metrics
        y_true, y_pred = self._extract_predictions(test_results)

        # Generate classification report
        from sklearn.metrics import classification_report

        report = classification_report(
            y_true,
            y_pred,
            target_names=self.data_processor.class_names,
            output_dict=True,
            zero_division=0,
        )

        self.logger.info("Model evaluation completed")

        return {
            "classification_report": report,
            "confusion_matrix": None,  # Will be created in visualize_results
            "accuracy": report["accuracy"],
            "y_true": y_true,
            "y_pred": y_pred,
            "class_names": self.data_processor.class_names,
        }

    def _extract_predictions(self, test_results) -> Tuple[list, list]:
        """Extract true and predicted labels from test results"""
        y_true = []
        y_pred = []

        for result in test_results:
            # Get predicted class
            pred_class = result.probs.top1
            y_pred.append(pred_class)

            # Get true class from filename/path
            image_path = result.path
            true_class_name = Path(image_path).parent.name
            true_class = self.data_processor.class_names.index(true_class_name)
            y_true.append(true_class)

        return y_true, y_pred

    def visualize_results(
        self, eval_results: Dict[str, Any], save_plots: bool = True
    ) -> None:
        """Create visualizations of results"""
        self.logger.info("Creating result visualizations...")

        plot_dir = "plots" if save_plots else None

        # Plot confusion matrix
        if eval_results["y_true"] and eval_results["y_pred"]:
            self.visualizer.plot_confusion_matrix(
                eval_results["y_true"],
                eval_results["y_pred"],
                eval_results["class_names"],
                save_path=f"{plot_dir}/confusion_matrix.png" if plot_dir else None,
            )

        # Plot class distribution
        class_distribution = self.data_processor.get_class_distribution(
            self.data_processor.labels if hasattr(self.data_processor, "labels") else []
        )
        if class_distribution:
            self.visualizer.plot_class_distribution(
                class_distribution,
                save_path=f"{plot_dir}/class_distribution.png" if plot_dir else None,
            )

        # Plot classification report heatmap
        if eval_results["classification_report"]:
            self.visualizer.create_classification_report_plot(
                eval_results["classification_report"],
                save_path=f"{plot_dir}/classification_report.png" if plot_dir else None,
            )

        self.logger.info("Visualization completed")

    def predict_sample(self, image_path: str) -> Dict[str, Any]:
        """Predict defect class for a single wafer map"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        results = self.model.predict(source=image_path, verbose=False)[0]

        pred_class = results.probs.top1
        confidence = results.probs.top1conf.item()

        return {
            "predicted_class": pred_class,
            "class_name": self.data_processor.class_names[pred_class],
            "confidence": confidence,
            "defect_pattern": self.data_processor.class_to_label[pred_class],
        }

    def run_complete_pipeline(
        self,
        epochs: int = ModelConfig.DEFAULT_EPOCHS,
        img_size: int = ModelConfig.DEFAULT_IMG_SIZE,
        use_subset: bool = False,
        subset_size: int = None,
        balance_classes: bool = False,
        visualize: bool = True,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """Run the complete classification pipeline"""
        self.logger.info("=== Starting Wafer Defect Classification Pipeline ===")

        try:
            # Prepare dataset
            label_analysis = self.prepare_dataset(
                use_subset=use_subset,
                subset_size=subset_size,
                balance_classes=balance_classes,
            )

            # Train model
            training_results = self.train_model(epochs=epochs, img_size=img_size)

            # Evaluate model
            eval_results = self.evaluate_model()

            # Visualize results
            if visualize:
                self.visualize_results(eval_results, save_plots=save_results)

            # Create summary report
            if save_results:
                summary_data = {
                    **eval_results,
                    **label_analysis,
                    "total_samples": len(self.data_processor.labels)
                    if hasattr(self.data_processor, "labels")
                    else 0,
                }
                create_summary_report(summary_data)

            self.logger.info("=== Pipeline Completed Successfully ===")
            self.logger.info(f"Final Accuracy: {eval_results['accuracy']:.4f}")

            return {
                "label_analysis": label_analysis,
                "training_results": training_results,
                "evaluation_results": eval_results,
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    @classmethod
    def from_optimized_params(
        cls,
        params_file: str,
        data_path: str = DataConfig.DEFAULT_DATA_PATH,
    ) -> "WaferDefectClassifier":
        """Create classifier instance with pre-optimized parameters"""
        with open(params_file, "r") as f:
            data = json.load(f)

        optimized_params = data["best_params"]

        classifier = cls(data_path=data_path, optimized_params=optimized_params)
        classifier.logger.info(f"Loaded optimized parameters from: {params_file}")
        classifier.logger.info(f"Optimization score: {data['best_score']:.4f}")

        return classifier

    def save_model(self, save_path: str) -> None:
        """Save the trained model"""
        if not self.is_trained or self.model is None:
            raise ValueError("No trained model to save")

        # The model is automatically saved by YOLO during training
        # We can also manually export it
        self.model.export(format="onnx", save_dir=save_path)
        self.logger.info(f"Model exported to: {save_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.model is None:
            return {"status": "No model loaded"}

        return {
            "status": "Trained" if self.is_trained else "Loaded",
            "model_path": str(self.model.model_path)
            if hasattr(self.model, "model_path")
            else "Unknown",
            "device": str(self.model.device)
            if hasattr(self.model, "device")
            else "Unknown",
            "class_names": self.data_processor.class_names
            if self.data_processor.class_names
            else [],
            "num_classes": len(self.data_processor.class_names)
            if self.data_processor.class_names
            else 0,
        }
