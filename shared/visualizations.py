"""
Visualization utilities for wafer defect classification tutorials.

This module provides consistent and professional visualizations across all tutorials,
including data exploration, training progress, and model evaluation plots.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Set style for consistent, professional plots
plt.style.use("default")
sns.set_palette("husl")


class WaferVisualization:
    """
    Comprehensive visualization toolkit for wafer defect classification.

    Provides consistent styling and professional plots across all tutorials.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize visualization settings.

        Args:
            figsize: Default figure size
            dpi: Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi

        # Color schemes
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "accent": "#F18F01",
            "success": "#C73E1D",
            "neutral": "#7F8C8D",
        }

        # Setup matplotlib defaults
        plt.rcParams.update(
            {
                "figure.figsize": figsize,
                "figure.dpi": dpi,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "font.size": 10,
            }
        )

    def plot_sample_wafer_maps(
        self,
        wafer_maps: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        n_samples: int = 16,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot sample wafer maps with their labels (and predictions if provided).

        Args:
            wafer_maps: Array of wafer maps (N, H, W) or (N, 1, H, W)
            labels: True labels
            class_names: List of class names
            predictions: Predicted labels (optional)
            n_samples: Number of samples to display
            save_path: Path to save the plot
        """
        # Convert to numpy if needed
        if isinstance(wafer_maps, torch.Tensor):
            wafer_maps = wafer_maps.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        # Handle different tensor shapes
        if wafer_maps.ndim == 4:
            wafer_maps = wafer_maps.squeeze(1)  # Remove channel dimension

        # Randomly select samples
        n_samples = min(n_samples, len(wafer_maps))
        indices = np.random.choice(len(wafer_maps), n_samples, replace=False)

        # Calculate grid size
        cols = int(np.ceil(np.sqrt(n_samples)))
        rows = int(np.ceil(n_samples / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(n_samples):
            idx = indices[i]
            ax = axes[i]

            # Plot wafer map
            im = ax.imshow(wafer_maps[idx], cmap="viridis", aspect="equal")

            # Create title
            true_label = (
                class_names[labels[idx]]
                if labels[idx] < len(class_names)
                else f"Class_{labels[idx]}"
            )
            title = f"True: {true_label}"

            if predictions is not None:
                pred_label = (
                    class_names[predictions[idx]]
                    if predictions[idx] < len(class_names)
                    else f"Class_{predictions[idx]}"
                )
                title += f"\nPred: {pred_label}"

                # Color code based on correctness
                color = "green" if labels[idx] == predictions[idx] else "red"
                ax.set_title(title, fontsize=10, color=color)
            else:
                ax.set_title(title, fontsize=10)

            ax.axis("off")

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide extra subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Sample wafer maps saved to {save_path}")

        plt.show()

    def plot_class_distribution(
        self,
        labels: Union[np.ndarray, torch.Tensor],
        class_names: List[str],
        title: str = "Class Distribution",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot class distribution as both bar chart and pie chart.

        Args:
            labels: Array of class labels
            class_names: List of class names
            title: Plot title
            save_path: Path to save the plot
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Count classes
        unique, counts = np.unique(labels, return_counts=True)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar chart
        bars = ax1.bar(
            range(len(unique)), counts, color=sns.color_palette("husl", len(unique))
        )
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Number of Samples")
        ax1.set_title(f"{title} - Bar Chart")
        ax1.set_xticks(range(len(unique)))
        ax1.set_xticklabels(
            [class_names[i] if i < len(class_names) else f"Class_{i}" for i in unique],
            rotation=45,
        )

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{count}",
                ha="center",
                va="bottom",
            )

        # Pie chart
        labels_for_pie = [
            class_names[i] if i < len(class_names) else f"Class_{i}" for i in unique
        ]
        wedges, texts, autotexts = ax2.pie(
            counts, labels=labels_for_pie, autopct="%1.1f%%", startangle=90
        )
        ax2.set_title(f"{title} - Distribution")

        # Improve pie chart readability
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Class distribution plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot confusion matrix with professional styling.

        Args:
            cm: Confusion matrix
            class_names: List of class names
            normalize: Whether to normalize the matrix
            title: Plot title
            save_path: Path to save the plot
        """
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            fmt = "d"

        plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names))))

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.5,
        )

        plt.title(title, fontsize=16, pad=20)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot training and validation metrics over epochs.

        Args:
            history: Dictionary with lists of metric values per epoch
            title: Plot title
            save_path: Path to save the plot
        """
        metrics = list(history.keys())
        n_metrics = len(metrics)

        # Determine grid size
        cols = 2 if n_metrics > 1 else 1
        rows = int(np.ceil(n_metrics / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i] if i < len(axes) else axes[-1]

            values = history[metric]
            epochs = range(1, len(values) + 1)

            ax.plot(epochs, values, marker="o", linewidth=2, markersize=4, label=metric)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f"{metric.replace('_', ' ').title()} Over Time")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide extra subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis("off")

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Training history saved to {save_path}")

        plt.show()

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        title: str = "ROC Curves",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot ROC curves for multiclass classification.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            class_names: List of class names
            title: Plot title
            save_path: Path to save the plot
        """
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize

        # Binarize labels for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        if y_true_bin.shape[1] == 1:  # Binary case
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each class
        for i in range(len(class_names)):
            if i < y_prob.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)

                plt.plot(
                    fpr,
                    tpr,
                    linewidth=2,
                    label=f"{class_names[i]} (AUC = {roc_auc:.3f})",
                )

        # Plot random classifier line
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"ROC curves saved to {save_path}")

        plt.show()

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        title: str = "Precision-Recall Curves",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot precision-recall curves for multiclass classification.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            class_names: List of class names
            title: Plot title
            save_path: Path to save the plot
        """
        from sklearn.metrics import average_precision_score, precision_recall_curve
        from sklearn.preprocessing import label_binarize

        # Binarize labels for multiclass PR curves
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        if y_true_bin.shape[1] == 1:  # Binary case
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        plt.figure(figsize=(10, 8))

        # Plot PR curve for each class
        for i in range(len(class_names)):
            if i < y_prob.shape[1]:
                precision, recall, _ = precision_recall_curve(
                    y_true_bin[:, i], y_prob[:, i]
                )
                avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])

                plt.plot(
                    recall,
                    precision,
                    linewidth=2,
                    label=f"{class_names[i]} (AP = {avg_precision:.3f})",
                )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Precision-recall curves saved to {save_path}")

        plt.show()

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ["accuracy", "f1_weighted"],
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compare multiple models across different metrics.

        Args:
            results: Dictionary of {model_name: {metric: value}}
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save the plot
        """
        model_names = list(results.keys())
        n_metrics = len(metrics)

        # Prepare data for plotting
        data = []
        for metric in metrics:
            for model_name in model_names:
                if metric in results[model_name]:
                    data.append(
                        {
                            "Model": model_name,
                            "Metric": metric.replace("_", " ").title(),
                            "Value": results[model_name][metric],
                        }
                    )

        if not data:
            print("No data available for plotting")
            return

        import pandas as pd

        df = pd.DataFrame(data)

        plt.figure(figsize=(max(8, len(model_names) * 2), 6))

        # Create grouped bar plot
        sns.barplot(data=df, x="Model", y="Value", hue="Metric")

        plt.title(title, fontsize=16, pad=20)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, axis="y", alpha=0.3)

        # Add value labels on bars
        for container in plt.gca().containers:
            plt.gca().bar_label(container, fmt="%.3f", padding=3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Model comparison saved to {save_path}")

        plt.show()

    def plot_hyperparameter_optimization(
        self,
        study_results: List[Dict],
        title: str = "Hyperparameter Optimization Progress",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot Optuna study results showing optimization progress.

        Args:
            study_results: List of trial results from Optuna study
            title: Plot title
            save_path: Path to save the plot
        """
        if not study_results:
            print("No study results provided")
            return

        # Extract data
        trial_numbers = [i + 1 for i in range(len(study_results))]
        values = [trial["value"] for trial in study_results]

        # Calculate best values so far
        best_so_far = []
        current_best = float("inf")
        for value in values:
            if value < current_best:
                current_best = value
            best_so_far.append(current_best)

        plt.figure(figsize=(12, 8))

        # Plot individual trials
        plt.subplot(2, 1, 1)
        plt.scatter(trial_numbers, values, alpha=0.6, s=30)
        plt.plot(trial_numbers, best_so_far, "r-", linewidth=2, label="Best So Far")
        plt.xlabel("Trial Number")
        plt.ylabel("Objective Value")
        plt.title(f"{title} - Individual Trials")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot parameter importance (if available)
        plt.subplot(2, 1, 2)
        if "params" in study_results[0]:
            # Get all parameter names
            all_params = set()
            for trial in study_results:
                if "params" in trial:
                    all_params.update(trial["params"].keys())

            # Plot parameter distributions for best trials (top 20%)
            n_best = max(1, len(study_results) // 5)
            sorted_trials = sorted(study_results, key=lambda x: x["value"])[:n_best]

            param_values = {param: [] for param in all_params}
            for trial in sorted_trials:
                if "params" in trial:
                    for param, value in trial["params"].items():
                        param_values[param].append(value)

            # Plot histogram of best parameter values
            n_params = len(all_params)
            if n_params > 0:
                for i, (param, values) in enumerate(param_values.items()):
                    if values:  # Only plot if we have values
                        plt.hist(
                            values, alpha=0.7, label=param, bins=min(10, len(values))
                        )

                plt.xlabel("Parameter Value")
                plt.ylabel("Frequency")
                plt.title("Parameter Distribution (Best 20% Trials)")
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No parameter information available",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
        else:
            plt.text(
                0.5,
                0.5,
                "No parameter information available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Hyperparameter optimization plot saved to {save_path}")

        plt.show()


def create_results_dashboard(
    metrics: Dict[str, float],
    confusion_matrix: np.ndarray,
    class_names: List[str],
    training_history: Optional[Dict[str, List[float]]] = None,
    sample_predictions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    save_dir: Optional[str] = None,
) -> None:
    """
    Create a comprehensive results dashboard with all key visualizations.

    Args:
        metrics: Dictionary of calculated metrics
        confusion_matrix: Confusion matrix
        class_names: List of class names
        training_history: Optional training history
        sample_predictions: Optional tuple of (wafer_maps, true_labels, predictions)
        save_dir: Directory to save all plots
    """
    visualizer = WaferVisualization()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    print("Creating comprehensive results dashboard...")

    # 1. Confusion Matrix
    cm_path = save_dir / "confusion_matrix.png" if save_dir else None
    visualizer.plot_confusion_matrix(
        confusion_matrix,
        class_names,
        title="Model Performance - Confusion Matrix",
        save_path=cm_path,
    )

    # 2. Training History (if available)
    if training_history:
        history_path = save_dir / "training_history.png" if save_dir else None
        visualizer.plot_training_history(
            training_history, title="Training Progress", save_path=history_path
        )

    # 3. Sample Predictions (if available)
    if sample_predictions:
        wafer_maps, true_labels, predictions = sample_predictions
        samples_path = save_dir / "sample_predictions.png" if save_dir else None
        visualizer.plot_sample_wafer_maps(
            wafer_maps,
            true_labels,
            class_names,
            predictions,
            n_samples=16,
            save_path=samples_path,
        )

    # 4. Metrics Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    key_metrics = ["accuracy", "f1_weighted", "f1_macro"]
    for metric in key_metrics:
        if metric in metrics:
            print(f"{metric.replace('_', ' ').title():20}: {metrics[metric]:.4f}")

    print("=" * 60)
    print(f"Dashboard plots saved to: {save_dir}")


if __name__ == "__main__":
    # Demo: Test visualization functions
    np.random.seed(42)

    # Generate sample data for testing
    n_samples = 100
    n_classes = 9
    class_names = [
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

    # Sample wafer maps
    wafer_maps = np.random.rand(n_samples, 64, 64)
    labels = np.random.randint(0, n_classes, n_samples)
    predictions = np.random.randint(0, n_classes, n_samples)

    # Initialize visualizer
    visualizer = WaferVisualization()

    # Test sample plots
    print("Testing sample wafer maps visualization...")
    visualizer.plot_sample_wafer_maps(
        wafer_maps, labels, class_names, predictions, n_samples=9
    )

    # Test class distribution
    print("Testing class distribution plot...")
    visualizer.plot_class_distribution(labels, class_names)

    # Test confusion matrix
    print("Testing confusion matrix...")
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, predictions)
    visualizer.plot_confusion_matrix(cm, class_names)

    # Test training history
    print("Testing training history plot...")
    history = {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2],
        "val_loss": [0.9, 0.7, 0.5, 0.4, 0.3],
        "train_acc": [0.6, 0.7, 0.8, 0.85, 0.9],
        "val_acc": [0.5, 0.65, 0.75, 0.8, 0.85],
    }
    visualizer.plot_training_history(history)

    print("All visualization tests completed successfully!")
