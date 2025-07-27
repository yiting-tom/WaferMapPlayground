#!/usr/bin/env python3
"""
Visualization utilities for Wafer Defect Classification
Handles plotting, visualization, and result reporting
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import ValidationConfig
from sklearn.metrics import confusion_matrix


class WaferVisualization:
    """Handles visualization and plotting for wafer classification results"""

    def __init__(self, figure_size: Tuple[int, int] = ValidationConfig.FIGURE_SIZE):
        self.figure_size = figure_size
        self.dpi = ValidationConfig.DPI
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        save_path: Optional[str] = None,
        normalize: bool = False,
        title: str = "Confusion Matrix - Wafer Defect Classification",
    ) -> None:
        """Plot confusion matrix with proper formatting"""

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            title += " (Normalized)"
        else:
            fmt = "d"

        # Only show classes that have samples in test set
        present_classes = np.unique(y_true + y_pred)
        present_class_names = [class_names[i] for i in present_classes]
        cm_subset = cm[np.ix_(present_classes, present_classes)]

        plt.figure(figsize=self.figure_size)
        sns.heatmap(
            cm_subset,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=present_class_names,
            yticklabels=present_class_names,
            cbar_kws={"label": "Count" if not normalize else "Proportion"},
        )

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("True", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_class_distribution(
        self,
        class_distribution: Dict[str, int],
        save_path: Optional[str] = None,
        title: str = "Class Distribution",
    ) -> None:
        """Plot class distribution as a bar chart"""

        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())

        plt.figure(figsize=(15, 6))
        bars = plt.bar(range(len(classes)), counts, alpha=0.8)

        # Color bars based on count (gradient)
        max_count = max(counts)
        colors = plt.cm.viridis(np.array(counts) / max_count)
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Class", fontsize=12)
        plt.ylabel("Number of Samples", fontsize=12)
        plt.xticks(range(len(classes)), classes, rotation=45, ha="right")

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_count * 0.01,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Class distribution plot saved to: {save_path}")

        plt.show()

    def plot_sample_wafer_maps(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        label_to_class: Dict[Tuple, int],
        n_samples: int = 9,
        save_path: Optional[str] = None,
        title: str = "Sample Wafer Maps with Defect Classifications",
    ) -> None:
        """Plot sample wafer map images with their labels"""

        n_rows = int(np.sqrt(n_samples))
        n_cols = int(np.ceil(n_samples / n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figure_size)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Flatten axes for easier indexing
        if n_samples > 1:
            axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
        else:
            axes_flat = [axes]

        for i, ax in enumerate(axes_flat):
            if i < len(images) and i < n_samples:
                # Display image
                ax.imshow(images[i], cmap=ValidationConfig.CMAP)

                # Find class name
                label_tuple = tuple(labels[i])
                class_idx = label_to_class.get(label_tuple, -1)

                if class_idx != -1 and class_idx < len(class_names):
                    class_name = class_names[class_idx]
                else:
                    class_name = "Unknown"

                ax.set_title(f"Sample {i + 1}: {class_name}", fontsize=10)
                ax.set_xlabel(f"Pattern: {labels[i]}", fontsize=8)
            else:
                ax.set_visible(False)

            ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Sample wafer maps saved to: {save_path}")

        plt.show()

    def plot_training_metrics(
        self,
        metrics_history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        title: str = "Training Metrics",
    ) -> None:
        """Plot training metrics over epochs"""

        n_metrics = len(metrics_history)
        n_cols = min(2, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        if n_metrics == 1:
            axes = [axes]
        elif n_metrics > 1 and n_rows == 1:
            axes = axes if hasattr(axes, "__len__") else [axes]
        else:
            axes = axes.flat

        for i, (metric_name, values) in enumerate(metrics_history.items()):
            if i < len(axes):
                axes[i].plot(values, marker="o", linewidth=2, markersize=4)
                axes[i].set_title(f"{metric_name.capitalize()}", fontweight="bold")
                axes[i].set_xlabel("Epoch")
                axes[i].set_ylabel(metric_name.capitalize())
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(metrics_history), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Training metrics plot saved to: {save_path}")

        plt.show()

    def plot_model_comparison(
        self,
        comparison_data: Dict[str, float],
        metric_name: str = "Accuracy",
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """Plot comparison between different models or configurations"""

        if title is None:
            title = f"Model Comparison - {metric_name}"

        models = list(comparison_data.keys())
        values = list(comparison_data.values())

        plt.figure(figsize=(10, 6))

        # Create color map
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = plt.bar(
            models, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1
        )

        plt.title(title, fontsize=14, fontweight="bold")
        plt.ylabel(metric_name, fontsize=12)
        plt.xlabel("Model/Configuration", fontsize=12)

        # Add value labels on bars
        max_value = max(values)
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_value * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        plt.ylim(0, max_value * 1.1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Model comparison plot saved to: {save_path}")

        plt.show()

    def create_classification_report_plot(
        self,
        classification_report: Dict[str, Any],
        save_path: Optional[str] = None,
        title: str = "Classification Report Heatmap",
    ) -> None:
        """Create a heatmap visualization of the classification report"""

        # Extract metrics for each class
        metrics_data = []
        class_names = []

        for class_name, metrics in classification_report.items():
            if isinstance(metrics, dict) and class_name not in [
                "accuracy",
                "macro avg",
                "weighted avg",
            ]:
                class_names.append(class_name)
                metrics_data.append(
                    [
                        metrics.get("precision", 0),
                        metrics.get("recall", 0),
                        metrics.get("f1-score", 0),
                    ]
                )

        if not metrics_data:
            print("No class-level metrics found in classification report")
            return

        # Create DataFrame for heatmap
        df = pd.DataFrame(
            metrics_data, index=class_names, columns=["Precision", "Recall", "F1-Score"]
        )

        plt.figure(figsize=self.figure_size)
        sns.heatmap(
            df,
            annot=True,
            cmap="RdYlBu_r",
            center=0.5,
            fmt=".3f",
            cbar_kws={"label": "Score"},
        )

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Metrics", fontsize=12)
        plt.ylabel("Classes", fontsize=12)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Classification report plot saved to: {save_path}")

        plt.show()

    def plot_dataset_splits_distribution(
        self,
        dataset_stats: Dict[str, Dict[str, int]],
        save_path: Optional[str] = None,
        title: str = "Dataset Splits Distribution",
    ) -> None:
        """Plot distribution of samples across train/val/test splits"""

        splits = list(dataset_stats.keys())

        # Get all unique classes across splits
        all_classes = set()
        for split_data in dataset_stats.values():
            all_classes.update(split_data.keys())
        all_classes.discard("total")  # Remove total count if present
        all_classes = sorted(list(all_classes))

        # Prepare data for stacked bar chart
        split_data = {split: [] for split in splits}

        for class_name in all_classes:
            for split in splits:
                count = dataset_stats[split].get(class_name, 0)
                split_data[split].append(count)

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(15, 8))

        x = np.arange(len(all_classes))
        width = 0.6
        bottom = np.zeros(len(all_classes))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

        for i, (split, counts) in enumerate(split_data.items()):
            ax.bar(
                x,
                counts,
                width,
                label=split.capitalize(),
                bottom=bottom,
                color=colors[i % len(colors)],
                alpha=0.8,
            )
            bottom += np.array(counts)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Classes", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Dataset splits distribution plot saved to: {save_path}")

        plt.show()

    def save_all_plots(
        self, results: Dict[str, Any], output_dir: str = "plots"
    ) -> None:
        """Save all visualization plots to specified directory"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Saving all plots to: {output_path}")

        # Save confusion matrix if available
        if (
            "confusion_matrix" in results
            and "y_true" in results
            and "y_pred" in results
        ):
            self.plot_confusion_matrix(
                results["y_true"],
                results["y_pred"],
                results.get("class_names", []),
                save_path=str(output_path / "confusion_matrix.png"),
                title="Confusion Matrix",
            )

        # Save class distribution if available
        if "class_distribution" in results:
            self.plot_class_distribution(
                results["class_distribution"],
                save_path=str(output_path / "class_distribution.png"),
            )

        # Save classification report if available
        if "classification_report" in results:
            self.create_classification_report_plot(
                results["classification_report"],
                save_path=str(output_path / "classification_report_heatmap.png"),
            )

        print(f"All plots saved to: {output_path}")


def create_summary_report(
    results: Dict[str, Any], output_file: str = "classification_summary.txt"
) -> None:
    """Create a text summary report of classification results"""

    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("WAFER DEFECT CLASSIFICATION - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Basic metrics
        if "accuracy" in results:
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n\n")

        # Dataset information
        if "num_classes" in results:
            f.write(f"Number of Classes: {results['num_classes']}\n")

        if "total_samples" in results:
            f.write(f"Total Samples: {results['total_samples']}\n\n")

        # Class distribution
        if "class_distribution" in results:
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            total_samples = sum(results["class_distribution"].values())

            for class_name, count in results["class_distribution"].items():
                percentage = (count / total_samples) * 100
                f.write(f"{class_name:<20}: {count:>6} ({percentage:>5.1f}%)\n")
            f.write("\n")

        # Performance metrics
        if "classification_report" in results:
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 30 + "\n")

            report = results["classification_report"]
            if "weighted avg" in report:
                weighted = report["weighted avg"]
                f.write(f"Weighted Precision: {weighted.get('precision', 0):.4f}\n")
                f.write(f"Weighted Recall:    {weighted.get('recall', 0):.4f}\n")
                f.write(f"Weighted F1-Score:  {weighted.get('f1-score', 0):.4f}\n\n")

            if "macro avg" in report:
                macro = report["macro avg"]
                f.write(f"Macro Precision:    {macro.get('precision', 0):.4f}\n")
                f.write(f"Macro Recall:       {macro.get('recall', 0):.4f}\n")
                f.write(f"Macro F1-Score:     {macro.get('f1-score', 0):.4f}\n\n")

        f.write("=" * 60 + "\n")
        f.write("Report generated automatically by WaferVisualization\n")

    print(f"Summary report saved to: {output_file}")
