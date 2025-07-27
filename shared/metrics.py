"""
Metrics utilities for wafer defect classification tutorials.

This module provides consistent evaluation metrics and reporting across all tutorials,
ensuring standardized performance measurement and comparison.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


class MetricsCalculator:
    """
    Comprehensive metrics calculator for classification tasks.

    Supports both binary and multiclass classification with detailed
    per-class and macro/micro averaged metrics.
    """

    def __init__(self, class_names: List[str], task_type: str = "multiclass"):
        """
        Initialize metrics calculator.

        Args:
            class_names: List of class names for reporting
            task_type: 'binary' or 'multiclass'
        """
        self.class_names = class_names
        self.task_type = task_type
        self.num_classes = len(class_names)

    def calculate_all_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive set of classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional, for AUC metrics)

        Returns:
            Dictionary containing all calculated metrics
        """
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()

        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1 (macro and weighted averages)
        if self.task_type == "binary":
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

            # AUC metrics for binary classification
            if y_prob is not None:
                if y_prob.ndim > 1:
                    y_prob = y_prob[:, 1]  # Use positive class probability
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
                metrics["auc_pr"] = average_precision_score(y_true, y_prob)
        else:
            # Multiclass metrics
            metrics["precision_macro"] = precision_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            metrics["precision_weighted"] = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics["recall_macro"] = recall_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            metrics["recall_weighted"] = recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics["f1_macro"] = f1_score(
                y_true, y_pred, average="macro", zero_division=0
            )
            metrics["f1_weighted"] = f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

            # AUC metrics for multiclass
            if y_prob is not None and len(np.unique(y_true)) > 1:
                try:
                    # One-vs-rest AUC
                    y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                    if y_true_bin.shape[1] == 1:  # Binary case
                        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

                    metrics["auc_roc_macro"] = roc_auc_score(
                        y_true_bin, y_prob, average="macro", multi_class="ovr"
                    )
                    metrics["auc_roc_weighted"] = roc_auc_score(
                        y_true_bin, y_prob, average="weighted", multi_class="ovr"
                    )
                except ValueError:
                    # Handle cases where not all classes are present
                    pass

        # Per-class metrics
        per_class_metrics = self.calculate_per_class_metrics(y_true, y_pred)
        metrics.update(per_class_metrics)

        return metrics

    def calculate_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate per-class precision, recall, and F1 scores."""
        per_class = {}

        # Get per-class scores
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Store with class names
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                per_class[f"precision_{class_name}"] = precision_per_class[i]
                per_class[f"recall_{class_name}"] = recall_per_class[i]
                per_class[f"f1_{class_name}"] = f1_per_class[i]

        return per_class

    def get_confusion_matrix(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        return confusion_matrix(y_true, y_pred)

    def get_classification_report(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
    ) -> str:
        """Generate detailed classification report."""
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

        return classification_report(
            y_true, y_pred, target_names=self.class_names, zero_division=0
        )

    def print_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """Print a formatted summary of metrics."""
        print("\n" + "=" * 50)
        print("METRICS SUMMARY")
        print("=" * 50)

        # Main metrics
        print(f"Accuracy:     {metrics.get('accuracy', 0):.4f}")

        if self.task_type == "binary":
            print(f"Precision:    {metrics.get('precision', 0):.4f}")
            print(f"Recall:       {metrics.get('recall', 0):.4f}")
            print(f"F1-Score:     {metrics.get('f1', 0):.4f}")
            if "auc_roc" in metrics:
                print(f"AUC-ROC:      {metrics.get('auc_roc', 0):.4f}")
                print(f"AUC-PR:       {metrics.get('auc_pr', 0):.4f}")
        else:
            print(f"F1 (macro):   {metrics.get('f1_macro', 0):.4f}")
            print(f"F1 (weighted): {metrics.get('f1_weighted', 0):.4f}")
            if "auc_roc_macro" in metrics:
                print(f"AUC-ROC (macro): {metrics.get('auc_roc_macro', 0):.4f}")

        print("=" * 50)


class TorchMetrics:
    """
    PyTorch-native metrics for use during training.

    Provides efficient computation of metrics on GPU tensors
    without moving data to CPU.
    """

    @staticmethod
    def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate accuracy from logits or predictions."""
        if y_pred.dim() > 1:
            y_pred = y_pred.argmax(dim=1)
        return (y_pred == y_true).float().mean()

    @staticmethod
    def top_k_accuracy(
        y_pred: torch.Tensor, y_true: torch.Tensor, k: int = 3
    ) -> torch.Tensor:
        """Calculate top-k accuracy."""
        if y_pred.dim() == 1:
            return TorchMetrics.accuracy(y_pred, y_true)

        _, top_k_pred = y_pred.topk(k, dim=1)
        y_true_expanded = y_true.unsqueeze(1).expand_as(top_k_pred)
        return (top_k_pred == y_true_expanded).any(dim=1).float().mean()

    @staticmethod
    def precision_recall_f1(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        num_classes: int,
        average: str = "macro",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate precision, recall, and F1 score.

        Args:
            y_pred: Predicted logits or class indices
            y_true: True class indices
            num_classes: Number of classes
            average: 'macro', 'micro', or 'none'

        Returns:
            Tuple of (precision, recall, f1)
        """
        if y_pred.dim() > 1:
            y_pred = y_pred.argmax(dim=1)

        # Create confusion matrix
        cm = torch.zeros(num_classes, num_classes, device=y_pred.device)
        indices = num_classes * y_true + y_pred
        cm += torch.bincount(indices, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )

        # Calculate metrics
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if average == "macro":
            return precision.mean(), recall.mean(), f1.mean()
        elif average == "micro":
            tp_sum = tp.sum()
            fp_sum = fp.sum()
            fn_sum = fn.sum()
            precision_micro = tp_sum / (tp_sum + fp_sum + 1e-8)
            recall_micro = tp_sum / (tp_sum + fn_sum + 1e-8)
            f1_micro = (
                2
                * precision_micro
                * recall_micro
                / (precision_micro + recall_micro + 1e-8)
            )
            return precision_micro, recall_micro, f1_micro
        else:  # none
            return precision, recall, f1


class ModelEvaluator:
    """
    Comprehensive model evaluation helper.

    Provides easy-to-use evaluation functions that work with PyTorch models
    and data loaders to compute comprehensive metrics.
    """

    def __init__(
        self, model, device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator.

        Args:
            model: PyTorch model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device

    def evaluate_on_loader(
        self,
        data_loader,
        class_names: List[str],
        task_type: str = "multiclass",
        return_predictions: bool = False,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Evaluate model on a data loader.

        Args:
            data_loader: PyTorch DataLoader
            class_names: List of class names
            task_type: 'binary' or 'multiclass'
            return_predictions: Whether to return predictions and probabilities

        Returns:
            Dictionary containing metrics and optionally predictions
        """
        self.model.eval()

        all_predictions = []
        all_probabilities = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_x)

                # Get predictions
                if task_type == "binary" and outputs.dim() == 1:
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > 0.5).long()
                    probabilities = torch.stack(
                        [1 - probabilities, probabilities], dim=1
                    )
                else:
                    probabilities = F.softmax(outputs, dim=1)
                    predictions = outputs.argmax(dim=1)

                # Store results
                all_predictions.append(predictions.cpu())
                all_probabilities.append(probabilities.cpu())
                all_targets.append(batch_y.cpu())

        # Concatenate all results
        predictions = torch.cat(all_predictions).numpy()
        probabilities = torch.cat(all_probabilities).numpy()
        targets = torch.cat(all_targets).numpy()

        # Calculate metrics
        calculator = MetricsCalculator(class_names, task_type)
        metrics = calculator.calculate_all_metrics(targets, predictions, probabilities)

        # Add additional useful information
        metrics["confusion_matrix"] = calculator.get_confusion_matrix(
            targets, predictions
        )

        if return_predictions:
            metrics["predictions"] = predictions
            metrics["probabilities"] = probabilities
            metrics["targets"] = targets

        return metrics

    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        data_loader,
        class_names: List[str],
        task_type: str = "multiclass",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same dataset.

        Args:
            models: Dictionary of {model_name: model}
            data_loader: PyTorch DataLoader for evaluation
            class_names: List of class names
            task_type: 'binary' or 'multiclass'

        Returns:
            Dictionary of {model_name: metrics}
        """
        results = {}

        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            evaluator = ModelEvaluator(model, self.device)
            metrics = evaluator.evaluate_on_loader(data_loader, class_names, task_type)
            results[model_name] = metrics

            # Print summary
            calculator = MetricsCalculator(class_names, task_type)
            calculator.print_metrics_summary(metrics)

        return results


# Utility functions
def calculate_class_imbalance_ratio(labels: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Calculate class imbalance ratio (max count / min count).

    Args:
        labels: Array of class labels

    Returns:
        Imbalance ratio (1.0 = perfectly balanced)
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    unique, counts = np.unique(labels, return_counts=True)
    if len(counts) == 0:
        return 1.0

    return counts.max() / counts.min()


def compute_confidence_intervals(
    metric_values: List[float], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence intervals for a metric across multiple runs.

    Args:
        metric_values: List of metric values from different runs
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    metric_values = np.array(metric_values)
    mean_val = np.mean(metric_values)

    # Use t-distribution for small samples
    from scipy import stats

    n = len(metric_values)
    std_err = stats.sem(metric_values)
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)

    margin_error = t_val * std_err

    return mean_val, mean_val - margin_error, mean_val + margin_error


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence intervals for a metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_func: Function that takes (y_true, y_pred) and returns metric
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Dictionary with mean, lower_bound, upper_bound
    """
    n_samples = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Calculate metric
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)
    mean_score = np.mean(bootstrap_scores)

    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = 100 - alpha / 2

    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)

    return {
        "mean": mean_score,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "std": np.std(bootstrap_scores),
    }


if __name__ == "__main__":
    # Demo: Test metrics calculation
    np.random.seed(42)

    # Generate sample data
    n_samples = 1000
    n_classes = 9

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize

    # Test metrics calculator
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

    calculator = MetricsCalculator(class_names, "multiclass")
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_prob)

    # Print results
    calculator.print_metrics_summary(metrics)

    # Test confusion matrix
    cm = calculator.get_confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix Shape: {cm.shape}")

    # Test classification report
    report = calculator.get_classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)

    # Test imbalance ratio
    imbalance = calculate_class_imbalance_ratio(y_true)
    print(f"\nClass Imbalance Ratio: {imbalance:.2f}")

    print("\nMetrics module test completed successfully!")
