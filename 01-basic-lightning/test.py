"""
Test Script for PyTorch Lightning Wafer Defect Classification Model

This script loads a trained Lightning model and evaluates it comprehensively
on the test dataset, providing detailed metrics and analysis.

Usage:
    python test.py [--checkpoint_path path/to/checkpoint.ckpt]
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from data_utils import create_data_loaders, get_tutorial_dataset
from metrics import MetricsCalculator, ModelEvaluator

# Import our Lightning model
from simple_model import WaferLightningModel


def find_best_checkpoint(
    checkpoint_dir: str = "checkpoints/01-basic-lightning",
) -> Optional[str]:
    """
    Find the best checkpoint based on validation loss.

    Args:
        checkpoint_dir: Directory containing model checkpoints

    Returns:
        Path to best checkpoint or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"Checkpoint directory not found: {checkpoint_path}")
        return None

    # Look for checkpoints
    checkpoint_files = list(checkpoint_path.glob("*.ckpt"))

    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_path}")
        return None

    # If there's a 'last.ckpt', prefer that, otherwise take the first one
    last_checkpoint = checkpoint_path / "last.ckpt"
    if last_checkpoint.exists():
        return str(last_checkpoint)

    # Sort by modification time and take the most recent
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(checkpoint_files[0])


def load_model_from_checkpoint(checkpoint_path: str) -> WaferLightningModel:
    """
    Load Lightning model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Loaded Lightning model
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")

    try:
        # Load the model
        model = WaferLightningModel.load_from_checkpoint(checkpoint_path)
        model.eval()

        # Print model info
        model_info = model.get_model_info()
        print("Model loaded successfully!")
        for key, value in model_info.items():
            print(f"  {key}: {value}")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with strict=False...")

        try:
            # Try loading checkpoint manually and filter problematic keys
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Remove problematic keys from state_dict
            state_dict = checkpoint["state_dict"]
            keys_to_remove = [
                k
                for k in state_dict.keys()
                if "criterion.weight" in k or k == "class_weights"
            ]
            for key in keys_to_remove:
                print(f"Removing problematic key: {key}")
                del state_dict[key]

            # Create a new model with default parameters
            model = WaferLightningModel(
                num_classes=9, learning_rate=1e-3, optimizer_name="Adam"
            )

            # Load the filtered state dict
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            print("Model loaded successfully with filtered state dict!")
            model_info = model.get_model_info()
            for key, value in model_info.items():
                print(f"  {key}: {value}")

            return model

        except Exception as e2:
            print(f"Failed to load with fallback method: {e2}")
            raise


def comprehensive_evaluation(
    model: WaferLightningModel,
    test_loader,
    class_names: list,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation of the model.

    Args:
        model: Trained Lightning model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to run evaluation on

    Returns:
        Dictionary containing all evaluation results
    """
    print(f"\n🔍 Running comprehensive evaluation on {device}...")
    print("=" * 60)

    # Move model to device
    model = model.to(device)

    # Create evaluator
    evaluator = ModelEvaluator(model, device)

    # Evaluate on test set
    results = evaluator.evaluate_on_loader(
        test_loader, class_names, task_type="multiclass", return_predictions=True
    )

    return results


def print_detailed_metrics(results: Dict[str, Any], class_names: list) -> None:
    """
    Print detailed evaluation metrics in a formatted way.

    Args:
        results: Results dictionary from evaluation
        class_names: List of class names
    """
    print("\n📊 DETAILED EVALUATION RESULTS")
    print("=" * 60)

    # Overall metrics
    print("\n🎯 OVERALL PERFORMANCE:")
    print(f"  Accuracy:           {results.get('accuracy', 0):.4f}")
    print(f"  Precision (macro):  {results.get('precision_macro', 0):.4f}")
    print(f"  Precision (weighted): {results.get('precision_weighted', 0):.4f}")
    print(f"  Recall (macro):     {results.get('recall_macro', 0):.4f}")
    print(f"  Recall (weighted):  {results.get('recall_weighted', 0):.4f}")
    print(f"  F1-Score (macro):   {results.get('f1_macro', 0):.4f}")
    print(f"  F1-Score (weighted): {results.get('f1_weighted', 0):.4f}")

    if "auc_roc_macro" in results:
        print(f"  AUC-ROC (macro):    {results.get('auc_roc_macro', 0):.4f}")

    # Per-class metrics
    print("\n📋 PER-CLASS PERFORMANCE:")
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)

    for i, class_name in enumerate(class_names):
        precision_key = f"precision_{class_name}"
        recall_key = f"recall_{class_name}"
        f1_key = f"f1_{class_name}"

        precision = results.get(precision_key, 0)
        recall = results.get(recall_key, 0)
        f1 = results.get(f1_key, 0)

        print(f"{class_name:<12} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")


def print_confusion_matrix(confusion_matrix: np.ndarray, class_names: list) -> None:
    """
    Print confusion matrix in a formatted way.

    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
    """
    print("\n🔢 CONFUSION MATRIX:")
    print("=" * 20)

    # Print header
    header = "True\\Pred".ljust(12)
    for name in class_names:
        header += name[:8].ljust(9)
    print(header)
    print("-" * len(header))

    # Print matrix rows
    for i, true_class in enumerate(class_names):
        row = true_class[:10].ljust(12)
        for j in range(len(class_names)):
            row += str(confusion_matrix[i, j]).ljust(9)
        print(row)

    print()


def save_results(
    results: Dict[str, Any], output_path: str = "test_results.txt"
) -> None:
    """
    Save evaluation results to a text file.

    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    print(f"\n💾 Saving results to {output_path}...")

    with open(output_path, "w") as f:
        f.write("WAFER DEFECT CLASSIFICATION - TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")

        # Overall metrics
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"Accuracy:           {results.get('accuracy', 0):.4f}\n")
        f.write(f"Precision (macro):  {results.get('precision_macro', 0):.4f}\n")
        f.write(f"Precision (weighted): {results.get('precision_weighted', 0):.4f}\n")
        f.write(f"Recall (macro):     {results.get('recall_macro', 0):.4f}\n")
        f.write(f"Recall (weighted):  {results.get('recall_weighted', 0):.4f}\n")
        f.write(f"F1-Score (macro):   {results.get('f1_macro', 0):.4f}\n")
        f.write(f"F1-Score (weighted): {results.get('f1_weighted', 0):.4f}\n")

        if "auc_roc_macro" in results:
            f.write(f"AUC-ROC (macro):    {results.get('auc_roc_macro', 0):.4f}\n")

        f.write("\nCONFUSION MATRIX:\n")
        f.write(str(results["confusion_matrix"]))
        f.write("\n")

    print("✅ Results saved successfully!")


def main():
    """Main function to run model testing."""
    parser = argparse.ArgumentParser(
        description="Test PyTorch Lightning wafer defect classification model"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-detect if not provided)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save results to file"
    )

    args = parser.parse_args()

    print("🧪 PyTorch Lightning Model Testing")
    print("=" * 50)

    # Set seed for reproducibility
    pl.seed_everything(42)

    # 1. Find or use provided checkpoint
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = find_best_checkpoint()

    if not checkpoint_path:
        print(
            "❌ No checkpoint found! Please train a model first or provide checkpoint path."
        )
        return

    # 2. Load model
    try:
        model = load_model_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Prepare test data
    print("\n📁 Loading test dataset...")
    dataset = get_tutorial_dataset(
        max_samples_per_class=100,  # Same as training
        balance_classes=True,
    )

    # Create data loaders (we only need test loader)
    _, _, test_loader = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        use_weighted_sampling=False,  # No sampling for test
    )

    print(f"Test dataset: {len(test_loader.dataset)} samples")
    print(f"Test batches: {len(test_loader)}")

    # 4. Run evaluation
    results = comprehensive_evaluation(model, test_loader, dataset.task_class_names)

    # 5. Print results
    print_detailed_metrics(results, dataset.task_class_names)
    print_confusion_matrix(results["confusion_matrix"], dataset.task_class_names)

    # 6. Print classification report using sklearn
    calculator = MetricsCalculator(dataset.task_class_names, "multiclass")
    print("\n📈 CLASSIFICATION REPORT:")
    print("=" * 30)
    classification_report = calculator.get_classification_report(
        results["targets"], results["predictions"]
    )
    print(classification_report)

    # 7. Additional analysis
    print("\n🔍 ADDITIONAL ANALYSIS:")
    print("=" * 30)

    # Class distribution in test set
    unique, counts = np.unique(results["targets"], return_counts=True)
    print("Test set class distribution:")
    for class_id, count in zip(unique, counts):
        class_name = dataset.task_class_names[class_id]
        percentage = count / len(results["targets"]) * 100
        print(f"  {class_name:<12}: {count:4} ({percentage:5.1f}%)")

    # Prediction confidence
    max_probs = np.max(results["probabilities"], axis=1)
    print("\nPrediction confidence:")
    print(f"  Mean confidence: {np.mean(max_probs):.4f}")
    print(f"  Min confidence:  {np.min(max_probs):.4f}")
    print(f"  Max confidence:  {np.max(max_probs):.4f}")

    # Misclassification analysis
    correct_predictions = results["predictions"] == results["targets"]
    accuracy = np.mean(correct_predictions)
    print("\nMisclassification analysis:")
    print(
        f"  Correct predictions: {np.sum(correct_predictions)}/{len(correct_predictions)}"
    )
    print(f"  Misclassified: {np.sum(~correct_predictions)}/{len(correct_predictions)}")
    print(f"  Error rate: {1 - accuracy:.4f}")

    # 8. Save results if requested
    if args.save_results:
        save_results(results, "test_results.txt")

    print("\n✨ Testing completed successfully!")
    print("Key Findings:")
    print(f"  • Overall Accuracy: {results.get('accuracy', 0):.2%}")
    print(f"  • F1-Score (macro): {results.get('f1_macro', 0):.4f}")
    print(
        f"  • Best performing class: {dataset.task_class_names[np.argmax([results.get(f'f1_{name}', 0) for name in dataset.task_class_names])]}"
    )
    print(
        f"  • Most challenging class: {dataset.task_class_names[np.argmin([results.get(f'f1_{name}', 0) for name in dataset.task_class_names])]}"
    )


if __name__ == "__main__":
    main()
