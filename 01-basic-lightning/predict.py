"""
Prediction Script for PyTorch Lightning Wafer Defect Classification Model

This script loads a trained Lightning model and makes predictions on input data,
saving the results with detailed output including predictions, probabilities, and confidence.

Usage:
    python predict.py [--checkpoint_path path/to/checkpoint.ckpt] [--input_data path/to/data]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# Add shared utilities to path
sys.path.append(str(Path(__file__).parent.parent / "shared"))
from data_utils import WaferMapDataset, get_tutorial_dataset

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


def prepare_data_for_prediction(
    data_source: Optional[str] = None, batch_size: int = 32
) -> Tuple[DataLoader, List[str]]:
    """
    Prepare data for prediction.

    Args:
        data_source: Path to custom data or None for test dataset
        batch_size: Batch size for prediction

    Returns:
        Tuple of (data_loader, class_names)
    """
    if data_source and Path(data_source).exists():
        print(f"Loading custom data from: {data_source}")
        # Load custom dataset
        dataset = WaferMapDataset(
            data_source, task_type="multiclass", balance_classes=False
        )

        # Create data loader for all data
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

        class_names = dataset.task_class_names

    else:
        print("Using tutorial test dataset for prediction...")
        # Use test dataset
        dataset = get_tutorial_dataset(
            max_samples_per_class=100,
            balance_classes=True,
        )

        # Create data loaders and use test set
        from data_utils import create_data_loaders

        _, _, data_loader = create_data_loaders(
            dataset,
            batch_size=batch_size,
            use_weighted_sampling=False,
        )

        class_names = dataset.task_class_names

    print(
        f"Data prepared: {len(data_loader.dataset)} samples, {len(data_loader)} batches"
    )
    return data_loader, class_names


def make_predictions(
    model: WaferLightningModel,
    data_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, np.ndarray]:
    """
    Make predictions on the provided data.

    Args:
        model: Trained Lightning model
        data_loader: Data loader for prediction
        device: Device to run prediction on

    Returns:
        Dictionary containing predictions, probabilities, and targets
    """
    print(f"\n🔮 Making predictions on {device}...")
    print("=" * 40)

    # Move model to device
    model = model.to(device)

    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_logits = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            logits = model(batch_x)

            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    # Concatenate all results
    results = {
        "predictions": np.concatenate(all_predictions),
        "probabilities": np.concatenate(all_probabilities),
        "targets": np.concatenate(all_targets),
        "logits": np.concatenate(all_logits),
    }

    print(f"✅ Predictions completed for {len(results['predictions'])} samples")
    return results


def analyze_predictions(
    results: Dict[str, np.ndarray], class_names: List[str]
) -> Dict[str, Any]:
    """
    Analyze prediction results and compute statistics.

    Args:
        results: Prediction results dictionary
        class_names: List of class names

    Returns:
        Dictionary containing analysis results
    """
    print("\n📊 Analyzing predictions...")

    predictions = results["predictions"]
    probabilities = results["probabilities"]
    targets = results["targets"]

    # Confidence scores (max probability for each prediction)
    confidence_scores = np.max(probabilities, axis=1)

    # Prediction distribution
    pred_unique, pred_counts = np.unique(predictions, return_counts=True)
    prediction_distribution = {
        class_names[class_id]: count
        for class_id, count in zip(pred_unique, pred_counts)
    }

    # Confidence statistics
    confidence_stats = {
        "mean": float(np.mean(confidence_scores)),
        "std": float(np.std(confidence_scores)),
        "min": float(np.min(confidence_scores)),
        "max": float(np.max(confidence_scores)),
        "median": float(np.median(confidence_scores)),
    }

    # High/Low confidence predictions
    high_confidence_threshold = 0.9
    low_confidence_threshold = 0.6

    high_confidence_count = np.sum(confidence_scores >= high_confidence_threshold)
    low_confidence_count = np.sum(confidence_scores <= low_confidence_threshold)

    analysis = {
        "total_samples": int(len(predictions)),
        "prediction_distribution": {
            k: int(v) for k, v in prediction_distribution.items()
        },
        "confidence_stats": confidence_stats,
        "high_confidence_count": int(high_confidence_count),
        "low_confidence_count": int(low_confidence_count),
        "high_confidence_ratio": float(high_confidence_count / len(predictions)),
        "low_confidence_ratio": float(low_confidence_count / len(predictions)),
    }

    return analysis


def create_prediction_dataframe(
    results: Dict[str, np.ndarray], class_names: List[str]
) -> pd.DataFrame:
    """
    Create a pandas DataFrame with prediction results.

    Args:
        results: Prediction results dictionary
        class_names: List of class names

    Returns:
        DataFrame containing all prediction information
    """
    predictions = results["predictions"]
    probabilities = results["probabilities"]
    targets = results["targets"]

    # Create base DataFrame
    df_data = {
        "sample_id": range(len(predictions)),
        "true_label": [class_names[label] for label in targets],
        "predicted_label": [class_names[pred] for pred in predictions],
        "confidence": np.max(probabilities, axis=1),
        "is_correct": predictions == targets,
    }

    # Add probability columns for each class
    for i, class_name in enumerate(class_names):
        df_data[f"prob_{class_name}"] = probabilities[:, i]

    df = pd.DataFrame(df_data)
    return df


def save_predictions(
    results: Dict[str, np.ndarray],
    analysis: Dict[str, Any],
    class_names: List[str],
    output_dir: str = "prediction_results",
) -> None:
    """
    Save prediction results to files.

    Args:
        results: Prediction results dictionary
        analysis: Analysis results dictionary
        class_names: List of class names
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\n💾 Saving prediction results to {output_path}...")

    # 1. Save detailed predictions as CSV
    df = create_prediction_dataframe(results, class_names)
    csv_path = output_path / "predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Detailed predictions saved to {csv_path}")

    # 2. Save summary analysis as JSON
    json_path = output_path / "prediction_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  ✅ Analysis summary saved to {json_path}")

    # 3. Save raw probabilities as NPZ
    npz_path = output_path / "raw_predictions.npz"
    np.savez_compressed(
        npz_path,
        predictions=results["predictions"],
        probabilities=results["probabilities"],
        targets=results["targets"],
        logits=results["logits"],
        class_names=class_names,
    )
    print(f"  ✅ Raw predictions saved to {npz_path}")

    # 4. Save human-readable summary
    summary_path = output_path / "prediction_summary.txt"
    with open(summary_path, "w") as f:
        f.write("WAFER DEFECT CLASSIFICATION - PREDICTION RESULTS\n")
        f.write("=" * 55 + "\n\n")

        f.write(f"Total Samples: {analysis['total_samples']}\n\n")

        f.write("PREDICTION DISTRIBUTION:\n")
        for class_name, count in analysis["prediction_distribution"].items():
            percentage = count / analysis["total_samples"] * 100
            f.write(f"  {class_name:<12}: {count:5} ({percentage:5.1f}%)\n")

        f.write("\nCONFIDENCE STATISTICS:\n")
        conf_stats = analysis["confidence_stats"]
        f.write(f"  Mean:   {conf_stats['mean']:.4f}\n")
        f.write(f"  Std:    {conf_stats['std']:.4f}\n")
        f.write(f"  Min:    {conf_stats['min']:.4f}\n")
        f.write(f"  Max:    {conf_stats['max']:.4f}\n")
        f.write(f"  Median: {conf_stats['median']:.4f}\n")

        f.write(
            f"\nHigh Confidence (>0.9):  {analysis['high_confidence_count']} ({analysis['high_confidence_ratio']:.1%})\n"
        )
        f.write(
            f"Low Confidence (<=0.6): {analysis['low_confidence_count']} ({analysis['low_confidence_ratio']:.1%})\n"
        )

    print(f"  ✅ Summary report saved to {summary_path}")


def print_prediction_summary(analysis: Dict[str, Any], class_names: List[str]) -> None:
    """
    Print a summary of prediction results.

    Args:
        analysis: Analysis results dictionary
        class_names: List of class names
    """
    print("\n📈 PREDICTION SUMMARY")
    print("=" * 40)

    print(f"Total samples processed: {analysis['total_samples']}")

    print("\n🎯 Prediction Distribution:")
    for class_name, count in analysis["prediction_distribution"].items():
        percentage = count / analysis["total_samples"] * 100
        print(f"  {class_name:<12}: {count:5} ({percentage:5.1f}%)")

    print("\n🎲 Confidence Statistics:")
    conf_stats = analysis["confidence_stats"]
    print(f"  Mean confidence: {conf_stats['mean']:.4f}")
    print(f"  Std deviation:   {conf_stats['std']:.4f}")
    print(f"  Min confidence:  {conf_stats['min']:.4f}")
    print(f"  Max confidence:  {conf_stats['max']:.4f}")

    print(
        f"\n🔥 High confidence predictions (>0.9): {analysis['high_confidence_count']} ({analysis['high_confidence_ratio']:.1%})"
    )
    print(
        f"⚠️  Low confidence predictions (<=0.6): {analysis['low_confidence_count']} ({analysis['low_confidence_ratio']:.1%})"
    )


def main():
    """Main function to run model prediction."""
    parser = argparse.ArgumentParser(
        description="Make predictions with PyTorch Lightning wafer defect classification model"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to model checkpoint (auto-detect if not provided)",
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default=None,
        help="Path to input data file (.npz format, uses test set if not provided)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for prediction"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="prediction_results",
        help="Directory to save prediction results",
    )

    args = parser.parse_args()

    print("🔮 PyTorch Lightning Model Prediction")
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

    # 3. Prepare data
    try:
        data_loader, class_names = prepare_data_for_prediction(
            args.input_data, args.batch_size
        )
    except Exception as e:
        print(f"❌ Failed to prepare data: {e}")
        return

    # 4. Make predictions
    results = make_predictions(model, data_loader)

    # 5. Analyze results
    analysis = analyze_predictions(results, class_names)

    # 6. Print summary
    print_prediction_summary(analysis, class_names)

    # 7. Save results
    save_predictions(results, analysis, class_names, args.output_dir)

    print("\n✨ Prediction completed successfully!")
    print(f"📁 Results saved in: {args.output_dir}/")
    print("Generated files:")
    print("  • predictions.csv - Detailed predictions with probabilities")
    print("  • prediction_analysis.json - Summary statistics")
    print("  • raw_predictions.npz - Raw numpy arrays")
    print("  • prediction_summary.txt - Human-readable summary")


if __name__ == "__main__":
    main()
