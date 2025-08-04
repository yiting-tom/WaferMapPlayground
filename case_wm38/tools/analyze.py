# %%
#!/usr/bin/env python3
"""
Image Dataset Analysis Tool
A comprehensive tool for analyzing image classification datasets

Author: Dataset Analysis Tool
Version: 1.0
"""

import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

warnings.filterwarnings("ignore")


class ImageDatasetAnalyzer:
    """
    A comprehensive image dataset analyzer that provides statistical insights
    into image classification datasets.
    """

    def __init__(
        self, image_dir: str, labels_file: str = None, labels_dict: Dict = None
    ):
        """
        Initialize the analyzer.

        Args:
            image_dir: Path to directory containing images
            labels_file: Path to CSV file with columns ['filename', 'label'] (optional)
            labels_dict: Dictionary mapping filename to label (optional)
        """
        self.image_dir = Path(image_dir)
        self.labels = {}
        self.analysis_results = {}
        self.image_stats = []

        # Load labels
        if labels_file:
            self._load_labels_from_csv(labels_file)
        elif labels_dict:
            self.labels = labels_dict
        else:
            # Try to infer labels from directory structure
            self._infer_labels_from_structure()

        print(f"Initialized analyzer for {len(self.labels)} images")

    def _load_labels_from_csv(self, labels_file: str):
        """Load labels from CSV file."""
        df = pd.read_csv(labels_file)
        if "filename" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'filename' and 'label' columns")
        self.labels = dict(zip(df["filename"], df["label"]))

    def _infer_labels_from_structure(self):
        """Infer labels from directory structure (class_name/image.jpg)."""
        for img_path in self.image_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
            ]:
                if img_path.parent != self.image_dir:
                    # Use parent directory name as label
                    label = img_path.parent.name
                    self.labels[img_path.name] = label

    def analyze_dataset(self) -> Dict:
        """
        Perform comprehensive dataset analysis.

        Returns:
            Dictionary containing all analysis results
        """
        print("Starting comprehensive dataset analysis...")

        # 1. Basic dataset statistics
        self._analyze_basic_stats()

        # 2. Class distribution analysis
        self._analyze_class_distribution()

        # 3. Image quality analysis
        self._analyze_image_quality()

        # 4. Duplicate detection
        self._detect_duplicates()

        # 5. Statistical tests
        self._perform_statistical_tests()

        # 6. Generate visualizations
        self._create_visualizations()

        print("Analysis completed!")
        return self.analysis_results

    def _analyze_basic_stats(self):
        """Analyze basic dataset statistics."""
        print("Analyzing basic statistics...")

        total_images = len(self.labels)
        unique_classes = len(set(self.labels.values()))

        # Calculate file sizes and image dimensions
        file_sizes = []
        dimensions = []
        formats = []

        for filename in self.labels.keys():
            img_path = self._find_image_path(filename)
            if img_path and img_path.exists():
                # File size
                file_sizes.append(img_path.stat().st_size / 1024)  # KB

                # Image dimensions and format
                try:
                    with Image.open(img_path) as img:
                        dimensions.append(img.size)  # (width, height)
                        formats.append(img.format)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue

        self.analysis_results["basic_stats"] = {
            "total_images": total_images,
            "unique_classes": unique_classes,
            "avg_file_size_kb": np.mean(file_sizes) if file_sizes else 0,
            "file_size_std": np.std(file_sizes) if file_sizes else 0,
            "formats": Counter(formats),
            "avg_width": np.mean([d[0] for d in dimensions]) if dimensions else 0,
            "avg_height": np.mean([d[1] for d in dimensions]) if dimensions else 0,
            "dimension_variety": len(set(dimensions)) if dimensions else 0,
        }

    def _analyze_class_distribution(self):
        """Analyze class distribution and balance."""
        print("Analyzing class distribution...")

        class_counts = Counter(self.labels.values())
        total_samples = sum(class_counts.values())

        # Calculate balance metrics
        class_proportions = {k: v / total_samples for k, v in class_counts.items()}

        # Gini coefficient for imbalance measure
        proportions = list(class_proportions.values())
        gini = 1 - sum(p**2 for p in proportions)

        # Entropy
        entropy = -sum(p * np.log2(p) for p in proportions if p > 0)

        # Imbalance ratio (majority class / minority class)
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        self.analysis_results["class_distribution"] = {
            "class_counts": dict(class_counts),
            "class_proportions": class_proportions,
            "gini_coefficient": gini,
            "entropy": entropy,
            "imbalance_ratio": imbalance_ratio,
            "is_balanced": imbalance_ratio <= 2.0,  # Rule of thumb
        }

    def _analyze_image_quality(self):
        """Analyze image quality metrics."""
        print("Analyzing image quality...")

        brightness_scores = []
        contrast_scores = []
        sharpness_scores = []
        color_diversity = []
        aspect_ratios = []

        for filename in list(self.labels.keys())[:100]:  # Sample first 100 for speed
            img_path = self._find_image_path(filename)
            if not img_path or not img_path.exists():
                continue

            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Convert to different color spaces
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # Brightness (mean intensity)
                brightness = np.mean(gray)
                brightness_scores.append(brightness)

                # Contrast (standard deviation of intensity)
                contrast = np.std(gray)
                contrast_scores.append(contrast)

                # Sharpness (Laplacian variance)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_scores.append(sharpness)

                # Color diversity (unique colors / total pixels)
                unique_colors = len(np.unique(img.reshape(-1, img.shape[-1]), axis=0))
                total_pixels = img.shape[0] * img.shape[1]
                color_div = unique_colors / total_pixels
                color_diversity.append(color_div)

                # Aspect ratio
                h, w = img.shape[:2]
                aspect_ratio = w / h
                aspect_ratios.append(aspect_ratio)

            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
                continue

        self.analysis_results["image_quality"] = {
            "brightness": {
                "mean": np.mean(brightness_scores) if brightness_scores else 0,
                "std": np.std(brightness_scores) if brightness_scores else 0,
                "distribution": brightness_scores,
            },
            "contrast": {
                "mean": np.mean(contrast_scores) if contrast_scores else 0,
                "std": np.std(contrast_scores) if contrast_scores else 0,
                "distribution": contrast_scores,
            },
            "sharpness": {
                "mean": np.mean(sharpness_scores) if sharpness_scores else 0,
                "std": np.std(sharpness_scores) if sharpness else 0,
                "distribution": sharpness_scores,
            },
            "color_diversity": {
                "mean": np.mean(color_diversity) if color_diversity else 0,
                "std": np.std(color_diversity) if color_diversity else 0,
            },
            "aspect_ratios": {
                "mean": np.mean(aspect_ratios) if aspect_ratios else 0,
                "std": np.std(aspect_ratios) if aspect_ratios else 0,
                "distribution": aspect_ratios,
            },
        }

    def _detect_duplicates(self):
        """Detect potential duplicate images using perceptual hashing."""
        print("Detecting duplicates...")

        hashes = {}
        duplicates = []

        for filename in list(self.labels.keys())[:200]:  # Sample for speed
            img_path = self._find_image_path(filename)
            if not img_path or not img_path.exists():
                continue

            try:
                # Simple perceptual hash using image histogram
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Calculate histogram
                hist = cv2.calcHist(
                    [img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
                )
                hist_hash = hashlib.md5(hist.tobytes()).hexdigest()

                if hist_hash in hashes:
                    duplicates.append((filename, hashes[hist_hash]))
                else:
                    hashes[hist_hash] = filename

            except Exception as e:
                print(f"Error hashing {filename}: {e}")
                continue

        self.analysis_results["duplicates"] = {
            "potential_duplicates": duplicates,
            "duplicate_count": len(duplicates),
            "duplicate_percentage": len(duplicates) / len(self.labels) * 100
            if self.labels
            else 0,
        }

    def _perform_statistical_tests(self):
        """Perform statistical tests on the dataset."""
        print("Performing statistical tests...")

        # Test for normal distribution of image properties
        quality_data = self.analysis_results.get("image_quality", {})

        tests = {}

        # Shapiro-Wilk test for normality
        for metric in ["brightness", "contrast", "sharpness"]:
            if metric in quality_data and quality_data[metric]["distribution"]:
                data = quality_data[metric]["distribution"]
                if len(data) >= 3:  # Minimum for Shapiro-Wilk
                    statistic, p_value = stats.shapiro(data[:50])  # Max 50 samples
                    tests[f"{metric}_normality"] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "is_normal": p_value > 0.05,
                    }

        # Chi-square test for class distribution uniformity
        class_counts = list(
            self.analysis_results["class_distribution"]["class_counts"].values()
        )
        if len(class_counts) > 1:
            expected = [sum(class_counts) / len(class_counts)] * len(class_counts)
            chi2_stat, chi2_p = stats.chisquare(class_counts, expected)
            tests["class_uniformity"] = {
                "chi2_statistic": chi2_stat,
                "p_value": chi2_p,
                "is_uniform": chi2_p > 0.05,
            }

        self.analysis_results["statistical_tests"] = tests

    def _create_visualizations(self):
        """Create visualization plots."""
        print("Creating visualizations...")

        # Set up the plotting style
        plt.style.use("default")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Class distribution
        class_counts = self.analysis_results["class_distribution"]["class_counts"]
        axes[0, 0].bar(range(len(class_counts)), list(class_counts.values()))
        axes[0, 0].set_title("Class Distribution")
        axes[0, 0].set_xlabel("Classes")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Brightness distribution
        brightness_data = self.analysis_results["image_quality"]["brightness"][
            "distribution"
        ]
        if brightness_data:
            axes[0, 1].hist(brightness_data, bins=20, alpha=0.7)
            axes[0, 1].set_title("Brightness Distribution")
            axes[0, 1].set_xlabel("Brightness")
            axes[0, 1].set_ylabel("Frequency")

        # 3. Contrast distribution
        contrast_data = self.analysis_results["image_quality"]["contrast"][
            "distribution"
        ]
        if contrast_data:
            axes[0, 2].hist(contrast_data, bins=20, alpha=0.7, color="orange")
            axes[0, 2].set_title("Contrast Distribution")
            axes[0, 2].set_xlabel("Contrast")
            axes[0, 2].set_ylabel("Frequency")

        # 4. Sharpness distribution
        sharpness_data = self.analysis_results["image_quality"]["sharpness"][
            "distribution"
        ]
        if sharpness_data:
            axes[1, 0].hist(sharpness_data, bins=20, alpha=0.7, color="green")
            axes[1, 0].set_title("Sharpness Distribution")
            axes[1, 0].set_xlabel("Sharpness")
            axes[1, 0].set_ylabel("Frequency")

        # 5. Aspect ratio distribution
        aspect_data = self.analysis_results["image_quality"]["aspect_ratios"][
            "distribution"
        ]
        if aspect_data:
            axes[1, 1].hist(aspect_data, bins=20, alpha=0.7, color="red")
            axes[1, 1].set_title("Aspect Ratio Distribution")
            axes[1, 1].set_xlabel("Aspect Ratio")
            axes[1, 1].set_ylabel("Frequency")

        # 6. Class balance visualization
        proportions = list(
            self.analysis_results["class_distribution"]["class_proportions"].values()
        )
        axes[1, 2].pie(proportions, labels=class_counts.keys(), autopct="%1.1f%%")
        axes[1, 2].set_title("Class Proportion")

        plt.tight_layout()
        plt.savefig("dataset_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def _find_image_path(self, filename: str) -> Optional[Path]:
        """Find the full path of an image file."""
        # Try direct path first
        direct_path = self.image_dir / filename
        if direct_path.exists():
            return direct_path

        # Search in subdirectories
        for img_path in self.image_dir.rglob(filename):
            return img_path

        return None

    def generate_report(self, output_file: str = "dataset_analysis_report.txt"):
        """Generate a comprehensive text report."""
        with open(output_file, "w") as f:
            f.write("IMAGE DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Basic Statistics
            f.write("BASIC STATISTICS\n")
            f.write("-" * 20 + "\n")
            basic = self.analysis_results["basic_stats"]
            f.write(f"Total Images: {basic['total_images']}\n")
            f.write(f"Unique Classes: {basic['unique_classes']}\n")
            f.write(f"Average File Size: {basic['avg_file_size_kb']:.2f} KB\n")
            f.write(
                f"Average Dimensions: {basic['avg_width']:.0f}x{basic['avg_height']:.0f}\n"
            )
            f.write(f"Dimension Variety: {basic['dimension_variety']} unique sizes\n\n")

            # Class Distribution
            f.write("CLASS DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            dist = self.analysis_results["class_distribution"]
            f.write(f"Gini Coefficient: {dist['gini_coefficient']:.3f}\n")
            f.write(f"Entropy: {dist['entropy']:.3f}\n")
            f.write(f"Imbalance Ratio: {dist['imbalance_ratio']:.2f}\n")
            f.write(f"Is Balanced: {dist['is_balanced']}\n\n")

            # Image Quality
            f.write("IMAGE QUALITY\n")
            f.write("-" * 20 + "\n")
            quality = self.analysis_results["image_quality"]
            f.write(f"Average Brightness: {quality['brightness']['mean']:.2f}\n")
            f.write(f"Average Contrast: {quality['contrast']['mean']:.2f}\n")
            f.write(f"Average Sharpness: {quality['sharpness']['mean']:.2f}\n")
            f.write(f"Color Diversity: {quality['color_diversity']['mean']:.4f}\n\n")

            # Duplicates
            f.write("DUPLICATE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            dup = self.analysis_results["duplicates"]
            f.write(f"Potential Duplicates: {dup['duplicate_count']}\n")
            f.write(f"Duplicate Percentage: {dup['duplicate_percentage']:.2f}%\n\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            self._generate_recommendations(f)

    def _generate_recommendations(self, f):
        """Generate recommendations based on analysis results."""
        recommendations = []

        # Class balance recommendations
        if not self.analysis_results["class_distribution"]["is_balanced"]:
            recommendations.append(
                "- Dataset is imbalanced. Consider data augmentation or resampling."
            )

        # Quality recommendations
        quality = self.analysis_results["image_quality"]
        if quality["brightness"]["std"] > 50:
            recommendations.append(
                "- High brightness variation detected. Consider normalization."
            )

        if quality["contrast"]["std"] > 30:
            recommendations.append(
                "- High contrast variation detected. Consider histogram equalization."
            )

        # Duplicate recommendations
        if self.analysis_results["duplicates"]["duplicate_percentage"] > 5:
            recommendations.append(
                "- High percentage of potential duplicates. Manual review recommended."
            )

        if not recommendations:
            recommendations.append(
                "- Dataset appears to be well-balanced and of good quality."
            )

        for rec in recommendations:
            f.write(rec + "\n")

    def export_results(self, output_file: str = "analysis_results.json"):
        """Export analysis results to JSON."""

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(item) for item in d]
            else:
                return convert_numpy(d)

        cleaned_results = clean_dict(self.analysis_results)

        with open(output_file, "w") as f:
            json.dump(cleaned_results, f, indent=2)

        print(f"Results exported to {output_file}")


def main():
    """Example usage of the ImageDatasetAnalyzer."""

    # Example 1: Using directory structure (class_name/images.jpg)
    # analyzer = ImageDatasetAnalyzer('path/to/your/dataset')

    # Example 2: Using CSV labels file
    # analyzer = ImageDatasetAnalyzer('path/to/images', labels_file='labels.csv')

    # Example 3: Using labels dictionary
    # labels_dict = {'image1.jpg': 'cat', 'image2.jpg': 'dog', ...}
    # analyzer = ImageDatasetAnalyzer('path/to/images', labels_dict=labels_dict)

    print("Image Dataset Analyzer Tool")
    print("Please modify the main() function with your dataset path")
    print("\nExample usage:")
    print("analyzer = ImageDatasetAnalyzer('/path/to/dataset')")
    print("results = analyzer.analyze_dataset()")
    print("analyzer.generate_report()")
    print("analyzer.export_results()")
