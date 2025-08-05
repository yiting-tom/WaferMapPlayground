# %%
import warnings
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance
from skimage import color, feature, filters
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class ImageDataDriftDetector:
    """
    Data drift detection algorithm specifically designed for image classification tasks
    Based on traditional computer vision features rather than deep learning models
    """

    def __init__(
        self,
        pca_components: int = 50,
        drift_threshold: float = 0.1,
        histogram_bins: int = 256,
    ):
        """
        Initialize the drift detector

        Args:
            pca_components: Number of PCA components after dimensionality reduction
            drift_threshold: Drift warning threshold
            histogram_bins: Number of bins for color histograms
        """
        self.pca_components = pca_components
        self.drift_threshold = drift_threshold
        self.histogram_bins = histogram_bins
        self.reference_stats = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components)
        self.is_fitted = False

    def extract_comprehensive_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive image features (color, texture, shape, etc.)

        Args:
            images: Image array with shape (n_samples, height, width, channels)

        Returns:
            Feature matrix with shape (n_samples, n_features)
        """
        features_list = []

        for img in images:
            img_features = []

            # Ensure image is in 3-channel RGB format
            if len(img.shape) == 3 and img.shape[2] == 3:
                rgb_img = img
            else:
                rgb_img = (
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if len(img.shape) == 3
                    else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                )

            # 1. Color features
            color_features = self._extract_color_features(rgb_img)
            img_features.extend(color_features)

            # 2. Texture features
            texture_features = self._extract_texture_features(rgb_img)
            img_features.extend(texture_features)

            # 3. Edge features
            edge_features = self._extract_edge_features(rgb_img)
            img_features.extend(edge_features)

            # 4. Statistical features
            statistical_features = self._extract_statistical_features(rgb_img)
            img_features.extend(statistical_features)

            features_list.append(img_features)

        return np.array(features_list)

    def _extract_color_features(self, img: np.ndarray) -> List[float]:
        """Extract color features"""
        features = []

        # RGB color histograms
        for channel in range(3):
            hist = cv2.calcHist([img], [channel], None, [self.histogram_bins], [0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize
            # Take first 20 major bins to reduce dimensionality
            features.extend(hist[:20])

        # HSV color space features
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        for channel in range(3):
            hist = cv2.calcHist([hsv_img], [channel], None, [32], [0, 256])
            hist = hist.flatten() / hist.sum()
            features.extend(hist[:10])  # Take first 10 values

        # Average color values
        features.extend([np.mean(img[:, :, i]) for i in range(3)])

        return features

    def _extract_texture_features(self, img: np.ndarray) -> List[float]:
        """Extract texture features"""
        features = []

        # Convert to grayscale
        gray_img = color.rgb2gray(img)
        gray_img = (gray_img * 255).astype(np.uint8)

        # LBP (Local Binary Pattern) features
        lbp = feature.local_binary_pattern(gray_img, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist / lbp_hist.sum()
        features.extend(lbp_hist)

        # GLCM (Gray Level Co-occurrence Matrix) features
        try:
            glcm = graycomatrix(
                gray_img,
                distances=[1],
                angles=[0, 45, 90, 135],
                levels=256,
                symmetric=True,
                normed=True,
            )

            # GLCM properties
            contrast = graycoprops(glcm, "contrast").mean()
            dissimilarity = graycoprops(glcm, "dissimilarity").mean()
            homogeneity = graycoprops(glcm, "homogeneity").mean()
            energy = graycoprops(glcm, "energy").mean()

            features.extend([contrast, dissimilarity, homogeneity, energy])
        except:
            features.extend(
                [0, 0, 0, 0]
            )  # Use default values if GLCM calculation fails

        return features

    def _extract_edge_features(self, img: np.ndarray) -> List[float]:
        """Extract edge features"""
        features = []

        # Convert to grayscale
        gray_img = color.rgb2gray(img)

        # Canny edge detection
        edges = feature.canny(gray_img, sigma=1.0)
        edge_density = np.sum(edges) / edges.size
        features.append(edge_density)

        # Sobel edge detection
        sobel_h = filters.sobel_h(gray_img)
        sobel_v = filters.sobel_v(gray_img)
        sobel_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)

        features.extend(
            [np.mean(sobel_magnitude), np.std(sobel_magnitude), np.max(sobel_magnitude)]
        )

        return features

    def _extract_statistical_features(self, img: np.ndarray) -> List[float]:
        """Extract statistical features"""
        features = []

        for channel in range(3):
            channel_data = img[:, :, channel].flatten()

            # Basic statistics
            features.extend(
                [
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.median(channel_data),
                    stats.skew(channel_data),
                    stats.kurtosis(channel_data),
                ]
            )

        # Overall image statistics
        gray_img = color.rgb2gray(img)
        features.extend(
            [np.mean(gray_img), np.std(gray_img), np.min(gray_img), np.max(gray_img)]
        )

        return features

    def fit_reference(self, reference_images: np.ndarray):
        """
        Train the detector on reference dataset

        Args:
            reference_images: Reference image array
        """
        print("Extracting reference data features...")
        reference_features = self.extract_comprehensive_features(reference_images)

        # Standardize features
        reference_features_scaled = self.scaler.fit_transform(reference_features)

        # PCA dimensionality reduction
        reference_features_pca = self.pca.fit_transform(reference_features_scaled)

        # Store reference statistics
        self.reference_stats = {
            "features_mean": np.mean(reference_features_pca, axis=0),
            "features_std": np.std(reference_features_pca, axis=0),
            "features_distribution": reference_features_pca,
            "sample_size": len(reference_images),
        }

        self.is_fitted = True
        print(
            f"Reference model trained successfully with {len(reference_images)} samples"
        )

    def detect_drift(self, new_images: np.ndarray) -> Dict:
        """
        Detect drift in new image data

        Args:
            new_images: New image data

        Returns:
            Dictionary containing drift detection results
        """
        if not self.is_fitted:
            raise ValueError("Please train reference model first using fit_reference()")

        print("Extracting new data features...")
        new_features = self.extract_comprehensive_features(new_images)

        # Transform new features using trained scaler and PCA
        new_features_scaled = self.scaler.transform(new_features)
        new_features_pca = self.pca.transform(new_features_scaled)

        # Calculate various drift metrics
        drift_results = {}

        # 1. Population Stability Index (PSI)
        psi_score = self._calculate_psi(
            self.reference_stats["features_distribution"], new_features_pca
        )
        drift_results["psi_score"] = psi_score

        # 2. Wasserstein distance
        wasserstein_scores = []
        for dim in range(
            min(10, new_features_pca.shape[1])
        ):  # Only calculate first 10 principal components
            wd = wasserstein_distance(
                self.reference_stats["features_distribution"][:, dim],
                new_features_pca[:, dim],
            )
            wasserstein_scores.append(wd)

        drift_results["wasserstein_distance"] = np.mean(wasserstein_scores)

        # 3. KS test
        ks_scores = []
        for dim in range(min(10, new_features_pca.shape[1])):
            ks_stat, ks_p_value = stats.ks_2samp(
                self.reference_stats["features_distribution"][:, dim],
                new_features_pca[:, dim],
            )
            ks_scores.append(ks_stat)

        drift_results["ks_statistic"] = np.mean(ks_scores)

        # 4. Statistical differences
        new_mean = np.mean(new_features_pca, axis=0)
        new_std = np.std(new_features_pca, axis=0)

        mean_drift = np.mean(np.abs(new_mean - self.reference_stats["features_mean"]))
        std_drift = np.mean(np.abs(new_std - self.reference_stats["features_std"]))

        drift_results["mean_drift"] = mean_drift
        drift_results["std_drift"] = std_drift

        # 5. Overall drift score
        overall_drift_score = (
            0.3 * min(psi_score / 0.25, 1.0)  # PSI normalized to 0-1
            + 0.3 * min(drift_results["wasserstein_distance"] / 0.5, 1.0)
            + 0.2 * min(drift_results["ks_statistic"] / 0.5, 1.0)
            + 0.2 * min(mean_drift / 0.5, 1.0)
        )

        drift_results["overall_drift_score"] = overall_drift_score
        drift_results["drift_detected"] = overall_drift_score > self.drift_threshold
        drift_results["sample_size"] = len(new_images)

        return drift_results

    def _calculate_psi(
        self, reference_data: np.ndarray, new_data: np.ndarray, bins: int = 10
    ) -> float:
        """Calculate Population Stability Index"""
        psi_values = []

        for dim in range(
            min(10, reference_data.shape[1])
        ):  # Only calculate first 10 dimensions
            ref_dim = reference_data[:, dim]
            new_dim = new_data[:, dim]

            # Create bins
            min_val = min(ref_dim.min(), new_dim.min())
            max_val = max(ref_dim.max(), new_dim.max())
            bin_edges = np.linspace(min_val, max_val, bins + 1)

            # Calculate proportions
            ref_counts, _ = np.histogram(ref_dim, bins=bin_edges)
            new_counts, _ = np.histogram(new_dim, bins=bin_edges)

            ref_props = ref_counts / len(ref_dim)
            new_props = new_counts / len(new_dim)

            # Avoid division by zero
            ref_props = np.where(ref_props == 0, 0.0001, ref_props)
            new_props = np.where(new_props == 0, 0.0001, new_props)

            # Calculate PSI
            psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
            psi_values.append(psi)

        return np.mean(psi_values)

    def generate_drift_report(self, drift_results: Dict) -> str:
        """Generate drift detection report"""
        report = "=== Image Data Drift Detection Report ===\n\n"

        report += f"Detection Result: {'⚠️  Drift Detected' if drift_results['drift_detected'] else '✅ No Significant Drift'}\n"
        report += f"Sample Size: {drift_results['sample_size']}\n"
        report += f"Overall Drift Score: {drift_results['overall_drift_score']:.4f}\n"
        report += f"Drift Threshold: {self.drift_threshold}\n\n"

        report += "Detailed Metrics:\n"
        report += f"  • PSI Score: {drift_results['psi_score']:.4f}\n"
        report += (
            f"  • Wasserstein Distance: {drift_results['wasserstein_distance']:.4f}\n"
        )
        report += f"  • KS Statistic: {drift_results['ks_statistic']:.4f}\n"
        report += f"  • Mean Drift: {drift_results['mean_drift']:.4f}\n"
        report += f"  • Standard Deviation Drift: {drift_results['std_drift']:.4f}\n\n"

        # Drift severity interpretation
        score = drift_results["overall_drift_score"]
        if score < 0.1:
            severity = "No Drift"
        elif score < 0.25:
            severity = "Mild Drift"
        elif score < 0.5:
            severity = "Moderate Drift"
        else:
            severity = "Severe Drift"

        report += f"Drift Severity: {severity}\n"

        # Recommendations
        if drift_results["drift_detected"]:
            report += "\nRecommended Actions:\n"
            report += "  1. Check if there are changes in data collection process\n"
            report += "  2. Analyze factors like image quality, lighting, angles\n"
            report += "  3. Consider retraining or fine-tuning the model\n"
            report += "  4. Add data augmentation or preprocessing steps\n"

        return report

    def visualize_drift(
        self,
        drift_results: Dict,
        reference_features: np.ndarray = None,
        new_features: np.ndarray = None,
        save_path: str = None,
    ):
        """Visualize drift detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Image Data Drift Analysis Visualization", fontsize=16)

        # 1. Drift metrics radar chart
        ax1 = axes[0, 0]
        metrics = ["PSI", "Wasserstein", "KS Test", "Mean Shift", "Std Shift"]
        values = [
            min(drift_results["psi_score"] / 0.25, 1.0),
            min(drift_results["wasserstein_distance"] / 0.5, 1.0),
            min(drift_results["ks_statistic"] / 0.5, 1.0),
            min(drift_results["mean_drift"] / 0.5, 1.0),
            min(drift_results["std_drift"] / 0.5, 1.0),
        ]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        values += values[:1]  # Close radar chart
        angles = np.concatenate((angles, [angles[0]]))

        ax1 = plt.subplot(2, 2, 1, projection="polar")
        ax1.plot(angles, values, "o-", linewidth=2)
        ax1.fill(angles, values, alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 1)
        ax1.set_title("Drift Metrics Radar Chart")

        # 2. Overall drift score
        ax2 = axes[0, 1]
        colors = ["green" if not drift_results["drift_detected"] else "red"]
        ax2.bar(
            ["Overall Drift Score"],
            [drift_results["overall_drift_score"]],
            color=colors,
        )
        ax2.axhline(
            y=self.drift_threshold, color="orange", linestyle="--", label="Threshold"
        )
        ax2.set_ylabel("Drift Score")
        ax2.set_title("Overall Drift Score")
        ax2.legend()

        # 3. Individual metrics comparison
        ax3 = axes[1, 0]
        metric_names = ["PSI", "Wasserstein", "KS", "Mean Drift", "Std Drift"]
        metric_values = [
            drift_results["psi_score"],
            drift_results["wasserstein_distance"],
            drift_results["ks_statistic"],
            drift_results["mean_drift"],
            drift_results["std_drift"],
        ]

        bars = ax3.bar(metric_names, metric_values)
        ax3.set_ylabel("Metric Value")
        ax3.set_title("Individual Drift Metrics")
        ax3.tick_params(axis="x", rotation=45)

        # Set colors based on values
        for i, bar in enumerate(bars):
            if metric_values[i] > 0.1:
                bar.set_color("red")
            elif metric_values[i] > 0.05:
                bar.set_color("orange")
            else:
                bar.set_color("green")

        # 4. Sample size information
        ax4 = axes[1, 1]
        sample_info = [
            self.reference_stats["sample_size"],
            drift_results["sample_size"],
        ]
        ax4.bar(
            ["Reference Data", "New Data"], sample_info, color=["blue", "lightblue"]
        )
        ax4.set_ylabel("Sample Count")
        ax4.set_title("Dataset Size Comparison")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {save_path}")

        plt.show()


# Usage example
def example_usage():
    """Usage example"""

    # Create mock data
    def create_mock_images(n_samples, image_size=(64, 64), shift_type="none"):
        """Create mock image data"""
        images = []
        for i in range(n_samples):
            if shift_type == "none":
                # Normal images
                img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            elif shift_type == "brightness":
                # Brightness shift
                img = np.random.randint(50, 255, (*image_size, 3), dtype=np.uint8)
            elif shift_type == "contrast":
                # Contrast shift
                base_img = np.random.randint(100, 155, (*image_size, 3), dtype=np.uint8)
                img = np.clip(base_img * 1.5, 0, 255).astype(np.uint8)
            else:
                img = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)

            images.append(img)
        return np.array(images)

    # Initialize detector
    detector = ImageDataDriftDetector(pca_components=30, drift_threshold=0.15)

    # Create reference data
    print("Creating reference data...")
    reference_images = create_mock_images(100, shift_type="none")

    # Train detector
    detector.fit_reference(reference_images)

    # Test no-drift data
    print("\nTesting no-drift data...")
    normal_images = create_mock_images(50, shift_type="none")
    normal_results = detector.detect_drift(normal_images)
    print(detector.generate_drift_report(normal_results))

    # Test drift data
    print("\nTesting brightness drift data...")
    drift_images = create_mock_images(50, shift_type="brightness")
    drift_results = detector.detect_drift(drift_images)
    print(detector.generate_drift_report(drift_results))

    # Visualize results
    detector.visualize_drift(drift_results)


if __name__ == "__main__":
    example_usage()

# %%
