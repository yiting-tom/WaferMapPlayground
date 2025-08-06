# %%
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import ndimage, stats
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest


class WaferMapVisualizationAnalyzer:
    """
    Enhanced wafer map dataset analyzer with comprehensive visualization capabilities.
    """

    def __init__(
        self,
        dataset_path: str = None,
        images: List[np.ndarray] = None,
        labels: List[str] = None,
        metadata: Dict = None,
    ):
        """Initialize the analyzer with dataset."""
        self.dataset_path = dataset_path
        self.images = images or []
        self.labels = labels or []
        self.metadata = metadata or {}

        # Analysis results storage
        self.analysis_results = {}
        self.quality_metrics = {}
        self.statistical_test_results = {}

        # Configuration
        self.class_mapping = {0: "background", 1: "wafer", 2: "defect"}
        self.class_colors = {0: "#2E86AB", 1: "#A23B72", 2: "#F18F01"}
        self.random_state = 42

        # Style configuration
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def load_dataset(self, path: str, file_pattern: str = "*.png") -> None:
        """Load dataset from directory."""
        from glob import glob

        from PIL import Image

        dataset_path = Path(path)
        image_files = glob(str(dataset_path / file_pattern))

        self.images = []
        self.labels = []

        for img_path in image_files:
            try:
                img = np.array(Image.open(img_path))
                if len(img.shape) == 3:
                    img = img[:, :, 0]
                self.images.append(img)
                self.labels.append(Path(img_path).stem)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

        print(f"Loaded {len(self.images)} images from {path}")

    def analyze_dataset(self) -> Dict:
        """Run complete analysis of the dataset."""
        print("Running comprehensive dataset analysis...")

        # Basic statistics
        self.basic_statistics()
        self.spatial_analysis()
        self.quality_assessment()
        self.statistical_tests()
        self.outlier_detection()

        print("Analysis complete!")
        return self.analysis_results

    def basic_statistics(self) -> Dict:
        """Calculate basic descriptive statistics."""
        if not self.images:
            raise ValueError("No images loaded. Please load dataset first.")

        stats_dict = {
            "dataset_size": len(self.images),
            "image_dimensions": [],
            "class_distributions": [],
            "total_pixels_per_class": {0: 0, 1: 0, 2: 0},
        }

        for img in self.images:
            stats_dict["image_dimensions"].append(img.shape)

            # Class distribution per image
            unique, counts = np.unique(img, return_counts=True)
            img_dist = {int(k): int(v) for k, v in zip(unique, counts)}
            stats_dict["class_distributions"].append(img_dist)

            # Accumulate total counts
            for class_val, count in img_dist.items():
                if class_val in stats_dict["total_pixels_per_class"]:
                    stats_dict["total_pixels_per_class"][class_val] += count

        # Calculate global statistics
        total_pixels = sum(stats_dict["total_pixels_per_class"].values())
        stats_dict["global_class_proportions"] = {
            k: v / total_pixels for k, v in stats_dict["total_pixels_per_class"].items()
        }

        # Shannon entropy for class diversity
        proportions = list(stats_dict["global_class_proportions"].values())
        proportions = [p for p in proportions if p > 0]
        stats_dict["shannon_entropy"] = -sum(p * np.log2(p) for p in proportions)

        # Image dimension consistency
        dims = stats_dict["image_dimensions"]
        stats_dict["consistent_dimensions"] = len(set(dims)) == 1
        stats_dict["unique_dimensions"] = list(set(dims))

        self.analysis_results["basic_statistics"] = stats_dict
        return stats_dict

    def spatial_analysis(self) -> Dict:
        """Analyze spatial distribution of defects."""
        spatial_stats = {
            "defect_locations": [],
            "defect_sizes": [],
            "defect_shapes": [],
            "spatial_clustering": [],
            "center_of_mass": [],
            "spatial_moments": [],
            "defect_density_maps": [],
        }

        for i, img in enumerate(self.images):
            defect_mask = img == 2

            if not np.any(defect_mask):
                continue

            # Connected component analysis
            labeled_defects = label(defect_mask)
            regions = regionprops(labeled_defects)

            image_defect_info = []
            for region in regions:
                defect_info = {
                    "area": region.area,
                    "centroid": region.centroid,
                    "bbox": region.bbox,
                    "eccentricity": region.eccentricity,
                    "solidity": region.solidity,
                    "aspect_ratio": region.bbox[2] / region.bbox[3]
                    if region.bbox[3] > 0
                    else 0,
                }
                image_defect_info.append(defect_info)
                spatial_stats["defect_sizes"].append(region.area)
                spatial_stats["defect_locations"].append(region.centroid)

            spatial_stats["defect_shapes"].append(image_defect_info)

            # Create density map
            density_map = ndimage.gaussian_filter(defect_mask.astype(float), sigma=10)
            spatial_stats["defect_density_maps"].append(density_map)

            # Center of mass analysis
            if np.any(defect_mask):
                com = ndimage.center_of_mass(defect_mask)
                spatial_stats["center_of_mass"].append(com)

                # Spatial moments
                defect_coords = np.where(defect_mask)
                if len(defect_coords[0]) > 0:
                    spatial_var = [np.var(defect_coords[0]), np.var(defect_coords[1])]
                    spatial_stats["spatial_moments"].append(spatial_var)

            # Spatial clustering analysis
            defect_coords = np.column_stack(np.where(defect_mask))
            if len(defect_coords) > 5:
                clustering = DBSCAN(eps=5, min_samples=3).fit(defect_coords)
                n_clusters = len(set(clustering.labels_)) - (
                    1 if -1 in clustering.labels_ else 0
                )
                spatial_stats["spatial_clustering"].append(
                    {
                        "n_clusters": n_clusters,
                        "n_noise_points": list(clustering.labels_).count(-1),
                        "clustering_labels": clustering.labels_,
                    }
                )

        # Global spatial statistics
        if spatial_stats["defect_sizes"]:
            spatial_stats["size_statistics"] = {
                "mean_size": np.mean(spatial_stats["defect_sizes"]),
                "std_size": np.std(spatial_stats["defect_sizes"]),
                "median_size": np.median(spatial_stats["defect_sizes"]),
                "size_range": [
                    np.min(spatial_stats["defect_sizes"]),
                    np.max(spatial_stats["defect_sizes"]),
                ],
            }

        self.analysis_results["spatial_analysis"] = spatial_stats
        return spatial_stats

    def quality_assessment(self) -> Dict:
        """Assess image quality metrics."""
        quality_stats = {
            "snr_values": [],
            "contrast_values": [],
            "brightness_values": [],
            "edge_density": [],
            "noise_levels": [],
            "sharpness_values": [],
        }

        for img in self.images:
            # Signal-to-noise ratio
            wafer_pixels = img[img == 1]
            background_pixels = img[img == 0]

            if len(wafer_pixels) > 0 and len(background_pixels) > 0:
                signal = np.mean(wafer_pixels)
                noise = (
                    np.std(background_pixels) if np.std(background_pixels) > 0 else 1e-6
                )
                snr = signal / noise
                quality_stats["snr_values"].append(snr)

            # RMS Contrast
            rms_contrast = np.sqrt(np.mean((img - np.mean(img)) ** 2))
            quality_stats["contrast_values"].append(rms_contrast)

            # Brightness
            quality_stats["brightness_values"].append(np.mean(img))

            # Edge density
            edges = cv2.Sobel(img.astype(np.float32), cv2.CV_64F, 1, 1, ksize=3)
            edge_density = np.sum(np.abs(edges)) / img.size
            quality_stats["edge_density"].append(edge_density)

            # Noise estimation
            try:
                noise_level = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F).var()
            except:
                # Fallback: use simple gradient-based noise estimation
                noise_level = np.var(np.diff(img.astype(np.float32), axis=0)) + np.var(
                    np.diff(img.astype(np.float32), axis=1)
                )
            quality_stats["noise_levels"].append(noise_level)

            # Sharpness (using variance of Laplacian)
            try:
                sharpness = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F).var()
            except:
                # Fallback: use gradient magnitude for sharpness
                grad_x = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
                sharpness = np.var(np.sqrt(grad_x**2 + grad_y**2))
            quality_stats["sharpness_values"].append(sharpness)

        self.quality_metrics = quality_stats
        return quality_stats

    def statistical_tests(self) -> Dict:
        """Perform statistical tests."""
        test_results = {}

        if not self.images:
            return test_results

        # Collect data for testing
        all_defect_sizes = []
        defect_locations_x = []
        defect_locations_y = []

        for img in self.images:
            defect_mask = img == 2
            if np.any(defect_mask):
                labeled = label(defect_mask)
                regions = regionprops(labeled)

                for region in regions:
                    all_defect_sizes.append(region.area)
                    defect_locations_x.append(region.centroid[1])
                    defect_locations_y.append(region.centroid[0])

        if all_defect_sizes:
            # Normality test
            try:
                shapiro_stat, shapiro_p = stats.shapiro(all_defect_sizes[:5000])
                test_results["defect_size_normality"] = {
                    "shapiro_stat": float(shapiro_stat),
                    "shapiro_p_value": float(shapiro_p),
                    "is_normal": shapiro_p > 0.05,
                }
            except Exception as e:
                test_results["defect_size_normality"] = {"error": str(e)}

            # Spatial uniformity test
            if len(defect_locations_x) > 10:
                try:
                    ks_stat_x, ks_p_x = stats.kstest(defect_locations_x, "uniform")
                    ks_stat_y, ks_p_y = stats.kstest(defect_locations_y, "uniform")

                    test_results["spatial_uniformity"] = {
                        "x_ks_stat": float(ks_stat_x),
                        "x_p_value": float(ks_p_x),
                        "y_ks_stat": float(ks_stat_y),
                        "y_p_value": float(ks_p_y),
                        "is_uniform_x": ks_p_x > 0.05,
                        "is_uniform_y": ks_p_y > 0.05,
                    }
                except Exception as e:
                    test_results["spatial_uniformity"] = {"error": str(e)}

        self.statistical_test_results = test_results
        return test_results

    def outlier_detection(self) -> Dict:
        """Detect outliers in dataset."""
        outlier_results = {"image_outliers": [], "defect_outliers": []}

        # Feature extraction
        image_features = []
        for i, img in enumerate(self.images):
            features = {
                "defect_ratio": np.sum(img == 2) / img.size,
                "wafer_ratio": np.sum(img == 1) / img.size,
                "total_defects": len(regionprops(label(img == 2))),
                "mean_intensity": np.mean(img),
                "std_intensity": np.std(img),
            }
            image_features.append(list(features.values()))

        if len(image_features) > 3:
            # Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1, random_state=self.random_state
            )
            outlier_labels = iso_forest.fit_predict(image_features)

            outlier_indices = np.where(outlier_labels == -1)[0]
            outlier_results["image_outliers"] = [int(idx) for idx in outlier_indices]

        return outlier_results

    def plot_overview_dashboard(self, figsize: Tuple[int, int] = (20, 16)) -> None:
        """Create comprehensive overview dashboard."""
        if not self.analysis_results:
            self.analyze_dataset()

        # Create figure with custom layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Main title
        fig.suptitle(
            "Wafer Map Dataset Analysis Dashboard", fontsize=20, fontweight="bold"
        )

        # 1. Class Distribution Pie Chart
        ax1 = fig.add_subplot(gs[0, 0])
        if "basic_statistics" in self.analysis_results:
            class_props = self.analysis_results["basic_statistics"][
                "global_class_proportions"
            ]
            classes = [self.class_mapping[k] for k in class_props.keys()]
            props = list(class_props.values())
            colors = [self.class_colors[k] for k in class_props.keys()]

            wedges, texts, autotexts = ax1.pie(
                props, labels=classes, autopct="%1.2f%%", colors=colors, startangle=90
            )
            ax1.set_title("Class Distribution", fontweight="bold")

        # 2. Defect Size Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if (
            "spatial_analysis" in self.analysis_results
            and self.analysis_results["spatial_analysis"]["defect_sizes"]
        ):
            defect_sizes = self.analysis_results["spatial_analysis"]["defect_sizes"]
            ax2.hist(defect_sizes, bins=30, alpha=0.7, color="coral", edgecolor="black")
            ax2.set_title("Defect Size Distribution", fontweight="bold")
            ax2.set_xlabel("Size (pixels)")
            ax2.set_ylabel("Frequency")
            ax2.axvline(
                np.mean(defect_sizes),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(defect_sizes):.1f}",
            )
            ax2.legend()

        # 3. Quality Metrics Box Plot
        ax3 = fig.add_subplot(gs[0, 2])
        if hasattr(self, "quality_metrics"):
            quality_data = []
            quality_labels = []
            for metric in ["snr_values", "contrast_values", "brightness_values"]:
                if metric in self.quality_metrics and self.quality_metrics[metric]:
                    quality_data.append(self.quality_metrics[metric])
                    quality_labels.append(metric.replace("_values", "").upper())

            if quality_data:
                ax3.boxplot(quality_data, tick_labels=quality_labels)
                ax3.set_title("Quality Metrics Distribution", fontweight="bold")
                ax3.tick_params(axis="x", rotation=45)

        # 4. Sample Wafer Map
        ax4 = fig.add_subplot(gs[0, 3])
        if self.images:
            sample_img = self.images[0]
            # Create colored version
            colored_img = np.zeros((*sample_img.shape, 3))
            for class_val, color in self.class_colors.items():
                mask = sample_img == class_val
                colored_img[mask] = matplotlib.colors.to_rgb(color)

            ax4.imshow(colored_img)
            ax4.set_title("Sample Wafer Map", fontweight="bold")
            ax4.axis("off")

        # 5. Spatial Heatmap (large plot)
        ax5 = fig.add_subplot(gs[1, :2])
        if (
            "spatial_analysis" in self.analysis_results
            and self.analysis_results["spatial_analysis"]["defect_locations"]
        ):
            locations = self.analysis_results["spatial_analysis"]["defect_locations"]
            if locations:
                y_coords = [loc[0] for loc in locations]
                x_coords = [loc[1] for loc in locations]

                # Create 2D histogram
                hist, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=50)

                im = ax5.imshow(
                    hist.T,
                    origin="lower",
                    cmap="hot",
                    interpolation="gaussian",
                    extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)],
                )
                ax5.set_title("Defect Spatial Distribution Heatmap", fontweight="bold")
                ax5.set_xlabel("X Position")
                ax5.set_ylabel("Y Position")
                plt.colorbar(im, ax=ax5, label="Defect Density")

        # 6. Quality Metrics Correlation
        ax6 = fig.add_subplot(gs[1, 2:])
        if hasattr(self, "quality_metrics"):
            quality_data = {}
            for metric in [
                "snr_values",
                "contrast_values",
                "brightness_values",
                "sharpness_values",
            ]:
                if metric in self.quality_metrics and self.quality_metrics[metric]:
                    quality_data[metric.replace("_values", "")] = self.quality_metrics[
                        metric
                    ]

            if len(quality_data) > 1:
                quality_df = pd.DataFrame(quality_data)
                corr_matrix = quality_df.corr()

                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(
                    corr_matrix,
                    mask=mask,
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    ax=ax6,
                    square=True,
                    cbar_kws={"label": "Correlation"},
                )
                ax6.set_title("Quality Metrics Correlation Matrix", fontweight="bold")

        # 7. Defect Shape Analysis
        ax7 = fig.add_subplot(gs[2, 0])
        if "spatial_analysis" in self.analysis_results:
            shapes_data = self.analysis_results["spatial_analysis"]["defect_shapes"]
            if shapes_data:
                all_eccentricity = []
                all_solidity = []
                for img_shapes in shapes_data:
                    for shape in img_shapes:
                        all_eccentricity.append(shape["eccentricity"])
                        all_solidity.append(shape["solidity"])

                ax7.scatter(all_eccentricity, all_solidity, alpha=0.6, s=30)
                ax7.set_xlabel("Eccentricity")
                ax7.set_ylabel("Solidity")
                ax7.set_title("Defect Shape Analysis", fontweight="bold")
                ax7.grid(True, alpha=0.3)

        # 8. Statistical Test Results
        ax8 = fig.add_subplot(gs[2, 1])
        if hasattr(self, "statistical_test_results") and self.statistical_test_results:
            test_results = []
            test_names = []

            if "defect_size_normality" in self.statistical_test_results:
                result = self.statistical_test_results["defect_size_normality"]
                if "shapiro_p_value" in result:
                    test_results.append(result["shapiro_p_value"])
                    test_names.append("Size\nNormality")

            if "spatial_uniformity" in self.statistical_test_results:
                result = self.statistical_test_results["spatial_uniformity"]
                if "x_p_value" in result:
                    test_results.append(result["x_p_value"])
                    test_names.append("Spatial\nUniformity X")
                if "y_p_value" in result:
                    test_results.append(result["y_p_value"])
                    test_names.append("Spatial\nUniformity Y")

            if test_results:
                colors = ["green" if p > 0.05 else "red" for p in test_results]
                bars = ax8.bar(test_names, test_results, color=colors, alpha=0.7)
                ax8.axhline(y=0.05, color="red", linestyle="--", label="α = 0.05")
                ax8.set_ylabel("p-value")
                ax8.set_title("Statistical Tests", fontweight="bold")
                ax8.legend()
                ax8.tick_params(axis="x", rotation=45)

        # 9. Center of Mass Distribution
        ax9 = fig.add_subplot(gs[2, 2])
        if (
            "spatial_analysis" in self.analysis_results
            and self.analysis_results["spatial_analysis"]["center_of_mass"]
        ):
            centers = self.analysis_results["spatial_analysis"]["center_of_mass"]
            if centers:
                x_centers = [c[1] for c in centers]
                y_centers = [c[0] for c in centers]

                ax9.scatter(x_centers, y_centers, alpha=0.6, s=50, c="purple")
                ax9.set_xlabel("X Position")
                ax9.set_ylabel("Y Position")
                ax9.set_title("Defect Centers of Mass", fontweight="bold")
                ax9.invert_yaxis()
                ax9.grid(True, alpha=0.3)

        # 10. Dataset Health Summary
        ax10 = fig.add_subplot(gs[2, 3])
        ax10.axis("off")

        # Generate health summary
        health_text = "DATASET HEALTH SUMMARY\n" + "=" * 25 + "\n\n"

        if "basic_statistics" in self.analysis_results:
            total_images = self.analysis_results["basic_statistics"]["dataset_size"]
            health_text += f"Total Images: {total_images}\n"

            class_props = self.analysis_results["basic_statistics"][
                "global_class_proportions"
            ]
            defect_ratio = class_props.get(2, 0)

            if defect_ratio > 0.05:
                health_text += "✓ Good defect representation\n"
            elif defect_ratio > 0.01:
                health_text += "⚠ Low defect representation\n"
            else:
                health_text += "✗ Very low defect representation\n"

        if hasattr(self, "quality_metrics") and "snr_values" in self.quality_metrics:
            avg_snr = np.mean(self.quality_metrics["snr_values"])
            if avg_snr > 10:
                health_text += "✓ Good signal-to-noise ratio\n"
            elif avg_snr > 5:
                health_text += "⚠ Moderate SNR\n"
            else:
                health_text += "✗ Low SNR\n"

        if "spatial_analysis" in self.analysis_results:
            defect_sizes = self.analysis_results["spatial_analysis"]["defect_sizes"]
            if defect_sizes and len(defect_sizes) > 50:
                health_text += "✓ Sufficient defect samples\n"
            elif defect_sizes and len(defect_sizes) > 20:
                health_text += "⚠ Limited defect samples\n"
            else:
                health_text += "✗ Very few defect samples\n"

        ax10.text(
            0.05,
            0.95,
            health_text,
            transform=ax10.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
        )

        # 11. Quality Trends (bottom row)
        ax11 = fig.add_subplot(gs[3, :])
        if hasattr(self, "quality_metrics") and len(self.images) > 1:
            image_indices = list(range(len(self.images)))

            # Plot multiple quality metrics
            if "contrast_values" in self.quality_metrics:
                ax11.plot(
                    image_indices,
                    self.quality_metrics["contrast_values"],
                    "o-",
                    label="Contrast",
                    alpha=0.7,
                    linewidth=2,
                )

            if "snr_values" in self.quality_metrics:
                # Normalize SNR for plotting
                snr_norm = np.array(self.quality_metrics["snr_values"]) / max(
                    self.quality_metrics["snr_values"]
                )
                ax11.plot(
                    image_indices,
                    snr_norm,
                    "s-",
                    label="SNR (normalized)",
                    alpha=0.7,
                    linewidth=2,
                )

            if "brightness_values" in self.quality_metrics:
                # Normalize brightness
                bright_norm = np.array(self.quality_metrics["brightness_values"]) / max(
                    self.quality_metrics["brightness_values"]
                )
                ax11.plot(
                    image_indices,
                    bright_norm,
                    "^-",
                    label="Brightness (normalized)",
                    alpha=0.7,
                    linewidth=2,
                )

            ax11.set_xlabel("Image Index")
            ax11.set_ylabel("Quality Metric Value")
            ax11.set_title("Quality Metrics Trends Across Dataset", fontweight="bold")
            ax11.legend()
            ax11.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_spatial_analysis(self, figsize: Tuple[int, int] = (16, 12)) -> None:
        """Create detailed spatial analysis plots."""
        if "spatial_analysis" not in self.analysis_results:
            self.spatial_analysis()

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("Detailed Spatial Analysis", fontsize=16, fontweight="bold")

        spatial_data = self.analysis_results["spatial_analysis"]

        # 1. Defect size histogram with statistics
        if spatial_data["defect_sizes"]:
            sizes = spatial_data["defect_sizes"]
            axes[0, 0].hist(
                sizes, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
            )
            axes[0, 0].axvline(
                np.mean(sizes),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(sizes):.1f}",
            )
            axes[0, 0].axvline(
                np.median(sizes),
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Median: {np.median(sizes):.1f}",
            )
            axes[0, 0].set_xlabel("Defect Size (pixels)")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("Defect Size Distribution")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Defect locations scatter plot
        if spatial_data["defect_locations"]:
            locations = spatial_data["defect_locations"]
            y_coords = [loc[0] for loc in locations]
            x_coords = [loc[1] for loc in locations]

            axes[0, 1].scatter(x_coords, y_coords, alpha=0.6, s=30, c="red")
            axes[0, 1].set_xlabel("X Position")
            axes[0, 1].set_ylabel("Y Position")
            axes[0, 1].set_title("Defect Locations")
            axes[0, 1].invert_yaxis()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Clustering analysis visualization
        axes[0, 2].text(
            0.5,
            0.5,
            "Clustering Analysis\n\n"
            + f"Images with defects: {len(spatial_data['spatial_clustering'])}\n"
            + f"Avg clusters per image: {np.mean([c['n_clusters'] for c in spatial_data['spatial_clustering']]) if spatial_data['spatial_clustering'] else 0:.2f}",
            ha="center",
            va="center",
            transform=axes[0, 2].transAxes,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            fontsize=12,
        )
        axes[0, 2].set_title("Spatial Clustering Summary")
        axes[0, 2].axis("off")

        # 4. Shape analysis scatter plot
        if spatial_data["defect_shapes"]:
            all_eccentricity = []
            all_solidity = []
            all_aspect_ratios = []

            for img_shapes in spatial_data["defect_shapes"]:
                for shape in img_shapes:
                    all_eccentricity.append(shape["eccentricity"])
                    all_solidity.append(shape["solidity"])
                    all_aspect_ratios.append(shape["aspect_ratio"])

            scatter = axes[1, 0].scatter(
                all_eccentricity,
                all_solidity,
                c=all_aspect_ratios,
                cmap="viridis",
                alpha=0.6,
                s=50,
            )
            axes[1, 0].set_xlabel("Eccentricity")
            axes[1, 0].set_ylabel("Solidity")
            axes[1, 0].set_title("Defect Shape Analysis")
            axes[1, 0].grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=axes[1, 0])
            cbar.set_label("Aspect Ratio")

        # 5. Center of mass distribution
        if spatial_data["center_of_mass"]:
            centers = spatial_data["center_of_mass"]
            x_centers = [c[1] for c in centers]
            y_centers = [c[0] for c in centers]

            # Create 2D histogram
            axes[1, 1].hist2d(x_centers, y_centers, bins=20, cmap="hot")
            axes[1, 1].set_xlabel("X Position")
            axes[1, 1].set_ylabel("Y Position")
            axes[1, 1].set_title("Center of Mass Density")
            axes[1, 1].invert_yaxis()

        # 6. Spatial moments visualization
        if spatial_data["spatial_moments"]:
            moments = spatial_data["spatial_moments"]
            x_var = [m[1] for m in moments]  # X variance
            y_var = [m[0] for m in moments]  # Y variance

            axes[1, 2].scatter(x_var, y_var, alpha=0.7, s=50, c="purple")
            axes[1, 2].set_xlabel("X Variance")
            axes[1, 2].set_ylabel("Y Variance")
            axes[1, 2].set_title("Spatial Variance Distribution")
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_quality_analysis(self, figsize: Tuple[int, int] = (16, 10)) -> None:
        """Create detailed quality analysis plots."""
        if not hasattr(self, "quality_metrics"):
            self.quality_assessment()

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("Image Quality Analysis", fontsize=16, fontweight="bold")

        quality_data = self.quality_metrics

        # 1. SNR Distribution
        if "snr_values" in quality_data and quality_data["snr_values"]:
            snr_values = quality_data["snr_values"]
            axes[0, 0].hist(
                snr_values, bins=30, alpha=0.7, color="lightgreen", edgecolor="black"
            )
            axes[0, 0].axvline(
                np.mean(snr_values),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(snr_values):.2f}",
            )
            axes[0, 0].set_xlabel("Signal-to-Noise Ratio")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_title("SNR Distribution")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Contrast vs Brightness scatter
        if "contrast_values" in quality_data and "brightness_values" in quality_data:
            contrast = quality_data["contrast_values"]
            brightness = quality_data["brightness_values"]

            scatter = axes[0, 1].scatter(
                brightness, contrast, alpha=0.6, s=50, c="blue"
            )
            axes[0, 1].set_xlabel("Brightness")
            axes[0, 1].set_ylabel("Contrast")
            axes[0, 1].set_title("Brightness vs Contrast")
            axes[0, 1].grid(True, alpha=0.3)

            # Add correlation coefficient
            if len(brightness) > 1:
                corr_coef = np.corrcoef(brightness, contrast)[0, 1]
                axes[0, 1].text(
                    0.05,
                    0.95,
                    f"Correlation: {corr_coef:.3f}",
                    transform=axes[0, 1].transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        # 3. Sharpness distribution
        if "sharpness_values" in quality_data and quality_data["sharpness_values"]:
            sharpness = quality_data["sharpness_values"]
            axes[0, 2].hist(
                sharpness, bins=30, alpha=0.7, color="orange", edgecolor="black"
            )
            axes[0, 2].axvline(
                np.mean(sharpness),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(sharpness):.2f}",
            )
            axes[0, 2].set_xlabel("Sharpness (Laplacian Variance)")
            axes[0, 2].set_ylabel("Frequency")
            axes[0, 2].set_title("Image Sharpness Distribution")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Quality metrics box plot comparison
        quality_metrics_data = []
        quality_labels = []
        for metric in [
            "snr_values",
            "contrast_values",
            "brightness_values",
            "sharpness_values",
        ]:
            if metric in quality_data and quality_data[metric]:
                # Normalize for comparison
                values = np.array(quality_data[metric])
                normalized_values = (values - np.min(values)) / (
                    np.max(values) - np.min(values) + 1e-8
                )
                quality_metrics_data.append(normalized_values)
                quality_labels.append(metric.replace("_values", "").title())

        if quality_metrics_data:
            bp = axes[1, 0].boxplot(
                quality_metrics_data, tick_labels=quality_labels, patch_artist=True
            )
            colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow"]
            for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
                patch.set_facecolor(color)
            axes[1, 0].set_ylabel("Normalized Value")
            axes[1, 0].set_title("Quality Metrics Comparison")
            axes[1, 0].tick_params(axis="x", rotation=45)
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Quality trends over dataset
        if len(self.images) > 1:
            image_indices = list(range(len(self.images)))

            for i, (metric, color) in enumerate(
                zip(
                    ["snr_values", "contrast_values", "brightness_values"],
                    ["blue", "green", "red"],
                )
            ):
                if metric in quality_data and quality_data[metric]:
                    values = quality_data[metric]
                    # Normalize for plotting
                    normalized = (np.array(values) - np.min(values)) / (
                        np.max(values) - np.min(values) + 1e-8
                    )
                    axes[1, 1].plot(
                        image_indices,
                        normalized,
                        "o-",
                        label=metric.replace("_values", "").title(),
                        color=color,
                        alpha=0.7,
                        linewidth=2,
                    )

            axes[1, 1].set_xlabel("Image Index")
            axes[1, 1].set_ylabel("Normalized Quality Value")
            axes[1, 1].set_title("Quality Trends Across Dataset")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Quality correlation heatmap
        quality_df_data = {}
        for metric in [
            "snr_values",
            "contrast_values",
            "brightness_values",
            "sharpness_values",
            "edge_density",
        ]:
            if metric in quality_data and quality_data[metric]:
                quality_df_data[
                    metric.replace("_values", "").replace("_", " ").title()
                ] = quality_data[metric]

        if len(quality_df_data) > 1:
            quality_df = pd.DataFrame(quality_df_data)
            corr_matrix = quality_df.corr()

            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                cmap="RdBu_r",
                center=0,
                ax=axes[1, 2],
                square=True,
                cbar_kws={"label": "Correlation"},
            )
            axes[1, 2].set_title("Quality Metrics Correlation")

        plt.tight_layout()
        plt.show()

    def plot_statistical_tests(self, figsize: Tuple[int, int] = (14, 10)) -> None:
        """Create detailed statistical test visualization."""
        if not hasattr(self, "statistical_test_results"):
            self.statistical_tests()

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Statistical Test Results", fontsize=16, fontweight="bold")

        # 1. Normality test visualization
        if (
            "spatial_analysis" in self.analysis_results
            and self.analysis_results["spatial_analysis"]["defect_sizes"]
        ):
            defect_sizes = self.analysis_results["spatial_analysis"]["defect_sizes"]

            # Q-Q plot for normality
            stats.probplot(defect_sizes, dist="norm", plot=axes[0, 0])
            axes[0, 0].set_title("Q-Q Plot: Defect Size Normality")
            axes[0, 0].grid(True, alpha=0.3)

            # Add test result
            if "defect_size_normality" in self.statistical_test_results:
                test_result = self.statistical_test_results["defect_size_normality"]
                if "shapiro_p_value" in test_result:
                    p_val = test_result["shapiro_p_value"]
                    result_text = f"Shapiro-Wilk Test\np-value: {p_val:.4f}\n"
                    result_text += "Normal" if p_val > 0.05 else "Not Normal"
                    axes[0, 0].text(
                        0.05,
                        0.95,
                        result_text,
                        transform=axes[0, 0].transAxes,
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                        verticalalignment="top",
                    )

        # 2. Spatial uniformity test
        if (
            "spatial_analysis" in self.analysis_results
            and self.analysis_results["spatial_analysis"]["defect_locations"]
        ):
            locations = self.analysis_results["spatial_analysis"]["defect_locations"]
            x_coords = [loc[1] for loc in locations]
            y_coords = [loc[0] for loc in locations]

            # Histogram of x-coordinates
            axes[0, 1].hist(
                x_coords,
                bins=30,
                alpha=0.7,
                color="skyblue",
                density=True,
                edgecolor="black",
                label="Observed",
            )

            # Expected uniform distribution
            x_min, x_max = min(x_coords), max(x_coords)
            uniform_line = np.ones(100) / (x_max - x_min)
            x_uniform = np.linspace(x_min, x_max, 100)
            axes[0, 1].plot(
                x_uniform, uniform_line, "r--", linewidth=2, label="Uniform"
            )

            axes[0, 1].set_xlabel("X Position")
            axes[0, 1].set_ylabel("Density")
            axes[0, 1].set_title("Spatial Distribution Test (X-axis)")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Add test result
            if "spatial_uniformity" in self.statistical_test_results:
                test_result = self.statistical_test_results["spatial_uniformity"]
                if "x_p_value" in test_result:
                    p_val = test_result["x_p_value"]
                    result_text = f"KS Test (X)\np-value: {p_val:.4f}\n"
                    result_text += "Uniform" if p_val > 0.05 else "Not Uniform"
                    axes[0, 1].text(
                        0.05,
                        0.95,
                        result_text,
                        transform=axes[0, 1].transAxes,
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
                        verticalalignment="top",
                    )

        # 3. Class balance test
        if "basic_statistics" in self.analysis_results:
            class_counts = list(
                self.analysis_results["basic_statistics"][
                    "total_pixels_per_class"
                ].values()
            )
            class_names = [self.class_mapping[i] for i in range(len(class_counts))]

            # Bar plot of class counts
            bars = axes[1, 0].bar(
                class_names,
                class_counts,
                color=[self.class_colors[i] for i in range(len(class_counts))],
                alpha=0.7,
                edgecolor="black",
            )
            axes[1, 0].set_ylabel("Pixel Count")
            axes[1, 0].set_title("Class Distribution")
            axes[1, 0].tick_params(axis="x", rotation=45)

            # Add count labels on bars
            for bar, count in zip(bars, class_counts):
                height = bar.get_height()
                axes[1, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{count:,}",
                    ha="center",
                    va="bottom",
                )

            # Expected uniform distribution line
            expected_uniform = sum(class_counts) / len(class_counts)
            axes[1, 0].axhline(
                y=expected_uniform,
                color="red",
                linestyle="--",
                label=f"Expected Uniform: {expected_uniform:,.0f}",
            )
            axes[1, 0].legend()

        # 4. Statistical test summary
        axes[1, 1].axis("off")

        # Create summary text
        summary_text = "STATISTICAL TEST SUMMARY\n" + "=" * 30 + "\n\n"

        if hasattr(self, "statistical_test_results"):
            tests = self.statistical_test_results

            if (
                "defect_size_normality" in tests
                and "shapiro_p_value" in tests["defect_size_normality"]
            ):
                p_val = tests["defect_size_normality"]["shapiro_p_value"]
                summary_text += "Defect Size Normality:\n"
                summary_text += f"  Shapiro-Wilk p-value: {p_val:.4f}\n"
                summary_text += (
                    f"  Result: {'Normal' if p_val > 0.05 else 'Not Normal'}\n\n"
                )

            if "spatial_uniformity" in tests:
                x_p = tests["spatial_uniformity"].get("x_p_value", "N/A")
                y_p = tests["spatial_uniformity"].get("y_p_value", "N/A")
                summary_text += "Spatial Uniformity:\n"
                x_p_str = f"{x_p:.4f}" if isinstance(x_p, float) else str(x_p)
                y_p_str = f"{y_p:.4f}" if isinstance(y_p, float) else str(y_p)
                summary_text += f"  X-axis KS p-value: {x_p_str}\n"
                summary_text += f"  Y-axis KS p-value: {y_p_str}\n"
                if isinstance(x_p, float) and isinstance(y_p, float):
                    summary_text += f"  Result: {'Uniform' if min(x_p, y_p) > 0.05 else 'Not Uniform'}\n\n"

        # Add interpretation guide
        summary_text += "\nINTERPRETATION:\n"
        summary_text += "• p > 0.05: Null hypothesis not rejected\n"
        summary_text += "• p ≤ 0.05: Null hypothesis rejected\n"
        summary_text += "• Normal: Data follows normal distribution\n"
        summary_text += "• Uniform: Spatial distribution is uniform"

        axes[1, 1].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

    def plot_outlier_analysis(self, figsize: Tuple[int, int] = (14, 8)) -> None:
        """Create outlier detection visualization."""
        outlier_results = self.outlier_detection()

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("Outlier Detection Analysis", fontsize=16, fontweight="bold")

        # Feature extraction for visualization
        image_features = []
        feature_names = [
            "Defect Ratio",
            "Wafer Ratio",
            "Total Defects",
            "Mean Intensity",
            "Std Intensity",
        ]

        for img in self.images:
            features = [
                np.sum(img == 2) / img.size,  # defect ratio
                np.sum(img == 1) / img.size,  # wafer ratio
                len(regionprops(label(img == 2))),  # total defects
                np.mean(img),  # mean intensity
                np.std(img),  # std intensity
            ]
            image_features.append(features)

        image_features = np.array(image_features)
        outlier_indices = set(outlier_results.get("image_outliers", []))

        # 1. Feature scatter plot (first 2 features)
        if len(image_features) > 0:
            colors = [
                "red" if i in outlier_indices else "blue"
                for i in range(len(image_features))
            ]
            sizes = [
                100 if i in outlier_indices else 50 for i in range(len(image_features))
            ]

            scatter = axes[0].scatter(
                image_features[:, 0], image_features[:, 1], c=colors, s=sizes, alpha=0.7
            )
            axes[0].set_xlabel(feature_names[0])
            axes[0].set_ylabel(feature_names[1])
            axes[0].set_title("Feature Space (Outliers in Red)")
            axes[0].grid(True, alpha=0.3)

            # Add legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="blue",
                    markersize=8,
                    alpha=0.7,
                    label="Normal",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=10,
                    alpha=0.7,
                    label="Outlier",
                ),
            ]
            axes[0].legend(handles=legend_elements)

        # 2. Box plot of all features
        if len(image_features) > 0:
            # Normalize features for comparison
            normalized_features = []
            for i in range(image_features.shape[1]):
                feature = image_features[:, i]
                normalized = (feature - np.min(feature)) / (
                    np.max(feature) - np.min(feature) + 1e-8
                )
                normalized_features.append(normalized)

            bp = axes[1].boxplot(
                normalized_features,
                tick_labels=[name.replace(" ", "\n") for name in feature_names],
                patch_artist=True,
            )

            # Color boxes
            colors = [
                "lightblue",
                "lightgreen",
                "lightcoral",
                "lightyellow",
                "lightpink",
            ]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)

            axes[1].set_ylabel("Normalized Feature Value")
            axes[1].set_title("Feature Distribution (Outlier Detection)")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].grid(True, alpha=0.3)

        # 3. Outlier summary
        axes[2].axis("off")

        summary_text = "OUTLIER ANALYSIS SUMMARY\n" + "=" * 25 + "\n\n"
        summary_text += f"Total Images: {len(self.images)}\n"
        summary_text += f"Detected Outliers: {len(outlier_indices)}\n"
        summary_text += f"Outlier Percentage: {len(outlier_indices) / len(self.images) * 100:.1f}%\n\n"

        if outlier_indices:
            summary_text += "Outlier Image Indices:\n"
            outlier_list = sorted(list(outlier_indices))
            if len(outlier_list) <= 10:
                summary_text += f"{outlier_list}\n\n"
            else:
                summary_text += f"{outlier_list[:10]}...\n(showing first 10)\n\n"

            # Show outlier characteristics
            if len(image_features) > 0:
                outlier_features = image_features[list(outlier_indices)]
                normal_features = image_features[
                    [i for i in range(len(image_features)) if i not in outlier_indices]
                ]

                summary_text += "Outlier Characteristics:\n"
                for i, feature_name in enumerate(feature_names):
                    if len(outlier_features) > 0 and len(normal_features) > 0:
                        outlier_mean = np.mean(outlier_features[:, i])
                        normal_mean = np.mean(normal_features[:, i])
                        ratio = outlier_mean / (normal_mean + 1e-8)
                        summary_text += f"  {feature_name}: {ratio:.2f}x normal\n"
        else:
            summary_text += "No outliers detected.\n"
            summary_text += "Dataset appears consistent."

        axes[2].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[2].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

    def create_sample_visualization(
        self, n_samples: int = 6, figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """Create visualization of sample wafer maps."""
        if not self.images:
            print("No images loaded.")
            return

        # Select diverse samples (including outliers if any)
        outlier_results = self.outlier_detection()
        outlier_indices = set(outlier_results.get("image_outliers", []))

        sample_indices = []

        # Add some outliers if they exist
        outlier_list = list(outlier_indices)
        if outlier_list:
            sample_indices.extend(outlier_list[: min(2, len(outlier_list))])

        # Add random normal samples
        normal_indices = [
            i for i in range(len(self.images)) if i not in outlier_indices
        ]
        remaining_samples = n_samples - len(sample_indices)
        if remaining_samples > 0 and normal_indices:
            np.random.seed(self.random_state)
            additional_samples = np.random.choice(
                normal_indices,
                min(remaining_samples, len(normal_indices)),
                replace=False,
            )
            sample_indices.extend(additional_samples)

        # Ensure we have enough samples
        while len(sample_indices) < n_samples and len(sample_indices) < len(
            self.images
        ):
            remaining = [i for i in range(len(self.images)) if i not in sample_indices]
            if remaining:
                sample_indices.append(remaining[0])

        sample_indices = sample_indices[:n_samples]

        # Create subplot grid
        rows = 2
        cols = (n_samples + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_samples == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        fig.suptitle("Sample Wafer Map Visualizations", fontsize=16, fontweight="bold")

        for i, idx in enumerate(sample_indices):
            if i >= len(axes):
                break

            img = self.images[idx]

            # Create colored version for better visualization
            colored_img = np.zeros((*img.shape, 3))
            for class_val, color in self.class_colors.items():
                mask = img == class_val
                colored_img[mask] = matplotlib.colors.to_rgb(color)

            axes[i].imshow(colored_img)

            # Add defect statistics
            defect_count = len(regionprops(label(img == 2)))
            defect_ratio = np.sum(img == 2) / img.size * 100

            title = f"Sample {idx}"
            if idx in outlier_indices:
                title += " (Outlier)"
            title += f"\nDefects: {defect_count} ({defect_ratio:.2f}%)"

            axes[i].set_title(title, fontsize=10)
            axes[i].axis("off")

        # Hide unused subplots
        for i in range(len(sample_indices), len(axes)):
            axes[i].axis("off")

        # Add legend
        legend_elements = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=self.class_colors[i],
                label=self.class_mapping[i].title(),
            )
            for i in sorted(self.class_colors.keys())
        ]
        fig.legend(
            handles=legend_elements, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=3
        )

        plt.tight_layout()
        plt.show()


# Function to generate synthetic wafer data for demonstration
def generate_synthetic_wafer_dataset(
    n_images: int = 20, image_size: Tuple[int, int] = (512, 512)
) -> List[np.ndarray]:
    """Generate synthetic wafer map dataset for testing."""
    np.random.seed(42)
    images = []

    for i in range(n_images):
        img = np.zeros(image_size, dtype=np.uint8)

        # Create circular wafer region
        center = (image_size[0] // 2, image_size[1] // 2)
        radius = min(image_size) // 2 - 20

        y, x = np.ogrid[: image_size[0], : image_size[1]]
        mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius**2
        img[mask] = 1

        # Add defects with varying patterns
        if i < 5:  # High defect density
            n_defects = np.random.randint(15, 30)
        elif i < 15:  # Normal defect density
            n_defects = np.random.randint(5, 15)
        else:  # Low defect density
            n_defects = np.random.randint(1, 5)

        for _ in range(n_defects):
            # Random defect location within wafer
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, radius - 10)
            dx = int(r * np.cos(angle))
            dy = int(r * np.sin(angle))
            defect_x = center[1] + dx
            defect_y = center[0] + dy

            # Create defect with random size and shape
            if i % 4 == 0:  # Circular defects
                defect_radius = np.random.randint(2, 8)
                cy, cx = np.ogrid[: image_size[0], : image_size[1]]
                defect_mask = (cx - defect_x) ** 2 + (
                    cy - defect_y
                ) ** 2 <= defect_radius**2
                img[defect_mask & mask] = 2
            else:  # Rectangular defects
                defect_size = np.random.randint(3, 12)
                y1 = max(0, defect_y - defect_size // 2)
                y2 = min(image_size[0], defect_y + defect_size // 2)
                x1 = max(0, defect_x - defect_size // 2)
                x2 = min(image_size[1], defect_x + defect_size // 2)
                img[y1:y2, x1:x2] = np.where(
                    img[y1:y2, x1:x2] == 1, 2, img[y1:y2, x1:x2]
                )

        images.append(img)

    return images


# Example usage function
def run_comprehensive_analysis():
    """Run comprehensive analysis with all visualizations."""
    print("Generating synthetic wafer dataset...")
    synthetic_images = generate_synthetic_wafer_dataset(n_images=25)

    print("Initializing analyzer...")
    analyzer = WaferMapVisualizationAnalyzer(
        images=synthetic_images,
        labels=[f"wafer_{i:03d}" for i in range(len(synthetic_images))],
    )

    print("Running analysis...")
    analyzer.analyze_dataset()

    print("Creating visualizations...")

    # 1. Overview Dashboard
    print("1. Creating Overview Dashboard...")
    analyzer.plot_overview_dashboard()

    # 2. Spatial Analysis
    print("2. Creating Spatial Analysis...")
    analyzer.plot_spatial_analysis()

    # 3. Quality Analysis
    print("3. Creating Quality Analysis...")
    analyzer.plot_quality_analysis()

    # 4. Statistical Tests
    print("4. Creating Statistical Test Results...")
    analyzer.plot_statistical_tests()

    # 5. Outlier Analysis
    print("5. Creating Outlier Analysis...")
    analyzer.plot_outlier_analysis()

    # 6. Sample Visualizations
    print("6. Creating Sample Visualizations...")
    analyzer.create_sample_visualization(n_samples=6)

    print("Analysis complete! All visualizations have been generated.")
    return analyzer


def run_quick_analysis():
    """Run quick analysis with essential visualizations only."""
    print("Generating quick synthetic dataset...")
    synthetic_images = generate_synthetic_wafer_dataset(n_images=10)

    analyzer = WaferMapVisualizationAnalyzer(
        images=synthetic_images,
        labels=[f"wafer_{i:03d}" for i in range(len(synthetic_images))],
    )

    print("Running quick analysis...")
    analyzer.analyze_dataset()

    # Show only the most important visualizations
    print("Creating overview dashboard...")
    analyzer.plot_overview_dashboard()

    print("Creating sample visualizations...")
    analyzer.create_sample_visualization(n_samples=4)

    return analyzer


# Additional utility functions for advanced analysis
def compare_datasets(
    analyzer1: WaferMapVisualizationAnalyzer,
    analyzer2: WaferMapVisualizationAnalyzer,
    figsize: Tuple[int, int] = (16, 10),
) -> None:
    """Compare two datasets side by side."""
    if not analyzer1.analysis_results or not analyzer2.analysis_results:
        print("Both analyzers must have completed analysis first.")
        return

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Dataset Comparison", fontsize=16, fontweight="bold")

    # Compare class distributions
    for i, (analyzer, title) in enumerate(
        [(analyzer1, "Dataset 1"), (analyzer2, "Dataset 2")]
    ):
        class_props = analyzer.analysis_results["basic_statistics"][
            "global_class_proportions"
        ]
        classes = [analyzer.class_mapping[k] for k in class_props.keys()]
        props = list(class_props.values())
        colors = [analyzer.class_colors[k] for k in class_props.keys()]

        axes[0, i].pie(props, labels=classes, autopct="%1.2f%%", colors=colors)
        axes[0, i].set_title(f"{title}: Class Distribution")

    # Compare defect size distributions
    sizes1 = analyzer1.analysis_results["spatial_analysis"]["defect_sizes"]
    sizes2 = analyzer2.analysis_results["spatial_analysis"]["defect_sizes"]

    if sizes1 and sizes2:
        axes[0, 2].hist(
            [sizes1, sizes2],
            bins=30,
            alpha=0.7,
            label=["Dataset 1", "Dataset 2"],
            color=["blue", "red"],
        )
        axes[0, 2].set_title("Defect Size Comparison")
        axes[0, 2].set_xlabel("Defect Size")
        axes[0, 2].set_ylabel("Frequency")
        axes[0, 2].legend()

    # Compare quality metrics
    quality_metrics = ["snr_values", "contrast_values", "brightness_values"]
    for i, metric in enumerate(quality_metrics):
        if (
            hasattr(analyzer1, "quality_metrics")
            and metric in analyzer1.quality_metrics
            and hasattr(analyzer2, "quality_metrics")
            and metric in analyzer2.quality_metrics
        ):
            data1 = analyzer1.quality_metrics[metric]
            data2 = analyzer2.quality_metrics[metric]

            axes[1, i].boxplot([data1, data2], labels=["Dataset 1", "Dataset 2"])
            axes[1, i].set_title(f"{metric.replace('_values', '').title()} Comparison")
            axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def export_analysis_report(
    analyzer: WaferMapVisualizationAnalyzer, output_dir: str = "wafer_analysis_output"
) -> None:
    """Export comprehensive analysis report with all visualizations."""
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Exporting analysis report to {output_path}...")

    # Save JSON report
    if analyzer.analysis_results:
        report_data = {
            "basic_statistics": analyzer.analysis_results.get("basic_statistics", {}),
            "spatial_analysis": {
                k: v
                for k, v in analyzer.analysis_results.get(
                    "spatial_analysis", {}
                ).items()
                if k != "defect_density_maps"
            },  # Skip large arrays
            "quality_metrics": analyzer.quality_metrics,
            "statistical_tests": analyzer.statistical_tests,
        }

        with open(output_path / "analysis_report.json", "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    # Save individual plots
    original_backend = plt.get_backend()
    plt.switch_backend("Agg")  # Non-interactive backend for saving

    try:
        # Override show method to save instead
        def save_fig(name):
            plt.savefig(output_path / f"{name}.png", dpi=300, bbox_inches="tight")
            plt.close()

        # Temporarily replace show with save
        original_show = plt.show
        plt.show = lambda: save_fig("current_plot")

        # Generate and save all plots
        plot_names = [
            ("overview_dashboard", analyzer.plot_overview_dashboard),
            ("spatial_analysis", analyzer.plot_spatial_analysis),
            ("quality_analysis", analyzer.plot_quality_analysis),
            ("statistical_tests", analyzer.plot_statistical_tests),
            ("outlier_analysis", analyzer.plot_outlier_analysis),
        ]

        for name, plot_func in plot_names:
            try:
                plt.show = lambda n=name: save_fig(n)
                plot_func()
            except Exception as e:
                print(f"Error saving {name}: {e}")

        # Save sample visualization
        plt.show = lambda: save_fig("sample_visualization")
        analyzer.create_sample_visualization()

    finally:
        plt.show = original_show
        plt.switch_backend(original_backend)

    print(f"Analysis report exported successfully to {output_path}")


if __name__ == "__main__":
    # Run the comprehensive analysis
    print("Starting Wafer Map Dataset Analysis...")
    print("=" * 50)

    # Uncomment the analysis you want to run:

    # Full comprehensive analysis (recommended)
    analyzer = run_comprehensive_analysis()

    # Quick analysis (faster, fewer plots)
    # analyzer = run_quick_analysis()

    # Export results
    # export_analysis_report(analyzer, "my_wafer_analysis")

    print("\nAnalysis completed successfully!")
    print(
        "You can now use the analyzer object to access all results and create custom visualizations."
    )
