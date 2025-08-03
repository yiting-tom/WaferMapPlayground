# %%
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import MultiLabelBinarizer


class MultilabelDatasetAnalyzer:
    def __init__(self, labels, class_names=None):
        """
        Initialize analyzer for multilabel dataset

        Args:
            labels: List of lists/sets containing labels for each sample
                   e.g., [['cat', 'animal'], ['dog', 'animal'], ['cat']]
            class_names: Optional list of all possible class names
        """
        self.labels = labels
        self.n_samples = len(labels)

        # Convert to binary matrix format
        self.mlb = MultiLabelBinarizer()
        self.binary_labels = self.mlb.fit_transform(labels)
        self.class_names = self.mlb.classes_ if class_names is None else class_names
        self.n_classes = len(self.class_names)

    def basic_statistics(self):
        """Calculate basic multilabel statistics"""
        # Label cardinality (average labels per sample)
        cardinality = np.mean(np.sum(self.binary_labels, axis=1))

        # Label density (cardinality / total possible labels)
        density = cardinality / self.n_classes

        # Individual label frequencies
        label_frequencies = np.sum(self.binary_labels, axis=0)
        label_proportions = label_frequencies / self.n_samples

        # Number of distinct label combinations
        unique_combinations = len(set(tuple(row) for row in self.binary_labels))

        return {
            "cardinality": cardinality,
            "density": density,
            "label_frequencies": dict(zip(self.class_names, label_frequencies)),
            "label_proportions": dict(zip(self.class_names, label_proportions)),
            "unique_combinations": unique_combinations,
            "combination_diversity": unique_combinations / self.n_samples,
        }

    def calculate_entropies(self):
        """Calculate various entropy measures"""
        results = {}

        # Individual label entropies
        individual_entropies = {}
        for i, class_name in enumerate(self.class_names):
            p = np.mean(self.binary_labels[:, i])
            if p > 0 and p < 1:
                h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            else:
                h = 0
            individual_entropies[class_name] = h

        results["individual_entropies"] = individual_entropies
        results["mean_individual_entropy"] = np.mean(
            list(individual_entropies.values())
        )

        # Joint entropy of label combinations
        combination_counts = Counter(tuple(row) for row in self.binary_labels)
        combination_probs = np.array(list(combination_counts.values())) / self.n_samples
        joint_entropy = entropy(combination_probs, base=2)
        results["joint_entropy"] = joint_entropy

        # Conditional entropies (simplified for pairs)
        conditional_entropies = {}
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                # H(Y_j | Y_i)
                y_i = self.binary_labels[:, i]
                y_j = self.binary_labels[:, j]

                # P(Y_i = 1)
                p_i_1 = np.mean(y_i)
                p_i_0 = 1 - p_i_1

                if p_i_1 > 0:
                    # P(Y_j = 1 | Y_i = 1)
                    p_j_1_given_i_1 = np.mean(y_j[y_i == 1]) if np.sum(y_i) > 0 else 0
                    h_j_given_i_1 = -p_j_1_given_i_1 * np.log2(
                        p_j_1_given_i_1 + 1e-10
                    ) - (1 - p_j_1_given_i_1) * np.log2(1 - p_j_1_given_i_1 + 1e-10)
                else:
                    h_j_given_i_1 = 0

                if p_i_0 > 0:
                    # P(Y_j = 1 | Y_i = 0)
                    p_j_1_given_i_0 = (
                        np.mean(y_j[y_i == 0]) if np.sum(1 - y_i) > 0 else 0
                    )
                    h_j_given_i_0 = -p_j_1_given_i_0 * np.log2(
                        p_j_1_given_i_0 + 1e-10
                    ) - (1 - p_j_1_given_i_0) * np.log2(1 - p_j_1_given_i_0 + 1e-10)
                else:
                    h_j_given_i_0 = 0

                conditional_entropy = p_i_1 * h_j_given_i_1 + p_i_0 * h_j_given_i_0
                conditional_entropies[
                    f"{self.class_names[j]}|{self.class_names[i]}"
                ] = conditional_entropy

        results["conditional_entropies"] = conditional_entropies

        return results

    def mutual_information_matrix(self):
        """Calculate pairwise mutual information between labels"""
        mi_matrix = np.zeros((self.n_classes, self.n_classes))

        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i == j:
                    mi_matrix[i, j] = 0
                    continue

                y_i = self.binary_labels[:, i]
                y_j = self.binary_labels[:, j]

                # Joint distribution
                joint_counts = np.zeros((2, 2))
                joint_counts[0, 0] = np.sum((y_i == 0) & (y_j == 0))
                joint_counts[0, 1] = np.sum((y_i == 0) & (y_j == 1))
                joint_counts[1, 0] = np.sum((y_i == 1) & (y_j == 0))
                joint_counts[1, 1] = np.sum((y_i == 1) & (y_j == 1))

                joint_probs = joint_counts / self.n_samples
                marginal_i = np.array([np.mean(y_i == 0), np.mean(y_i == 1)])
                marginal_j = np.array([np.mean(y_j == 0), np.mean(y_j == 1)])

                mi = 0
                for ii in range(2):
                    for jj in range(2):
                        if joint_probs[ii, jj] > 0:
                            mi += joint_probs[ii, jj] * np.log2(
                                joint_probs[ii, jj]
                                / (marginal_i[ii] * marginal_j[jj] + 1e-10)
                            )

                mi_matrix[i, j] = mi

        return mi_matrix

    def label_cooccurrence_matrix(self):
        """Calculate label co-occurrence matrix"""
        cooccurrence = np.dot(self.binary_labels.T, self.binary_labels)
        return cooccurrence

    def analyze_image_entropies(self, images):
        """Analyze image entropies in context of multilabel structure"""
        if images is None:
            return None

        image_entropies = []
        for img in images:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = np.mean(img, axis=2).astype(np.uint8)
            else:
                gray = img

            # Calculate histogram and entropy
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
            hist = hist / hist.sum()  # Normalize
            img_entropy = entropy(hist[hist > 0], base=2)
            image_entropies.append(img_entropy)

        # Analyze entropy vs label characteristics
        cardinalities = np.sum(self.binary_labels, axis=1)

        results = {
            "image_entropies": image_entropies,
            "mean_entropy": np.mean(image_entropies),
            "entropy_vs_cardinality_corr": np.corrcoef(image_entropies, cardinalities)[
                0, 1
            ],
            "entropy_by_cardinality": {},
        }

        # Group by cardinality
        for card in np.unique(cardinalities):
            mask = cardinalities == card
            results["entropy_by_cardinality"][card] = {
                "mean_entropy": np.mean(np.array(image_entropies)[mask]),
                "count": np.sum(mask),
            }

        return results

    def visualize_analysis(self, save_path=None):
        """Create comprehensive visualization of the analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Label frequency distribution
        stats = self.basic_statistics()
        frequencies = list(stats["label_frequencies"].values())
        axes[0, 0].bar(range(len(frequencies)), frequencies)
        axes[0, 0].set_title("Label Frequencies")
        axes[0, 0].set_xlabel("Label Index")
        axes[0, 0].set_ylabel("Frequency")

        # 2. Cardinality distribution
        cardinalities = np.sum(self.binary_labels, axis=1)
        axes[0, 1].hist(cardinalities, bins=range(max(cardinalities) + 2), alpha=0.7)
        axes[0, 1].set_title("Label Cardinality Distribution")
        axes[0, 1].set_xlabel("Number of Labels per Sample")
        axes[0, 1].set_ylabel("Count")

        # 3. Co-occurrence heatmap
        cooccurrence = self.label_cooccurrence_matrix()
        im = axes[0, 2].imshow(cooccurrence, cmap="Blues")
        axes[0, 2].set_title("Label Co-occurrence Matrix")
        plt.colorbar(im, ax=axes[0, 2])

        # 4. Mutual information heatmap
        mi_matrix = self.mutual_information_matrix()
        im2 = axes[1, 0].imshow(mi_matrix, cmap="Reds")
        axes[1, 0].set_title("Mutual Information Matrix")
        plt.colorbar(im2, ax=axes[1, 0])

        # 5. Individual label entropies
        entropies = self.calculate_entropies()
        individual_ents = list(entropies["individual_entropies"].values())
        axes[1, 1].bar(range(len(individual_ents)), individual_ents)
        axes[1, 1].set_title("Individual Label Entropies")
        axes[1, 1].set_xlabel("Label Index")
        axes[1, 1].set_ylabel("Entropy (bits)")

        # 6. Label combination frequency (top 10)
        combination_counts = Counter(tuple(row) for row in self.binary_labels)
        top_combinations = combination_counts.most_common(10)
        combo_labels = [f"Combo {i + 1}" for i in range(len(top_combinations))]
        combo_counts = [count for _, count in top_combinations]

        axes[1, 2].bar(combo_labels, combo_counts)
        axes[1, 2].set_title("Top 10 Label Combinations")
        axes[1, 2].set_xlabel("Label Combination")
        axes[1, 2].set_ylabel("Frequency")
        axes[1, 2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def generate_report(self):
        """Generate comprehensive analysis report"""
        stats = self.basic_statistics()
        entropies = self.calculate_entropies()

        print("=== MULTILABEL DATASET ANALYSIS REPORT ===\n")

        print("Basic Statistics:")
        print(f"  Number of samples: {self.n_samples}")
        print(f"  Number of classes: {self.n_classes}")
        print(f"  Label cardinality: {stats['cardinality']:.3f}")
        print(f"  Label density: {stats['density']:.3f}")
        print(f"  Unique combinations: {stats['unique_combinations']}")
        print(f"  Combination diversity: {stats['combination_diversity']:.3f}")

        print("\nEntropy Analysis:")
        print(f"  Joint entropy: {entropies['joint_entropy']:.3f} bits")
        print(
            f"  Mean individual entropy: {entropies['mean_individual_entropy']:.3f} bits"
        )

        print("\nLabel Imbalance:")
        proportions = list(stats["label_proportions"].values())
        print(f"  Most frequent label: {max(proportions):.3f}")
        print(f"  Least frequent label: {min(proportions):.3f}")
        print(f"  Imbalance ratio: {max(proportions) / min(proportions):.2f}")

        print("\nLabel Dependencies:")
        mi_matrix = self.mutual_information_matrix()
        max_mi_idx = np.unravel_index(np.argmax(mi_matrix), mi_matrix.shape)
        print(f"  Highest mutual information: {mi_matrix[max_mi_idx]:.3f} bits")
        print(
            f"  Between: {self.class_names[max_mi_idx[0]]} and {self.class_names[max_mi_idx[1]]}"
        )


# %%
npz = np.load("../sparse_wm38.npz")
images = npz["images"]
labels = npz["labels"]

analyzer = MultilabelDatasetAnalyzer(labels)
# %%
analyzer.visualize_analysis()
