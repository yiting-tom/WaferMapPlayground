# Wafer Map Dataset Visualization & Analysis Tool

A comprehensive Python tool for analyzing semiconductor wafer map datasets with advanced visualization capabilities, statistical testing, and quality assessment features.

## Overview

The `WaferMapVisualizationAnalyzer` provides a complete suite of analysis methods for semiconductor wafer maps, enabling engineers and researchers to:

- Perform statistical analysis of defect patterns
- Assess image quality metrics
- Detect outliers and anomalies
- Visualize spatial defect distributions
- Generate comprehensive analysis reports

## Key Features

### 🔍 **Multi-dimensional Analysis Pipeline**
- Basic statistical analysis
- Spatial defect distribution analysis
- Image quality assessment
- Statistical hypothesis testing
- Outlier detection using machine learning
- Comprehensive visualization dashboard

### 📊 **Rich Visualizations**
- Interactive dashboards with multiple plot types
- Spatial heatmaps and clustering analysis
- Quality metrics correlation matrices
- Statistical test result visualizations
- Sample wafer map displays with defect highlighting

## Analysis Methods Explained

### 1. Basic Statistics Analysis (`basic_statistics()`)

**Purpose**: Provides foundational dataset characteristics and class distribution metrics.

**Key Metrics**:
- **Dataset Size**: Total number of wafer images
- **Image Dimensions**: Consistency check across all images
- **Class Distributions**: Pixel-level distribution of background (0), wafer (1), and defect (2) classes
- **Shannon Entropy**: Measures class diversity using information theory
- **Global Class Proportions**: Overall balance between different pixel types

**Mathematical Foundation**:
```
Shannon Entropy = -Σ(pi × log2(pi))
where pi = proportion of class i
```

**Output**: Dictionary containing distribution statistics, entropy measures, and dimension consistency checks.

---

### 2. Spatial Analysis (`spatial_analysis()`)

**Purpose**: Analyzes the geometric and spatial characteristics of defects across wafer surfaces.

#### Connected Component Analysis
- Uses `skimage.measure.label()` to identify individual defect regions
- Extracts geometric properties for each defect:
  - **Area**: Number of pixels in defect region
  - **Centroid**: Center of mass coordinates
  - **Bounding Box**: Rectangular region containing the defect
  - **Eccentricity**: Shape elongation measure (0 = circle, 1 = line)
  - **Solidity**: Ratio of defect area to convex hull area
  - **Aspect Ratio**: Width to height ratio of bounding box

#### Spatial Clustering Analysis
- Implements **DBSCAN clustering** on defect coordinates
- Parameters: `eps=5` (neighborhood distance), `min_samples=3`
- Identifies clustered vs. scattered defect patterns
- Counts noise points (isolated defects)

#### Density Mapping
- Creates **Gaussian-filtered density maps** using `scipy.ndimage`
- Sigma parameter: 10 pixels for smooth density estimation
- Visualizes defect concentration areas

#### Statistical Moments
- Calculates spatial variance in X and Y directions
- Measures defect spread patterns
- Identifies directional bias in defect distribution

**Output**: Comprehensive spatial statistics including defect locations, sizes, shapes, clustering metrics, and density maps.

---

### 3. Quality Assessment (`quality_assessment()`)

**Purpose**: Evaluates image quality using computer vision metrics relevant to semiconductor inspection.

#### Signal-to-Noise Ratio (SNR)
```
SNR = mean(wafer_pixels) / std(background_pixels)
```
- Higher values indicate better signal quality
- Critical for automated defect detection systems

#### RMS Contrast
```
RMS Contrast = √(mean((pixel_intensity - mean_intensity)²))
```
- Measures overall image contrast
- Important for visual defect detection

#### Edge Density Analysis
- Uses **Sobel edge detection** to quantify edge content
- Formula: `sum(|sobel_edges|) / total_pixels`
- Indicates image detail and sharpness

#### Noise Level Estimation
- **Laplacian variance method**: `variance(Laplacian(image))`
- Fallback: Gradient-based estimation using pixel differences
- Lower values indicate cleaner images

#### Sharpness Assessment
- Primary: Laplacian variance (Tenengrad operator)
- Fallback: Gradient magnitude variance
- Critical for focus quality evaluation

**Output**: Dictionary containing SNR values, contrast measurements, brightness levels, edge density, noise estimates, and sharpness metrics.

---

### 4. Statistical Testing (`statistical_tests()`)

**Purpose**: Performs rigorous statistical hypothesis testing on defect characteristics and spatial distributions.

#### Normality Testing
- **Shapiro-Wilk Test** on defect size distributions
- Null hypothesis: Data follows normal distribution
- Critical for choosing appropriate statistical models
- Sample limit: 5000 points for computational efficiency

#### Spatial Uniformity Testing
- **Kolmogorov-Smirnov (KS) Test** for spatial coordinates
- Tests X and Y coordinates separately against uniform distribution
- Null hypothesis: Defects are uniformly distributed spatially
- Important for identifying spatial bias in defect occurrence

**Statistical Interpretation**:
- **p > 0.05**: Fail to reject null hypothesis (data fits assumed distribution)
- **p ≤ 0.05**: Reject null hypothesis (data deviates significantly)

**Output**: Test statistics, p-values, and binary classification results for normality and spatial uniformity.

---

### 5. Outlier Detection (`outlier_detection()`)

**Purpose**: Identifies anomalous images in the dataset using machine learning techniques.

#### Feature Extraction
Extracts five key features per image:
1. **Defect Ratio**: `defects_pixels / total_pixels`
2. **Wafer Ratio**: `wafer_pixels / total_pixels`
3. **Total Defects**: Count of connected defect regions
4. **Mean Intensity**: Average pixel value
5. **Standard Deviation**: Intensity variation measure

#### Isolation Forest Algorithm
- **Contamination Rate**: 0.1 (assumes 10% outliers)
- **Random State**: 42 (reproducible results)
- **Method**: Isolates anomalies using random forest partitioning
- **Advantage**: Effective for high-dimensional data without requiring normal distribution assumptions

**Mathematical Principle**:
Isolation Forest works by:
1. Randomly selecting features and split values
2. Creating isolation trees that separate data points
3. Anomalies require fewer splits to isolate (shorter path lengths)
4. Anomaly score based on average path length across trees

**Output**: List of image indices identified as outliers based on feature anomalies.

---

## Visualization Components

### 1. Overview Dashboard (`plot_overview_dashboard()`)
**11-panel comprehensive visualization**:
- Class distribution pie chart
- Defect size histogram with statistics
- Quality metrics box plots
- Sample wafer map with color coding
- Spatial distribution heatmap
- Quality correlation matrix
- Shape analysis scatter plot
- Statistical test results
- Center of mass distribution
- Dataset health summary
- Quality trends across images

### 2. Spatial Analysis Plots (`plot_spatial_analysis()`)
**6-panel spatial focus**:
- Defect size distribution with mean/median markers
- Spatial scatter plot of all defect locations
- Clustering analysis summary
- Shape analysis (eccentricity vs. solidity)
- Center of mass density plot
- Spatial variance distribution

### 3. Quality Analysis Plots (`plot_quality_analysis()`)
**6-panel quality focus**:
- SNR distribution histogram
- Brightness vs. contrast scatter plot
- Sharpness distribution
- Normalized quality metrics comparison
- Quality trends over dataset sequence
- Quality correlation heatmap

### 4. Statistical Test Visualization (`plot_statistical_tests()`)
**4-panel statistical focus**:
- Q-Q plot for normality assessment
- Spatial uniformity histogram with expected distribution
- Class balance bar chart
- Statistical test summary with interpretation guide

### 5. Outlier Analysis (`plot_outlier_analysis()`)
**3-panel outlier focus**:
- Feature space scatter plot (outliers highlighted)
- Normalized feature box plots
- Outlier summary statistics and characteristics

### 6. Sample Visualizations (`create_sample_visualization()`)
**Multi-panel sample display**:
- Color-coded wafer maps showing defects
- Includes outlier samples when available
- Defect statistics overlay
- Legend for class interpretation

---

## Usage Examples

### Basic Usage
```python
# Initialize with your dataset
analyzer = WaferMapVisualizationAnalyzer(
    images=your_image_list,
    labels=your_labels,
    metadata=your_metadata
)

# Run complete analysis
results = analyzer.analyze_dataset()

# Generate comprehensive dashboard
analyzer.plot_overview_dashboard()
```

### Advanced Usage
```python
# Load from directory
analyzer = WaferMapVisualizationAnalyzer()
analyzer.load_dataset("/path/to/wafer/images", "*.png")

# Run specific analyses
basic_stats = analyzer.basic_statistics()
spatial_results = analyzer.spatial_analysis()
quality_metrics = analyzer.quality_assessment()
test_results = analyzer.statistical_tests()
outliers = analyzer.outlier_detection()

# Generate specific visualizations
analyzer.plot_spatial_analysis()
analyzer.plot_quality_analysis()
analyzer.plot_statistical_tests()
```

### Export Results
```python
# Export comprehensive report
export_analysis_report(analyzer, "output_directory")

# Compare two datasets
compare_datasets(analyzer1, analyzer2)
```

---

## Synthetic Data Generation

The tool includes a synthetic wafer dataset generator for testing and demonstration:

```python
synthetic_images = generate_synthetic_wafer_dataset(
    n_images=25,
    image_size=(512, 512)
)
```

**Features of synthetic data**:
- Circular wafer regions with realistic geometry
- Variable defect densities (high, normal, low)
- Multiple defect shapes (circular, rectangular)
- Random spatial distribution within wafer boundaries
- Controllable parameters for testing different scenarios

---

## Class Mapping and Visualization

**Pixel Classification**:
- `0`: Background (blue, #2E86AB)
- `1`: Wafer surface (purple, #A23B72) 
- `2`: Defects (orange, #F18F01)

**Color Scheme**: Optimized for accessibility and print visibility using the "husl" palette.

---

## Mathematical Foundations

### Information Theory
- **Shannon Entropy**: Measures dataset diversity and class balance
- **Applications**: Dataset quality assessment, class imbalance detection

### Computer Vision
- **Morphological Operations**: Connected component analysis for defect segmentation
- **Gradient Operators**: Sobel, Laplacian for edge detection and sharpness assessment
- **Gaussian Filtering**: Density map generation and noise reduction

### Statistical Analysis
- **Hypothesis Testing**: Shapiro-Wilk, Kolmogorov-Smirnov tests
- **Non-parametric Methods**: Distribution-free statistical analysis
- **Descriptive Statistics**: Moments, percentiles, correlation analysis

### Machine Learning
- **Anomaly Detection**: Isolation Forest for outlier identification
- **Clustering**: DBSCAN for spatial pattern analysis
- **Feature Engineering**: Multi-dimensional feature extraction for comprehensive analysis

---

## Dependencies

### Core Libraries
- **NumPy**: Numerical computations and array operations
- **SciPy**: Statistical functions and advanced mathematical operations
- **Scikit-learn**: Machine learning algorithms (DBSCAN, Isolation Forest)
- **OpenCV**: Computer vision operations (edge detection, noise estimation)
- **Scikit-image**: Image processing (connected components, region properties)

### Visualization
- **Matplotlib**: Primary plotting library with extensive customization
- **Seaborn**: Statistical visualization with enhanced aesthetics
- **Pandas**: Data manipulation and correlation analysis

### Utilities
- **Pathlib**: Modern file path handling
- **JSON**: Results export and configuration management
- **PIL**: Image loading and basic operations

---

## Performance Considerations

### Memory Optimization
- Lazy loading of large arrays (density maps)
- Efficient numpy operations for pixel-level analysis
- Selective feature extraction to minimize memory footprint

### Computational Efficiency
- Vectorized operations using NumPy
- Optimized OpenCV functions for image processing
- Sample size limits for statistical tests (5000 samples max)
- Random state control for reproducible results

### Scalability
- Batch processing capabilities for large datasets
- Modular design allowing selective analysis components
- Export functionality for distributed analysis workflows

---

## Applications

### Semiconductor Manufacturing
- **Process Monitoring**: Quality control in wafer fabrication
- **Yield Analysis**: Defect pattern correlation with manufacturing parameters
- **Equipment Validation**: Tool performance assessment through defect metrics

### Research and Development
- **Algorithm Validation**: Benchmarking defect detection algorithms
- **Statistical Modeling**: Understanding defect generation mechanisms
- **Quality Metrics**: Developing new assessment criteria for wafer inspection

### Academic Use
- **Teaching Tool**: Demonstrating statistical analysis and computer vision concepts
- **Research Platform**: Extensible framework for wafer analysis research
- **Comparative Studies**: Multi-dataset analysis capabilities

---

## Future Enhancements

### Planned Features
- **Deep Learning Integration**: CNN-based defect classification
- **Time Series Analysis**: Temporal defect pattern evolution
- **Multi-scale Analysis**: Hierarchical defect pattern recognition
- **Real-time Processing**: Streaming analysis capabilities

### Extension Points
- **Custom Metrics**: User-defined quality and spatial metrics
- **Algorithm Plugins**: Modular algorithm integration
- **Export Formats**: Additional output formats (PDF reports, Excel summaries)
- **Interactive Visualizations**: Web-based dashboard with user interaction

---

## Contributing

The codebase is designed for extensibility and community contribution. Key areas for enhancement:

1. **New Analysis Methods**: Additional statistical tests and quality metrics
2. **Visualization Improvements**: Interactive plots and 3D visualizations
3. **Performance Optimization**: GPU acceleration and parallel processing
4. **Documentation**: Additional examples and use cases
5. **Testing**: Unit tests and validation datasets

---

## License and Citation

This tool is provided for educational and research purposes. When using in academic work, please cite the statistical methods and computer vision techniques employed.

**Key References**:
- Shapiro-Wilk Test: Shapiro, S.S. and Wilk, M.B. (1965)
- Isolation Forest: Liu, F.T., Ting, K.M. and Zhou, Z.H. (2008)
- DBSCAN Clustering: Ester, M., Kriegel, H.P., Sander, J. and Xu, X. (1996)