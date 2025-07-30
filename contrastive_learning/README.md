# MobileNetV3 Contrastive Learning on Fashion-MNIST

This repository implements MobileNetV3-Large with contrastive learning using the SimCLR framework on Fashion-MNIST dataset. The implementation uses PyTorch Lightning for clean, scalable training.

## 📋 Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [File Structure](#file-structure)

## 🔍 Overview

Contrastive learning is a self-supervised learning approach that learns representations by pulling similar samples together and pushing dissimilar samples apart in the embedding space. This implementation uses SimCLR (Simple Framework for Contrastive Learning of Visual Representations) with MobileNetV3-Large as the backbone.

### Key Concepts:

- **Self-supervised learning**: No manual labels required during pre-training
- **Contrastive learning**: Learn by comparing augmented views of the same image
- **Linear evaluation**: Standard protocol to evaluate representation quality
- **MobileNetV3**: Efficient CNN architecture optimized for mobile devices

## 📊 Mathematical Foundation

### 1. Contrastive Loss (NT-Xent)

The core of contrastive learning is the Normalized Temperature-scaled Cross-Entropy (NT-Xent) loss, also known as InfoNCE loss.

For a batch of N samples, we create 2N augmented views. For each sample i, we have:

- Positive pair: (z*i, z*{i+N}) - two augmented views of the same image
- Negative pairs: All other samples in the batch

The loss for sample i is:

```
ℓ_i = -log(exp(sim(z_i, z_{i+N}) / τ) / Σ_{k=1}^{2N} 𝟙_{k≠i} exp(sim(z_i, z_k) / τ))
```

Where:

- `sim(u, v) = u^T v / (||u|| ||v||)` is cosine similarity
- `τ` is the temperature parameter (controls concentration)
- `𝟙_{k≠i}` is an indicator function excluding self-similarity

The total loss is the average over all samples:

$L = \frac{1}{2N} \sum_{i=1}^{2N} l_i$

### 2. Temperature Parameter (τ)

The temperature parameter controls the concentration of the distribution:

- **Low τ (< 0.1)**: Hard negatives, steep gradients, faster convergence but may overfit
- **High τ (> 0.5)**: Soft negatives, gentler gradients, more stable but slower learning
- **Typical range**: 0.05 - 0.2 (we use 0.07)

### 3. Cosine Similarity

Cosine similarity measures the angle between vectors, making it scale-invariant:

```
cos(θ) = (A · B) / (||A|| ||B||)
```

This is preferred over Euclidean distance because:

- Scale invariant
- Focuses on direction rather than magnitude
- Works well with normalized embeddings

### 4. Projection Head

The projection head is a small MLP that maps backbone features to contrastive space:

```
z = g(f(x)) = W_2 · ReLU(W_1 · f(x))
```

Where:

- `f(x)` is the backbone feature extractor
- `g()` is the projection head
- The projection head is discarded after training

## 🏗️ Architecture

### Overall Pipeline

```
Input Image → Data Augmentation → Backbone → Projection Head → Normalized Embeddings
     ↓
Fashion-MNIST → ContrastiveTransform → MobileNetV3 → ProjectionHead → L2 Normalize
```

### 1. Data Augmentation Pipeline

```python
ContrastiveTransform:
├── Resize(224, 224)
├── RandomHorizontalFlip(p=0.5)
├── RandomRotation(10°)
├── ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
├── RandomGrayscale(p=0.2)
├── ToTensor()
└── Normalize(ImageNet stats)
```

Strong augmentations are crucial for contrastive learning to prevent trivial solutions.

### 2. MobileNetV3-Large Backbone

- **Input**: 224×224×3 RGB images
- **Output**: 960-dimensional feature vectors
- **Pretrained**: ImageNet weights (transfer learning)
- **Modification**: Remove final classification head

### 3. Projection Head

```python
ProjectionHead:
├── Linear(960 → 512)
├── ReLU()
└── Linear(512 → 128)
```

The projection head maps backbone features to a lower-dimensional space optimized for contrastive learning.

### 4. L2 Normalization

All embeddings are L2 normalized before computing similarities:

```python
z_normalized = z / ||z||_2
```

This ensures cosine similarity is computed correctly and prevents magnitude domination.

## 💻 Implementation Details

### Training Strategy

The training follows a two-phase approach:

#### Phase 1: Contrastive Pre-training

- **Objective**: Learn general visual representations without labels
- **Data**: Augmented image pairs from Fashion-MNIST
- **Loss**: NT-Xent contrastive loss
- **Duration**: 100 epochs with early stopping

#### Phase 2: Linear Evaluation

- **Objective**: Evaluate representation quality
- **Method**: Freeze backbone, train only linear classifier
- **Data**: Original Fashion-MNIST with labels
- **Metric**: Classification accuracy

### Key Hyperparameters

| Parameter       | Value | Rationale                                             |
| --------------- | ----- | ----------------------------------------------------- |
| Temperature (τ) | 0.07  | Standard SimCLR value, balances hard/soft negatives   |
| Batch Size      | 128   | Larger batches provide more negatives                 |
| Learning Rate   | 1e-3  | Conservative rate for stable training                 |
| Weight Decay    | 1e-4  | L2 regularization                                     |
| Projection Dim  | 128   | Common choice, balances expressiveness and efficiency |
| Embedding Norm  | L2    | Required for cosine similarity                        |

### Data Augmentation Rationale

Each augmentation serves a specific purpose:

- **RandomHorizontalFlip**: Horizontal invariance (clothing can be worn either way)
- **RandomRotation**: Small rotational invariance
- **ColorJitter**: Robustness to lighting conditions
- **RandomGrayscale**: Reduces color bias, focuses on shape/texture
- **Resize**: Ensures consistent input size for MobileNet

### Loss Implementation Details

The contrastive loss implementation handles several important details:

1. **Batch construction**: Each batch contains 2N samples (N originals + N augmented)
2. **Similarity matrix**: Computed efficiently using matrix multiplication
3. **Masking**: Self-similarities are masked out (set to -∞)
4. **Numerical stability**: Uses logsumexp for stable computation

```python
# Key insight: Positive pairs are at specific indices
pos_indices = torch.arange(batch_size)
pos_sim_1 = sim_matrix[pos_indices, pos_indices + batch_size]  # z1 -> z2
pos_sim_2 = sim_matrix[pos_indices + batch_size, pos_indices]  # z2 -> z1
```

## 🚀 Installation

```bash
# Clone the repository
git clone <repository-url>
cd mobilenetv3-contrastive

# Install dependencies
pip install torch torchvision pytorch-lightning tensorboard

# Optional: Install in virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Requirements

```
torch>=1.12.0
torchvision>=0.13.0
pytorch-lightning>=1.8.0
tensorboard>=2.10.0
numpy>=1.21.0
```

## 📖 Usage

### Basic Training

```bash
# Run complete training pipeline
python mobilenet_contrastive.py
```

### Custom Configuration

```python
# Modify hyperparameters
model = MobileNetV3Contrastive(
    temperature=0.05,      # Lower temperature for harder negatives
    learning_rate=5e-4,    # Lower learning rate
    weight_decay=1e-3,     # Higher regularization
    max_epochs=200         # Longer training
)
```

### Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir logs/

# View at http://localhost:6006
```

### Loading Pretrained Model

```python
# Load contrastive model
model = MobileNetV3Contrastive.load_from_checkpoint('checkpoints/best_model.ckpt')

# Extract features
features = model.backbone(images)
embeddings = model.projection_head(features)
```

## 📈 Results

### Expected Performance

- **Contrastive Training**: Validation loss should decrease and stabilize
- **Linear Evaluation**: Test accuracy should reach 85-90% on Fashion-MNIST
- **Training Time**: ~2-3 hours on single GPU (depends on hardware)

### Evaluation Metrics

1. **Contrastive Loss**: Lower is better (typical range: 2-4)
2. **Linear Probe Accuracy**: Classification accuracy using frozen features
3. **Representation Quality**: t-SNE visualization of learned embeddings

### Comparison Baselines

| Method              | Fashion-MNIST Accuracy |
| ------------------- | ---------------------- |
| Random Features     | ~30%                   |
| ImageNet Pretrained | ~85%                   |
| **Our Method**      | **~87-90%**            |
| Supervised Training | ~92%                   |

## 📁 File Structure

```
mobilenetv3-contrastive/
├── mobilenet_contrastive.py     # Main implementation
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── checkpoints/                 # Saved models
├── logs/                       # TensorBoard logs
├── data/                       # Fashion-MNIST dataset
└── notebooks/                  # Analysis notebooks (optional)
    ├── visualization.ipynb     # Embedding visualization
    └── analysis.ipynb         # Performance analysis
```

## 🔬 Advanced Topics

### 1. Negative Sampling

The effectiveness of contrastive learning heavily depends on negative sampling:

- **In-batch negatives**: Other samples in the same batch
- **Hard negatives**: Semantically similar but different samples
- **Memory bank**: Store embeddings from previous batches for more negatives

### 2. Temperature Annealing

Some implementations anneal temperature during training:

```python
# Linear annealing
current_temp = initial_temp * (1 - epoch / max_epochs) + final_temp * (epoch / max_epochs)
```

### 3. Momentum Encoders

Advanced methods like MoCo use momentum encoders:

```python
# EMA update of momentum encoder
for param_q, param_k in zip(encoder.parameters(), momentum_encoder.parameters()):
    param_k.data = param_k.data * momentum + param_q.data * (1 - momentum)
```

### 4. Multi-crop Strategy

Using multiple crops of different sizes can improve performance:

- Standard crops: 224×224
- Small crops: 96×96
- More views → better representations but higher computational cost

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**

   ```python
   # Reduce batch size
   batch_size = 64  # Instead of 128
   ```

2. **Loss not decreasing**

   - Check data augmentation strength
   - Verify positive pair construction
   - Adjust temperature parameter

3. **Poor linear evaluation**
   - Increase contrastive training epochs
   - Check feature normalization
   - Verify backbone is frozen during evaluation

### Debugging Tips

```python
# Visualize augmentations
import matplotlib.pyplot as plt

def visualize_augmentations(dataset, idx=0):
    aug1, aug2, label = dataset[idx]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(aug1.permute(1, 2, 0))
    ax2.imshow(aug2.permute(1, 2, 0))
    plt.show()

# Check similarity distribution
def analyze_similarities(model, dataloader):
    similarities = []
    for batch in dataloader:
        aug1, aug2, _ = batch
        z1, z2 = model(aug1), model(aug2)
        sim = F.cosine_similarity(z1, z2, dim=1)
        similarities.extend(sim.tolist())

    plt.hist(similarities, bins=50)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Positive Pair Similarities')
    plt.show()
```

## 📚 References

1. **SimCLR**: Chen, T., et al. "A simple framework for contrastive learning of visual representations." ICML 2020.
2. **MobileNetV3**: Howard, A., et al. "Searching for mobilenetv3." ICCV 2019.
3. **NT-Xent Loss**: Oord, A., et al. "Representation learning with contrastive predictive coding." arXiv 2018.
4. **Fashion-MNIST**: Xiao, H., et al. "Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms." arXiv 2017.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch Lightning team for the excellent framework
- Google Research for SimCLR methodology
- Fashion-MNIST dataset creators
- MobileNet architecture developers
