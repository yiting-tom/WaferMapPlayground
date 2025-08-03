import torch
from lightning import seed_everything
from torch import nn

seed_everything(42)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization and dropout"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First residual connection
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual

        # Second normalization after residual
        x = self.norm2(x)
        return x


class MultiLayerResidualHead(nn.Module):
    """Multi-layer classification head with residual connections"""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Input projection if needed
        if in_features != hidden_dim:
            self.input_proj = nn.Linear(in_features, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )

        # Final classification layer
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.dropout_final = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 2D (already pooled) and 4D (feature maps) inputs
        if x.dim() == 4:
            x = self.global_pool(x)
        x = self.flatten(x)

        # Project to hidden dimension
        x = self.input_proj(x)

        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Final classification
        x = self.norm_final(x)
        x = self.dropout_final(x)
        x = self.classifier(x)

        return x
