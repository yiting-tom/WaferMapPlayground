import timm
import torch
from lightning import seed_everything
from timm.layers.classifier import NormMlpClassifierHead
from torch import nn

from .blocks import MultiLayerResidualHead

seed_everything(42)


"""
TinyVit(
  (patch_embed): PatchEmbed(
    (conv1): ConvNorm(
      (conv): Conv2d(3, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (act): GELU(approximate='none')
    (conv2): ConvNorm(
      (conv): Conv2d(48, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (stages): Sequential(
    (0): ConvLayer(...
    (1): TinyVitStage(...)
    (2): TinyVitStage(...)
    (3): TinyVitStage(...)
  )
  (head): NormMlpClassifierHead(
    (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Identity())
    (norm): LayerNorm2d((576,), eps=1e-05, elementwise_affine=True)
    (flatten): Flatten(start_dim=1, end_dim=-1)
    (pre_logits): Identity()
    (drop): Dropout(p=0.0, inplace=False)
    (fc): Identity()
  )
)
"""


class TinyViT(nn.Module):
    def __init__(
        self,
        num_labels: int = 8,
        pretrained: str = "tiny_vit_21m_512.dist_in22k_ft_in1k",
        train_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "tiny_vit_21m_512.dist_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,  # Remove the classification head
            cache_dir="./timm_cache",
        )

        if train_backbone:
            self.backbone.train()
        else:
            self.backbone.requires_grad_(False)
        self.backbone.head: NormMlpClassifierHead

        head = MultiLayerResidualHead(
            in_features=self.backbone.head.in_features,
            num_classes=num_labels,
            hidden_dim=512,
            num_layers=3,
            dropout=0.1,
        )
        self.backbone.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
