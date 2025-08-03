import torch
from lightning import seed_everything
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .blocks import MultiLayerResidualHead

seed_everything(42)


"""
MobileNetV3(
  (features): Sequential(
    (0): Conv2dNormActivation(...)
    (2): InvertedResidual(...)
    ...
    (15): InvertedResidual(...)
    (16): Conv2dNormActivation(
      (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): Hardswish()
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=1)
  (classifier): Sequential(
    (0): Linear(in_features=960, out_features=1280, bias=True)
    (1): Hardswish()
    (2): Dropout(p=0.2, inplace=True)
    (3): Linear(in_features=1280, out_features=1000, bias=True)
  )
)
"""


class MobileNetV3Large(nn.Module):
    def __init__(
        self,
        num_labels: int = 8,
        pretrained: str
        | MobileNet_V3_Large_Weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        train_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = mobilenet_v3_large(weights=pretrained)

        if not train_backbone:
            self.backbone.features.requires_grad_(False)

        input_features = self.backbone.features[-1].out_channels  # This is 960

        self.backbone.head = MultiLayerResidualHead(
            in_features=input_features,
            num_classes=num_labels,
            hidden_dim=512,
            num_layers=3,
            dropout=0.1,
        )
        self.backbone.avgpool = nn.Identity()
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        return self.backbone.head(features)
