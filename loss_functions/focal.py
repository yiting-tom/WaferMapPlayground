from collections import Counter
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float | Literal["auto"] | None = None,
        gamma: float = 2.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """
        alpha: Tensor of shape [num_classes] (or None)
        gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B, C] raw logits
        targets: [B] class indices
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        focal_term = (1 - probs) ** self.gamma
        loss = -focal_term * log_probs

        if self.alpha == "auto":
            alpha = self.get_alpha(targets).to(inputs.device)
            alpha_factor = alpha[targets]
            loss = loss * alpha_factor.unsqueeze(1)

        elif isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(inputs.device)
            alpha_factor = alpha[targets]
            loss = loss * alpha_factor.unsqueeze(1)

        loss = (loss * targets_one_hot).sum(dim=1)  # pick only the target class loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # [B]

    def get_alpha(self, train_dataset: torch.utils.data.Dataset) -> torch.Tensor:
        labels = [label for _, label in train_dataset]
        class_counts = Counter(labels)
        num_classes = len(class_counts)
        class_counts_list = [class_counts[i] for i in range(num_classes)]
        total = sum(class_counts_list)
        alpha = [total / c for c in class_counts_list]
        alpha = torch.tensor(alpha, dtype=torch.float)
        alpha = alpha / alpha.sum()  # normalize
        return alpha


if __name__ == "__main__":
    train_dataset = ...
    # get all labels
    labels = [label for _, label in train_dataset]

    # count the number of each class
    class_counts = Counter(labels)
    print(class_counts)

    # assume class_counts is the result of the Counter
    num_classes = 10
    class_counts_list = [class_counts[i] for i in range(num_classes)]
    total = sum(class_counts_list)

    # use inverse proportional method, so that the class with less samples has higher weight
    alpha = [total / c for c in class_counts_list]

    # convert to tensor and normalize
    alpha = torch.tensor(alpha, dtype=torch.float)
    alpha = alpha / alpha.sum()  # normalize
    print(alpha)

    loss = FocalLoss(alpha=alpha, gamma=2.0)
    print(loss)
