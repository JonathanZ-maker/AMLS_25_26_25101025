"""
Model B - ResNet-18 adapted for 28x28, 1-channel input.

- Input: (N, 1, 28, 28)
- Modifications:
  * conv1: 3x3, stride=1, padding=1
  * remove maxpool
  * fc: output 2 classes
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_1ch_28(num_classes: int = 2) -> nn.Module:
    model = resnet18(weights=None)

    # Adapt first conv for 1-channel and small images
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_trainable_backbone(model: nn.Module, train_backbone: bool) -> None:
    """
    Freeze/unfreeze backbone. Always keep final fc trainable.
    """
    for name, p in model.named_parameters():
        p.requires_grad = True

    if not train_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
