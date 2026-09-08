"""Small image classifier and deterministic inference preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from .lpq import preprocess


class SmallHandwritingCNN(nn.Module):
    """Compact four-block CNN for the small PaHaW participant cohort."""

    def __init__(self, dropout: float = 0.30) -> None:
        super().__init__()
        channels = (3, 16, 32, 64, 128)
        blocks = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True):
            blocks.extend(
                (
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=False),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128 * 8 * 8, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features.flatten(1)).squeeze(1)

    @property
    def gradcam_layer(self) -> nn.Module:
        """Last convolutional activation layer used by the later XAI stage."""

        return self.features[-2]


def prepared_rgb(path: Path, image_size: int) -> np.ndarray:
    """Crop foreground, square-pad, and resize a source image for the CNN."""

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return preprocess(rgb, output_size=image_size)


def normalized_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Map RGB uint8 data to channel-first float values in [-1, 1]."""

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 array.")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0
    return (tensor - 0.5) / 0.5


def parameter_count(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
