from typing import Any

import timm
from timm import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src import config, model


def get_vision_base(
    config: config.VisionConfig,
) -> tuple[nn.Module, int, transforms.Compose]:
    base: nn.Module = timm.create_model(config.vision_model, num_classes=0, pretrained=True)

    # Get actual output dimension by doing a forward pass with a dummy tensor
    # This is more reliable than base.num_features for some models
    with torch.no_grad():
        timm_config: dict[str, Any] = data.resolve_data_config({}, model=base)
        sample_input = torch.randn(2, *timm_config["input_size"])
        sample_output: torch.Tensor = base(sample_input)
        num_features = sample_output.shape[-1]

    transform = data.transforms_factory.create_transform(**timm_config)
    return base, num_features, transform


def get_inverse_transform(
    transform: transforms.Compose,
) -> transforms.Normalize:
    # Extract mean and std from the last Normalize transform
    for t in reversed(transform.transforms):
        if isinstance(t, transforms.Normalize):
            return transforms.Normalize(
                mean=-t.mean / t.std,
                std=1.0 / t.std,
            )
    raise ValueError("No Normalize transform found in the transform chain")


class VisionModel(nn.Module):
    def __init__(self, base: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.projection = projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return self.projection(x)


def get_vision_model(
    config: config.VisionConfig,
) -> tuple[nn.Module, transforms.Compose, transforms.Normalize]:
    base, num_features, transform = get_vision_base(config)
    projection = model.ProjectionLayers(num_features, config.embed_dims, config.projection_layers)
    vision_model = VisionModel(base, projection)
    inverse_transform = get_inverse_transform(transform)
    return vision_model, transform, inverse_transform
