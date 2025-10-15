import timm
from timm import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src import config


def get_vision_base(
    config: config.VisionConfig,
) -> tuple[nn.Module, int]:
    base = timm.create_model(config.vision_model, num_classes=0, pretrained=True)
    num_features = base.num_features
    return base, num_features


def get_vision_transform(config: config.VisionConfig) -> transforms.Compose:
    timm_config = data.resolve_data_config({}, model=config.vision_model)
    transform = data.transforms_factory.create_transform(**timm_config)
    return transform  # type: ignore


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


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class Normalize(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=self.dim)


def projection_layers(d_in: int, d_out: int, num_layers: int) -> list[nn.Module]:
    layers = []
    for _ in range(num_layers - 1):
        layers.extend([Projection(d_in, d_in), nn.GELU()])
    layers += [Projection(d_in, d_out)]
    return layers


def get_vision_model(
    config: config.VisionConfig,
) -> tuple[nn.Module, transforms.Compose, transforms.Normalize]:
    base, num_features = get_vision_base(config)
    projection = projection_layers(num_features, config.embed_dims, config.projection_layers)
    vision_model = nn.Sequential(*[base, *projection, Normalize(dim=-1)])
    transform = get_vision_transform(config)
    inverse_transform = get_inverse_transform(transform)
    return vision_model, transform, inverse_transform
