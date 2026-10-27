from __future__ import annotations

import torch
from torch import nn
from torchvision import models


MODEL_NAMES = ("resnet18", "efficientnet_b0", "convnext_tiny", "resnet50")
MODEL_LABELS = {
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
    "resnet50": "ResNet50",
}
MODEL_TRAINING_DEFAULTS = {
    "resnet18": {"max_epochs": 25, "patience": 5, "learning_rate": 1e-4, "batch_size": 32},
    "efficientnet_b0": {"max_epochs": 20, "patience": 4, "learning_rate": 1e-4, "batch_size": 32},
    "convnext_tiny": {"max_epochs": 50, "patience": 8, "learning_rate": 5e-5, "batch_size": 16},
    "resnet50": {"max_epochs": 30, "patience": 6, "learning_rate": 1e-4, "batch_size": 16},
}


def _replace_input_conv(model: nn.Module, path: str, in_channels: int, pretrained: bool) -> None:
    parent: nn.Module = model
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    old_conv = parent[int(last)] if last.isdigit() else getattr(parent, last)
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels, old_conv.kernel_size, old_conv.stride,
        old_conv.padding, dilation=old_conv.dilation, groups=old_conv.groups, bias=False,
    )
    with torch.no_grad():
        if pretrained:
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    if last.isdigit():
        parent[int(last)] = new_conv
    else:
        setattr(parent, last, new_conv)


def create_binary_classifier(
    model_name: str = "resnet18", in_channels: int = 3, pretrained: bool = False
) -> nn.Module:
    """Create one of the supported two-class image classifiers."""
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        _replace_input_conv(model, "conv1", in_channels, pretrained)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        _replace_input_conv(model, "conv1", in_channels, pretrained)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        _replace_input_conv(model, "features.0.0", in_channels, pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        )
        _replace_input_conv(model, "features.0.0", in_channels, pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    else:
        raise ValueError(f"Unknown model {model_name!r}; expected one of {MODEL_NAMES}")
    model.model_name = model_name
    model.input_channels = in_channels
    return model


def get_gradcam_layer(model: nn.Module) -> nn.Module:
    if getattr(model, "model_name", "resnet18") in {"resnet18", "resnet50"}:
        return model.layer4[-1]
    return model.features[-1]


def get_input_channels(model: nn.Module) -> int:
    return int(getattr(model, "input_channels", 3))


def create_resnet18_binary(in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """Backward-compatible alias for legacy callers."""
    return create_binary_classifier("resnet18", in_channels, pretrained)

