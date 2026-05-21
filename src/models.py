from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet18


CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def build_model(name: str = "resnet18", num_classes: int = 10) -> nn.Module:
    if name.lower() != "resnet18":
        raise ValueError(f"Unsupported model: {name}")

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    return model


def save_checkpoint(
    path: str,
    model: nn.Module,
    epoch: int,
    best_acc: float,
    extra: dict | None = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return checkpoint
