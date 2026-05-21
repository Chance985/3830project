from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import entropy_from_logits, expected_calibration_error, nll_from_logits


@dataclass
class EvalResult:
    accuracy: float
    error_rate: float
    nll: float
    entropy: float
    ece: float
    runtime_sec: float
    selected_fraction: float
    num_samples: int


def clone_model(model: nn.Module, device: torch.device) -> nn.Module:
    cloned = copy.deepcopy(model)
    cloned.to(device)
    return cloned


def _set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def configure_source(model: nn.Module) -> nn.Module:
    model.eval()
    _set_requires_grad(model, False)
    return model


def configure_bn_adapt(model: nn.Module) -> nn.Module:
    model.train()
    _set_requires_grad(model, False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0.1
            module.track_running_stats = True
    return model


def configure_entropy_adapt(model: nn.Module) -> Iterable[nn.Parameter]:
    model.train()
    _set_requires_grad(model, False)
    params = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            if module.weight is not None:
                params.append(module.weight)
            if module.bias is not None:
                params.append(module.bias)
            module.track_running_stats = True
    return params


def _collect_batch_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    totals: Dict[str, torch.Tensor],
) -> None:
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    totals["correct"] += predictions.eq(targets).sum()
    totals["nll"] += nll_from_logits(logits, targets).sum()
    totals["entropy"] += entropy_from_logits(logits).sum()
    totals["count"] += torch.tensor(targets.numel(), device=targets.device)
    totals["confidences"].append(confidences.detach())
    totals["predictions"].append(predictions.detach())
    totals["targets"].append(targets.detach())


def _finalize(totals: Dict[str, torch.Tensor], runtime_sec: float, selected_fraction: float) -> EvalResult:
    count = int(totals["count"].item())
    confidences = torch.cat(totals["confidences"]) if totals["confidences"] else torch.empty(0)
    predictions = torch.cat(totals["predictions"]) if totals["predictions"] else torch.empty(0, dtype=torch.long)
    targets = torch.cat(totals["targets"]) if totals["targets"] else torch.empty(0, dtype=torch.long)
    accuracy = (totals["correct"] / totals["count"]).item()
    return EvalResult(
        accuracy=accuracy,
        error_rate=1.0 - accuracy,
        nll=(totals["nll"] / totals["count"]).item(),
        entropy=(totals["entropy"] / totals["count"]).item(),
        ece=expected_calibration_error(confidences, predictions, targets),
        runtime_sec=runtime_sec,
        selected_fraction=selected_fraction,
        num_samples=count,
    )


def evaluate_stream(
    model: nn.Module,
    loader,
    device: torch.device,
    method: str,
    lr: float = 1e-3,
    steps: int = 1,
    threshold: Optional[float] = None,
) -> EvalResult:
    method = method.lower()
    if method == "source":
        configure_source(model)
        optimizer = None
    elif method == "bn":
        configure_bn_adapt(model)
        optimizer = None
    elif method in {"entropy", "gated"}:
        params = list(configure_entropy_adapt(model))
        if not params:
            raise RuntimeError("No BatchNorm affine parameters found for entropy adaptation.")
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unknown TTA method: {method}")

    totals = {
        "correct": torch.zeros((), device=device),
        "nll": torch.zeros((), device=device),
        "entropy": torch.zeros((), device=device),
        "count": torch.zeros((), device=device),
        "confidences": [],
        "predictions": [],
        "targets": [],
    }
    selected_total = 0
    seen_total = 0
    start = time.perf_counter()

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        seen_total += targets.numel()

        if method in {"source", "bn"}:
            with torch.no_grad():
                logits = model(inputs)
        else:
            for _ in range(max(1, steps)):
                optimizer.zero_grad(set_to_none=True)
                logits_for_loss = model(inputs)
                entropies = entropy_from_logits(logits_for_loss)
                if method == "gated":
                    if threshold is None:
                        raise ValueError("Confidence threshold is required for gated TTA.")
                    confidences = F.softmax(logits_for_loss.detach(), dim=1).max(dim=1).values
                    mask = confidences >= threshold
                    selected_total += int(mask.sum().item())
                    if mask.any():
                        loss = entropies[mask].mean()
                    else:
                        loss = None
                else:
                    selected_total += targets.numel()
                    loss = entropies.mean()
                if loss is not None:
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                logits = model(inputs)

        _collect_batch_stats(logits, targets, totals)

    runtime = time.perf_counter() - start
    if method == "source":
        selected_fraction = 0.0
    elif method == "bn":
        selected_fraction = 1.0
    else:
        selected_fraction = selected_total / max(1, seen_total * max(1, steps))
    return _finalize(totals, runtime, selected_fraction)
