from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_config(path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1)


def nll_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets, reduction="none")


def expected_calibration_error(
    confidences: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confidences.device)
    ece = torch.zeros((), device=confidences.device)
    correct = predictions.eq(targets)
    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = correct[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return ece.item()


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def write_rows_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
