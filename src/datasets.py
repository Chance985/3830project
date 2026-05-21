from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .corruptions import apply_corruption


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _subset(dataset: Dataset, subset_size: Optional[int], seed: int) -> Dataset:
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=subset_size, replace=False)
    return Subset(dataset, indices.tolist())


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


class SyntheticCorruptionDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        corruption: str | None,
        severity: int | None,
        seed: int,
        transform=None,
    ) -> None:
        self.base_dataset = base_dataset
        self.corruption = corruption
        self.severity = severity
        self.seed = seed
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _sample_seed(self, index: int) -> int:
        key = f"{self.seed}:{index}:{self.corruption}:{self.severity}".encode("utf-8")
        return int(hashlib.sha256(key).hexdigest()[:8], 16)

    def __getitem__(self, index: int):
        image, target = self.base_dataset[index]
        if self.corruption is not None:
            if self.severity is None:
                raise ValueError("Severity is required when corruption is set.")
            np_state = np.random.get_state()
            np.random.seed(self._sample_seed(index))
            image = apply_corruption(image, self.corruption, self.severity)
            np.random.set_state(np_state)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def make_train_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    subset_size: Optional[int] = None,
) -> DataLoader:
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform(),
    )
    dataset = _subset(dataset, subset_size, seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def make_eval_loader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    subset_size: Optional[int] = None,
    corruption: str | None = None,
    severity: int | None = None,
) -> DataLoader:
    base = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=None,
    )
    base = _subset(base, subset_size, seed)
    dataset = SyntheticCorruptionDataset(
        base_dataset=base,
        corruption=corruption,
        severity=severity,
        seed=seed,
        transform=eval_transform(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )


def parse_list(values: str | Iterable[str]) -> list[str]:
    if isinstance(values, str):
        return [value.strip() for value in values.split(",") if value.strip()]
    return list(values)
