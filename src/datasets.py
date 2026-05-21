from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
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


class CIFAR10CDataset(Dataset):
    """CIFAR-10-C severity slice loaded from local .npy files.

    Expected layout:
      cifar10c_dir/
        labels.npy
        gaussian_noise.npy
        motion_blur.npy
        ...

    Each corruption file is expected to contain 50,000 images, ordered as five
    consecutive 10,000-image severity slices. This class never downloads data.
    """

    def __init__(
        self,
        cifar10c_dir: str | Path,
        corruption: str,
        severity: int,
        transform=None,
    ) -> None:
        if severity not in {1, 2, 3, 4, 5}:
            raise ValueError(f"CIFAR-10-C severity must be in 1..5; got {severity}")

        root = Path(cifar10c_dir)
        image_path = root / f"{corruption}.npy"
        if not image_path.exists() and corruption == "jpeg":
            image_path = root / "jpeg_compression.npy"
        labels_path = root / "labels.npy"
        if not image_path.exists():
            raise FileNotFoundError(
                f"CIFAR-10-C corruption file not found: {image_path}. "
                "Download/extract CIFAR-10-C manually and pass --cifar10c-dir."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"CIFAR-10-C labels file not found: {labels_path}. "
                "Expected labels.npy in the same local directory."
            )

        images = np.load(image_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        start = (severity - 1) * 10000
        end = severity * 10000
        if len(images) < end:
            raise ValueError(
                f"{image_path} has {len(images)} images, but severity {severity} "
                f"requires at least {end} images."
            )

        self.images = images[start:end]
        if len(labels) >= end:
            self.labels = labels[start:end]
        elif len(labels) == 10000:
            self.labels = labels
        else:
            raise ValueError(
                f"{labels_path} has {len(labels)} labels; expected 10,000 or at least {end}."
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.fromarray(np.asarray(self.images[index], dtype=np.uint8)).convert("RGB")
        target = int(self.labels[index])
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
    dataset: str = "synthetic",
    cifar10c_dir: str | None = None,
) -> DataLoader:
    dataset = dataset.lower()
    if dataset not in {"synthetic", "cifar10c"}:
        raise ValueError("dataset must be either 'synthetic' or 'cifar10c'")

    if dataset == "cifar10c" and corruption is not None:
        if cifar10c_dir is None:
            raise ValueError("--cifar10c-dir is required when --dataset cifar10c is used.")
        eval_dataset = CIFAR10CDataset(
            cifar10c_dir=cifar10c_dir,
            corruption=corruption,
            severity=severity if severity is not None else 1,
            transform=eval_transform(),
        )
        eval_dataset = _subset(eval_dataset, subset_size, seed)
    else:
        base = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=None,
        )
        base = _subset(base, subset_size, seed)
        eval_dataset = SyntheticCorruptionDataset(
            base_dataset=base,
            corruption=corruption,
            severity=severity,
            seed=seed,
            transform=eval_transform(),
        )

    return DataLoader(
        eval_dataset,
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
