from __future__ import annotations

from io import BytesIO
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


SEVERITIES: List[int] = [1, 2, 3, 4, 5]


def _as_uint8_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def gaussian_noise(image: Image.Image, severity: int) -> Image.Image:
    scales = [8, 12, 18, 25, 35]
    arr = _as_uint8_array(image).astype(np.float32)
    noise = np.random.normal(0.0, scales[severity - 1], size=arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def gaussian_blur(image: Image.Image, severity: int) -> Image.Image:
    radii = [0.4, 0.7, 1.0, 1.3, 1.7]
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radii[severity - 1]))


def motion_blur(image: Image.Image, severity: int) -> Image.Image:
    sizes = [3, 5, 7, 9, 11]
    kernel_size = sizes[severity - 1]
    arr = _as_uint8_array(image).astype(np.float32)
    pad = kernel_size // 2
    padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    blurred = np.zeros_like(arr)
    for offset in range(kernel_size):
        blurred += padded[:, offset : offset + arr.shape[1], :]
    blurred /= kernel_size
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


def brightness(image: Image.Image, severity: int) -> Image.Image:
    factors = [1.25, 1.45, 1.70, 2.00, 2.35]
    return ImageEnhance.Brightness(image.convert("RGB")).enhance(factors[severity - 1])


def contrast(image: Image.Image, severity: int) -> Image.Image:
    factors = [0.75, 0.60, 0.45, 0.30, 0.18]
    return ImageEnhance.Contrast(image.convert("RGB")).enhance(factors[severity - 1])


def pixelate(image: Image.Image, severity: int) -> Image.Image:
    sizes = [24, 20, 16, 12, 8]
    image = image.convert("RGB")
    small = image.resize((sizes[severity - 1], sizes[severity - 1]), Image.Resampling.BILINEAR)
    return small.resize(image.size, Image.Resampling.NEAREST)


def jpeg(image: Image.Image, severity: int) -> Image.Image:
    qualities = [60, 45, 30, 20, 12]
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=qualities[severity - 1])
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


CORRUPTIONS: Dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "brightness": brightness,
    "contrast": contrast,
    "pixelate": pixelate,
    "jpeg": jpeg,
}


def apply_corruption(image: Image.Image, corruption: str, severity: int) -> Image.Image:
    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{corruption}'. Valid: {sorted(CORRUPTIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"Severity must be one of {SEVERITIES}; got {severity}")
    return CORRUPTIONS[corruption](image, severity)
