"""Fixed channel-wise Local Phase Quantization descriptor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


LPQ_IMAGE_SIZE = 64
LPQ_WINDOW_SIZE = 5
LPQ_BINS = 256
LPQ_RGB_DIMENSION = 768


def load_rgb(path: Path) -> np.ndarray:
    """Load one image as an RGB uint8 array."""

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def preprocess(rgb: np.ndarray, output_size: int = LPQ_IMAGE_SIZE) -> np.ndarray:
    """Crop foreground, pad to a square, and resize deterministically."""

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got {rgb.shape}.")
    mask = np.any(rgb < 250, axis=2)
    if not np.any(mask):
        return np.full((output_size, output_size, 3), 255, dtype=np.uint8)
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    height, width = y1 - y0, x1 - x0
    margin = int(round(max(height, width) * 0.05))
    y0, y1 = max(0, y0 - margin), min(rgb.shape[0], y1 + margin)
    x0, x1 = max(0, x0 - margin), min(rgb.shape[1], x1 + margin)
    crop = rgb[y0:y1, x0:x1]
    side = max(crop.shape[:2])
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    oy = (side - crop.shape[0]) // 2
    ox = (side - crop.shape[1]) // 2
    canvas[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
    resized = Image.fromarray(canvas, mode="RGB").resize(
        (output_size, output_size), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.uint8)


def channel_codes(channel: np.ndarray) -> np.ndarray:
    """Compute the 8-bit LPQ code for each channel pixel."""

    values = np.asarray(channel, dtype=np.float32) / 255.0
    radius = LPQ_WINDOW_SIZE // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    base = np.ones(LPQ_WINDOW_SIZE, dtype=np.complex64)
    wave = np.exp(-2j * np.pi * coordinates / float(LPQ_WINDOW_SIZE)).astype(np.complex64)
    filters = (
        np.outer(base, wave),
        np.outer(wave, base),
        np.outer(wave, wave),
        np.outer(wave, np.conj(wave)),
    )
    codes = np.zeros(values.shape, dtype=np.uint8)
    for index, kernel in enumerate(filters):
        response = ndimage.convolve(values, kernel, mode="reflect")
        codes |= (response.real >= 0).astype(np.uint8) << (2 * index)
        codes |= (response.imag >= 0).astype(np.uint8) << (2 * index + 1)
    return codes


def extract(rgb: np.ndarray) -> np.ndarray:
    """Extract 256 Hellinger-transformed LPQ bins from each RGB channel."""

    prepared = preprocess(rgb)
    blocks: list[np.ndarray] = []
    for channel in range(3):
        codes = channel_codes(prepared[:, :, channel])
        histogram = np.bincount(codes.ravel(), minlength=LPQ_BINS).astype(np.float32)
        histogram /= max(float(histogram.sum()), 1.0)
        blocks.append(np.sqrt(histogram).astype(np.float32))
    vector = np.concatenate(blocks).astype(np.float32)
    if vector.shape != (LPQ_RGB_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise ValueError("Invalid LPQ feature vector.")
    return vector


def extract_path(path: Path) -> np.ndarray:
    """Load an image and extract its LPQ descriptor."""

    return extract(load_rgb(path))


def feature_names() -> list[str]:
    """Return stable names for all descriptor positions."""

    return [f"lpq_{channel}_{index:03d}" for channel in "rgb" for index in range(LPQ_BINS)]
