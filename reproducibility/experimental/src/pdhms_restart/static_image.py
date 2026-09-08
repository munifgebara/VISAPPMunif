"""Render a static reconstruction from only the on-surface trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import Sample, load_svc


@dataclass(frozen=True)
class StaticRenderConfig:
    """Parameters that fully define the static reconstruction."""

    image_size: int = 512
    margin: int = 20
    antialias_scale: int = 3
    line_width_px: int = 4
    foreground_rgb: tuple[int, int, int] = (30, 30, 30)
    background_rgb: tuple[int, int, int] = (255, 255, 255)


def fit_surface_points(points: np.ndarray, config: StaticRenderConfig) -> np.ndarray:
    """Fit all coordinates using bounds derived only from pen-down points."""

    if points.ndim != 2 or points.shape[1] != 7:
        raise ValueError(f"Expected shape (n, 7), got {points.shape}.")
    surface = points[:, 3] > 0
    if not np.any(surface):
        raise ValueError("Recording has no on-surface points.")
    x = points[:, 1]
    y = points[:, 0]
    x_surface = x[surface]
    y_surface = y[surface]
    x_min, x_max = float(x_surface.min()), float(x_surface.max())
    y_min, y_max = float(y_surface.min()), float(y_surface.max())
    drawable = float(config.image_size - (2 * config.margin))
    if drawable <= 0:
        raise ValueError("Image size must exceed twice the margin.")
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    scale = min(drawable / x_range, drawable / y_range)
    used_width = (x_max - x_min) * scale
    used_height = (y_max - y_min) * scale
    x_offset = config.margin + (drawable - used_width) / 2.0
    y_offset = config.margin + (drawable - used_height) / 2.0
    return np.column_stack(
        (x_offset + (x - x_min) * scale, y_offset + (y - y_min) * scale)
    ).astype(np.float64)


def render_static_array(points: np.ndarray, config: StaticRenderConfig) -> np.ndarray:
    """Return an RGB raster with constant-width pen-down segments only."""

    if not np.all(np.isfinite(points)):
        raise ValueError("Cannot render non-finite coordinates.")
    scale = max(config.antialias_scale, 1)
    large_config = StaticRenderConfig(
        image_size=config.image_size * scale,
        margin=config.margin * scale,
        antialias_scale=1,
        line_width_px=config.line_width_px * scale,
        foreground_rgb=config.foreground_rgb,
        background_rgb=config.background_rgb,
    )
    fitted = fit_surface_points(points, large_config)
    image = Image.new("RGB", (large_config.image_size, large_config.image_size), config.background_rgb)
    draw = ImageDraw.Draw(image)
    surface = points[:, 3] > 0
    for index in range(points.shape[0] - 1):
        if surface[index] and surface[index + 1]:
            draw.line(
                [tuple(fitted[index]), tuple(fitted[index + 1])],
                fill=config.foreground_rgb,
                width=large_config.line_width_px,
            )
    if scale > 1:
        image = image.resize((config.image_size, config.image_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def render_sample(sample: Sample, output_path: Path, config: StaticRenderConfig) -> dict[str, object]:
    """Render and save one sample, returning manifest fields."""

    _, points = load_svc(sample.path)
    rgb = render_static_array(points, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG", optimize=True)
    foreground = np.any(rgb < 250, axis=2)
    return {
        "subject_id": sample.subject_id,
        "task_id": sample.task_id,
        "repetition_id": sample.repetition_id,
        "label": sample.label,
        "image_path": str(output_path),
        "foreground_pixels": int(np.count_nonzero(foreground)),
    }
