"""Controlled single-signal dynamic image encodings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import Sample, load_svc
from .static_image import StaticRenderConfig, fit_surface_points


SUPPORTED_SIGNALS = ("speed", "pressure", "altitude", "azimuth")


@dataclass(frozen=True)
class DynamicRenderConfig:
    """Parameters shared by every isolated dynamic signal."""

    image_size: int = 512
    margin: int = 20
    antialias_scale: int = 3
    line_width_px: int = 4
    channel_min: int = 30
    channel_max: int = 220
    constant_channels: tuple[int, int] = (30, 30)
    background_rgb: tuple[int, int, int] = (255, 255, 255)
    normalization_low_percentile: float = 5.0
    normalization_high_percentile: float = 95.0
    speed_width_min_px: int = 1
    speed_width_max_px: int = 8


@dataclass(frozen=True)
class SegmentSignal:
    """Values and diagnostics for consecutive pen-down segments."""

    surface: np.ndarray
    raw: np.ndarray
    transformed: np.ndarray
    normalized: np.ndarray
    low: float
    high: float
    degenerate_range: bool
    invalid_surface_time_steps: int


def surface_segment_signal(
    points: np.ndarray,
    signal: str,
    config: DynamicRenderConfig,
) -> SegmentSignal:
    """Compute and normalize one signal on pen-down segments only."""

    if signal not in SUPPORTED_SIGNALS:
        raise ValueError(f"Unsupported signal {signal!r}.")
    if points.ndim != 2 or points.shape[1] != 7:
        raise ValueError(f"Expected shape (n, 7), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Cannot encode non-finite signal values.")
    surface = (points[:-1, 3] > 0) & (points[1:, 3] > 0)
    if not np.any(surface):
        raise ValueError("Recording has no pen-down segments.")

    if signal == "speed":
        dx = np.diff(points[:, 1])
        dy = np.diff(points[:, 0])
        dt = np.diff(points[:, 2])
        invalid_surface = surface & (dt <= 0)
        invalid_count = int(np.count_nonzero(invalid_surface))
        if invalid_count:
            raise ValueError(f"Found {invalid_count} non-positive pen-down timestamp steps.")
        raw = np.zeros(points.shape[0] - 1, dtype=np.float64)
        raw[surface] = np.hypot(dx[surface], dy[surface]) / dt[surface]
        transformed = np.log1p(raw)
    elif signal == "azimuth":
        surface_points = points[:, 3] > 0
        radians = np.deg2rad(points[:, 4] / 10.0)
        surface_radians = radians[surface_points]
        reference = np.arctan2(np.mean(np.sin(surface_radians)), np.mean(np.cos(surface_radians)))
        centered = np.angle(np.exp(1j * (radians - reference)))
        centered_tenths = np.rad2deg(centered) * 10.0
        raw = (centered_tenths[:-1] + centered_tenths[1:]) / 2.0
        transformed = raw.copy()
        invalid_count = 0
    else:
        column = {"altitude": 5, "pressure": 6}[signal]
        raw = (points[:-1, column] + points[1:, column]) / 2.0
        transformed = raw.copy()
        invalid_count = 0

    surface_values = transformed[surface]
    low = float(np.percentile(surface_values, config.normalization_low_percentile))
    high = float(np.percentile(surface_values, config.normalization_high_percentile))
    degenerate = not np.isfinite(low) or not np.isfinite(high) or high <= low
    normalized = np.zeros_like(transformed, dtype=np.float64)
    if degenerate:
        normalized[surface] = 0.5
    else:
        normalized[surface] = np.clip((surface_values - low) / (high - low), 0.0, 1.0)
    return SegmentSignal(
        surface=surface,
        raw=raw,
        transformed=transformed,
        normalized=normalized,
        low=low,
        high=high,
        degenerate_range=degenerate,
        invalid_surface_time_steps=invalid_count,
    )


def render_dynamic_array(
    points: np.ndarray,
    signal: str,
    config: DynamicRenderConfig,
) -> tuple[np.ndarray, SegmentSignal]:
    """Render one isolated signal into red with fixed width and no air path."""

    segment_signal = surface_segment_signal(points, signal, config)
    scale = max(config.antialias_scale, 1)
    large_static_config = StaticRenderConfig(
        image_size=config.image_size * scale,
        margin=config.margin * scale,
        antialias_scale=1,
        line_width_px=config.line_width_px * scale,
        foreground_rgb=(config.channel_min, *config.constant_channels),
        background_rgb=config.background_rgb,
    )
    fitted = fit_surface_points(points, large_static_config)
    image = Image.new(
        "RGB",
        (large_static_config.image_size, large_static_config.image_size),
        config.background_rgb,
    )
    draw = ImageDraw.Draw(image)
    span = config.channel_max - config.channel_min
    red = np.rint(config.channel_min + span * segment_signal.normalized).astype(np.uint8)
    for index in np.flatnonzero(segment_signal.surface):
        colour = (int(red[index]), *config.constant_channels)
        draw.line(
            [tuple(fitted[index]), tuple(fitted[index + 1])],
            fill=colour,
            width=large_static_config.line_width_px,
        )
    if scale > 1:
        image = image.resize((config.image_size, config.image_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8), segment_signal


def render_sample(
    sample: Sample,
    signal: str,
    output_path: Path,
    config: DynamicRenderConfig,
) -> dict[str, object]:
    """Render and save one isolated-signal image with diagnostics."""

    _, points = load_svc(sample.path)
    rgb, values = render_dynamic_array(points, signal, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG", optimize=True)
    return {
        "encoding": signal,
        "subject_id": sample.subject_id,
        "task_id": sample.task_id,
        "repetition_id": sample.repetition_id,
        "label": sample.label,
        "image_path": str(output_path),
        "surface_segments": int(np.count_nonzero(values.surface)),
        "normalization_low": values.low,
        "normalization_high": values.high,
        "degenerate_range": values.degenerate_range,
        "invalid_surface_time_steps": values.invalid_surface_time_steps,
    }


def render_dynamic_triplet_array(
    points: np.ndarray,
    signals: tuple[str, str, str],
    config: DynamicRenderConfig,
) -> tuple[np.ndarray, dict[str, SegmentSignal]]:
    """Render three distinct signals in the declared RGB channel order."""

    if len(signals) != 3 or len(set(signals)) != 3:
        raise ValueError("A dynamic triplet must contain three distinct signals.")
    if any(signal not in SUPPORTED_SIGNALS for signal in signals):
        raise ValueError(f"Unsupported dynamic triplet: {signals!r}.")
    values = {signal: surface_segment_signal(points, signal, config) for signal in signals}
    surfaces = [values[signal].surface for signal in signals]
    if not all(np.array_equal(surfaces[0], current) for current in surfaces[1:]):
        raise AssertionError("Signal surface masks differ.")

    scale = max(config.antialias_scale, 1)
    large_static_config = StaticRenderConfig(
        image_size=config.image_size * scale,
        margin=config.margin * scale,
        antialias_scale=1,
        line_width_px=config.line_width_px * scale,
        foreground_rgb=(config.channel_min,) * 3,
        background_rgb=config.background_rgb,
    )
    fitted = fit_surface_points(points, large_static_config)
    image = Image.new(
        "RGB",
        (large_static_config.image_size, large_static_config.image_size),
        config.background_rgb,
    )
    draw = ImageDraw.Draw(image)
    span = config.channel_max - config.channel_min
    colours = {
        signal: np.rint(config.channel_min + span * values[signal].normalized).astype(np.uint8)
        for signal in signals
    }
    for index in np.flatnonzero(surfaces[0]):
        colour = tuple(int(colours[signal][index]) for signal in signals)
        draw.line(
            [tuple(fitted[index]), tuple(fitted[index + 1])],
            fill=colour,
            width=large_static_config.line_width_px,
        )
    if scale > 1:
        image = image.resize((config.image_size, config.image_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8), values


def render_speed_width_triplet_array(
    points: np.ndarray,
    signals: tuple[str, str, str],
    config: DynamicRenderConfig,
) -> tuple[np.ndarray, dict[str, SegmentSignal], np.ndarray]:
    """Render an RGB triplet while mapping lower speed to thicker strokes.

    Width is the only rendering difference from :func:`render_dynamic_triplet_array`.
    The speed signal uses the same surface-only ``log1p`` and recording-level p05/p95
    normalization used by the colour encodings. Returned widths are expressed at the
    final image resolution and are zero for segments that are not pen-down.
    """

    if len(signals) != 3 or len(set(signals)) != 3:
        raise ValueError("A dynamic triplet must contain three distinct signals.")
    if any(signal not in SUPPORTED_SIGNALS for signal in signals):
        raise ValueError(f"Unsupported dynamic triplet: {signals!r}.")
    if config.speed_width_min_px < 1:
        raise ValueError("Minimum speed width must be at least one pixel.")
    if config.speed_width_max_px < config.speed_width_min_px:
        raise ValueError("Maximum speed width must not be smaller than minimum width.")

    requested = tuple(dict.fromkeys((*signals, "speed")))
    values = {signal: surface_segment_signal(points, signal, config) for signal in requested}
    surfaces = [values[signal].surface for signal in requested]
    if not all(np.array_equal(surfaces[0], current) for current in surfaces[1:]):
        raise AssertionError("Signal surface masks differ.")
    surface = surfaces[0]

    widths = np.zeros(points.shape[0] - 1, dtype=np.int16)
    width_span = config.speed_width_max_px - config.speed_width_min_px
    widths[surface] = np.rint(
        config.speed_width_max_px - width_span * values["speed"].normalized[surface]
    ).astype(np.int16)
    widths[surface] = np.clip(
        widths[surface], config.speed_width_min_px, config.speed_width_max_px
    )

    scale = max(config.antialias_scale, 1)
    large_static_config = StaticRenderConfig(
        image_size=config.image_size * scale,
        margin=config.margin * scale,
        antialias_scale=1,
        line_width_px=config.line_width_px * scale,
        foreground_rgb=(config.channel_min,) * 3,
        background_rgb=config.background_rgb,
    )
    fitted = fit_surface_points(points, large_static_config)
    image = Image.new(
        "RGB",
        (large_static_config.image_size, large_static_config.image_size),
        config.background_rgb,
    )
    draw = ImageDraw.Draw(image)
    colour_span = config.channel_max - config.channel_min
    colours = {
        signal: np.rint(config.channel_min + colour_span * values[signal].normalized).astype(
            np.uint8
        )
        for signal in signals
    }
    for index in np.flatnonzero(surface):
        colour = tuple(int(colours[signal][index]) for signal in signals)
        draw.line(
            [tuple(fitted[index]), tuple(fitted[index + 1])],
            fill=colour,
            width=int(widths[index]) * scale,
        )
    if scale > 1:
        image = image.resize((config.image_size, config.image_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8), values, widths


def render_triplet_sample(
    sample: Sample,
    encoding: str,
    signals: tuple[str, str, str],
    output_path: Path,
    config: DynamicRenderConfig,
) -> dict[str, object]:
    """Render and save one three-signal RGB image with diagnostics."""

    _, points = load_svc(sample.path)
    rgb, values = render_dynamic_triplet_array(points, signals, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG", optimize=True)
    row: dict[str, object] = {
        "encoding": encoding,
        "red_signal": signals[0],
        "green_signal": signals[1],
        "blue_signal": signals[2],
        "subject_id": sample.subject_id,
        "task_id": sample.task_id,
        "repetition_id": sample.repetition_id,
        "label": sample.label,
        "image_path": str(output_path),
        "surface_segments": int(np.count_nonzero(values[signals[0]].surface)),
    }
    for signal in signals:
        row[f"{signal}_normalization_low"] = values[signal].low
        row[f"{signal}_normalization_high"] = values[signal].high
        row[f"{signal}_degenerate_range"] = values[signal].degenerate_range
        row[f"{signal}_invalid_surface_time_steps"] = values[signal].invalid_surface_time_steps
    return row


def render_speed_width_triplet_sample(
    sample: Sample,
    encoding: str,
    signals: tuple[str, str, str],
    output_path: Path,
    config: DynamicRenderConfig,
) -> dict[str, object]:
    """Render and save one RGB triplet with inverse-speed stroke width."""

    _, points = load_svc(sample.path)
    rgb, values, widths = render_speed_width_triplet_array(points, signals, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG", optimize=True)
    surface = values[signals[0]].surface
    surface_widths = widths[surface]
    row: dict[str, object] = {
        "encoding": encoding,
        "red_signal": signals[0],
        "green_signal": signals[1],
        "blue_signal": signals[2],
        "width_signal": "inverse_speed",
        "subject_id": sample.subject_id,
        "task_id": sample.task_id,
        "repetition_id": sample.repetition_id,
        "label": sample.label,
        "image_path": str(output_path),
        "surface_segments": int(np.count_nonzero(surface)),
        "stroke_width_min_observed_px": int(surface_widths.min()),
        "stroke_width_max_observed_px": int(surface_widths.max()),
        "stroke_width_mean_px": float(surface_widths.mean()),
        "stroke_width_unique_count": int(np.unique(surface_widths).size),
    }
    for signal in values:
        row[f"{signal}_normalization_low"] = values[signal].low
        row[f"{signal}_normalization_high"] = values[signal].high
        row[f"{signal}_degenerate_range"] = values[signal].degenerate_range
        row[f"{signal}_invalid_surface_time_steps"] = values[signal].invalid_surface_time_steps
    return row
