"""Controls for the association between pen signals and positions within strokes.

Normalization is delegated to the original dynamic renderer. A control only moves
the resulting signal vectors between consecutive pen-down segments of the same
stroke. It does not move points, join pen lifts, or change channel semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from operator import index as integer_index
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from .dynamic_image import DynamicRenderConfig, SUPPORTED_SIGNALS, surface_segment_signal
from .static_image import StaticRenderConfig, fit_surface_points


Control = Literal["identity", "circular_shift", "joint_permutation"]


def _array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class StrokeShift:
    """One maximal contiguous run of pen-down segments; stop is exclusive."""

    ordinal: int
    start: int
    stop: int
    shift: int | None

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class SpatialControlDiagnostics:
    """Segment-level provenance, with a compact JSON-compatible summary.

    Row i of controlled_normalized comes from source_indices[i] of the original
    array. Columns follow signals; colour arrays additionally contain the fixed
    green/blue channels for a single-signal rendering. The arrays include all
    consecutive-point segments, with non-surface rows unchanged.
    """

    control: str
    record_key: str
    seed: int
    signals: tuple[str, ...]
    surface: np.ndarray
    source_indices: np.ndarray
    original_normalized: np.ndarray
    controlled_normalized: np.ndarray
    original_colours: np.ndarray
    controlled_colours: np.ndarray
    strokes: tuple[StrokeShift, ...]
    normalization: dict[str, dict[str, float | bool | int]]
    geometry_sha256: str

    def summary(self) -> dict[str, object]:
        """Report actual changed values separately from moved source indices."""
        segment_indices = np.arange(len(self.surface), dtype=np.int64)
        moved = self.source_indices != segment_indices
        changed = np.any(self.controlled_normalized != self.original_normalized, axis=1)
        colour_changed = np.any(self.controlled_colours != self.original_colours, axis=1)
        count = int(np.count_nonzero(self.surface))
        stroke_rows = []
        for stroke in self.strokes:
            span = slice(stroke.start, stroke.stop)
            expected_indices = segment_indices[span]
            source_indices = self.source_indices[span]
            preserved = bool(
                np.array_equal(np.sort(source_indices), expected_indices)
                and np.array_equal(
                    self.controlled_normalized[span], self.original_normalized[source_indices]
                )
            )
            stroke_rows.append({
                **asdict(stroke),
                "length": stroke.length,
                "shift_eligible": stroke.length > 1,
                "shift_fraction": None if stroke.shift is None else stroke.shift / stroke.length,
                "moved_segments": int(np.count_nonzero(moved[span])),
                "changed_signal_vectors": int(np.count_nonzero(changed[span])),
                "changed_colour_vectors": int(np.count_nonzero(colour_changed[span])),
                "joint_vectors_preserved": preserved,
            })
        return {
            "control": self.control,
            "record_key": self.record_key,
            "seed": self.seed,
            "signals": list(self.signals),
            "surface_segments": count,
            "stroke_count": len(self.strokes),
            "shift_eligible_strokes": sum(stroke.length > 1 for stroke in self.strokes),
            "moved_segment_fraction": float(np.count_nonzero(moved & self.surface) / count),
            "changed_signal_vector_fraction": float(np.count_nonzero(changed & self.surface) / count),
            "changed_colour_vector_fraction": float(np.count_nonzero(colour_changed & self.surface) / count),
            "surface_mask_preserved": bool(np.array_equal(self.surface[self.source_indices], self.surface)),
            "non_surface_rows_unchanged": bool(
                np.array_equal(self.source_indices[~self.surface], segment_indices[~self.surface])
                and np.array_equal(self.controlled_normalized[~self.surface], self.original_normalized[~self.surface])
            ),
            "joint_vectors_preserved_per_stroke": all(row["joint_vectors_preserved"] for row in stroke_rows),
            "surface_mask_sha256": _array_hash(self.surface),
            "geometry_sha256": self.geometry_sha256,
            "original_normalized_sha256": _array_hash(self.original_normalized),
            "controlled_normalized_sha256": _array_hash(self.controlled_normalized),
            "source_indices_sha256": _array_hash(self.source_indices),
            "normalization": self.normalization,
            "strokes": stroke_rows,
        }


def _stroke_spans(surface: np.ndarray) -> list[tuple[int, int]]:
    transitions = np.diff(np.r_[False, surface, False].astype(np.int8))
    return list(zip(np.flatnonzero(transitions == 1).tolist(),
                    np.flatnonzero(transitions == -1).tolist()))


def _shift_for_stroke(record_key: str, seed: int, ordinal: int, start: int, stop: int) -> int:
    length = stop - start
    if length < 2:
        return 0
    low = max(1, (length + 3) // 4)
    high = min(length - 1, (3 * length) // 4)
    # SHA-256 fixes the mapping across processes and avoids Python's salted hash
    # and shared RNG state. Signals and diagnosis are deliberately absent.
    payload = json.dumps(["pdhms-spatial-control-v1", seed, record_key, ordinal, start, stop],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    number = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return low + number % (high - low + 1)


def render_spatial_control_array(
    points: np.ndarray,
    signals: str | tuple[str, ...],
    config: DynamicRenderConfig,
    *,
    control: Control = "circular_shift",
    record_key: str,
    seed: int,
) -> tuple[np.ndarray, SpatialControlDiagnostics]:
    """Render an isolated signal or RGB triplet after a joint within-stroke shift.

    Use a stable recording key (for example, its relative SVC filename) independent
    of diagnosis. No Sample object, label, prediction or cohort statistic enters
    this API. Positive shifts follow numpy.roll: a vector moves toward larger
    segment indices, wrapping at its stroke boundary. Shifts are nonzero and lie
    between ceil(n/4) and floor(3*n/4); one-segment strokes remain unchanged.

    joint_permutation instead applies a seeded joint permutation in each stroke,
    including possible fixed points; diagnostics report how many values change.
    Identity reproduces the original renderer, including acquisition-order
    overpainting, per-recording normalization, antialiasing and colour rounding.
    Preserved multisets refer to pre-rasterization segment vectors. Overpainting
    and antialiasing can change raster histograms and thresholded foreground masks
    after a shift, despite identical geometry and segment masks.
    """
    if control not in ("identity", "circular_shift", "joint_permutation"):
        raise ValueError(f"Unsupported spatial control {control!r}.")
    if not isinstance(record_key, str) or not record_key:
        raise ValueError("record_key must be a nonempty diagnosis-independent string.")
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer, not a boolean.")
    seed = integer_index(seed)
    signals = (signals,) if isinstance(signals, str) else tuple(signals)
    if len(signals) not in (1, 3) or len(set(signals)) != len(signals):
        raise ValueError("Choose one signal or three distinct signals in RGB order.")
    if any(signal not in SUPPORTED_SIGNALS for signal in signals):
        raise ValueError(f"Unsupported dynamic signals {signals!r}.")

    values = {signal: surface_segment_signal(points, signal, config) for signal in signals}
    surface = values[signals[0]].surface.copy()
    if not all(np.array_equal(surface, value.surface) for value in values.values()):
        raise AssertionError("Signal surface masks differ.")
    original = np.column_stack([values[signal].normalized for signal in signals])
    source_indices = np.arange(len(surface), dtype=np.int64)
    strokes = []
    for ordinal, (start, stop) in enumerate(_stroke_spans(surface)):
        indices = np.arange(start, stop, dtype=np.int64)
        if control == "joint_permutation":
            payload = json.dumps(["pdhms-spatial-permutation-v1", seed, record_key, ordinal, start, stop],
                                 ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest(), "big"))
            source_indices[start:stop] = rng.permutation(indices)
            shift = None
        else:
            shift = 0 if control == "identity" else _shift_for_stroke(record_key, seed, ordinal, start, stop)
            source_indices[start:stop] = np.roll(indices, shift)
        strokes.append(StrokeShift(ordinal, start, stop, shift))
    controlled = original[source_indices].copy()
    span = config.channel_max - config.channel_min
    original_colours = np.rint(config.channel_min + span * original).astype(np.uint8)
    controlled_colours = np.rint(config.channel_min + span * controlled).astype(np.uint8)
    if len(signals) == 1:
        constants = np.tile(np.asarray(config.constant_channels, dtype=np.uint8), (len(surface), 1))
        original_colours = np.column_stack([original_colours, constants])
        controlled_colours = np.column_stack([controlled_colours, constants])

    scale = max(config.antialias_scale, 1)
    large_config = StaticRenderConfig(
        image_size=config.image_size * scale,
        margin=config.margin * scale,
        antialias_scale=1,
        line_width_px=config.line_width_px * scale,
        foreground_rgb=(config.channel_min, *config.constant_channels)
        if len(signals) == 1 else (config.channel_min,) * 3,
        background_rgb=config.background_rgb,
    )
    fitted = fit_surface_points(points, large_config)
    image = Image.new("RGB", (large_config.image_size, large_config.image_size), config.background_rgb)
    draw = ImageDraw.Draw(image)
    for index in np.flatnonzero(surface):
        draw.line([tuple(fitted[index]), tuple(fitted[index + 1])],
                  fill=tuple(int(value) for value in controlled_colours[index]),
                  width=large_config.line_width_px)
    if scale > 1:
        image = image.resize((config.image_size, config.image_size), Image.Resampling.LANCZOS)

    geometry = np.stack([fitted[:-1][surface], fitted[1:][surface]], axis=1)
    geometry_hash = hashlib.sha256(
        (_array_hash(geometry) + json.dumps(asdict(large_config), sort_keys=True)).encode("ascii")
    ).hexdigest()
    normalization = {
        signal: {
            "low": value.low,
            "high": value.high,
            "degenerate_range": bool(value.degenerate_range),
            "invalid_surface_time_steps": value.invalid_surface_time_steps,
        }
        for signal, value in values.items()
    }
    for array in (surface, source_indices, original, controlled, original_colours, controlled_colours):
        array.setflags(write=False)
    diagnostics = SpatialControlDiagnostics(
        control, record_key, seed, signals, surface, source_indices, original, controlled,
        original_colours, controlled_colours, tuple(strokes), normalization, geometry_hash,
    )
    return np.asarray(image, dtype=np.uint8), diagnostics
