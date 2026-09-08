"""Fixed, label-free kinematic and pressure summaries of PaHaW recordings.

This is an independently specified baseline inspired by the signal families in
Drotar et al. (2016), not a reproduction of that paper's feature set. Coordinates
and pressure retain their native tablet units; millisecond differences become
seconds. No image fitting, recording-level normalization or learned transform is
applied here. See ``kinematic_descriptor.json`` beside the experiment config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TIMESTAMP_SECONDS_PER_UNIT = 0.001
MAX_VALID_INTERVAL_MS = 60_000.0
SUMMARY_NAMES = ("mean", "std", "median", "p05", "p95", "iqr")
GLOBAL_FEATURE_NAMES = (
    "valid_acquisition_time_s",
    "contact_acquisition_time_s",
    "air_acquisition_time_s",
    "transition_acquisition_time_s",
    "contact_time_fraction",
    "contact_stroke_count",
    "air_stroke_count",
    "contact_path_length_native",
    "air_path_length_native",
    "contact_bbox_width_native",
    "contact_bbox_height_native",
)
DISTRIBUTION_NAMES = (
    "contact_stroke_duration_s", "contact_stroke_length_native",
    "air_stroke_duration_s", "air_stroke_length_native",
    "contact_speed_native_per_s", "contact_acceleration_native_per_s2",
    "contact_jerk_native_per_s3", "air_speed_native_per_s",
    "air_acceleration_native_per_s2", "air_jerk_native_per_s3",
    "contact_pressure_native", "contact_absolute_pressure_rate_native_per_s",
)
FEATURE_NAMES = GLOBAL_FEATURE_NAMES + tuple(
    f"{distribution}__{summary}"
    for distribution in DISTRIBUTION_NAMES
    for summary in SUMMARY_NAMES
)


@dataclass(frozen=True)
class KinematicFeatures:
    values: np.ndarray
    diagnostics: dict[str, object]


def summarize(values: np.ndarray) -> tuple[float, ...]:
    """Six fixed summaries; a nonexistent distribution is represented by zeros."""
    if values.size == 0:
        return (0.0,) * len(SUMMARY_NAMES)
    q05, q25, median, q75, q95 = np.percentile(values, (5, 25, 50, 75, 95))
    return (
        float(np.mean(values)), float(np.std(values, ddof=0)),
        float(median), float(q05), float(q95), float(q75 - q25),
    )


def _runs(cuts: np.ndarray, size: int) -> list[tuple[int, int]]:
    """Half-open point ranges split after each true entry of the interval mask."""
    boundaries = np.r_[0, np.flatnonzero(cuts) + 1, size]
    return [(int(start), int(end)) for start, end in zip(boundaries[:-1], boundaries[1:])]


def extract_kinematic_features(points: np.ndarray) -> KinematicFeatures:
    """Return exactly 83 attributes without access to a label or identifier.

    State runs count as strokes, including single-point runs. Invalid intervals
    (nonpositive or >60 seconds) split derivative runs, contribute no duration or
    path length and never connect derivatives. A valid contact/air transition is
    divided equally between the two acquisition times, but is excluded from
    stroke lengths and derivatives. Each state-run duration sums its valid
    internal intervals. Pressure levels use all recorded contact samples.

    Velocity vectors are secants located at interval midpoints. Their vector
    differences divided by midpoint time differences yield acceleration; the
    same operation yields jerk. The norms of all three vectors are summarized.
    These are unsmoothed finite differences, sensitive to quantization. A long
    valid interval measures average velocity over its gap, not instantaneous
    velocity during unobserved movement.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 7 or len(points) == 0:
        raise ValueError(f"Expected nonempty shape (n, 7), got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError("Kinematic extraction requires finite input values.")
    if not np.isin(points[:, 3], (0.0, 1.0)).all():
        raise ValueError("Pen state must be zero or one.")

    contact = points[:, 3] == 1.0
    coordinates = points[:, [1, 0]]
    dt_ms = np.diff(points[:, 2])
    valid = (dt_ms > 0.0) & (dt_ms <= MAX_VALID_INTERVAL_MS)
    dt_s = np.where(valid, dt_ms * TIMESTAMP_SECONDS_PER_UNIT, 0.0)
    transitions = contact[:-1] != contact[1:]
    contact_intervals = contact[:-1] & contact[1:] & valid
    air_intervals = ~contact[:-1] & ~contact[1:] & valid
    distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    transition_time = float(dt_s[transitions].sum())
    total_time = float(dt_s.sum())
    contact_time = float(dt_s[contact_intervals].sum()) + transition_time / 2.0
    air_time = float(dt_s[air_intervals].sum()) + transition_time / 2.0
    state_runs = _runs(transitions, len(points))

    distributions: dict[str, list[float]] = {name: [] for name in DISTRIBUTION_NAMES}
    stroke_counts = {"contact": 0, "air": 0}
    for start, end in state_runs:
        state = "contact" if contact[start] else "air"
        stroke_counts[state] += 1
        inside = slice(start, end - 1)
        distributions[f"{state}_stroke_duration_s"].append(float(dt_s[inside].sum()))
        distributions[f"{state}_stroke_length_native"].append(
            float(distances[inside][valid[inside]].sum())
        )

    derivative_runs = _runs(transitions | ~valid, len(points))
    for start, end in derivative_runs:
        if end - start < 2:
            continue
        state = "contact" if contact[start] else "air"
        block_dt = dt_s[start:end - 1]
        # Relative cumulative times avoid cancellation with absolute timestamps.
        times = np.r_[0.0, np.cumsum(block_dt)]
        derivative_times = (times[:-1] + times[1:]) / 2.0
        vectors = np.diff(coordinates[start:end], axis=0) / block_dt[:, None]
        for order, unit in (("speed", "s"), ("acceleration", "s2"), ("jerk", "s3")):
            distributions[f"{state}_{order}_native_per_{unit}"].extend(
                np.linalg.norm(vectors, axis=1).tolist()
            )
            if len(vectors) < 2:
                break
            vectors = np.diff(vectors, axis=0) / np.diff(derivative_times)[:, None]
            derivative_times = (derivative_times[:-1] + derivative_times[1:]) / 2.0
        if state == "contact":
            distributions["contact_absolute_pressure_rate_native_per_s"].extend(
                np.abs(np.diff(points[start:end, 6]) / block_dt).tolist()
            )
    distributions["contact_pressure_native"] = points[contact, 6].tolist()

    contact_xy = coordinates[contact]
    bbox = np.ptp(contact_xy, axis=0) if len(contact_xy) else np.zeros(2)
    global_values = (
        total_time, contact_time, air_time, transition_time,
        contact_time / total_time if total_time else 0.0,
        float(stroke_counts["contact"]), float(stroke_counts["air"]),
        float(distances[contact_intervals].sum()), float(distances[air_intervals].sum()),
        float(bbox[0]), float(bbox[1]),
    )
    summaries = tuple(
        value
        for name in DISTRIBUTION_NAMES
        for value in summarize(np.asarray(distributions[name], dtype=np.float64))
    )
    values = np.asarray(global_values + summaries, dtype=np.float64)
    if values.shape != (83,) or not np.isfinite(values).all():
        raise ValueError("Feature extraction did not produce 83 finite values.")

    diagnostics: dict[str, object] = {
        "observed_points": len(points), "intervals": len(dt_ms),
        "negative_intervals": int(np.count_nonzero(dt_ms < 0.0)),
        "zero_intervals": int(np.count_nonzero(dt_ms == 0.0)),
        "over_60s_intervals": int(np.count_nonzero(dt_ms > MAX_VALID_INTERVAL_MS)),
        "excluded_intervals": int(np.count_nonzero(~valid)),
        "excluded_interval_fraction": float(np.mean(~valid)) if len(valid) else 0.0,
        "excluded_contact_intervals": int(np.count_nonzero(~valid & contact[:-1] & contact[1:])),
        "excluded_air_intervals": int(np.count_nonzero(~valid & ~contact[:-1] & ~contact[1:])),
        "excluded_transition_intervals": int(np.count_nonzero(~valid & transitions)),
        "valid_over_1s_intervals": int(np.count_nonzero(valid & (dt_ms > 1000.0))),
        "positive_dt_median_ms": float(np.median(dt_ms[dt_ms > 0])) if np.any(dt_ms > 0) else 0.0,
        "derivative_runs": len(derivative_runs),
        "empty_distributions": [name for name in DISTRIBUTION_NAMES if not distributions[name]],
        "distribution_counts": {name: len(distributions[name]) for name in DISTRIBUTION_NAMES},
        "zero_feature_count": int(np.count_nonzero(values == 0.0)),
    }
    return KinematicFeatures(values=values, diagnostics=diagnostics)
