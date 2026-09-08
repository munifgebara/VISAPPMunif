from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from pdhms_restart.dynamic_image import (
    DynamicRenderConfig,
    render_dynamic_array,
    render_dynamic_triplet_array,
    surface_segment_signal,
)
from pdhms_restart.spatial_controls import render_spatial_control_array


SAZ = ("speed", "altitude", "azimuth")
CONFIG = DynamicRenderConfig(image_size=96, margin=8, antialias_scale=1, line_width_px=3)


def synthetic_record(stroke_lengths: tuple[int, ...] = (7, 5, 1)) -> np.ndarray:
    """Lengths count segments; air gaps separate horizontally disjoint strokes."""
    rows = []
    for stroke, segments in enumerate(stroke_lengths):
        for point in range(segments + 1):
            rows.append([20 * stroke, 15 * stroke + point * (point + 1) / 4,
                         len(rows), 1, (3500 + 51 * point + 130 * stroke) % 3600,
                         30 + point * point + 20 * stroke,
                         60 + 13 * point + 15 * stroke])
        if stroke < len(stroke_lengths) - 1:
            rows.append([-500, 500, len(rows), 0, 100, 20, 0])
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.parametrize("signals", ["speed", "pressure", "altitude", "azimuth",
                                    SAZ, ("pressure", "altitude", "azimuth"),
                                    ("speed", "pressure", "altitude"),
                                    ("speed", "pressure", "azimuth")])
@pytest.mark.parametrize("antialias", [1, 3])
def test_identity_is_byte_identical_to_original_renderers(signals, antialias: int) -> None:
    points = synthetic_record()
    # Also exercise acquisition-order overpainting without changing pen-up gaps.
    points[3, :2] = points[1, :2]
    before = points.copy()
    config = replace(CONFIG, antialias_scale=antialias, constant_channels=(43, 71))
    if isinstance(signals, str):
        expected, _ = render_dynamic_array(points, signals, config)
    else:
        expected, _ = render_dynamic_triplet_array(points, signals, config)
    actual, diagnostics = render_spatial_control_array(
        points, signals, config, control="identity", record_key="synthetic/record.svc", seed=17
    )
    assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(points, before)
    assert diagnostics.summary()["changed_signal_vector_fraction"] == 0.0


def test_joint_shift_preserves_vectors_within_each_stroke_without_crossing_air() -> None:
    points = synthetic_record()
    original_image, original = render_spatial_control_array(
        points, SAZ, CONFIG, control="identity", record_key="record.svc", seed=29
    )
    image, shifted = render_spatial_control_array(points, SAZ, CONFIG, record_key="record.svc", seed=29)
    expected_surface = (points[:-1, 3] > 0) & (points[1:, 3] > 0)
    np.testing.assert_array_equal(shifted.surface, expected_surface)
    np.testing.assert_array_equal(shifted.source_indices[~expected_surface], np.flatnonzero(~expected_surface))
    for stroke in shifted.strokes:
        indices = np.arange(stroke.start, stroke.stop)
        sources = shifted.source_indices[indices]
        np.testing.assert_array_equal(np.sort(sources), indices)
        np.testing.assert_array_equal(shifted.controlled_normalized[indices], original.original_normalized[sources])
        np.testing.assert_array_equal(shifted.controlled_colours[indices], original.original_colours[sources])
        if stroke.length > 1:
            assert 0.25 <= stroke.shift / stroke.length <= 0.75
            assert np.all(indices != sources)
        else:
            assert stroke.shift == 0
    # With no antialiasing and channels below the foreground threshold, exact
    # pixel support is a meaningful extra check of unaltered drawing geometry.
    np.testing.assert_array_equal(np.any(image < 250, axis=2), np.any(original_image < 250, axis=2))
    assert shifted.geometry_sha256 == original.geometry_sha256
    assert shifted.summary()["joint_vectors_preserved_per_stroke"]
    assert shifted.summary()["non_surface_rows_unchanged"]
    assert shifted.summary()["surface_mask_preserved"]
    assert shifted.summary()["changed_colour_vector_fraction"] > 0
    json.dumps(shifted.summary(), allow_nan=False)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6, 9, 17])
def test_short_strokes_and_declared_shift_range(length: int) -> None:
    _, diagnostics = render_spatial_control_array(
        synthetic_record((length,)), SAZ, CONFIG, record_key="short.svc", seed=3
    )
    assert len(diagnostics.strokes) == 1
    stroke = diagnostics.strokes[0]
    assert stroke.length == length
    if length == 1:
        assert stroke.shift == 0
        assert diagnostics.summary()["moved_segment_fraction"] == 0
    else:
        assert max(1, (length + 3) // 4) <= stroke.shift <= (3 * length) // 4
        assert diagnostics.summary()["moved_segment_fraction"] == 1


def test_preserves_original_normalization_and_ignores_global_rng_and_signal_choice() -> None:
    points = synthetic_record((19, 11))
    baseline, first = render_spatial_control_array(points, SAZ, CONFIG, record_key="case.svc", seed=5)
    np.random.seed(782)
    np.random.random(100)
    repeated, second = render_spatial_control_array(points, SAZ, CONFIG, record_key="case.svc", seed=5)
    _, other_signals = render_spatial_control_array(
        points, ("pressure", "altitude", "azimuth"), CONFIG, record_key="case.svc", seed=5
    )
    np.testing.assert_array_equal(baseline, repeated)
    assert first.summary() == second.summary()
    np.testing.assert_array_equal(first.source_indices, other_signals.source_indices)
    for column, signal in enumerate(SAZ):
        original = surface_segment_signal(points, signal, CONFIG)
        np.testing.assert_array_equal(first.original_normalized[:, column], original.normalized)
        assert first.normalization[signal]["low"] == original.low
        assert first.normalization[signal]["high"] == original.high
    choices = {
        render_spatial_control_array(points, SAZ, CONFIG, record_key="case.svc", seed=seed)[1]
        .source_indices.tobytes()
        for seed in range(8)
    }
    assert len(choices) > 1


def test_constant_signals_report_moved_indices_without_claiming_changed_values() -> None:
    points = synthetic_record((7,))
    points[:, 5] = 40
    points[:, 6] = 80
    _, diagnostics = render_spatial_control_array(points, "pressure", CONFIG, record_key="constant.svc", seed=7)
    summary = diagnostics.summary()
    assert summary["moved_segment_fraction"] == 1
    assert summary["changed_signal_vector_fraction"] == 0
    assert summary["changed_colour_vector_fraction"] == 0
    assert summary["normalization"]["pressure"]["degenerate_range"]


def test_no_label_or_diagnosis_parameter_and_no_input_mutation() -> None:
    points = synthetic_record()
    before = points.copy()
    with pytest.raises(TypeError, match="label"):
        render_spatial_control_array(points, SAZ, CONFIG, record_key="opaque.svc", seed=1, label="PD")
    render_spatial_control_array(points, SAZ, CONFIG, record_key="opaque.svc", seed=1)
    np.testing.assert_array_equal(points, before)


def test_single_contact_points_do_not_create_or_join_strokes() -> None:
    points = synthetic_record((1, 0, 2))
    _, diagnostics = render_spatial_control_array(points, SAZ, CONFIG, record_key="isolated.svc", seed=9)
    assert [stroke.length for stroke in diagnostics.strokes] == [1, 2]
    assert diagnostics.summary()["surface_segments"] == 3
    assert diagnostics.summary()["non_surface_rows_unchanged"]


def test_rejects_records_without_segments_and_invalid_times_like_original() -> None:
    points = synthetic_record((0, 0))
    with pytest.raises(ValueError, match="no pen-down segments"):
        render_spatial_control_array(points, SAZ, CONFIG, record_key="empty.svc", seed=1)
    points = synthetic_record((3,))
    points[1, 2] = points[0, 2]
    with pytest.raises(ValueError, match="non-positive"):
        render_spatial_control_array(points, SAZ, CONFIG, record_key="invalid.svc", seed=1)


def test_joint_permutation_preserves_rgb_vectors_and_stroke_boundaries() -> None:
    points = synthetic_record((19, 7, 1))
    image, first = render_spatial_control_array(
        points, SAZ, CONFIG, control="joint_permutation", record_key="shuffle.svc", seed=19
    )
    repeated, second = render_spatial_control_array(
        points, SAZ, CONFIG, control="joint_permutation", record_key="shuffle.svc", seed=19
    )
    np.testing.assert_array_equal(image, repeated)
    np.testing.assert_array_equal(first.source_indices, second.source_indices)
    for stroke in first.strokes:
        indices = np.arange(stroke.start, stroke.stop)
        sources = first.source_indices[indices]
        np.testing.assert_array_equal(np.sort(sources), indices)
        np.testing.assert_array_equal(first.controlled_normalized[indices], first.original_normalized[sources])
        assert stroke.shift is None
    assert first.summary()["joint_vectors_preserved_per_stroke"]
    assert first.summary()["non_surface_rows_unchanged"]
    assert first.summary()["changed_signal_vector_fraction"] > 0
    assert first.source_indices[-1] == len(first.source_indices) - 1
    json.dumps(first.summary(), allow_nan=False)
