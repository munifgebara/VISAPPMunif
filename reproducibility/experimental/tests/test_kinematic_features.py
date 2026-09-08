"""Analytical checks for the independent kinematic descriptor."""

import numpy as np
import pytest

from pdhms_restart.kinematic_features import FEATURE_NAMES, extract_kinematic_features


def recording(x, timestamps, states=None, pressure=None, y=None):
    n = len(x)
    points = np.zeros((n, 7))
    points[:, 0] = np.zeros(n) if y is None else y
    points[:, 1] = x
    points[:, 2] = timestamps
    points[:, 3] = np.ones(n) if states is None else states
    points[:, 6] = np.full(n, 20.0) if pressure is None else pressure
    return points


def feature_dict(points):
    result = extract_kinematic_features(points)
    return dict(zip(FEATURE_NAMES, result.values)), result.diagnostics


def test_milliseconds_and_constant_vector_motion():
    values, diagnostics = feature_dict(recording([0, 3, 6, 9], [0, 1000, 2000, 3000], y=[0, 4, 8, 12]))
    assert len(values) == 83
    assert values["valid_acquisition_time_s"] == 3.0
    assert values["contact_speed_native_per_s__mean"] == 5.0
    assert values["contact_path_length_native"] == 15.0
    assert values["contact_acceleration_native_per_s2__mean"] == 0.0
    assert values["contact_jerk_native_per_s3__mean"] == 0.0
    assert diagnostics["excluded_intervals"] == 0


def test_nonuniform_time_acceleration_and_pressure_rate():
    # x = t^2: secant velocity equals 2t at each interval midpoint.
    values, _ = feature_dict(recording([0, 1, 9, 16], [0, 1000, 3000, 4000], pressure=[0, 2, 6, 8]))
    assert values["contact_acceleration_native_per_s2__mean"] == pytest.approx(2.0)
    assert values["contact_jerk_native_per_s3__mean"] == pytest.approx(0.0)
    assert values["contact_absolute_pressure_rate_native_per_s__mean"] == 2.0


def test_derivative_norms_capture_curved_motion_at_constant_speed():
    values, _ = feature_dict(recording([0, 1, 1, 0], [0, 1000, 2000, 3000], y=[0, 0, 1, 1]))
    assert values["contact_speed_native_per_s__mean"] == 1.0
    assert values["contact_acceleration_native_per_s2__mean"] == pytest.approx(np.sqrt(2))
    assert values["contact_jerk_native_per_s3__mean"] == pytest.approx(2.0)


def test_pen_lift_prevents_derivative_bridge_and_splits_transition_time():
    points = recording([0, 1, 100, 101, 900, 901], np.arange(6) * 1000,
                       states=[1, 1, 0, 0, 1, 1], pressure=[0, 2, 500, 800, 10, 12])
    values, diagnostic = feature_dict(points)
    assert values["contact_speed_native_per_s__mean"] == 1.0
    assert values["air_speed_native_per_s__mean"] == 1.0
    assert values["contact_path_length_native"] == 2.0
    assert values["contact_absolute_pressure_rate_native_per_s__mean"] == 2.0
    assert values["contact_stroke_count"] == 2.0
    assert values["air_stroke_count"] == 1.0
    assert values["valid_acquisition_time_s"] == 5.0
    assert values["contact_acquisition_time_s"] == 3.0
    assert values["air_acquisition_time_s"] == 2.0
    assert values["transition_acquisition_time_s"] == 2.0
    assert diagnostic["distribution_counts"]["contact_acceleration_native_per_s2"] == 0


@pytest.mark.parametrize("bad_time", [1000, 900, 61001, 1e12])
def test_invalid_interval_splits_derivatives_without_inventing_strokes(bad_time):
    points = recording([0, 1, 100, 101], [0, 1000, bad_time, bad_time + 1000])
    values, diagnostic = feature_dict(points)
    assert values["contact_speed_native_per_s__mean"] == 1.0
    assert values["valid_acquisition_time_s"] == 2.0
    assert values["contact_path_length_native"] == 2.0
    assert values["contact_stroke_count"] == 1.0
    assert diagnostic["excluded_intervals"] == 1
    assert diagnostic["distribution_counts"]["contact_acceleration_native_per_s2"] == 0


def test_plausible_long_pause_preserved_and_sixty_second_boundary_included():
    values, diagnostic = feature_dict(recording([0, 10, 20], [0, 10_000, 70_000], states=[0, 0, 0]))
    assert values["valid_acquisition_time_s"] == 70.0
    assert values["air_acquisition_time_s"] == 70.0
    assert diagnostic["valid_over_1s_intervals"] == 2
    assert diagnostic["excluded_intervals"] == 0


def test_translation_and_timestamp_origin_invariance():
    points = recording([0, 1, 2, 4, 7], [0, 7, 15, 22, 30], pressure=[1, 2, 3, 4, 5])
    transformed = points.copy()
    transformed[:, :2] += [12345, -98765]
    transformed[:, 2] += 1_000_000_000_000
    np.testing.assert_array_equal(extract_kinematic_features(points).values,
                                  extract_kinematic_features(transformed).values)


def test_physical_scale_and_pressure_levels_are_retained():
    points = recording([0, 1, 2, 4], [0, 1000, 2000, 3000], pressure=[1, 2, 3, 4])
    scaled = points.copy()
    scaled[:, :2] *= 3.0
    scaled[:, 6] *= 2.0
    original, _ = feature_dict(points)
    result, _ = feature_dict(scaled)
    assert result["contact_speed_native_per_s__mean"] == 3 * original["contact_speed_native_per_s__mean"]
    assert result["contact_bbox_width_native"] == 3 * original["contact_bbox_width_native"]
    assert result["contact_pressure_native__mean"] == 2 * original["contact_pressure_native__mean"]


def test_single_point_empty_distributions_and_unknown_state():
    values, diagnostic = feature_dict(recording([5], [42], pressure=[7]))
    assert values["valid_acquisition_time_s"] == 0.0
    assert values["contact_pressure_native__mean"] == 7.0
    assert values["contact_speed_native_per_s__mean"] == 0.0
    assert "contact_speed_native_per_s" in diagnostic["empty_distributions"]
    with pytest.raises(ValueError, match="state"):
        extract_kinematic_features(recording([5], [42], states=[2]))


@pytest.mark.parametrize("points", [np.empty((0, 7)), np.zeros((3, 6)), np.full((3, 7), np.nan)])
def test_nonfinite_or_malformed_inputs_rejected(points):
    with pytest.raises(ValueError):
        extract_kinematic_features(points)
