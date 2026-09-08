import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pdhms_restart.comparison import paired_task_mean_macro_f1_comparison

spec = importlib.util.spec_from_file_location("selection_analysis", Path(__file__).resolve().parents[1] / "scripts/analyze_nested_selection_controls.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def make_frame(predictions, seed=0):
    return pd.DataFrame({"subject_id": ["00001", "00002", "00003", "00004"], "task_id": [1] * 4,
                         "repeat": [1] * 4, "outer_fold": [1, 1, 2, 2], "control_seed": [seed] * 4,
                         "y_true": [0, 0, 1, 1], "y_pred": predictions})


def test_seed_aggregation_is_mean_of_metrics_not_prediction_ensemble():
    seeds = [11, 12, 13]
    variants = [make_frame([0, 0, 0, 0], 11), make_frame([1, 1, 1, 1], 12), make_frame([0, 0, 1, 1], 13)]
    expanded = analysis.seed_expanded(pd.concat(variants), seeds, 5)
    summary = analysis.absolute_summary(expanded, resamples=128, seed=1)
    expected = np.mean([f1_score(row.y_true, row.y_pred, average="macro") for row in variants])
    assert np.isclose(summary["macro_f1_mean"], expected)
    assert summary["n_subjects"] == 4
    assert summary["macro_f1_mean"] < 1
    reference = analysis.duplicate_reference(make_frame([0, 0, 0, 0]), seeds, 5)
    contrast = paired_task_mean_macro_f1_comparison(expanded, reference, resamples=128, seed=3)
    assert np.isclose(contrast["delta_task_mean_macro_f1"], expected - 1 / 3)


def test_duplicate_seeds_do_not_shrink_participant_interval():
    original = make_frame([0, 1, 1, 1])
    triplicate = analysis.duplicate_reference(original, [11, 12, 13], 5)
    first = analysis.absolute_summary(original, resamples=256, seed=5)
    second = analysis.absolute_summary(triplicate, resamples=256, seed=5)
    for key in first:
        assert np.isclose(first[key], second[key], atol=1e-12)


def test_outer_folds_are_concatenated_before_macro_f1():
    predictions = make_frame([0, 0, 0, 1])
    summary = analysis.absolute_summary(predictions, resamples=128, seed=9)
    pooled = f1_score(predictions.y_true, predictions.y_pred, labels=[0, 1], average="macro")
    fold_mean = np.mean([f1_score(group.y_true, group.y_pred, labels=[0, 1], average="macro", zero_division=0) for _, group in predictions.groupby("outer_fold")])
    assert np.isclose(summary["macro_f1_mean"], pooled)
    assert not np.isclose(pooled, fold_mean)
