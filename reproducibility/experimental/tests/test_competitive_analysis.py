"""Participant clustering, fold pooling and independently seeded inference."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


path = Path(__file__).resolve().parents[1] / "scripts/analyze_competitive_task_subset.py"
spec = importlib.util.spec_from_file_location("competitive_analysis", path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def frames():
    truth = [0, 0, 0, 0, 1, 1, 1, 1]
    a, b = [0, 0, 0, 1, 1, 1, 0, 1], [0, 1, 0, 1, 1, 0, 0, 1]
    base = pd.DataFrame({"subject_id": list("abcdefgh"), "y_true": truth,
                         "task_id": 0, "repeat": 1, "outer_fold": [1]*4 + [2]*4})
    return base.assign(y_pred=a), base.assign(y_pred=b)


SETTINGS = {"bootstrap_resamples": 120, "bootstrap_seed": 123,
            "paired_randomization_resamples": 160, "paired_randomization_seed": 456}


def test_folds_are_pooled_before_macro_f1():
    a, _ = frames()
    summary = analysis.absolute_summary(a, SETTINGS)
    expected = f1_score(a.y_true, a.y_pred, average="macro", labels=[0, 1])
    assert summary["macro_f1_mean"] == expected
    assert expected != np.mean([f1_score(g.y_true, g.y_pred, average="macro", labels=[0, 1], zero_division=0) for _, g in a.groupby("outer_fold")])


def test_duplicate_repeats_do_not_create_independent_people():
    a, b = frames()
    original = analysis.paired_summary(a, b, SETTINGS)
    duplicated = analysis.paired_summary(pd.concat([a, a.assign(repeat=2)]), pd.concat([b, b.assign(repeat=2)]), SETTINGS)
    assert original == duplicated


def test_permutation_seed_does_not_change_bootstrap_and_zero_contrast():
    a, b = frames()
    original = analysis.paired_summary(a, b, SETTINGS)
    changed = analysis.paired_summary(a, b, {**SETTINGS, "paired_randomization_seed": 99})
    assert [original[k] for k in ("delta_macro_f1", "ci_low", "ci_high")] == [changed[k] for k in ("delta_macro_f1", "ci_low", "ci_high")]
    same = analysis.paired_summary(a, a, SETTINGS)
    assert same == {"delta_macro_f1": 0.0, "ci_low": 0.0, "ci_high": 0.0, "permutation_p": 1.0}
