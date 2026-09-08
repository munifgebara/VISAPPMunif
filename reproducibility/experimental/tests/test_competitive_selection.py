"""Leakage and endpoint checks for the new competitive task-subset comparison."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pdhms_restart.competitive_selection import majority_fusion, run_competitive_unit, select_for_tasks
from pdhms_restart.nested_selection import shared_folds


def test_task_scope_cannot_be_selected_from_excluded_tasks():
    tuned = {"a": {1: {"inner_macro_f1": 1.0}, 2: {"inner_macro_f1": 0.3}},
             "b": {1: {"inner_macro_f1": 0.0}, 2: {"inner_macro_f1": 0.8}}}
    assert select_for_tasks(tuned, [2], ["a", "b"])["encoding"] == "b"
    tuned["a"][1]["inner_macro_f1"] = 1000.0
    assert select_for_tasks(tuned, [2], ["a", "b"])["encoding"] == "b"
    tuned["a"][2]["inner_macro_f1"] = 0.8
    assert select_for_tasks(tuned, [2], ["a", "b"])["encoding"] == "a"


def test_majority_uses_available_tasks_and_fixed_pd_tie():
    common = dict(task_set="all8", method="m", repeat=1, outer_fold=1, encoding="static")
    rows = [{**common, "subject_id": "a", "task_id": task, "y_true": 0, "y_pred": pred}
            for task, pred in [(2, 1), (3, 0)]]
    rows += [{**common, "subject_id": "b", "task_id": task, "y_true": 1, "y_pred": pred}
             for task, pred in [(2, 0), (3, 1), (4, 0)]]
    fused = majority_fusion(pd.DataFrame(rows)).set_index("subject_id")
    assert fused.loc["a", "score_pd"] == 0.5
    assert fused.loc["a", "y_pred"] == 1
    assert fused.loc["a", "available_tasks"] == 2
    assert fused.loc["b", "y_pred"] == 0


def test_poisoned_outer_test_cannot_change_choices_or_fitted_models(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "experiments/2026-09-restart/competitive-task-subset-v1/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["encoding_order"] = ["static", "speed"]
    config["task_sets"] = {"subset": [2, 3], "all": [1, 2, 3]}
    config["svm_search"] = {"linear_c": [0.1, 1.0], "rbf_c": [], "rbf_gamma": []}
    config["logistic_search"]["c"] = [0.1, 1.0]
    records = [{"subject_id": f"{subject:05}", "task_id": task,
                "repetition_id": 1, "label": "PD" if subject % 2 else "H"}
               for subject in range(18) for task in (1, 2, 3)]
    metadata = pd.DataFrame(records)
    subjects = metadata[["subject_id", "label"]].drop_duplicates()
    train_ids, test_ids = subjects.subject_id.iloc[:12].tolist(), subjects.subject_id.iloc[12:].tolist()
    outer = {"repeat": 1, "outer_fold": 1, "train_subject_ids": train_ids,
             "test_subject_ids": test_ids,
             "tuning_folds": shared_folds(subjects, train_ids, 42, 3)}
    rng = np.random.default_rng(73)
    families = {"lpq": {name: rng.normal(size=(len(metadata), 5)) for name in ("static", "speed")},
                "kinematic": {"kinematic": rng.normal(size=(len(metadata), 4))},
                "transfer": {name: rng.normal(size=(len(metadata), 6)) for name in ("static", "speed")}}
    run_competitive_unit(metadata, families, outer, config, str(tmp_path / "original"))
    poisoned = metadata.copy()
    test_mask = poisoned.subject_id.isin(test_ids).to_numpy()
    poisoned.loc[test_mask, "label"] = poisoned.loc[test_mask, "label"].map({"PD": "H", "H": "PD"})
    changed_features = copy.deepcopy(families)
    for matrices in changed_features.values():
        for matrix in matrices.values():
            matrix[test_mask] = rng.normal(size=matrix[test_mask].shape) * 1e6
    run_competitive_unit(poisoned, changed_features, outer, config, str(tmp_path / "poisoned"))
    assert (tmp_path / "original/selection_before_test.json").read_bytes() == (tmp_path / "poisoned/selection_before_test.json").read_bytes()
    import joblib
    original = joblib.load(tmp_path / "original/models.joblib")
    altered = joblib.load(tmp_path / "poisoned/models.joblib")
    for key, model in original.items():
        matrix = families[key[0]][key[2]]
        np.testing.assert_array_equal(model.predict(matrix), altered[key].predict(matrix))
        np.testing.assert_allclose(model[0].mean_, altered[key][0].mean_, rtol=0, atol=0)
