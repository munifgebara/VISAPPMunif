import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pdhms_restart.evaluation import Candidate
from pdhms_restart.nested_selection import build_selection_plan, choose_encoding, macro_f1, run_unit, tune_encodings


def toy():
    metadata = pd.DataFrame([{"subject_id": f"{i:05}", "task_id": task, "label": "PD" if i % 2 else "H"} for task in (1, 2) for i in range(24) if not (task == 1 and i == 23)])
    features = np.random.default_rng(5).normal(size=(len(metadata), 6)).astype(np.float32)
    ids = [f"{i:05}" for i in range(24)]
    original = {"outer_splits": [{"repeat": 1, "outer_fold": 1, "train_subject_ids": ids[:16], "test_subject_ids": ids[16:]}]}
    config = {"validation": {"master_seed": 19, "meta_folds": 3, "tuning_folds": 3, "fusion_order": ["majority_vote_all8", "calibrated_mean_all8", "calibrated_reliability_weighted_all8"]}, "svm_search": {"linear_c": [0.1], "rbf_c": [1.0], "rbf_gamma": ["scale"]}, "execution": {"inner_threads": 1}}
    return metadata, features, build_selection_plan(metadata, original, config), config


def test_common_three_layer_partitions_and_missing_task():
    metadata, _, plan, _ = toy()
    outer = plan["outer_splits"][0]
    seen = []
    for meta in outer["meta_folds"]:
        assert not set(meta["train_subject_ids"]) & set(meta["validation_subject_ids"])
        assert not set(meta["train_subject_ids"] + meta["validation_subject_ids"]) & set(outer["test_subject_ids"])
        for inner in meta["tuning_folds"]:
            assert set(inner["train_subject_ids"] + inner["validation_subject_ids"]) == set(meta["train_subject_ids"])
        seen.extend(meta["validation_subject_ids"])
    assert sorted(seen) == sorted(outer["train_subject_ids"])
    assert len(metadata[metadata.task_id == 1]) == 23


def test_outer_test_labels_and_features_cannot_change_selection():
    metadata, features, plan, _ = toy()
    outer = plan["outer_splits"][0]
    candidates = [Candidate("linear", 0.1), Candidate("rbf", 1)]
    original, table = tune_encodings(metadata, {"static": features, "speed": features.copy()}, outer["train_subject_ids"], outer["tuning_folds"], candidates, "test")
    changed = metadata.copy()
    mask = changed.subject_id.isin(outer["test_subject_ids"]).to_numpy()
    changed.loc[mask, "label"] = np.where(changed.loc[mask, "label"].eq("PD"), "H", "PD")
    poisoned = features.copy()
    poisoned[mask] = 1e10
    result, changed_table = tune_encodings(changed, {"static": poisoned, "speed": poisoned.copy()}, outer["train_subject_ids"], outer["tuning_folds"], candidates, "test")
    assert original == result and table == changed_table
    assert choose_encoding(result, ["static", "speed"]) == "static"


def test_small_complete_unit_and_control_unit(tmp_path):
    metadata, features, plan, config = toy()
    outer = plan["outer_splits"][0]
    result_path = run_unit(metadata, {"static": features, "speed": features.copy()}, outer, config, str(tmp_path / "auth"))
    result = json.loads(open(result_path, encoding="utf-8").read())
    outer_selections = [row for row in result["selections"] if row["phase"] == "outer_train"]
    assert {row["policy"]: row["encoding"] for row in outer_selections} == {"static": "static", "auth9": "static", "auth8": "speed"}
    assert len(result["task_predictions"]) == 3 * 15
    assert len(result["fusion_predictions"]) == 3 * 8
    assert {row["subject_id"] for row in result["task_predictions"]} == set(outer["test_subject_ids"])
    control_path = run_unit(metadata, {"speed": features}, outer, config, str(tmp_path / "control"), "circular_shift", 3)
    control = json.loads(open(control_path, encoding="utf-8").read())
    assert len(control["task_predictions"]) == 15
    assert not control["fusion_predictions"]
    assert all(row["calibrated_probability_pd"] is None for row in control["task_predictions"])


def test_fast_metric_matches_sklearn_including_constant_predictions():
    rng = np.random.default_rng(29)
    for n in (1, 2, 15, 75):
        for _ in range(10):
            y = rng.integers(0, 2, n)
            pred = rng.integers(0, 2, n)
            assert np.isclose(macro_f1(y, pred), f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))
