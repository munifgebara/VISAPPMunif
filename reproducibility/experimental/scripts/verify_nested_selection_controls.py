"""Audit the frozen nested-selection/control run without fitting any model.

This verifier independently reconstructs choices and fusion from the persisted
search tables and predictions. It never imports or calls the experiment runner.
Only locally produced model artifacts are deserialized for prediction checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/2026-09-restart/nested-selection-controls-v1/config.json"
KEYS = ["repeat", "outer_fold", "task_id", "subject_id"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def frame(rows) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    if "subject_id" in result:
        result["subject_id"] = result.subject_id.astype(str).str.zfill(5)
    return result


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"subject_id": str}, float_precision="round_trip")


def score(truth, predicted) -> float:
    return float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0))


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures: list[dict] = []
        self.notes: list[str] = []

    def check(self, condition, message: str):
        self.checks += 1
        if not bool(condition):
            self.failures.append({"check": message})

    def close(self, actual, expected, message: str, atol: float = 1e-10):
        self.check(np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True), message)

    def guarded(self, label, function, *args):
        try:
            return function(*args)
        except Exception as error:
            self.failures.append({"check": label, "exception": f"{type(error).__name__}: {error}",
                                  "traceback": traceback.format_exc(limit=3)})
            return None


def candidate_names(config: dict) -> list[str]:
    search = config["svm_search"]
    return ([f"linear_C={float(c):g}_gamma=na" for c in search["linear_c"]]
            + [f"rbf_C={float(c):g}_gamma={g}" for c in search["rbf_c"] for g in search["rbf_gamma"]])


def audit_config(audit: Audit, config: dict):
    validation = config["validation"]
    audit.check(config["tasks"] == list(range(1, 9)), "eight original tasks")
    audit.check(len(config["encoding_order"]) == 9 and config["encoding_order"][0] == "static"
                and len(set(config["encoding_order"])) == 9, "nine ordered distinct encodings, static first")
    audit.check(set(config["signal_channels"]) == set(config["encoding_order"][1:]), "eight dynamic channel definitions")
    audit.check(len(candidate_names(config)) == 21, "21 SVM candidates")
    audit.check(config["svm_search"]["class_weight"] == "balanced", "fixed balanced SVM family")
    for name, count in (("repeats", 5), ("outer_folds", 5), ("meta_folds", 3), ("tuning_folds", 3)):
        audit.check(validation[name] == count, f"declared {name}={count}")
    audit.check(config["controls"]["methods"] == ["circular_shift", "joint_permutation"], "two declared spatial controls")
    audit.check(len(config["controls"]["seeds"]) == 3 and len(set(config["controls"]["seeds"])) == 3,
                "three distinct control seeds")
    audit.check(config["controls"]["endpoint"] == "task_mean_only", "no control fusion endpoint")
    audit.check(len(validation["fusion_order"]) == 3 and len(set(validation["fusion_order"])) == 3,
                "three ordered fusion rules")
    audit.check(len(validation["task_global_family"]) == 4, "global task family has four contrasts")
    audit.check(len(validation["fusion_secondary_family"]) == 2, "fusion family has two contrasts")
    audit.check(validation["task_specific_exploratory_family_size"] == 32, "task family has 32 contrasts")


def audit_lock(audit: Audit, root: Path, config_path: Path, complete: bool):
    lock = read_json(root / "manifests/run_lock.json")
    audit.check(lock["config_sha256"] == sha256(config_path), "configuration unchanged since fitting started")
    audit.check(lock["protocol_sha256"] == sha256(config_path.with_name("DECISION.md")), "protocol unchanged since fitting started")
    for name, expected in lock["source_sha256"].items():
        audit.check(sha256(resolve(name)) == expected, f"fitting source unchanged: {name}")
        audit.check(sha256(root / "manifests/source_snapshot" / name) == expected, f"source snapshot intact: {name}")
    for name, expected in lock["input_sha256"].items():
        audit.check(sha256(resolve(name)) == expected, f"original input unchanged: {name}")
    for path in sorted((root / "manifests").glob("*_seed*_inputs.json")):
        for name, expected in read_json(path).items():
            audit.check(sha256(resolve(name)) == expected, f"control input unchanged: {name}")
    if complete:
        manifest = read_json(root / "manifests/run_manifest.json")
        audit.check(manifest["completed_outer_units"] == 175 and all(count == 25 for count in manifest["unit_counts"].values()), "run manifest covers all units")
        audit.check(manifest["analysis_source_sha256"] == sha256(ROOT / "scripts/analyze_nested_selection_controls.py"), "analysis source hash")
        audit.check(manifest["analysis_config_sha256"] == sha256(config_path), "analysis uses frozen configuration")


def audit_partition(audit: Audit, folds: list[dict], allowed: set[str], labels: dict, label: str):
    audit.check(len(folds) == 3, f"{label}: three folds")
    seen = []
    for item in folds:
        train, valid = item["train_subject_ids"], item["validation_subject_ids"]
        audit.check(len(train) == len(set(train)) and len(valid) == len(set(valid)), f"{label}: no duplicate IDs")
        audit.check(not set(train) & set(valid), f"{label}: training/validation disjoint")
        audit.check(set(train) | set(valid) == allowed, f"{label}: complete and confined partition")
        audit.check({labels[value] for value in train} == {"H", "PD"}, f"{label}: both training diagnoses")
        audit.check({labels[value] for value in valid} == {"H", "PD"}, f"{label}: both validation diagnoses")
        seen.extend(valid)
    audit.check(Counter(seen) == Counter(allowed), f"{label}: each person validates exactly once")


def audit_plan(audit: Audit, config: dict, plan: dict, metadata: pd.DataFrame):
    subjects = metadata[["subject_id", "label"]].drop_duplicates()
    audit.check(len(subjects) == 75 and subjects.subject_id.nunique() == 75, "75 distinct participants with unique labels")
    audit.check(subjects.label.value_counts().to_dict() == {"H": 38, "PD": 37}, "38 H and 37 PD participants")
    audit.check(len(metadata) == 597 and not metadata.duplicated(["subject_id", "task_id"]).any(), "597 distinct participant/task records")
    audit.check(metadata.task_id.value_counts().sort_index().tolist() == [72, 75, 75, 75, 75, 75, 75, 75], "original task availability")
    labels = subjects.set_index("subject_id").label.to_dict()
    all_ids = set(labels)
    original = read_json(resolve(config["validation"]["outer_splits_from"]))
    original_index = {(int(row["repeat"]), int(row["outer_fold"])): row for row in original["outer_splits"]}
    audit.check(len(plan["outer_splits"]) == 25, "25 outer assignments")
    seen = Counter()
    seen_keys = []
    for outer in plan["outer_splits"]:
        key = (int(outer["repeat"]), int(outer["outer_fold"]))
        seen_keys.append(key)
        train, test = set(outer["train_subject_ids"]), set(outer["test_subject_ids"])
        audit.check(len(train) == 60 and len(test) == 15, f"{key}: 60 train/15 test")
        audit.check(not train & test and train | test == all_ids, f"{key}: external separation")
        for name in ("train_subject_ids", "test_subject_ids"):
            audit.check(set(outer[name]) == {str(value).zfill(5) for value in original_index[key][name]}, f"{key}: original {name}")
        seen.update((key[0], value) for value in test)
        audit_partition(audit, outer["tuning_folds"], train, labels, f"{key}/outer tuning")
        audit_partition(audit, outer["meta_folds"], train, labels, f"{key}/meta")
        for tuning, meta in zip(outer["tuning_folds"], outer["meta_folds"], strict=True):
            audit.check(tuning["train_subject_ids"] == meta["train_subject_ids"] and
                        tuning["validation_subject_ids"] == meta["validation_subject_ids"], f"{key}: task tuning matches common meta partition")
            audit_partition(audit, meta["tuning_folds"], set(meta["train_subject_ids"]), labels, f"{key}/meta{meta['inner_fold']}/tuning")
    audit.check(set(seen_keys) == {(r, f) for r in range(1, 6) for f in range(1, 6)} and len(set(seen_keys)) == 25,
                "all repeat/fold keys exactly once")
    audit.check(seen == Counter({(r, value): 1 for r in range(1, 6) for value in all_ids}), "75 test people per repeat; 375 participant predictions")


def reconstruct_search(audit: Audit, table: pd.DataFrame, phases: list[str], encodings: list[str], config: dict, label: str):
    names = candidate_names(config)
    expected_keys = {(phase, encoding, task, i) for phase in phases for encoding in encodings for task in config["tasks"] for i in range(21)}
    observed_keys = list(table[["phase", "encoding", "task_id", "candidate_index"]].itertuples(index=False, name=None))
    audit.check(len(observed_keys) == len(expected_keys) and set(observed_keys) == expected_keys, f"{label}: complete search grid")
    scores = []
    for row in table.itertuples(index=False):
        values = np.asarray(json.loads(row.inner_fold_scores), dtype=float)
        audit.check(len(values) == 3 and np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 1)), f"{label}: valid three inner F1 scores")
        audit.close(float(row.inner_macro_f1), values.mean(), f"{label}: inner fold mean")
        audit.check(row.candidate == names[int(row.candidate_index)], f"{label}: declared candidate order")
        scores.append(float(row.inner_macro_f1))
    result = {}
    for phase in phases:
        result[phase] = {}
        for encoding in encodings:
            tasks = {}
            for task in config["tasks"]:
                selected = table[(table.phase == phase) & (table.encoding == encoding) & (table.task_id == task)].sort_values("candidate_index")
                row = selected.iloc[int(np.argmax(selected.inner_macro_f1.to_numpy()))]
                tasks[int(task)] = {"candidate": row.candidate, "candidate_index": int(row.candidate_index), "inner_macro_f1": float(row.inner_macro_f1)}
            result[phase][encoding] = {"score": float(np.mean([row["inner_macro_f1"] for row in tasks.values()])), "tasks": tasks}
    return result


def expected_fusions(predictions: pd.DataFrame, rules: list[str]) -> dict[str, pd.DataFrame]:
    rows = {rule: [] for rule in rules}
    for subject, group in predictions.groupby("subject_id", sort=True):
        values = {"majority_vote_all8": float(group.base_y_pred.mean()),
                  "calibrated_mean_all8": float(group.calibrated_probability_pd.mean()),
                  "calibrated_reliability_weighted_all8": float(np.average(group.calibrated_probability_pd, weights=group.reliability_weight))}
        if group.y_true.nunique() != 1:
            raise ValueError("Conflicting diagnosis in fusion input.")
        for rule in rules:
            rows[rule].append({"subject_id": subject, "y_true": int(group.y_true.iloc[0]), "y_pred": int(values[rule] >= 0.5),
                               "decision_score_pd": values[rule], "available_task_count": len(group)})
    return {rule: frame(values) for rule, values in rows.items()}


def audit_predictions(audit: Audit, predictions: pd.DataFrame, metadata: pd.DataFrame, ids: list[str], selected: dict, label: str, calibrated: bool):
    expected = metadata[metadata.subject_id.isin(ids)].set_index(["subject_id", "task_id"])
    keys = list(predictions[["subject_id", "task_id"]].itertuples(index=False, name=None))
    audit.check(len(keys) == len(expected) and set(keys) == set(expected.index), f"{label}: exact held-out participant/tasks")
    audit.check(set(predictions.y_pred) <= {0, 1} and np.all(np.isfinite(predictions.decision_score_pd)), f"{label}: finite predictions")
    for row in predictions.itertuples(index=False):
        choice = selected["tasks"][int(row.task_id)]
        truth = int(expected.loc[(row.subject_id, row.task_id), "label"] == "PD")
        audit.check(int(row.y_true) == truth, f"{label}: correct diagnosis")
        audit.check(row.selected_candidate == choice["candidate"], f"{label}: prediction uses selected candidate")
        audit.close(row.inner_macro_f1, choice["inner_macro_f1"], f"{label}: selection score propagated")
        audit.close(row.reliability_weight, max(choice["inner_macro_f1"] - 0.5, 0.01), f"{label}: training-only reliability")
    if calibrated:
        probabilities = predictions.calibrated_probability_pd.to_numpy(float)
        audit.check(np.all(np.isfinite(probabilities)) and np.all((probabilities >= 0) & (probabilities <= 1)), f"{label}: finite calibrated probability")
    else:
        audit.check(predictions.calibrated_probability_pd.isna().all(), f"{label}: no control calibration")


def same_frames(audit: Audit, actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], columns: list[str], label: str):
    audit.check(not actual.duplicated(keys).any(), f"{label}: unique keys")
    audit.check(len(actual) == len(expected), f"{label}: row count")
    left = actual[keys + columns].sort_values(keys).reset_index(drop=True)
    right = expected[keys + columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=0, atol=1e-9)
    audit.check(True, f"{label}: rows agree")


def audit_unit(audit: Audit, path: Path, outer: dict, config: dict, metadata: pd.DataFrame, condition: str, seed: int):
    label = path.as_posix().split("/units/")[-1]
    result = read_json(path / "result.json")
    before = read_json(path / "selection_before_test.json")
    authentic = condition == "authentic"
    encodings = config["encoding_order"] if authentic else config["encoding_order"][1:]
    policies = {"static": ["static"], "auth9": encodings, "auth8": encodings[1:]} if authentic else {condition: encodings}
    phases = ["meta_1", "meta_2", "meta_3", "outer_train"] if authentic else ["outer_train"]
    audit.check(result["repeat"] == outer["repeat"] and result["outer_fold"] == outer["outer_fold"] and
                result["condition"] == condition and result["control_seed"] == seed, f"{label}: unit identity")
    audit.check(before["selected"] == result["selections"] and before["fusion_search"] == result["fusion_search"], f"{label}: selections match pre-test record")
    audit.check((path / "selection_before_test.json").stat().st_mtime_ns <= (path / "result.json").stat().st_mtime_ns,
                f"{label}: selection recorded before final result")
    search = read_csv(path / "inner_search.csv.gz")
    tuning = reconstruct_search(audit, search, phases, encodings, config, label)
    audit.check(result["search_candidates_evaluated"] == len(search) and result["tuning_svm_fits"] == len(search) * 3, f"{label}: search accounting")
    choices = {}
    selection_keys = []
    for row in result["selections"]:
        phase, policy = row["phase"], row["policy"]
        selection_keys.append((phase, policy))
        chosen = max(policies[policy], key=lambda name: tuning[phase][name]["score"])
        audit.check(row["encoding"] == chosen, f"{label}/{phase}/{policy}: global encoding argmax and tie-break")
        audit.close(row["selection_score"], tuning[phase][chosen]["score"], f"{label}/{phase}/{policy}: global inner score")
        choices[(phase, policy)] = chosen
    audit.check(Counter(selection_keys) == Counter({(phase, policy): 1 for phase in phases for policy in policies}), f"{label}: complete policy/phase choices")
    for encoding, item in before["tuned_encodings"].items():
        audit.close(item["score"], tuning["outer_train"][encoding]["score"], f"{label}: persisted encoding score")
        for task, selection in item["tasks"].items():
            actual = tuning["outer_train"][encoding]["tasks"][int(task)]
            audit.check(selection["candidate"] == actual["candidate"] and selection["candidate_index"] == actual["candidate_index"], f"{label}: persisted hyperparameters")
            audit.close(selection["inner_macro_f1"], actual["inner_macro_f1"], f"{label}: persisted hyperparameter score")
    predictions = frame(result["task_predictions"])
    for policy in policies:
        selected = choices[("outer_train", policy)]
        current = predictions[predictions.policy == policy]
        audit.check(set(current.encoding) == {selected}, f"{label}: outer prediction encoding")
        audit_predictions(audit, current, metadata, outer["test_subject_ids"], tuning["outer_train"][selected], f"{label}/{policy}/outer", authentic)
    rules = config["validation"]["fusion_order"]
    if authentic:
        meta = read_csv(path / "meta_oof_predictions.csv")
        for entry in outer["meta_folds"]:
            for policy in policies:
                selected = choices[(f"meta_{entry['inner_fold']}", policy)]
                current = meta[(meta.meta_fold == entry["inner_fold"]) & (meta.policy == policy)]
                audit.check(set(current.encoding) == {selected}, f"{label}: meta prediction encoding")
                audit_predictions(audit, current, metadata, entry["validation_subject_ids"], tuning[f"meta_{entry['inner_fold']}"][selected], f"{label}/{policy}/meta{entry['inner_fold']}", True)
        audit.check(not meta.duplicated(["policy", "subject_id", "task_id"]).any(), f"{label}: each policy meta-OOF record exactly once")
        audit.check(len(result["fusion_search"]) == 9, f"{label}: three fusion rules per authentic policy")
        for policy in policies:
            recomputed = expected_fusions(meta[meta.policy == policy], rules)
            fusion_scores = {rule: score(values.y_true, values.y_pred) for rule, values in recomputed.items()}
            winner = max(rules, key=lambda rule: fusion_scores[rule])
            rows = [row for row in result["fusion_search"] if row["policy"] == policy]
            audit.check([row["fusion_rule"] for row in rows] == rules, f"{label}/{policy}: fusion rule order")
            for row in rows:
                audit.close(row["meta_oof_macro_f1"], fusion_scores[row["fusion_rule"]], f"{label}: independently reconstructed meta fusion F1")
                audit.check(row["selected"] == (row["fusion_rule"] == winner), f"{label}: selected fusion argmax")
            selected_row = next(row for row in result["selections"] if row["policy"] == policy and row["phase"] == "outer_train")
            audit.check(selected_row["fusion_rule"] == winner, f"{label}: selected fusion persisted before test")
            outer_fusion = frame(result["fusion_predictions"])
            current = outer_fusion[outer_fusion.policy == policy]
            audit.check(set(current.method) == {winner}, f"{label}: outer applies selected fusion only")
            expected = expected_fusions(predictions[predictions.policy == policy], rules)[winner]
            same_frames(audit, current, expected, ["subject_id"], ["y_true", "y_pred", "decision_score_pd", "available_task_count"], f"{label}/{policy}/outer fusion")
    else:
        audit.check(not result["fusion_predictions"] and not result["fusion_search"], f"{label}: no fusion in controls")
        audit.check(not (path / "meta_oof_predictions.csv").exists(), f"{label}: no control meta-OOF")
    return result


def feature_cache(config: dict, condition: str, seed: int, encoding: str):
    if condition == "authentic":
        if encoding == "static":
            root = resolve(config["static_run"]) / "features"
            return root / "metadata.csv", root / "static_lpq_rgb.npz"
        root = resolve(config["single_signal_run"] if len(config["signal_channels"][encoding]) == 1 else config["triplet_run"]) / "features"
        return root / f"{encoding}_metadata.csv", root / f"{encoding}_lpq_rgb.npz"
    root = resolve(config["controls_output_root"])
    # Prefer the declared deterministic control layout; never load an unrelated cache.
    stem = f"{encoding}__{condition}__seed{seed}"
    direct = (root / "features" / f"{stem}_metadata.csv", root / "features" / f"{stem}_lpq_rgb.npz")
    if all(path.is_file() for path in direct):
        return direct
    directories = [root / "features" / f"{condition}_seed{seed}", root / f"{condition}_seed{seed}" / "features"]
    for directory in directories:
        meta = directory / f"{encoding}_metadata.csv"
        values = directory / f"{encoding}_lpq_rgb.npz"
        if meta.is_file() and values.is_file():
            return meta, values
    raise FileNotFoundError(f"Control feature cache not found: {condition}, {seed}, {encoding}")


def audit_saved_models(audit: Audit, path: Path, config: dict, result: dict):
    models = joblib.load(path / "models.joblib")
    predictions = frame(result["task_predictions"])
    for encoding, task_models in models.items():
        meta_path, values_path = feature_cache(config, result["condition"], result["control_seed"], encoding)
        metadata = read_csv(meta_path)
        values = np.load(values_path, allow_pickle=False)["features"]
        audit.check(values.shape == (597, 768), f"{path.name}: feature cache shape")
        audit.check(not metadata.duplicated(["subject_id", "task_id"]).any(), f"{path.name}: feature cache row identities")
        lookup = {(row.subject_id, int(row.task_id)): i for i, row in enumerate(metadata.itertuples(index=False))}
        for task, artifact in task_models.items():
            current = predictions[(predictions.encoding == encoding) & (predictions.task_id == int(task))].drop_duplicates(["subject_id", "task_id"])
            indices = [lookup[(row.subject_id, int(row.task_id))] for row in current.itertuples(index=False)]
            actual = artifact["base"].predict(values[indices])
            audit.check(np.array_equal(actual, current.y_pred.to_numpy()), f"{path.name}/{encoding}/{task}: saved model reproduces labels")
            audit.close(artifact["base"].decision_function(values[indices]), current.decision_score_pd.to_numpy(), f"{path.name}/{encoding}/{task}: saved model reproduces margins", 2e-8)
            audit.check(set(current.selected_candidate) == {artifact["selected_candidate"]}, f"{path.name}/{encoding}/{task}: model candidate provenance")
            plan = read_json(resolve(config["output_root"]) / "manifests/selection_plan.json")
            outer = next(item for item in plan["outer_splits"] if item["repeat"] == result["repeat"] and item["outer_fold"] == result["outer_fold"])
            train_mask = (metadata.task_id == int(task)) & metadata.subject_id.isin(outer["train_subject_ids"])
            expected_mean = values[train_mask.to_numpy()].mean(axis=0, dtype=np.float64)
            audit.close(artifact["base"].named_steps["standardscaler"].mean_, expected_mean, f"{path.name}/{encoding}/{task}: scaler fits outer training only", 2e-10)
            if result["condition"] == "authentic":
                audit.check(artifact["calibrated"].ensemble is False, f"{path.name}: declared calibration ensemble=False")
                probabilities = artifact["calibrated"].predict_proba(values[indices])[:, 1]
                audit.close(probabilities, current.calibrated_probability_pd.to_numpy(), f"{path.name}/{encoding}/{task}: saved calibration reproduces probability", 2e-8)
            else:
                audit.check(artifact["calibrated"] is None, f"{path.name}: control model has no calibration")


def audit_aggregates(audit: Audit, root: Path, results: list[dict], config: dict):
    metrics = root / "metrics"
    actual_tasks = read_csv(metrics / "outer_task_predictions.csv")
    expected_tasks = frame([row for result in results for row in result["task_predictions"]])
    task_keys = ["policy", "control_seed"] + KEYS
    same_frames(audit, actual_tasks, expected_tasks, task_keys,
                ["encoding", "y_true", "y_pred", "decision_score_pd", "selected_candidate"], "aggregate task predictions")
    counts = actual_tasks.groupby(["policy", "control_seed"]).size()
    audit.check(len(counts) == 9 and counts.eq(2985).all(), "2,985 predictions for each of three authentic policies and six control/seed combinations")
    actual_fusion = read_csv(metrics / "outer_fusion_predictions.csv")
    expected_fusion = frame([row for result in results for row in result["fusion_predictions"]])
    same_frames(audit, actual_fusion, expected_fusion, ["policy"] + KEYS,
                ["encoding", "method", "y_true", "y_pred", "decision_score_pd"], "aggregate fusion predictions")
    audit.check(actual_fusion.groupby("policy").size().to_dict() == {"auth8": 375, "auth9": 375, "static": 375}, "375 fused predictions per authentic policy")
    recomputed = []
    for (policy, seed, task, repeat), group in actual_tasks.groupby(["policy", "control_seed", "task_id", "repeat"]):
        recomputed.append({"policy": policy, "control_seed": int(seed), "task_id": int(task), "repeat": int(repeat), "macro_f1": score(group.y_true, group.y_pred)})
    repeated = pd.DataFrame(recomputed)
    same_frames(audit, read_csv(metrics / "repeat_metrics.csv"), repeated, ["policy", "control_seed", "task_id", "repeat"], ["macro_f1"], "metrics concatenate folds before scoring")
    points = repeated.groupby("policy").macro_f1.mean().to_dict()
    task_points = repeated.groupby(["policy", "task_id"]).macro_f1.mean().to_dict()
    fused_points = {policy: float(np.mean([score(group.y_true, group.y_pred) for _, group in rows.groupby("repeat")])) for policy, rows in actual_fusion.groupby("policy")}
    for row in read_csv(metrics / "global_summary.csv").itertuples(index=False):
        expected = points[row.policy] if row.endpoint == "task_mean" else fused_points[row.policy]
        audit.close(row.macro_f1_mean, expected, "global point averages repetition/seed metrics, not predictions")
        audit.check(row.n_subjects == 75 and 0 <= row.ci_low <= row.ci_high <= 1, "global interval and observational unit")
    for row in read_csv(metrics / "task_summary.csv").itertuples(index=False):
        audit.close(row.macro_f1_mean, task_points[(row.policy, row.task_id)], "task point averages all declared seeds")
    for filename, expected_count in (("task_global_comparisons.csv", 4), ("fusion_comparisons.csv", 2), ("task_specific_comparisons.csv", 32)):
        comparisons = read_csv(metrics / filename)
        audit.check(len(comparisons) == expected_count, f"{filename}: declared family size")
        p_columns = [name for name in comparisons if name in {"permutation_p", "randomization_p", "p_value"}]
        adjusted_columns = [name for name in comparisons if name.startswith("holm_p") or name in {"p_adjusted", "adjusted_p"}]
        if len(p_columns) != 1 or len(adjusted_columns) != 1:
            raise ValueError(f"Ambiguous p-value schema in {filename}: {list(comparisons)}")
        p = comparisons[p_columns[0]].to_numpy(float)
        audit.check(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), f"{filename}: valid probabilities")
        order = np.argsort(p, kind="stable")
        adjusted = np.empty(len(p))
        adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
        audit.close(comparisons[adjusted_columns[0]].to_numpy(), adjusted, f"{filename}: independent Holm calculation")
        for row in comparisons.itertuples(index=False):
            if filename == "task_specific_comparisons.csv":
                expected_delta = task_points[(row.candidate, row.task_id)] - task_points[(row.reference, row.task_id)]
                actual_delta = row.delta_macro_f1
            else:
                source = fused_points if filename == "fusion_comparisons.csv" else points
                expected_delta = source[row.candidate] - source[row.reference]
                actual_delta = row.delta_task_mean_macro_f1
            audit.close(actual_delta, expected_delta, f"{filename}: independently reconstructed paired effect")
    return actual_tasks, actual_fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan-only", action="store_true", help="Audit configuration and participant plan only.")
    parser.add_argument("--allow-partial", action="store_true", help="Audit completed units while the run is in progress; never reports complete.")
    parser.add_argument("--models-per-condition", type=int, default=1, help="Number of deterministic units per condition/seed to reproduce without fitting (default: 1).")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = Audit()
    config = read_json(args.config)
    root = resolve(config["output_root"])
    metadata = read_csv(resolve(config["static_run"]) / "features/metadata.csv")
    plan = read_json(root / "manifests/selection_plan.json")
    audit.guarded("configuration", audit_config, audit, config)
    audit.guarded("frozen configuration/source/input integrity", audit_lock, audit, root, args.config, not args.allow_partial and not args.plan_only)
    audit.guarded("participant plan", audit_plan, audit, config, plan, metadata)
    conditions = [("authentic", 0)] + [(method, seed) for method in config["controls"]["methods"] for seed in config["controls"]["seeds"]]
    results = []
    completed = 0
    model_units = 0
    if not args.plan_only:
        for condition, seed in conditions:
            name = condition if condition == "authentic" else f"{condition}_seed{seed}"
            checked_models = 0
            for outer in plan["outer_splits"]:
                path = root / "units" / name / f"r{outer['repeat']:02}_f{outer['outer_fold']:02}"
                if not (path / "result.json").is_file():
                    if not args.allow_partial:
                        audit.check(False, f"missing completed unit: {name}/{path.name}")
                    continue
                completed += 1
                result = audit.guarded(f"unit {name}/{path.name}", audit_unit, audit, path, outer, config, metadata, condition, seed)
                if result is None:
                    continue
                results.append(result)
                if checked_models < args.models_per_condition:
                    audit.guarded(f"saved model {name}/{path.name}", audit_saved_models, audit, path, config, result)
                    checked_models += 1
                    model_units += 1
        if not args.allow_partial:
            audit.check(completed == 175 and len(results) == 175, "all 25 authentic and 150 control units verified")
            audit.guarded("aggregate metrics", audit_aggregates, audit, root, results, config)
    mode = "plan_only" if args.plan_only else "partial" if args.allow_partial else "complete"
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "passed": not audit.failures,
              "complete_run_verified": mode == "complete" and not audit.failures, "checks": audit.checks,
              "completed_units_found": completed, "units_fully_audited": len(results), "units_with_prediction_reproduction": model_units,
              "new_model_fits": 0, "config_sha256": sha256(args.config),
              "selection_plan_sha256": sha256(root / "manifests/selection_plan.json"),
              "verifier_sha256": sha256(Path(__file__)), "failures": audit.failures, "notes": audit.notes}
    destination = args.output or root / "manifests" / ("verification_plan.json" if args.plan_only else "verification_partial.json" if args.allow_partial else "verification.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("mode", "passed", "checks", "completed_units_found", "units_fully_audited", "units_with_prediction_reproduction")}, indent=2))
    if audit.failures:
        print(json.dumps(audit.failures[:8], indent=2))
    raise SystemExit(0 if not audit.failures else 1)


if __name__ == "__main__":
    main()
