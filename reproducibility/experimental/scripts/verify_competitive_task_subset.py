"""Independently audit the competitive run without fitting or selecting models.

The runner's input loader is reused only to read the aligned feature arrays.
Search choices, cohort coverage, fusion, scores and Holm adjustment are rebuilt
here from saved artifacts; no fitting/selection function is imported or called.
Only models produced locally in this run are deserialized for prediction checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["task_set", "method", "repeat", "outer_fold", "task_id", "subject_id"]
REPRODUCE_UNITS = {(1, 1), (5, 5)}


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    return pd.read_csv(path, dtype={"subject_id": str}, float_precision="round_trip")


def score(truth, predicted):
    return float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0))


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []
        self.notes = []
        self.units = 0
        self.reproduced_units = 0
        self.reproduced_models = 0

    def check(self, condition, message):
        self.checks += 1
        if not bool(condition):
            self.failures.append({"check": message})

    def close(self, actual, expected, message, atol=1e-10, rtol=0.0):
        self.check(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True), message)

    def guarded(self, label, function, *args):
        try:
            return function(*args)
        except Exception as error:
            self.failures.append({"check": label, "error": f"{type(error).__name__}: {error}",
                                  "traceback": traceback.format_exc(limit=4)})
            return None


def candidate_specs(learner, config):
    if learner == "logistic":
        return [(f"L2_C={float(c):g}", {"C": float(c)}) for c in config["logistic_search"]["c"]]
    search = config["svm_search"]
    return ([(f"linear_C={float(c):g}_gamma=na", {"kernel": "linear", "C": float(c), "gamma": "scale"})
             for c in search["linear_c"]]
            + [(f"rbf_C={float(c):g}_gamma={g}", {"kernel": "rbf", "C": float(c), "gamma": g})
               for c, g in product(search["rbf_c"], search["rbf_gamma"])])


def check_frame(audit, actual, expected, keys, label):
    audit.check(len(actual) == len(expected), f"{label}: row count")
    audit.check(not actual.duplicated(keys).any(), f"{label}: unique keys")
    a = actual.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    audit.check(list(a.columns) == list(b.columns) or set(a.columns) == set(b.columns), f"{label}: schema")
    if len(a) != len(b) or set(a.columns) != set(b.columns):
        return
    for column in b:
        if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
            audit.close(a[column], b[column], f"{label}: {column}")
        else:
            audit.check(a[column].fillna("").astype(str).tolist() == b[column].fillna("").astype(str).tolist(),
                        f"{label}: {column}")


def audit_lock(audit, config_path, config, output):
    lock = read_json(output / "manifests/run_lock.json")
    audit.check(lock["config_sha256"] == sha256(config_path), "frozen configuration hash")
    audit.check(lock["protocol_sha256"] == sha256(config_path.with_name("DECISION.md")), "frozen decision hash")
    audit.check(read_json(output / "manifests/config.json") == config, "configuration snapshot contents")
    audit.check(sha256(output / "manifests/DECISION.md") == lock["protocol_sha256"], "decision snapshot hash")
    for name, expected in lock["source_sha256"].items():
        audit.check(sha256(resolve(name)) == expected, f"fit source unchanged: {name}")
        audit.check(sha256(output / "manifests/source_snapshot" / name) == expected, f"fit snapshot: {name}")
    for name, expected in lock["input_sha256"].items():
        audit.check(sha256(resolve(name)) == expected, f"input unchanged: {name}")
    plan = read_json(output / "manifests/selection_plan.json")
    audit.check(plan == read_json(resolve(config["validation"]["selection_plan_from"])), "selection plan snapshot exact")
    return plan


def audit_features(audit, config, families, inputs, output):
    lock = read_json(output / "manifests/run_lock.json")
    for name, digest in inputs.items():
        audit.check(lock["input_sha256"].get(name) == digest, f"loaded matrix/metadata locked: {name}")
    for family, count, dimension in (("lpq", 9, 768), ("kinematic", 1, 83), ("transfer", 9, 512)):
        audit.check(len(families[family]) == count, f"{family}: matrix count")
        for name, matrix in families[family].items():
            audit.check(matrix.shape == (597, dimension) and np.isfinite(matrix).all(), f"{family}/{name}: finite shape")
    kinematic = resolve(config["kinematic_directory"])
    km = read_json(kinematic / "manifest.json")
    audit.check(km["status"] == "complete" and km["shape"] == [597, 83], "kinematic feature manifest complete")
    audit.check(km["models_fitted"] == 0, "kinematic extraction has no model fitting")
    audit.check(len(read_json(kinematic / "feature_names.json")) == 83, "83 named kinematic features")
    for name, expected in km["artifact_sha256"].items():
        audit.check(sha256(kinematic / name) == expected, f"kinematic artifact: {name}")
    for name, expected in km["source_sha256"].items():
        audit.check(sha256(resolve(name)) == expected, f"kinematic source: {name}")
    audit.check(sha256(config_path_from_output(config).with_name("kinematic_descriptor.json")) == km["descriptor_source_sha256"],
                "kinematic descriptor unchanged")
    for name, expected in km["input_sha256"].items():
        audit.check(sha256(Path(km["dataset_root"]) / name) == expected, f"kinematic signal input: {name}")
    transfer = resolve(config["transfer_directory"])
    tm = read_json(transfer / "manifests/feature_manifest.json")
    audit.check(tm["status"] == "complete", "transfer feature manifest complete")
    audit.check(tm["encoding_order"] == config["encoding_order"], "transfer encoding order")
    audit.check(set(tm["completed_encodings"]) == set(config["encoding_order"]), "all transfer encodings complete")
    audit.check(tm["image_input_count"] == 597 * 9, "transfer image count")
    audit.check(sha256(transfer / "metadata.csv") == tm["shared_metadata_sha256"], "transfer shared metadata")
    audit.check(sha256(transfer / "manifests/config.json") == tm["config_sha256"], "transfer extraction config")
    audit.check(sha256(transfer / "manifests/image_inputs.csv") == tm["image_inputs_sha256"], "transfer image inventory")
    for name, expected in tm["code_sha256"].items():
        audit.check(sha256(resolve(name)) == expected, f"transfer source: {name}")
        audit.check(sha256(transfer / "manifests/source_snapshot" / name) == expected, f"transfer source snapshot: {name}")
    shared = read_csv(transfer / "metadata.csv")
    record_keys = [f"{s}|{t}|{r}" for s, t, r in shared[["subject_id", "task_id", "repetition_id"]].itertuples(index=False, name=None)]
    for encoding, item in tm["completed_encodings"].items():
        path = transfer / item["path"]
        audit.check(sha256(path) == item["sha256"], f"transfer matrix hash: {encoding}")
        audit.check(item["shape"] == [597, 512] and item["dtype"] == "float32", f"transfer matrix schema: {encoding}")
        audit.check(item["first_batch_bitwise_repeatable"], f"transfer deterministic batch: {encoding}")
        audit.check(item["model_state_sha256_before"] == item["model_state_sha256_after"] == tm["backbone_state_sha256"],
                    f"transfer backbone unchanged: {encoding}")
        with np.load(path, allow_pickle=False) as archive:
            audit.check(archive["record_keys"].tolist() == record_keys, f"transfer row identity: {encoding}")
            audit.check(archive["feature_names"].tolist() == [f"resnet18_gap_{i:03d}" for i in range(512)],
                        f"transfer column identity: {encoding}")


def config_path_from_output(config):
    return ROOT / "experiments/2026-09-restart" / config["experiment_id"] / "config.json"


def audit_plan(audit, config, plan, metadata):
    subjects = metadata[["subject_id", "label"]].drop_duplicates().sort_values("subject_id")
    audit.check(len(subjects) == subjects.subject_id.nunique() == 75, "75 unique participants and labels")
    audit.check(subjects.label.value_counts().to_dict() == {"H": 38, "PD": 37}, "38 H and 37 PD")
    audit.check(len(metadata) == 597 and not metadata.duplicated(["subject_id", "task_id"]).any(), "597 unique recordings")
    audit.check(metadata.task_id.value_counts().sort_index().tolist() == [72, 75, 75, 75, 75, 75, 75, 75], "task availability")
    audit.check(subjects.to_dict("records") == pd.DataFrame(plan["subjects"]).sort_values("subject_id").to_dict("records"),
                "plan labels and participants match metadata")
    audit.check(config["task_sets"] == {"literature_T234": [2, 3, 4], "all8": list(range(1, 9))}, "two prespecified task sets")
    audit.check(len(candidate_specs("svm", config)) == 21 and len(candidate_specs("logistic", config)) == 7,
                "declared 21 SVM and seven logistic candidates")
    labels = subjects.set_index("subject_id").label.to_dict()
    original = read_json(resolve(config["validation"]["outer_splits_from"]))
    original_index = {(int(x["repeat"]), int(x["outer_fold"])): x for x in original["outer_splits"]}
    audit.check(len(plan["outer_splits"]) == 25, "25 external units")
    seen, keys = Counter(), []
    for outer in plan["outer_splits"]:
        key = int(outer["repeat"]), int(outer["outer_fold"])
        keys.append(key)
        train, test = outer["train_subject_ids"], outer["test_subject_ids"]
        audit.check(len(train) == len(set(train)) == 60 and len(test) == len(set(test)) == 15, f"{key}: 60/15 unique IDs")
        audit.check(not set(train) & set(test) and set(train) | set(test) == set(labels), f"{key}: outer isolation")
        for name in ("train_subject_ids", "test_subject_ids"):
            audit.check(set(outer[name]) == {str(x).zfill(5) for x in original_index[key][name]}, f"{key}: original {name}")
            audit.check({labels[x] for x in outer[name]} == {"H", "PD"}, f"{key}: both diagnoses in {name}")
        seen.update((key[0], person) for person in test)
        validated = []
        audit.check(len(outer["tuning_folds"]) == 3, f"{key}: three inner folds")
        for inner in outer["tuning_folds"]:
            a, b = inner["train_subject_ids"], inner["validation_subject_ids"]
            audit.check(len(a) == len(set(a)) == 40 and len(b) == len(set(b)) == 20, f"{key}: 40/20 inner IDs")
            audit.check(not set(a) & set(b) and set(a) | set(b) == set(train), f"{key}: inner confinement")
            validated.extend(b)
            for task in range(1, 9):
                ta = metadata[metadata.subject_id.isin(a) & metadata.task_id.eq(task)]
                tb = metadata[metadata.subject_id.isin(b) & metadata.task_id.eq(task)]
                audit.check(set(ta.label) == set(tb.label) == {"H", "PD"}, f"{key}/T{task}: both inner classes")
                audit.check(not set(ta.subject_id) & set(tb.subject_id), f"{key}/T{task}: task-level disjointness")
        audit.check(Counter(validated) == Counter(train), f"{key}: exactly one validation per train participant")
    audit.check(Counter(keys) == Counter(product(range(1, 6), range(1, 6))), "complete distinct repeat/fold grid")
    audit.check(seen == Counter(product(range(1, 6), labels)), "each participant tested once per repetition")


def audit_search(audit, config, directory, unit):
    table = read_csv(directory / "inner_search.csv.gz")
    families = {("lpq", "svm"): config["encoding_order"], ("kinematic", "svm"): ["kinematic"],
                ("transfer", "logistic"): config["encoding_order"]}
    names = {learner: candidate_specs(learner, config) for learner in ("svm", "logistic")}
    expected = {(family, learner, encoding, task, index)
                for (family, learner), encodings in families.items()
                for encoding, task, index in product(encodings, range(1, 9), range(len(names[learner])))}
    columns = ["feature_family", "learner", "encoding", "task_id", "candidate_index"]
    audit.check(Counter(table[columns].itertuples(index=False, name=None)) == Counter(expected),
                f"{directory.name}: complete unique search grid")
    winners = {}
    for (family, learner, encoding, task), group in table.groupby(columns[:-1], sort=False):
        scores = []
        for row in group.sort_values("candidate_index").itertuples(index=False):
            raw = json.loads(row.inner_fold_scores)
            label = f"{directory.name}/{family}/{encoding}/T{task}/c{row.candidate_index}"
            audit.check(len(raw) == 3 and all(np.isfinite(x) and 0 <= x <= 1 for x in raw), f"{label}: three valid scores")
            value = float(np.mean(raw))
            audit.close(row.inner_macro_f1, value, f"{label}: raw fold mean", atol=1e-15)
            audit.check(row.candidate == names[learner][row.candidate_index][0], f"{label}: ordered candidate name")
            scores.append((row.candidate_index, row.candidate, value))
        winner = max(scores, key=lambda item: item[2])
        winners[family, learner, encoding, int(task)] = winner
    expected_selection_keys = set(product(config["task_sets"], config["method_order"]))
    audit.check(Counter((x["task_set"], x["method"]) for x in unit["selections"]) == Counter(expected_selection_keys),
                f"{directory.name}: eight unique pipeline choices")
    for selected in unit["selections"]:
        method, task_set = selected["method"], selected["task_set"]
        settings, tasks = config["methods"][method], config["task_sets"][task_set]
        eligible = config["encoding_order"] if settings["eligible_encodings"] == "all9" else settings["eligible_encodings"]
        scores = {encoding: float(np.mean([winners[settings["feature_family"], settings["learner"], encoding, task][2]
                                          for task in tasks])) for encoding in eligible}
        chosen = max(eligible, key=lambda encoding: scores[encoding])
        label = f"{directory.name}/{task_set}/{method}"
        audit.check(selected["encoding"] == chosen, f"{label}: training-only encoding argmax with ordered ties")
        audit.check(set(selected["encoding_scores"]) == set(eligible), f"{label}: eligible score coverage")
        for encoding, value in scores.items():
            audit.close(selected["encoding_scores"][encoding], value, f"{label}/{encoding}: task-set mean", atol=1e-15)
        audit.close(selected["selection_score"], scores[chosen], f"{label}: winning mean", atol=1e-15)
        audit.check(set(map(int, selected["tasks"])) == set(tasks), f"{label}: selected tasks")
        for task in tasks:
            expected_index, expected_name, expected_mean = winners[settings["feature_family"], settings["learner"], chosen, task]
            item = selected["tasks"][str(task)]
            audit.check(item["candidate_index"] == expected_index and item["candidate"] == expected_name,
                        f"{label}/T{task}: task candidate argmax with ordered ties")
            audit.close(item["inner_macro_f1"], expected_mean, f"{label}/T{task}: selected task mean", atol=1e-15)
    audit.check(unit["tuning_fits"] == len(expected) * 3, f"{directory.name}: tuning fit count")


def audit_predictions(audit, config, metadata, outer, directory, unit):
    before = read_json(directory / "selection_before_test.json")
    audit.check(before["train_subject_ids"] == outer["train_subject_ids"] and before["test_subject_ids"] == outer["test_subject_ids"],
                f"{directory.name}: persisted partition")
    audit.check(before["selections"] == unit["selections"], f"{directory.name}: selection unchanged after test")
    tasks = pd.DataFrame(unit["task_predictions"])
    fused = pd.DataFrame(unit["fusion_predictions"])
    choices = {(x["task_set"], x["method"]): x for x in unit["selections"]}
    expected_keys = []
    for task_set, method in product(config["task_sets"], config["method_order"]):
        available = metadata[metadata.subject_id.isin(outer["test_subject_ids"]) & metadata.task_id.isin(config["task_sets"][task_set])]
        expected_keys.extend((task_set, method, outer["repeat"], outer["outer_fold"], int(row.task_id), row.subject_id)
                             for row in available.itertuples())
    audit.check(Counter(tasks[KEYS].itertuples(index=False, name=None)) == Counter(expected_keys), f"{directory.name}: exact test task rows")
    truth = metadata.set_index(["subject_id", "task_id"]).label.to_dict()
    for row in tasks.itertuples():
        chosen = choices[row.task_set, row.method]
        expected = chosen["tasks"][str(row.task_id)]
        label = f"{directory.name}/{row.task_set}/{row.method}/T{row.task_id}/{row.subject_id}"
        audit.check(row.y_true == int(truth[row.subject_id, row.task_id] == "PD"), f"{label}: true diagnosis")
        audit.check(row.y_pred in (0, 1) and np.isfinite(row.score_pd), f"{label}: valid prediction and margin")
        audit.check(row.encoding == chosen["encoding"] and row.selected_candidate == expected["candidate"], f"{label}: selected model identity")
        audit.close(row.inner_macro_f1, expected["inner_macro_f1"], f"{label}: frozen training score")
    rebuilt = []
    for key, group in tasks.groupby(["task_set", "method", "repeat", "outer_fold", "subject_id"], sort=True):
        task_set, method, repeat, fold, person = key
        fraction = int(group.y_pred.sum()) / len(group)
        rebuilt.append({"task_set": task_set, "method": method, "repeat": repeat, "outer_fold": fold,
                        "subject_id": person, "task_id": 0, "y_true": int(group.y_true.iloc[0]),
                        "y_pred": int(2 * int(group.y_pred.sum()) >= len(group)), "score_pd": fraction,
                        "available_tasks": len(group), "task_ids": ",".join(str(x) for x in sorted(group.task_id)),
                        "encoding": choices[task_set, method]["encoding"]})
    check_frame(audit, fused, pd.DataFrame(rebuilt), KEYS, f"{directory.name}: independent majority fusion")
    expected_fusion = {(task_set, method, outer["repeat"], outer["outer_fold"], 0, person)
                       for task_set, method, person in product(config["task_sets"], config["method_order"], outer["test_subject_ids"])}
    audit.check(Counter(fused[KEYS].itertuples(index=False, name=None)) == Counter(expected_fusion), f"{directory.name}: exact fusion rows")
    return tasks, fused


def audit_models(audit, config, metadata, families, outer, directory, unit):
    models = joblib.load(directory / "models.joblib")
    choices = unit["selections"]
    expected_keys = set()
    predictions = pd.DataFrame(unit["task_predictions"])
    for selected in choices:
        definition = config["methods"][selected["method"]]
        for task, choice in selected["tasks"].items():
            expected_keys.add((definition["feature_family"], definition["learner"], selected["encoding"], int(task), choice["candidate_index"]))
    audit.check(set(models) == expected_keys and len(models) == unit["final_fits"], f"{directory.name}: persisted model keys/count")
    for (family, learner, encoding, task, candidate_index), model in models.items():
        label = f"{directory.name}/{family}/{encoding}/T{task}/c{candidate_index}"
        matrix = families[family][encoding]
        train = np.flatnonzero(metadata.subject_id.isin(outer["train_subject_ids"]) & metadata.task_id.eq(task))
        audit.check(len(model.steps) == 2 and isinstance(model.steps[0][1], StandardScaler), f"{label}: training scaler pipeline")
        scaler, estimator = model.steps[0][1], model.steps[-1][1]
        # StandardScaler accumulates even float32 LPQ input in float64.
        training_values = np.asarray(matrix[train], dtype=np.float64)
        audit.close(scaler.mean_, training_values.mean(axis=0), f"{label}: scaler mean from training only", atol=1e-8, rtol=1e-12)
        audit.close(scaler.var_, training_values.var(axis=0), f"{label}: scaler variance from training only", atol=1e-8, rtol=1e-12)
        audit.check(int(scaler.n_samples_seen_) == len(train), f"{label}: scaler training sample count")
        specs = candidate_specs(learner, config)[candidate_index][1]
        audit.check(isinstance(estimator, SVC if learner == "svm" else LogisticRegression), f"{label}: estimator class")
        audit.check(estimator.class_weight == "balanced" and estimator.classes_.tolist() == [0, 1], f"{label}: weights/classes")
        for name, value in specs.items():
            audit.check(getattr(estimator, name) == value, f"{label}: selected {name}")
        if learner == "logistic":
            settings = config["logistic_search"]
            audit.check(estimator.l1_ratio == 0.0 and estimator.solver == settings["solver"], f"{label}: L2 logistic head")
            audit.check(int(np.max(estimator.n_iter_)) < settings["max_iter"], f"{label}: convergence before iteration limit")
        for selected in choices:
            settings = config["methods"][selected["method"]]
            if (settings["feature_family"], settings["learner"], selected["encoding"]) != (family, learner, encoding):
                continue
            choice = selected["tasks"].get(str(task))
            if choice is None or choice["candidate_index"] != candidate_index:
                continue
            frame = predictions[(predictions.task_set == selected["task_set"]) & (predictions.method == selected["method"]) & predictions.task_id.eq(task)].sort_values("subject_id")
            indices = metadata.reset_index().set_index(["subject_id", "task_id"])["index"]
            test = [indices.loc[(person, task)] for person in frame.subject_id]
            audit.check(np.array_equal(model.predict(matrix[test]), frame.y_pred.to_numpy()), f"{label}/{selected['task_set']}: reproduced labels")
            audit.close(model.decision_function(matrix[test]), frame.score_pd.to_numpy(), f"{label}/{selected['task_set']}: reproduced margins", atol=1e-8, rtol=1e-12)
        audit.reproduced_models += 1
    audit.reproduced_units += 1


def audit_units(audit, config, output, plan, metadata, families):
    expected_dirs = {f"r{x['repeat']:02}_f{x['outer_fold']:02}" for x in plan["outer_splits"]}
    actual_dirs = {path.parent.name for path in (output / "units").glob("*/result.json")}
    audit.check(actual_dirs == expected_dirs, "25 and only 25 complete unit directories")
    tasks, fused, selections, units = [], [], [], []
    for outer in plan["outer_splits"]:
        directory = output / "units" / f"r{outer['repeat']:02}_f{outer['outer_fold']:02}"
        unit = read_json(directory / "result.json")
        audit.check((unit["repeat"], unit["outer_fold"]) == (outer["repeat"], outer["outer_fold"]), f"{directory.name}: unit identity")
        audit.guarded(f"{directory.name}: search", audit_search, audit, config, directory, unit)
        frames = audit.guarded(f"{directory.name}: predictions", audit_predictions, audit, config, metadata, outer, directory, unit)
        if frames is not None:
            tasks.append(frames[0])
            fused.append(frames[1])
        if (outer["repeat"], outer["outer_fold"]) in REPRODUCE_UNITS:
            audit.guarded(f"{directory.name}: saved model reproduction", audit_models, audit, config, metadata, families, outer, directory, unit)
        for row in unit["selections"]:
            selections.append({"repeat": unit["repeat"], "outer_fold": unit["outer_fold"],
                               **{key: value for key, value in row.items() if key not in ("tasks", "encoding_scores")}})
        units.append(unit)
        audit.units += 1
    task = pd.concat(tasks, ignore_index=True)
    fusion = pd.concat(fused, ignore_index=True)
    audit.check(len(task) == 16440, "16,440 task prediction rows")
    audit.check(len(fusion) == 3000, "3,000 participant fusion rows")
    audit.check(len(selections) == 200, "200 pipeline encoding choices")
    check_frame(audit, read_csv(output / "metrics/outer_task_predictions.csv"), task, KEYS, "aggregated task predictions")
    check_frame(audit, read_csv(output / "metrics/outer_fusion_predictions.csv"), fusion, KEYS, "aggregated fusion predictions")
    check_frame(audit, read_csv(output / "metrics/selections.csv"), pd.DataFrame(selections),
                ["repeat", "outer_fold", "task_set", "method"], "aggregated selections")
    manifest = read_json(output / "manifests/run_manifest.json")
    for name, value in (("completed_outer_units", 25), ("expected_outer_units", 25), ("task_prediction_rows", 16440),
                        ("fusion_prediction_rows", 3000), ("tuning_fits", sum(x["tuning_fits"] for x in units)),
                        ("final_fits", sum(x["final_fits"] for x in units))):
        audit.check(manifest[name] == value, f"run manifest: {name}")
    audit.check(audit.reproduced_units == 2, "both deterministic model reproduction units completed")
    return task, fusion


def audit_analysis(audit, config, output, task, fusion):
    """Reconstruct aggregation and declared comparison families, never rescore folds."""
    manifest = read_json(output / "manifests/run_manifest.json")
    audit.check(manifest["status"] in {"analysis_complete", "complete"}, "analysis completed before complete audit")
    audit.check(manifest["analysis_source_sha256"] == sha256(ROOT / "scripts/analyze_competitive_task_subset.py"),
                "analysis source unchanged")
    destination = output / "metrics"
    methods = config["method_order"]
    frames = {"task_mean": task, "participant_fusion": fusion}
    repeat_rows = []
    expected_means = {}
    for endpoint, frame in frames.items():
        for (task_set, method), current in frame.groupby(["task_set", "method"]):
            metrics = []
            for (task_id, repeat), group in current.groupby(["task_id", "repeat"]):
                audit.check(not group.subject_id.duplicated().any(), f"{endpoint}/{task_set}/{method}/T{task_id}/r{repeat}: folds pooled once")
                y, p = group.y_true.to_numpy(), group.y_pred.to_numpy()
                sensitivity, specificity = float(np.mean(p[y == 1] == 1)), float(np.mean(p[y == 0] == 0))
                row = {"macro_f1": score(y, p), "balanced_accuracy": (sensitivity + specificity) / 2,
                       "accuracy": float(np.mean(y == p)), "sensitivity_pd": sensitivity,
                       "specificity_h": specificity, "f1_pd": float(f1_score(y, p, pos_label=1, zero_division=0)),
                       "roc_auc": float(roc_auc_score(y, group.score_pd))}
                metrics.append(row)
                repeat_rows.append({"task_set": task_set, "method": method, "endpoint": endpoint,
                                    "task_id": task_id, "repeat": repeat, **row})
            expected_means[task_set, method, endpoint] = {name: float(np.mean([row[name] for row in metrics])) for name in metrics[0]}
    repeats = pd.DataFrame(repeat_rows)
    check_frame(audit, read_csv(destination / "repeat_metrics.csv"), repeats,
                ["task_set", "method", "endpoint", "task_id", "repeat"], "repetition metrics reconstructed after pooling folds")
    summary = read_csv(destination / "global_summary.csv")
    audit.check(Counter(summary[["task_set", "method", "endpoint"]].itertuples(index=False, name=None)) == Counter(expected_means.keys()),
                "16 global method/task-set/endpoint summaries")
    for row in summary.itertuples(index=False):
        key = row.task_set, row.method, row.endpoint
        for name, expected in expected_means[key].items():
            actual_name = "macro_f1_mean" if name == "macro_f1" else name
            audit.close(getattr(row, actual_name), expected, f"summary {key}: {name}")
        audit.check(row.n_subjects == 75 and 0 <= row.ci_low <= row.ci_high <= 1, f"summary {key}: participant count/interval bounds")
    names = ["macro_f1", "balanced_accuracy", "accuracy", "sensitivity_pd", "specificity_h", "f1_pd", "roc_auc"]
    expected_tasks = repeats[repeats.endpoint == "task_mean"].groupby(["task_set", "method", "task_id"])[names].mean().reset_index()
    check_frame(audit, read_csv(destination / "task_summary.csv"), expected_tasks,
                ["task_set", "method", "task_id"], "task summaries averaged over five repetition scores")
    comparisons = read_csv(destination / "method_comparisons.csv")
    ordered_pairs = {(candidate, reference) for reference, candidate in combinations(methods, 2)}
    expected_pairs = {(task_set, endpoint, candidate, reference)
                      for task_set, endpoint in product(config["task_sets"], frames)
                      for candidate, reference in ordered_pairs}
    audit.check(Counter(comparisons[["task_set", "endpoint", "candidate", "reference"]].itertuples(index=False, name=None)) == Counter(expected_pairs),
                "24 declared method contrasts in four families")
    for key, group in comparisons.groupby(["task_set", "endpoint"]):
        audit_holm(audit, group, 6, f"method family {key}")
        for row in group.itertuples(index=False):
            expected = expected_means[row.task_set, row.candidate, row.endpoint]["macro_f1"] - expected_means[row.task_set, row.reference, row.endpoint]["macro_f1"]
            audit.close(row.delta_macro_f1, expected, f"contrast {key}/{row.candidate}/{row.reference}: observed paired difference")
            audit.check(-1 <= row.ci_low <= row.ci_high <= 1, f"contrast {key}/{row.candidate}/{row.reference}: interval bounds")
    reduction = read_csv(destination / "task_subset_comparisons.csv")
    audit.check(Counter(reduction.method) == Counter(methods), "four within-method task-subset contrasts")
    audit.check(set(reduction.candidate) == {"literature_T234"} and set(reduction.reference) == {"all8"}, "task-subset direction")
    audit_holm(audit, reduction, 4, "within-method subset family")
    for row in reduction.itertuples(index=False):
        expected = expected_means["literature_T234", row.method, "participant_fusion"]["macro_f1"] - expected_means["all8", row.method, "participant_fusion"]["macro_f1"]
        audit.close(row.delta_macro_f1, expected, f"subset {row.method}: participant fusion difference")
    choices = read_csv(destination / "selections.csv")
    frequency = choices.groupby(["task_set", "method", "encoding"]).size().reset_index(name="count")
    check_frame(audit, read_csv(destination / "encoding_selection_frequency.csv"), frequency,
                ["task_set", "method", "encoding"], "encoding frequencies include every choice")
    # The six primary contrasts are also checked by independently generated
    # stratified bootstrap sample indices and participant-level method swaps.
    primary = comparisons[(comparisons.task_set == config["primary_task_set"]) & (comparisons.endpoint == "participant_fusion")]
    for row in primary.itertuples(index=False):
        a = fusion[(fusion.task_set == row.task_set) & (fusion.method == row.candidate)]
        b = fusion[(fusion.task_set == row.task_set) & (fusion.method == row.reference)]
        reconstructed = reproduce_fusion_inference(a, b, config["validation"])
        for name, expected in reconstructed.items():
            audit.close(getattr(row, name), expected, f"primary inference {row.candidate}/{row.reference}: {name}", atol=1e-12)
    audit.notes.append("All scores, 28 contrasts and five Holm families reconstructed; the six primary bootstrap/randomization contrasts reproduced independently with the declared seeds. Secondary raw p-values and pointwise intervals were not independently regenerated.")


def audit_holm(audit, frame, size, label):
    audit.check(len(frame) == size and set(frame.family_size) == {size}, f"{label}: family size")
    pvalues = frame.permutation_p.to_numpy(dtype=float)
    audit.check(np.isfinite(pvalues).all() and np.all((pvalues >= 0) & (pvalues <= 1)), f"{label}: valid raw p-values")
    previous = 0.0
    adjusted = np.zeros(len(frame))
    for rank, index in enumerate(sorted(range(len(frame)), key=lambda i: pvalues[i])):
        previous = max(previous, min(1.0, (len(frame) - rank) * pvalues[index]))
        adjusted[index] = previous
    audit.close(frame.holm_p, adjusted, f"{label}: Holm values")
    audit.check(np.array_equal(frame.supported_holm_0_05.to_numpy(), adjusted < .05), f"{label}: significance flags")


def reproduce_fusion_inference(candidate, reference, settings):
    """Independent index-based bootstrap and binary-confusion swap arithmetic."""
    people = candidate[["subject_id", "y_true"]].drop_duplicates().sort_values("subject_id")
    ids, truth = people.subject_id.tolist(), people.y_true.to_numpy(dtype=np.int8)
    n_boot, n_swap = settings["bootstrap_resamples"], settings["paired_randomization_resamples"]
    rng = np.random.default_rng(settings["bootstrap_seed"])
    sampled = np.concatenate([rng.choice(np.flatnonzero(truth == label), (n_boot, int(np.sum(truth == label))), replace=True)
                              for label in (0, 1)], axis=1)
    switches = np.random.default_rng(settings["paired_randomization_seed"]).integers(0, 2, (n_swap, len(ids)), dtype=np.int8).astype(bool)

    def batch_score(y, p):
        tn = np.sum((y == 0) & (p == 0), axis=1)
        tp = np.sum((y == 1) & (p == 1), axis=1)
        wrong = np.sum(y != p, axis=1)
        den_h, den_pd = 2 * tn + wrong, 2 * tp + wrong
        fh = np.divide(2 * tn, den_h, out=np.zeros(len(tn), dtype=float), where=den_h != 0)
        fp = np.divide(2 * tp, den_pd, out=np.zeros(len(tp), dtype=float), where=den_pd != 0)
        return (fh + fp) / 2

    observed, boot, random = [], [], []
    for repeat in sorted(candidate.repeat.unique()):
        a = candidate[candidate.repeat == repeat].set_index("subject_id").y_pred.reindex(ids).to_numpy(dtype=np.int8)
        b = reference[reference.repeat == repeat].set_index("subject_id").y_pred.reindex(ids).to_numpy(dtype=np.int8)
        observed.append(batch_score(truth[None, :], a[None, :])[0] - batch_score(truth[None, :], b[None, :])[0])
        boot.append(batch_score(truth[sampled], a[sampled]) - batch_score(truth[sampled], b[sampled]))
        pa, pb = np.where(switches, b, a), np.where(switches, a, b)
        random.append(batch_score(truth[None, :], pa) - batch_score(truth[None, :], pb))
    value = float(np.mean(observed))
    low, high = np.percentile(np.mean(boot, axis=0), [2.5, 97.5])
    random_values = np.mean(random, axis=0)
    p = (1 + int(np.count_nonzero(np.abs(random_values) >= abs(value) - 1e-15))) / (n_swap + 1)
    return {"delta_macro_f1": value, "ci_low": float(low), "ci_high": float(high), "permutation_p": p}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("fits", "complete"), default="fits")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output = resolve(config["output_root"])
    audit = Audit()
    try:
        plan = audit_lock(audit, config_path, config, output)
        from run_competitive_task_subset import load_competitive_inputs
        metadata, families, inputs = load_competitive_inputs(config)
        audit_features(audit, config, families, inputs, output)
        audit_plan(audit, config, plan, metadata)
        task, fusion = audit_units(audit, config, output, plan, metadata, families)
        if args.phase == "complete":
            audit_analysis(audit, config, output, task, fusion)
    except Exception as error:
        audit.failures.append({"check": "audit execution", "error": f"{type(error).__name__}: {error}",
                               "traceback": traceback.format_exc(limit=6)})
    result = {"phase": args.phase, "passed": not audit.failures,
              "complete_run_verified": args.phase == "complete" and not audit.failures,
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "checks": audit.checks, "units_fully_audited": audit.units,
              "units_with_prediction_reproduction": audit.reproduced_units,
              "models_with_prediction_reproduction": audit.reproduced_models,
              "new_model_fits": 0, "config_sha256": sha256(config_path),
              "verifier_sha256": sha256(Path(__file__)), "failures": audit.failures, "notes": audit.notes}
    destination = output / "manifests" / ("verification.json" if args.phase == "complete" else "verification_fits.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "failures"}, indent=2), flush=True)
    if audit.failures:
        print(json.dumps(audit.failures[:20], indent=2), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
