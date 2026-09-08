"""Participant-nested selection of a global encoding, task SVMs and late fusion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .evaluation import Candidate, candidates_from_config, make_model
from .task_fusion import FUSION_METHODS, fuse_fold, reliability_weight


def macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Binary macro F1 with both labels present in the scoring definition."""
    tn = np.count_nonzero((truth == 0) & (predicted == 0))
    tp = np.count_nonzero((truth == 1) & (predicted == 1))
    wrong = np.count_nonzero(truth != predicted)
    h = 2 * tn / (2 * tn + wrong) if 2 * tn + wrong else 0.0
    pd_score = 2 * tp / (2 * tp + wrong) if 2 * tp + wrong else 0.0
    return float((h + pd_score) / 2)


def derived_seed(master: int, context: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{master}:{context}".encode()).digest()[:4], "big") % (2**31 - 1)


def shared_folds(subjects: pd.DataFrame, ids: list[str], seed: int, count: int = 3) -> list[dict]:
    current = subjects[subjects.subject_id.isin(ids)].sort_values("subject_id")
    if set(current.subject_id) != set(ids):
        raise ValueError("Partition contains an unknown participant.")
    if current.label.value_counts().min() < count:
        raise ValueError("Not enough participants of both diagnoses for the declared folds.")
    folds = []
    for index, (train, valid) in enumerate(StratifiedKFold(count, shuffle=True, random_state=seed).split(current.subject_id, current.label), 1):
        folds.append({"inner_fold": index,
                      "train_subject_ids": current.iloc[train].subject_id.tolist(),
                      "validation_subject_ids": current.iloc[valid].subject_id.tolist()})
    return folds


def build_selection_plan(metadata: pd.DataFrame, original: dict, config: dict) -> dict:
    subjects = metadata[["subject_id", "label"]].drop_duplicates().sort_values("subject_id")
    if subjects.subject_id.duplicated().any():
        raise ValueError("Conflicting participant labels.")
    outer_splits = []
    master = config["validation"]["master_seed"]
    for original_fold in original["outer_splits"]:
        repeat, fold = int(original_fold["repeat"]), int(original_fold["outer_fold"])
        outer = {key: original_fold[key] for key in ("repeat", "outer_fold", "train_subject_ids", "test_subject_ids")}
        for key in ("train_subject_ids", "test_subject_ids"):
            outer[key] = [str(value).zfill(5) for value in outer[key]]
        seed = derived_seed(master, f"r{repeat}_f{fold}_meta")
        outer["meta_seed"] = seed
        meta = shared_folds(subjects, outer["train_subject_ids"], seed, config["validation"]["meta_folds"])
        outer["tuning_folds"] = [dict(item) for item in meta]
        for entry in meta:
            tuning_seed = derived_seed(master, f"r{repeat}_f{fold}_meta{entry['inner_fold']}_tuning")
            entry["tuning_seed"] = tuning_seed
            entry["tuning_folds"] = shared_folds(subjects, entry["train_subject_ids"], tuning_seed, config["validation"]["tuning_folds"])
        outer["meta_folds"] = meta
        outer_splits.append(outer)
    return {"scope": "global encoding; task-specific SVM; participant-common inner folds",
            "outer_assignment_source_checksum": original.get("checksum"),
            "subjects": subjects.to_dict(orient="records"), "outer_splits": outer_splits}


def indices(metadata: pd.DataFrame, task: int, ids: list[str]) -> np.ndarray:
    return np.flatnonzero((metadata.task_id.to_numpy() == task) & metadata.subject_id.isin(ids).to_numpy())


def tune_encodings(metadata: pd.DataFrame, feature_sets: dict[str, np.ndarray], train_ids: list[str], folds: list[dict], candidates: list[Candidate], phase: str) -> tuple[dict, list[dict]]:
    """Use only train_ids for both the global encoding and task parameter choices."""
    tasks = sorted(metadata.task_id.unique())
    truth = metadata.label.eq("PD").to_numpy(dtype=np.int8)
    allowed = set(train_ids)
    seen = []
    for fold in folds:
        train, valid = set(fold["train_subject_ids"]), set(fold["validation_subject_ids"])
        if train & valid or train | valid != allowed:
            raise AssertionError("Invalid training-only tuning partition.")
        seen.extend(valid)
    if sorted(seen) != sorted(train_ids):
        raise AssertionError("Each fitting participant must be validated exactly once.")
    task_folds = {task: [(indices(metadata, task, item["train_subject_ids"]), indices(metadata, task, item["validation_subject_ids"])) for item in folds] for task in tasks}
    table, results = [], {}
    for encoding, features in feature_sets.items():
        best_per_task = {}
        for task in tasks:
            scores = []
            for candidate_index, candidate in enumerate(candidates):
                fold_scores = []
                for train, valid in task_folds[task]:
                    if len(np.unique(truth[train])) != 2 or not len(valid):
                        raise ValueError("Invalid task training or validation class coverage.")
                    fitted = make_model(candidate).fit(features[train], truth[train])
                    fold_scores.append(macro_f1(truth[valid], fitted.predict(features[valid])))
                mean_score = float(np.mean(fold_scores))
                scores.append(mean_score)
                table.append({"phase": phase, "encoding": encoding, "task_id": int(task),
                              "candidate_index": candidate_index, "candidate": candidate.name,
                              "inner_macro_f1": mean_score, "inner_fold_scores": json.dumps(fold_scores)})
            winner = int(np.argmax(scores))
            best_per_task[int(task)] = {"candidate_index": winner, "candidate": candidates[winner].name, "inner_macro_f1": scores[winner]}
        results[encoding] = {"tasks": best_per_task, "score": float(np.mean([row["inner_macro_f1"] for row in best_per_task.values()]))}
    return results, table


def choose_encoding(tuning: dict, eligible: list[str]) -> str:
    return max(eligible, key=lambda name: tuning[name]["score"])


def fit_predict_tasks(metadata: pd.DataFrame, features: np.ndarray, train_ids: list[str], evaluation_ids: list[str], folds: list[dict], selected: dict, candidates: list[Candidate], *, calibrate: bool) -> tuple[pd.DataFrame, dict]:
    if set(train_ids) & set(evaluation_ids):
        raise AssertionError("Evaluation participants overlap model fitting.")
    truth = metadata.label.eq("PD").to_numpy(dtype=np.int8)
    rows, models = [], {}
    for task, choice in selected["tasks"].items():
        task = int(task)
        train = indices(metadata, task, train_ids)
        evaluation = indices(metadata, task, evaluation_ids)
        candidate = candidates[choice["candidate_index"]]
        model = make_model(candidate).fit(features[train], truth[train])
        predicted = model.predict(features[evaluation]).astype(int)
        score = model.decision_function(features[evaluation]).astype(float)
        calibrated = None
        probabilities = np.full(len(evaluation), np.nan)
        if calibrate:
            train_subjects = metadata.iloc[train].subject_id.to_numpy()
            cv = [(np.flatnonzero(np.isin(train_subjects, fold["train_subject_ids"])), np.flatnonzero(np.isin(train_subjects, fold["validation_subject_ids"]))) for fold in folds]
            if any(set(a) & set(b) or len(a) + len(b) != len(train) for a, b in cv):
                raise AssertionError("Invalid calibration partition.")
            calibrated = CalibratedClassifierCV(estimator=make_model(candidate), method="sigmoid", cv=cv, ensemble=False, n_jobs=1).fit(features[train], truth[train])
            probabilities = calibrated.predict_proba(features[evaluation])[:, 1]
        models[task] = {"base": model, "calibrated": calibrated, "selected_candidate": candidate.name}
        for index, pred, value, probability in zip(evaluation, predicted, score, probabilities, strict=True):
            row = metadata.iloc[index]
            rows.append({"subject_id": row.subject_id, "task_id": task, "label": row.label,
                         "y_true": int(truth[index]), "y_pred": int(pred), "base_y_pred": int(pred),
                         "decision_score_pd": float(value), "calibrated_probability_pd": float(probability) if calibrate else None,
                         "selected_candidate": candidate.name, "inner_macro_f1": choice["inner_macro_f1"],
                         "reliability_weight": reliability_weight(choice["inner_macro_f1"])})
    return pd.DataFrame(rows), models


def score_fusion_rules(meta_predictions: pd.DataFrame, rule_order: list[str]) -> tuple[str, list[dict], pd.DataFrame]:
    fused = fuse_fold(meta_predictions)
    rows = []
    for rule in rule_order:
        group = fused[fused["method"] == rule]
        rows.append({"fusion_rule": rule, "meta_oof_macro_f1": macro_f1(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    winner = max(rows, key=lambda row: row["meta_oof_macro_f1"])["fusion_rule"]
    return winner, rows, fused


def run_unit(metadata: pd.DataFrame, feature_sets: dict[str, np.ndarray], outer: dict, config: dict, unit_path: str, condition: str = "authentic", control_seed: int = 0) -> str:
    """One resumable outer unit; selections are finalized before outer evaluation."""
    with threadpool_limits(limits=config["execution"]["inner_threads"]):
        return _run_unit(metadata, feature_sets, outer, config, Path(unit_path), condition, control_seed)


def _run_unit(metadata: pd.DataFrame, feature_sets: dict[str, np.ndarray], outer: dict, config: dict, unit_path: Path, condition: str, control_seed: int) -> str:
    unit_path.mkdir(parents=True, exist_ok=True)
    destination = unit_path / "result.json"
    if destination.is_file():
        return str(destination)
    candidates = candidates_from_config(config["svm_search"])
    encodings = list(feature_sets)
    authentic = condition == "authentic"
    policies = {"static": ["static"], "auth9": encodings, "auth8": [name for name in encodings if name != "static"]} if authentic else {condition: encodings}
    search_rows, selection_rows, meta_rows, meta_fusion_rows = [], [], [], []
    selected_rules = {}
    if authentic:
        for meta in outer["meta_folds"]:
            phase = f"meta_{meta['inner_fold']}"
            tuned, table = tune_encodings(metadata, feature_sets, meta["train_subject_ids"], meta["tuning_folds"], candidates, phase)
            search_rows.extend(table)
            chosen = {policy: choose_encoding(tuned, eligible) for policy, eligible in policies.items()}
            fitted_frames = {}
            for encoding in dict.fromkeys(chosen.values()):
                frame, _ = fit_predict_tasks(metadata, feature_sets[encoding], meta["train_subject_ids"], meta["validation_subject_ids"], meta["tuning_folds"], tuned[encoding], candidates, calibrate=True)
                fitted_frames[encoding] = frame
            for policy, encoding in chosen.items():
                frame = fitted_frames[encoding].copy()
                frame["policy"], frame["encoding"], frame["meta_fold"] = policy, encoding, meta["inner_fold"]
                meta_rows.extend(frame.to_dict(orient="records"))
                selection_rows.append({"phase": phase, "policy": policy, "encoding": encoding, "selection_score": tuned[encoding]["score"]})
        meta_all = pd.DataFrame(meta_rows)
        meta_all["repeat"], meta_all["outer_fold"] = outer["repeat"], outer["outer_fold"]
        for policy in policies:
            selected_rules[policy], scores, fused = score_fusion_rules(meta_all[meta_all.policy == policy], config["validation"]["fusion_order"])
            for row in scores:
                row.update(policy=policy, selected=row["fusion_rule"] == selected_rules[policy])
            meta_fusion_rows.extend(scores)
        meta_all.to_csv(unit_path / "meta_oof_predictions.csv", index=False)
    tuned, table = tune_encodings(metadata, feature_sets, outer["train_subject_ids"], outer["tuning_folds"], candidates, "outer_train")
    search_rows.extend(table)
    chosen = {policy: choose_encoding(tuned, eligible) for policy, eligible in policies.items()}
    # Persist the complete choice before producing any outer-test predictions.
    for policy, encoding in chosen.items():
        selection_rows.append({"phase": "outer_train", "policy": policy, "encoding": encoding,
                               "selection_score": tuned[encoding]["score"], "fusion_rule": selected_rules.get(policy)})
    (unit_path / "selection_before_test.json").write_text(json.dumps({"selected": selection_rows, "tuned_encodings": tuned, "fusion_search": meta_fusion_rows}, indent=2) + "\n", encoding="utf-8")
    fitted_frames, fitted_models = {}, {}
    for encoding in dict.fromkeys(chosen.values()):
        frame, models = fit_predict_tasks(metadata, feature_sets[encoding], outer["train_subject_ids"], outer["test_subject_ids"], outer["tuning_folds"], tuned[encoding], candidates, calibrate=authentic)
        fitted_frames[encoding], fitted_models[encoding] = frame, models
    task_rows, fusion_rows = [], []
    for policy, encoding in chosen.items():
        frame = fitted_frames[encoding].copy()
        frame["policy"], frame["encoding"], frame["control_seed"] = policy, encoding, control_seed
        frame["repeat"], frame["outer_fold"] = outer["repeat"], outer["outer_fold"]
        task_rows.extend(frame.to_dict(orient="records"))
        if authentic:
            fused = fuse_fold(frame)
            fused = fused[fused["method"] == selected_rules[policy]].copy()
            fused["policy"], fused["encoding"] = policy, encoding
            fusion_rows.extend(fused.to_dict(orient="records"))
    pd.DataFrame(search_rows).to_csv(unit_path / "inner_search.csv.gz", index=False, compression="gzip")
    joblib.dump(fitted_models, unit_path / "models.joblib", compress=3)
    result = {"repeat": outer["repeat"], "outer_fold": outer["outer_fold"], "condition": condition,
              "control_seed": control_seed, "search_candidates_evaluated": len(search_rows),
              "tuning_svm_fits": len(search_rows) * config["validation"]["tuning_folds"],
              "task_predictions": task_rows, "fusion_predictions": fusion_rows,
              "selections": selection_rows, "fusion_search": meta_fusion_rows}
    temporary = unit_path / "result.partial.json"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return str(destination)
