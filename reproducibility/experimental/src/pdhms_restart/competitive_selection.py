"""Training-only model selection for fixed literature-defined task sets."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .evaluation import candidates_from_config, make_model
from .nested_selection import indices, macro_f1


def candidate_names(learner: str, config: dict) -> list[str]:
    if learner == "svm":
        return [item.name for item in candidates_from_config(config["svm_search"])]
    if learner == "logistic":
        return [f"L2_C={value:g}" for value in config["logistic_search"]["c"]]
    raise ValueError(f"Unknown learner {learner}.")


def make_candidate(learner: str, index: int, config: dict):
    if learner == "svm":
        return make_model(candidates_from_config(config["svm_search"])[index])
    settings = config["logistic_search"]
    if learner != "logistic":
        raise ValueError(learner)
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=settings["c"][index], solver=settings["solver"],
        class_weight=settings["class_weight"], max_iter=settings["max_iter"],
        random_state=settings["random_state"], l1_ratio=0.0))


def tune_family(metadata: pd.DataFrame, features: dict, learner: str,
                train_ids: list[str], folds: list[dict], config: dict) -> tuple[dict, list[dict]]:
    allowed = set(train_ids)
    validated = []
    for fold in folds:
        train, valid = set(fold["train_subject_ids"]), set(fold["validation_subject_ids"])
        if train & valid or train | valid != allowed:
            raise AssertionError("Invalid training-only partition.")
        validated.extend(valid)
    if sorted(validated) != sorted(train_ids):
        raise AssertionError("Each training participant must be validated exactly once.")
    tasks = sorted(int(value) for value in metadata.task_id.unique())
    truth = metadata.label.eq("PD").to_numpy(dtype=np.int8)
    names = candidate_names(learner, config)
    tuned, rows = {}, []
    task_folds = {task: [(indices(metadata, task, fold["train_subject_ids"]),
                          indices(metadata, task, fold["validation_subject_ids"]))
                         for fold in folds] for task in tasks}
    for encoding, matrix in features.items():
        tuned[encoding] = {}
        for task in tasks:
            scores = []
            for candidate_index, name in enumerate(names):
                fold_scores = []
                for train, valid in task_folds[task]:
                    if len(np.unique(truth[train])) != 2 or not len(valid):
                        raise ValueError("Insufficient task/class coverage.")
                    model = make_candidate(learner, candidate_index, config)
                    model.fit(matrix[train], truth[train])
                    fold_scores.append(macro_f1(truth[valid], model.predict(matrix[valid])))
                score = float(np.mean(fold_scores))
                scores.append(score)
                rows.append({"encoding": encoding, "task_id": task,
                             "candidate_index": candidate_index, "candidate": name,
                             "inner_macro_f1": score, "inner_fold_scores": json.dumps(fold_scores)})
            winner = int(np.argmax(scores))
            tuned[encoding][task] = {"candidate_index": winner, "candidate": names[winner],
                                      "inner_macro_f1": scores[winner]}
    return tuned, rows


def select_for_tasks(tuned: dict, tasks: list[int], eligible: list[str]) -> dict:
    scores = {encoding: float(np.mean([tuned[encoding][int(task)]["inner_macro_f1"]
                                       for task in tasks])) for encoding in eligible}
    encoding = max(eligible, key=lambda value: scores[value])
    return {"encoding": encoding, "encoding_scores": scores,
            "selection_score": scores[encoding],
            "tasks": {int(task): tuned[encoding][int(task)] for task in tasks}}


def majority_fusion(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["task_set", "method", "repeat", "outer_fold", "subject_id"]
    for values, current in predictions.groupby(keys, sort=True):
        if current.task_id.duplicated().any() or current.y_true.nunique() != 1:
            raise ValueError("Duplicate task or conflicting participant truth.")
        fraction = float(current.y_pred.mean())
        rows.append({**dict(zip(keys, values, strict=True)),
                     "task_id": 0, "y_true": int(current.y_true.iloc[0]),
                     "y_pred": int(fraction >= 0.5), "score_pd": fraction,
                     "available_tasks": len(current),
                     "task_ids": ",".join(map(str, sorted(current.task_id))),
                     "encoding": current.encoding.iloc[0]})
    return pd.DataFrame(rows)


def run_competitive_unit(metadata: pd.DataFrame, families: dict, outer: dict,
                         config: dict, destination: str) -> str:
    with threadpool_limits(limits=config["execution"]["inner_threads"]):
        return _run_unit(metadata, families, outer, config, Path(destination))


def _run_unit(metadata, families, outer, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        return str(result_path)
    train_ids, test_ids = outer["train_subject_ids"], outer["test_subject_ids"]
    if set(train_ids) & set(test_ids):
        raise AssertionError("Outer participant leakage.")
    cache, search_rows, selection_rows = {}, [], []
    for method in config["method_order"]:
        definition = config["methods"][method]
        key = (definition["feature_family"], definition["learner"])
        if key not in cache:
            tuned, table = tune_family(metadata, families[key[0]], key[1], train_ids,
                                       outer["tuning_folds"], config)
            cache[key] = tuned
            search_rows.extend({**row, "feature_family": key[0], "learner": key[1]} for row in table)
        eligible = definition["eligible_encodings"]
        if eligible == "all9":
            eligible = config["encoding_order"]
        for task_set, tasks in config["task_sets"].items():
            selected = select_for_tasks(cache[key], tasks, eligible)
            selection_rows.append({"task_set": task_set, "method": method, **selected})
    # This file is committed to disk before any outer-test inference.
    selection_path = directory / "selection_before_test.json"
    selection_path.write_text(json.dumps({"train_subject_ids": train_ids,
                                          "test_subject_ids": test_ids,
                                          "selections": selection_rows}, indent=2) + "\n", encoding="utf-8")
    predictions, models = [], {}
    truth = metadata.label.eq("PD").to_numpy(dtype=np.int8)
    for selected in selection_rows:
        definition = config["methods"][selected["method"]]
        family, learner = definition["feature_family"], definition["learner"]
        encoding = selected["encoding"]
        matrix = families[family][encoding]
        for task, choice in selected["tasks"].items():
            model_key = (family, learner, encoding, task, choice["candidate_index"])
            train = indices(metadata, task, train_ids)
            test = indices(metadata, task, test_ids)
            if model_key not in models:
                model = make_candidate(learner, choice["candidate_index"], config)
                model.fit(matrix[train], truth[train])
                models[model_key] = model
            model = models[model_key]
            scores = model.decision_function(matrix[test])
            labels = model.predict(matrix[test])
            for index, predicted, score in zip(test, labels, scores, strict=True):
                row = metadata.iloc[index]
                predictions.append({"task_set": selected["task_set"], "method": selected["method"],
                                    "repeat": outer["repeat"], "outer_fold": outer["outer_fold"],
                                    "subject_id": row.subject_id, "task_id": int(task),
                                    "y_true": int(truth[index]), "y_pred": int(predicted),
                                    "score_pd": float(score), "encoding": encoding,
                                    "selected_candidate": choice["candidate"],
                                    "inner_macro_f1": choice["inner_macro_f1"]})
    frame = pd.DataFrame(predictions)
    fused = majority_fusion(frame)
    pd.DataFrame(search_rows).to_csv(directory / "inner_search.csv.gz", index=False, compression="gzip")
    joblib.dump(models, directory / "models.joblib", compress=3)
    result = {"repeat": outer["repeat"], "outer_fold": outer["outer_fold"],
              "tuning_fits": len(search_rows) * len(outer["tuning_folds"]),
              "final_fits": len(models), "selections": selection_rows,
              "task_predictions": predictions, "fusion_predictions": fused.to_dict(orient="records")}
    partial = directory / "result.partial.json"
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(result_path)
    return str(result_path)
