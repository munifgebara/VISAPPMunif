"""Nested evaluation for classical classifiers beyond the frozen SVM."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import metric_values


@dataclass(frozen=True)
class ClassicalCandidate:
    """One declared classifier configuration."""

    algorithm: str
    name: str
    parameters: dict[str, Any]


def candidates_from_config(algorithm: str, config: dict[str, Any]) -> list[ClassicalCandidate]:
    """Expand an ordered classical-model search space."""

    if algorithm == "logistic_regression":
        penalties = list(config["penalties"])
        if not penalties or any(penalty not in {"l1", "l2"} for penalty in penalties):
            raise ValueError("Logistic regularization must contain only l1 and/or l2.")
        return [
            ClassicalCandidate(
                algorithm=algorithm,
                name=f"logreg_penalty={penalty}_C={float(c):g}",
                parameters={
                    "regularization": penalty,
                    "l1_ratio": 1.0 if penalty == "l1" else 0.0,
                    "C": float(c),
                },
            )
            for penalty in penalties
            for c in config["c_values"]
        ]
    if algorithm == "random_forest":
        trees = int(config["n_estimators"])
        return [
            ClassicalCandidate(
                algorithm=algorithm,
                name=f"rf_trees={trees}_max_features={max_features}_min_leaf={int(minimum_leaf)}",
                parameters={
                    "n_estimators": trees,
                    "max_features": max_features,
                    "min_samples_leaf": int(minimum_leaf),
                },
            )
            for max_features in config["max_features"]
            for minimum_leaf in config["min_samples_leaf"]
        ]
    raise ValueError(f"Unsupported algorithm {algorithm!r}.")


def make_model(candidate: ClassicalCandidate, *, random_state: int):
    """Build a deterministic model for one candidate."""

    if candidate.algorithm == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                l1_ratio=candidate.parameters["l1_ratio"],
                C=candidate.parameters["C"],
                solver="liblinear",
                class_weight="balanced",
                max_iter=10_000,
                random_state=random_state,
            ),
        )
    if candidate.algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=candidate.parameters["n_estimators"],
            max_features=candidate.parameters["max_features"],
            min_samples_leaf=candidate.parameters["min_samples_leaf"],
            criterion="gini",
            class_weight="balanced_subsample",
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported algorithm {candidate.algorithm!r}.")


def _seed(master_seed: int, repeat: int, outer_fold: int, task_id: int, stage: int) -> int:
    return int(
        np.random.SeedSequence([master_seed, repeat, outer_fold, task_id, stage])
        .generate_state(1, dtype=np.uint32)[0]
    )


def _scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    class_index = int(np.flatnonzero(np.asarray(model.classes_) == 1)[0])
    return probabilities[:, class_index]


def run_nested_classical_cv(
    metadata: pd.DataFrame,
    features: np.ndarray,
    split_payload: dict[str, Any],
    candidates: list[ClassicalCandidate],
    *,
    master_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune one algorithm in inner folds and predict every outer test participant."""

    if not candidates:
        raise ValueError("At least one candidate is required.")
    algorithms = {candidate.algorithm for candidate in candidates}
    if len(algorithms) != 1:
        raise ValueError("Candidates in one nested run must use the same algorithm.")
    algorithm = next(iter(algorithms))
    metadata = metadata.reset_index(drop=True).copy()
    metadata["subject_id"] = metadata["subject_id"].astype(str).str.zfill(5)
    predictions: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for outer in split_payload["outer_splits"]:
        repeat = int(outer["repeat"])
        fold = int(outer["outer_fold"])
        train_subjects = set(str(value).zfill(5) for value in outer["train_subject_ids"])
        test_subjects = set(str(value).zfill(5) for value in outer["test_subject_ids"])
        for task_id in sorted(int(value) for value in metadata["task_id"].unique()):
            task_mask = metadata["task_id"].to_numpy() == task_id
            train_mask = task_mask & metadata["subject_id"].isin(train_subjects).to_numpy()
            test_mask = task_mask & metadata["subject_id"].isin(test_subjects).to_numpy()
            train_rows = metadata[train_mask]
            test_rows = metadata[test_mask]
            inner_splits = outer["tasks"][str(task_id)]["splits"]
            candidate_scores: list[tuple[float, ClassicalCandidate, list[float]]] = []
            for candidate in candidates:
                fold_scores: list[float] = []
                for inner in inner_splits:
                    inner_fold = int(inner["inner_fold"])
                    inner_train_subjects = set(
                        str(value).zfill(5) for value in inner["train_subject_ids"]
                    )
                    validation_subjects = set(
                        str(value).zfill(5) for value in inner["validation_subject_ids"]
                    )
                    inner_train_mask = train_mask & metadata["subject_id"].isin(inner_train_subjects).to_numpy()
                    validation_mask = train_mask & metadata["subject_id"].isin(validation_subjects).to_numpy()
                    model = make_model(
                        candidate,
                        random_state=_seed(master_seed, repeat, fold, task_id, inner_fold),
                    )
                    y_inner = (metadata.loc[inner_train_mask, "label"].to_numpy() == "PD").astype(int)
                    y_validation = (metadata.loc[validation_mask, "label"].to_numpy() == "PD").astype(int)
                    model.fit(features[inner_train_mask], y_inner)
                    predicted = model.predict(features[validation_mask])
                    fold_scores.append(
                        float(
                            f1_score(
                                y_validation,
                                predicted,
                                labels=[0, 1],
                                average="macro",
                                zero_division=0,
                            )
                        )
                    )
                mean_score = float(np.mean(fold_scores))
                candidate_scores.append((mean_score, candidate, fold_scores))
                selections.append(
                    {
                        "algorithm": algorithm,
                        "repeat": repeat,
                        "outer_fold": fold,
                        "task_id": task_id,
                        "candidate": candidate.name,
                        "parameters_json": json.dumps(candidate.parameters, sort_keys=True),
                        "inner_macro_f1": mean_score,
                        "inner_fold_scores": json.dumps(fold_scores),
                        "selected": False,
                    }
                )
            best_score, best_candidate, _ = max(candidate_scores, key=lambda item: item[0])
            for row in reversed(selections):
                if (
                    row["repeat"] == repeat
                    and row["outer_fold"] == fold
                    and row["task_id"] == task_id
                    and row["candidate"] == best_candidate.name
                ):
                    row["selected"] = True
                    break
            model = make_model(
                best_candidate,
                random_state=_seed(master_seed, repeat, fold, task_id, 999),
            )
            y_train = (train_rows["label"].to_numpy() == "PD").astype(int)
            y_test = (test_rows["label"].to_numpy() == "PD").astype(int)
            model.fit(features[train_mask], y_train)
            y_pred = np.asarray(model.predict(features[test_mask]), dtype=int)
            scores = _scores(model, features[test_mask])
            for (_, sample), actual, predicted, score in zip(
                test_rows.iterrows(), y_test, y_pred, scores, strict=True
            ):
                predictions.append(
                    {
                        "algorithm": algorithm,
                        "repeat": repeat,
                        "outer_fold": fold,
                        "task_id": task_id,
                        "subject_id": sample["subject_id"],
                        "y_true": int(actual),
                        "label": sample["label"],
                        "y_pred": int(predicted),
                        "decision_score_pd": float(score),
                        "selected_candidate": best_candidate.name,
                    }
                )
            values = metric_values(y_test, y_pred, scores)
            fold_rows.append(
                {
                    "algorithm": algorithm,
                    "repeat": repeat,
                    "outer_fold": fold,
                    "task_id": task_id,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "selected_candidate": best_candidate.name,
                    "best_inner_macro_f1": best_score,
                    **values,
                }
            )
        print(
            f"{algorithm}: evaluated repeat {repeat}/{split_payload['repeats']} "
            f"outer fold {fold}/{split_payload['outer_folds']}",
            flush=True,
        )
    return pd.DataFrame(predictions), pd.DataFrame(selections), pd.DataFrame(fold_rows)
