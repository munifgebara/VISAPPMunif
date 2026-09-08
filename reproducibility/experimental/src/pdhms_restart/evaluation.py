"""Nested subject-level SVM evaluation and statistical summaries."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class Candidate:
    """One SVM configuration considered only in inner validation."""

    kernel: str
    c: float
    gamma: str | float = "scale"

    @property
    def name(self) -> str:
        gamma = self.gamma if self.kernel == "rbf" else "na"
        return f"{self.kernel}_C={self.c:g}_gamma={gamma}"


def candidates_from_config(config: dict[str, Any]) -> list[Candidate]:
    """Expand the declared, ordered SVM search space."""

    candidates = [Candidate("linear", float(c)) for c in config["linear_c"]]
    candidates.extend(
        Candidate("rbf", float(c), gamma)
        for c in config["rbf_c"]
        for gamma in config["rbf_gamma"]
    )
    return candidates


def make_model(candidate: Candidate):
    """Build the leakage-safe feature-scaling and SVM pipeline."""

    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel=candidate.kernel,
            C=candidate.c,
            gamma=candidate.gamma,
            class_weight="balanced",
        ),
    )


def build_splits(
    metadata: pd.DataFrame,
    *,
    repeats: int,
    outer_folds: int,
    inner_folds: int,
    master_seed: int,
) -> dict[str, Any]:
    """Create explicit outer and inner subject lists shared across tasks."""

    subject_table = (
        metadata[["subject_id", "label"]]
        .drop_duplicates()
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    if subject_table.groupby("subject_id")["label"].nunique().max() != 1:
        raise ValueError("A subject has conflicting labels.")
    seed_rng = np.random.default_rng(master_seed)
    outer_seeds = seed_rng.integers(0, 2**31 - 1, size=repeats).tolist()
    task_ids = sorted(int(value) for value in metadata["task_id"].unique())
    outer: list[dict[str, Any]] = []
    for repeat, outer_seed in enumerate(outer_seeds, start=1):
        splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=int(outer_seed))
        for fold, (train_index, test_index) in enumerate(
            splitter.split(subject_table["subject_id"], subject_table["label"]), start=1
        ):
            train_subjects = subject_table.iloc[train_index]["subject_id"].astype(str).tolist()
            test_subjects = subject_table.iloc[test_index]["subject_id"].astype(str).tolist()
            if set(train_subjects) & set(test_subjects):
                raise AssertionError("Subject leakage in outer split.")
            task_splits: dict[str, Any] = {}
            for task_id in task_ids:
                available = metadata[
                    (metadata["task_id"] == task_id)
                    & metadata["subject_id"].astype(str).isin(train_subjects)
                ][["subject_id", "label"]].sort_values("subject_id")
                inner_seed = int(seed_rng.integers(0, 2**31 - 1))
                inner_splitter = StratifiedKFold(
                    n_splits=inner_folds, shuffle=True, random_state=inner_seed
                )
                inner: list[dict[str, Any]] = []
                for inner_fold, (inner_train, inner_validation) in enumerate(
                    inner_splitter.split(available["subject_id"], available["label"]), start=1
                ):
                    inner_train_subjects = available.iloc[inner_train]["subject_id"].astype(str).tolist()
                    validation_subjects = available.iloc[inner_validation]["subject_id"].astype(str).tolist()
                    if set(inner_train_subjects) & set(validation_subjects):
                        raise AssertionError("Subject leakage in inner split.")
                    inner.append(
                        {
                            "inner_fold": inner_fold,
                            "train_subject_ids": inner_train_subjects,
                            "validation_subject_ids": validation_subjects,
                        }
                    )
                task_splits[str(task_id)] = {"seed": inner_seed, "splits": inner}
            outer.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "seed": int(outer_seed),
                    "train_subject_ids": train_subjects,
                    "test_subject_ids": test_subjects,
                    "tasks": task_splits,
                }
            )
    return {
        "splitter": "StratifiedKFold on one row per subject",
        "master_seed": master_seed,
        "outer_seeds": outer_seeds,
        "repeats": repeats,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "subjects": subject_table.to_dict(orient="records"),
        "outer_splits": outer,
    }


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute the prespecified binary classification metrics."""

    values = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity_pd": float(np.mean(y_pred[y_true == 1] == 1)),
        "specificity_h": float(np.mean(y_pred[y_true == 0] == 0)),
        "f1_pd": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    values["roc_auc"] = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
    return values


def run_nested_cv(
    metadata: pd.DataFrame,
    features: np.ndarray,
    split_payload: dict[str, Any],
    candidates: list[Candidate],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune in inner splits and predict every outer test participant."""

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
            candidate_scores: list[tuple[float, Candidate, list[float]]] = []
            for candidate in candidates:
                fold_scores: list[float] = []
                for inner in inner_splits:
                    inner_train_subjects = set(str(value).zfill(5) for value in inner["train_subject_ids"])
                    validation_subjects = set(str(value).zfill(5) for value in inner["validation_subject_ids"])
                    inner_train_mask = train_mask & metadata["subject_id"].isin(inner_train_subjects).to_numpy()
                    validation_mask = train_mask & metadata["subject_id"].isin(validation_subjects).to_numpy()
                    model = make_model(candidate)
                    y_inner = (metadata.loc[inner_train_mask, "label"].to_numpy() == "PD").astype(int)
                    y_validation = (metadata.loc[validation_mask, "label"].to_numpy() == "PD").astype(int)
                    model.fit(features[inner_train_mask], y_inner)
                    predicted = model.predict(features[validation_mask])
                    fold_scores.append(
                        float(f1_score(y_validation, predicted, labels=[0, 1], average="macro", zero_division=0))
                    )
                mean_score = float(np.mean(fold_scores))
                candidate_scores.append((mean_score, candidate, fold_scores))
                selections.append(
                    {
                        "repeat": repeat,
                        "outer_fold": fold,
                        "task_id": task_id,
                        "candidate": candidate.name,
                        "kernel": candidate.kernel,
                        "C": candidate.c,
                        "gamma": candidate.gamma,
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
            model = make_model(best_candidate)
            y_train = (train_rows["label"].to_numpy() == "PD").astype(int)
            y_test = (test_rows["label"].to_numpy() == "PD").astype(int)
            model.fit(features[train_mask], y_train)
            y_pred = model.predict(features[test_mask]).astype(int)
            scores = model.decision_function(features[test_mask]).astype(float)
            for (_, sample), actual, predicted, score in zip(
                test_rows.iterrows(), y_test, y_pred, scores, strict=True
            ):
                predictions.append(
                    {
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
            fold_metrics = metric_values(y_test, y_pred, scores)
            fold_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": fold,
                    "task_id": task_id,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "selected_candidate": best_candidate.name,
                    "best_inner_macro_f1": best_score,
                    **fold_metrics,
                }
            )
        print(
            f"evaluated repeat {repeat}/{split_payload['repeats']} "
            f"outer fold {fold}/{split_payload['outer_folds']}",
            flush=True,
        )
    return pd.DataFrame(predictions), pd.DataFrame(selections), pd.DataFrame(fold_rows)


def summarize_predictions(
    predictions: pd.DataFrame, *, bootstrap_resamples: int, bootstrap_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-repeat metrics and subject-bootstrap task summaries."""

    repeat_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for task_id, task_frame in predictions.groupby("task_id", sort=True):
        for repeat, repeat_frame in task_frame.groupby("repeat", sort=True):
            values = metric_values(
                repeat_frame["y_true"].to_numpy(),
                repeat_frame["y_pred"].to_numpy(),
                repeat_frame["decision_score_pd"].to_numpy(),
            )
            repeat_rows.append(
                {"task_id": int(task_id), "repeat": int(repeat), "n_subjects": len(repeat_frame), **values}
            )
        current = pd.DataFrame([row for row in repeat_rows if row["task_id"] == int(task_id)])
        prediction_matrix = task_frame.pivot(
            index="repeat", columns="subject_id", values="y_pred"
        ).sort_index(axis=1)
        truth = (
            task_frame[["subject_id", "y_true"]]
            .drop_duplicates()
            .set_index("subject_id")
            .loc[prediction_matrix.columns, "y_true"]
            .to_numpy(dtype=np.int8)
        )
        predicted = prediction_matrix.to_numpy(dtype=np.int8)
        h_indices = np.flatnonzero(truth == 0)
        pd_indices = np.flatnonzero(truth == 1)
        sampled_h = rng.choice(h_indices, size=(bootstrap_resamples, len(h_indices)), replace=True)
        sampled_pd = rng.choice(pd_indices, size=(bootstrap_resamples, len(pd_indices)), replace=True)
        bootstrap_values = np.zeros(bootstrap_resamples, dtype=np.float64)
        for repeat_predictions in predicted:
            true_negatives = np.sum(repeat_predictions[sampled_h] == 0, axis=1)
            false_positives = len(h_indices) - true_negatives
            true_positives = np.sum(repeat_predictions[sampled_pd] == 1, axis=1)
            false_negatives = len(pd_indices) - true_positives
            f1_h = np.divide(
                2.0 * true_negatives,
                2.0 * true_negatives + false_positives + false_negatives,
                out=np.zeros(bootstrap_resamples, dtype=np.float64),
                where=(2.0 * true_negatives + false_positives + false_negatives) != 0,
            )
            f1_pd = np.divide(
                2.0 * true_positives,
                2.0 * true_positives + false_positives + false_negatives,
                out=np.zeros(bootstrap_resamples, dtype=np.float64),
                where=(2.0 * true_positives + false_positives + false_negatives) != 0,
            )
            bootstrap_values += (f1_h + f1_pd) / (2.0 * predicted.shape[0])
        selected_per_fold = task_frame[
            ["repeat", "outer_fold", "selected_candidate"]
        ].drop_duplicates()
        selected_counts = Counter(selected_per_fold["selected_candidate"])
        summary_rows.append(
            {
                "task_id": int(task_id),
                "n_subjects": int(task_frame["subject_id"].nunique()),
                "macro_f1_mean": float(current["macro_f1"].mean()),
                "macro_f1_sd": float(current["macro_f1"].std(ddof=1)),
                "macro_f1_ci_low": float(np.percentile(bootstrap_values, 2.5)),
                "macro_f1_ci_high": float(np.percentile(bootstrap_values, 97.5)),
                "balanced_accuracy_mean": float(current["balanced_accuracy"].mean()),
                "accuracy_mean": float(current["accuracy"].mean()),
                "sensitivity_pd_mean": float(current["sensitivity_pd"].mean()),
                "specificity_h_mean": float(current["specificity_h"].mean()),
                "f1_pd_mean": float(current["f1_pd"].mean()),
                "roc_auc_mean": float(current["roc_auc"].mean()),
                "selected_svm_counts": json.dumps(selected_counts, sort_keys=True),
            }
        )
    return pd.DataFrame(repeat_rows), pd.DataFrame(summary_rows)
