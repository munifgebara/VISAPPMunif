"""Leakage-safe late fusion of task-specific participant predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


FUSION_METHODS = (
    "majority_vote_all8",
    "calibrated_mean_all8",
    "calibrated_reliability_weighted_all8",
)


def reliability_weight(inner_macro_f1: float) -> float:
    """Convert an inner-validation score to the declared non-zero fusion weight."""

    if not np.isfinite(inner_macro_f1):
        raise ValueError("Inner macro F1 must be finite.")
    return max(float(inner_macro_f1) - 0.5, 0.01)


def fuse_fold(base_predictions: pd.DataFrame) -> pd.DataFrame:
    """Fuse all available tasks for each participant in one external fold.

    Expected probabilities and reliabilities must have been obtained without using
    the external test participants. A missing task is ignored for that participant.
    """

    required = {
        "repeat",
        "outer_fold",
        "task_id",
        "subject_id",
        "y_true",
        "base_y_pred",
        "calibrated_probability_pd",
        "reliability_weight",
    }
    missing = required - set(base_predictions)
    if missing:
        raise ValueError(f"Missing fusion columns: {sorted(missing)}.")
    fold_keys = base_predictions[["repeat", "outer_fold"]].drop_duplicates()
    if len(fold_keys) != 1:
        raise ValueError("fuse_fold expects exactly one repeat and outer fold.")
    if base_predictions.groupby("subject_id")["y_true"].nunique().max() != 1:
        raise ValueError("A participant has conflicting labels across tasks.")
    probabilities = base_predictions["calibrated_probability_pd"].to_numpy(dtype=float)
    weights = base_predictions["reliability_weight"].to_numpy(dtype=float)
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Calibrated probabilities must be finite and within [0, 1].")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Reliability weights must be finite and positive.")

    repeat, outer_fold = fold_keys.iloc[0]
    rows: list[dict[str, object]] = []
    for subject_id, subject in base_predictions.groupby("subject_id", sort=True):
        truth = int(subject["y_true"].iloc[0])
        mean_probability = float(subject["calibrated_probability_pd"].mean())
        values = {
            "majority_vote_all8": float(subject["base_y_pred"].mean()),
            "calibrated_mean_all8": mean_probability,
            "calibrated_reliability_weighted_all8": float(
                np.average(
                    subject["calibrated_probability_pd"].to_numpy(dtype=float),
                    weights=subject["reliability_weight"].to_numpy(dtype=float),
                )
            ),
        }
        for method, score in values.items():
            rows.append(
                {
                    "method": method,
                    "repeat": int(repeat),
                    "outer_fold": int(outer_fold),
                    "task_id": 0,
                    "subject_id": str(subject_id).zfill(5),
                    "y_true": truth,
                    "y_pred": int(score >= 0.5),
                    "decision_score_pd": score,
                    "available_task_count": int(subject["task_id"].nunique()),
                    "available_tasks": " ".join(
                        str(int(value)) for value in sorted(subject["task_id"].unique())
                    ),
                }
            )
    return pd.DataFrame(rows)
