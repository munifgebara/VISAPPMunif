"""Paired participant-level comparisons between image encodings."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import metric_values


def _macro_f1_for_samples(predictions: np.ndarray, truth: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Vectorized macro F1 for resampled participant indices."""

    sampled_truth = truth[indices]
    if predictions.ndim == 1:
        sampled_predictions = predictions[indices]
    elif predictions.ndim == 2 and predictions.shape == indices.shape:
        sampled_predictions = np.take_along_axis(predictions, indices, axis=1)
    else:
        raise ValueError("Predictions must be one-dimensional or match the index matrix.")
    true_h = np.sum((sampled_truth == 0) & (sampled_predictions == 0), axis=1)
    false_pd = np.sum((sampled_truth == 0) & (sampled_predictions == 1), axis=1)
    true_pd = np.sum((sampled_truth == 1) & (sampled_predictions == 1), axis=1)
    false_h = np.sum((sampled_truth == 1) & (sampled_predictions == 0), axis=1)
    f1_h = np.divide(
        2.0 * true_h,
        2.0 * true_h + false_pd + false_h,
        out=np.zeros(len(indices), dtype=np.float64),
        where=(2.0 * true_h + false_pd + false_h) != 0,
    )
    f1_pd = np.divide(
        2.0 * true_pd,
        2.0 * true_pd + false_pd + false_h,
        out=np.zeros(len(indices), dtype=np.float64),
        where=(2.0 * true_pd + false_pd + false_h) != 0,
    )
    return (f1_h + f1_pd) / 2.0


def paired_macro_f1_comparison(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Compare repeat-mean macro F1 with paired bootstrap and randomization."""

    keys = ["repeat", "outer_fold", "task_id", "subject_id"]
    merged = candidate.merge(reference, on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one")
    if len(merged) != len(candidate) or len(merged) != len(reference):
        raise ValueError("Candidate and reference predictions are not fully paired.")
    if not np.array_equal(merged["y_true_candidate"], merged["y_true_reference"]):
        raise ValueError("Candidate and reference labels differ.")
    pivot_candidate = merged.pivot(index="repeat", columns="subject_id", values="y_pred_candidate").sort_index(axis=1)
    pivot_reference = merged.pivot(index="repeat", columns="subject_id", values="y_pred_reference").sort_index(axis=1)
    if not pivot_candidate.columns.equals(pivot_reference.columns):
        raise ValueError("Candidate and reference subject columns differ.")
    truth = (
        merged[["subject_id", "y_true_candidate"]]
        .drop_duplicates()
        .set_index("subject_id")
        .loc[pivot_candidate.columns, "y_true_candidate"]
        .to_numpy(dtype=np.int8)
    )
    candidate_matrix = pivot_candidate.to_numpy(dtype=np.int8)
    reference_matrix = pivot_reference.to_numpy(dtype=np.int8)
    observed_candidate = []
    observed_reference = []
    for candidate_repeat, reference_repeat in zip(candidate_matrix, reference_matrix, strict=True):
        observed_candidate.append(metric_values(truth, candidate_repeat, candidate_repeat)["macro_f1"])
        observed_reference.append(metric_values(truth, reference_repeat, reference_repeat)["macro_f1"])
    observed = float(np.mean(observed_candidate) - np.mean(observed_reference))

    rng = np.random.default_rng(seed)
    h_indices = np.flatnonzero(truth == 0)
    pd_indices = np.flatnonzero(truth == 1)
    bootstrap_indices = np.concatenate(
        (
            rng.choice(h_indices, size=(resamples, len(h_indices)), replace=True),
            rng.choice(pd_indices, size=(resamples, len(pd_indices)), replace=True),
        ),
        axis=1,
    )
    bootstrap_delta = np.zeros(resamples, dtype=np.float64)
    for candidate_repeat, reference_repeat in zip(candidate_matrix, reference_matrix, strict=True):
        bootstrap_delta += (
            _macro_f1_for_samples(candidate_repeat, truth, bootstrap_indices)
            - _macro_f1_for_samples(reference_repeat, truth, bootstrap_indices)
        ) / candidate_matrix.shape[0]

    swap = rng.integers(0, 2, size=(resamples, len(truth)), dtype=np.int8).astype(bool)
    all_indices = np.broadcast_to(np.arange(len(truth)), (resamples, len(truth)))
    random_delta = np.zeros(resamples, dtype=np.float64)
    for candidate_repeat, reference_repeat in zip(candidate_matrix, reference_matrix, strict=True):
        permuted_candidate = np.where(swap, reference_repeat, candidate_repeat)
        permuted_reference = np.where(swap, candidate_repeat, reference_repeat)
        random_delta += (
            _macro_f1_for_samples(permuted_candidate, truth, all_indices)
            - _macro_f1_for_samples(permuted_reference, truth, all_indices)
        ) / candidate_matrix.shape[0]
    p_value = (1.0 + float(np.count_nonzero(np.abs(random_delta) >= abs(observed)))) / (resamples + 1.0)
    return {
        "delta_macro_f1": observed,
        "ci_low": float(np.percentile(bootstrap_delta, 2.5)),
        "ci_high": float(np.percentile(bootstrap_delta, 97.5)),
        "permutation_p": p_value,
    }


def holm_adjust(frame: pd.DataFrame, p_column: str = "permutation_p") -> pd.DataFrame:
    """Add Holm-adjusted p-values over all rows in the frame."""

    result = frame.copy()
    order = np.argsort(result[p_column].to_numpy(dtype=float), kind="stable")
    raw = result[p_column].to_numpy(dtype=float)[order]
    adjusted_sorted = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * raw)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    result["holm_p_global_32"] = adjusted
    result["supported_holm_0_05"] = adjusted < 0.05
    return result


def compare_all_to_static(
    dynamic_predictions: pd.DataFrame,
    static_predictions: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Run 32 prespecified signal-by-task comparisons to the static baseline."""

    rows: list[dict[str, Any]] = []
    seed_rng = np.random.default_rng(seed)
    for encoding, encoding_frame in dynamic_predictions.groupby("encoding", sort=True):
        for task_id, candidate in encoding_frame.groupby("task_id", sort=True):
            reference = static_predictions[static_predictions["task_id"] == task_id]
            result = paired_macro_f1_comparison(
                candidate,
                reference,
                resamples=resamples,
                seed=int(seed_rng.integers(0, 2**31 - 1)),
            )
            rows.append({"encoding": encoding, "task_id": int(task_id), **result})
    return holm_adjust(pd.DataFrame(rows))


def compare_all_to_reference(
    candidate_predictions: pd.DataFrame,
    reference_predictions: pd.DataFrame,
    *,
    group_column: str,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Compare every candidate group and task with one fixed reference."""

    rows: list[dict[str, Any]] = []
    seed_rng = np.random.default_rng(seed)
    for group_value, group_frame in candidate_predictions.groupby(group_column, sort=True):
        for task_id, candidate in group_frame.groupby("task_id", sort=True):
            reference = reference_predictions[reference_predictions["task_id"] == task_id]
            result = paired_macro_f1_comparison(
                candidate,
                reference,
                resamples=resamples,
                seed=int(seed_rng.integers(0, 2**31 - 1)),
            )
            rows.append({group_column: group_value, "task_id": int(task_id), **result})
    frame = holm_adjust(pd.DataFrame(rows))
    family_size = len(frame)
    return frame.rename(
        columns={
            "holm_p_global_32": f"holm_p_global_{family_size}",
            "supported_holm_0_05": f"supported_holm_0_05_global_{family_size}",
        }
    )


def _weighted_macro_f1(
    predictions: np.ndarray,
    truth: np.ndarray,
    available: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Macro F1 under participant bootstrap weights."""

    true_h = weights @ (available & (truth == 0) & (predictions == 0))
    false_pd = weights @ (available & (truth == 0) & (predictions == 1))
    true_pd = weights @ (available & (truth == 1) & (predictions == 1))
    false_h = weights @ (available & (truth == 1) & (predictions == 0))
    f1_h = np.divide(
        2.0 * true_h,
        2.0 * true_h + false_pd + false_h,
        out=np.zeros(weights.shape[0], dtype=np.float64),
        where=(2.0 * true_h + false_pd + false_h) != 0,
    )
    f1_pd = np.divide(
        2.0 * true_pd,
        2.0 * true_pd + false_pd + false_h,
        out=np.zeros(weights.shape[0], dtype=np.float64),
        where=(2.0 * true_pd + false_pd + false_h) != 0,
    )
    return (f1_h + f1_pd) / 2.0


def _permuted_macro_f1(
    candidate: np.ndarray,
    reference: np.ndarray,
    truth: np.ndarray,
    available: np.ndarray,
    swap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Macro F1 after participant-level method-label swaps."""

    candidate_permuted = np.where(swap, reference, candidate)
    reference_permuted = np.where(swap, candidate, reference)

    def score(predictions: np.ndarray) -> np.ndarray:
        true_h = np.sum(available & (truth == 0) & (predictions == 0), axis=1)
        false_pd = np.sum(available & (truth == 0) & (predictions == 1), axis=1)
        true_pd = np.sum(available & (truth == 1) & (predictions == 1), axis=1)
        false_h = np.sum(available & (truth == 1) & (predictions == 0), axis=1)
        f1_h = np.divide(
            2.0 * true_h,
            2.0 * true_h + false_pd + false_h,
            out=np.zeros(predictions.shape[0], dtype=np.float64),
            where=(2.0 * true_h + false_pd + false_h) != 0,
        )
        f1_pd = np.divide(
            2.0 * true_pd,
            2.0 * true_pd + false_pd + false_h,
            out=np.zeros(predictions.shape[0], dtype=np.float64),
            where=(2.0 * true_pd + false_pd + false_h) != 0,
        )
        return (f1_h + f1_pd) / 2.0

    return score(candidate_permuted), score(reference_permuted)


def paired_task_mean_macro_f1_comparison(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Compare the unweighted task-mean macro F1 with participant clustering."""

    keys = ["repeat", "outer_fold", "task_id", "subject_id"]
    merged = candidate.merge(reference, on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one")
    if len(merged) != len(candidate) or len(merged) != len(reference):
        raise ValueError("Candidate and reference predictions are not fully paired.")
    if not np.array_equal(merged["y_true_candidate"], merged["y_true_reference"]):
        raise ValueError("Candidate and reference labels differ.")

    subject_truth = (
        merged[["subject_id", "y_true_candidate"]]
        .drop_duplicates()
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    if subject_truth.groupby("subject_id")["y_true_candidate"].nunique().max() != 1:
        raise ValueError("A participant has conflicting labels.")
    subjects = subject_truth["subject_id"].tolist()
    truth = subject_truth["y_true_candidate"].to_numpy(dtype=np.int8)
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    task_repeats = sorted(
        (int(task), int(repeat))
        for task, repeat in merged[["task_id", "repeat"]].drop_duplicates().itertuples(index=False, name=None)
    )

    aligned: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    observed_candidate: list[float] = []
    observed_reference: list[float] = []
    for task_id, repeat in task_repeats:
        frame = merged[(merged["task_id"] == task_id) & (merged["repeat"] == repeat)]
        candidate_values = np.full(len(subjects), -1, dtype=np.int8)
        reference_values = np.full(len(subjects), -1, dtype=np.int8)
        for row in frame.itertuples(index=False):
            index = subject_index[row.subject_id]
            candidate_values[index] = int(row.y_pred_candidate)
            reference_values[index] = int(row.y_pred_reference)
        available = candidate_values >= 0
        if not np.array_equal(available, reference_values >= 0):
            raise ValueError("Candidate and reference availability differs.")
        aligned.append((candidate_values, reference_values, available))
        observed_candidate.append(
            metric_values(truth[available], candidate_values[available], candidate_values[available])["macro_f1"]
        )
        observed_reference.append(
            metric_values(truth[available], reference_values[available], reference_values[available])["macro_f1"]
        )
    observed = float(np.mean(observed_candidate) - np.mean(observed_reference))

    rng = np.random.default_rng(seed)
    h_indices = np.flatnonzero(truth == 0)
    pd_indices = np.flatnonzero(truth == 1)
    sampled = np.concatenate(
        (
            rng.choice(h_indices, size=(resamples, len(h_indices)), replace=True),
            rng.choice(pd_indices, size=(resamples, len(pd_indices)), replace=True),
        ),
        axis=1,
    )
    weights = np.zeros((resamples, len(subjects)), dtype=np.int16)
    np.add.at(
        weights,
        (np.repeat(np.arange(resamples), sampled.shape[1]), sampled.ravel()),
        1,
    )
    bootstrap_delta = np.zeros(resamples, dtype=np.float64)
    for candidate_values, reference_values, available in aligned:
        bootstrap_delta += (
            _weighted_macro_f1(candidate_values, truth, available, weights)
            - _weighted_macro_f1(reference_values, truth, available, weights)
        ) / len(aligned)

    swap = rng.integers(0, 2, size=(resamples, len(subjects)), dtype=np.int8).astype(bool)
    random_delta = np.zeros(resamples, dtype=np.float64)
    for candidate_values, reference_values, available in aligned:
        candidate_score, reference_score = _permuted_macro_f1(
            candidate_values[None, :],
            reference_values[None, :],
            truth[None, :],
            available[None, :],
            swap,
        )
        random_delta += (candidate_score - reference_score) / len(aligned)
    p_value = (1.0 + float(np.count_nonzero(np.abs(random_delta) >= abs(observed)))) / (resamples + 1.0)
    return {
        "delta_task_mean_macro_f1": observed,
        "ci_low": float(np.percentile(bootstrap_delta, 2.5)),
        "ci_high": float(np.percentile(bootstrap_delta, 97.5)),
        "permutation_p": p_value,
    }


def compare_global_method_pairs(
    prediction_frames: dict[str, pd.DataFrame],
    method_order: list[str],
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Compare all method pairs on mean macro F1 across tasks."""

    rows: list[dict[str, Any]] = []
    seed_rng = np.random.default_rng(seed)
    for candidate_name, reference_name in combinations(method_order, 2):
        result = paired_task_mean_macro_f1_comparison(
            prediction_frames[candidate_name],
            prediction_frames[reference_name],
            resamples=resamples,
            seed=int(seed_rng.integers(0, 2**31 - 1)),
        )
        rows.append(
            {
                "candidate": candidate_name,
                "reference": reference_name,
                **result,
            }
        )
    frame = holm_adjust(pd.DataFrame(rows))
    family_size = len(frame)
    return frame.rename(
        columns={
            "holm_p_global_32": f"holm_p_global_{family_size}",
            "supported_holm_0_05": f"supported_holm_0_05_global_{family_size}",
        }
    )
