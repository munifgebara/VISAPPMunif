"""Compare SVM, random forest, and regularized logistic regression."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdhms_restart.classical_models import candidates_from_config, run_nested_classical_cv
from pdhms_restart.comparison import compare_all_to_reference, compare_global_method_pairs
from pdhms_restart.data import sha256_file
from pdhms_restart.evaluation import metric_values, summarize_predictions


LABELS = {
    "svm": "SVM",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def artifact_manifest(output_root: Path) -> dict:
    target = output_root / "manifests" / "artifact_manifest.json"
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != target:
            rows.append(
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"artifact_count": len(rows), "artifacts": rows}


def task_mean_macro_f1(predictions: pd.DataFrame) -> float:
    values = []
    for (_, _), frame in predictions.groupby(["task_id", "repeat"], sort=True):
        values.append(
            metric_values(
                frame["y_true"].to_numpy(),
                frame["y_pred"].to_numpy(),
                frame["decision_score_pd"].to_numpy(),
            )["macro_f1"]
        )
    return float(np.mean(values))


def normalize_svm_search(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "algorithm", "svm")
    frame["parameters_json"] = frame.apply(
        lambda row: json.dumps(
            {"kernel": row["kernel"], "C": row["C"], "gamma": row["gamma"]},
            sort_keys=True,
        ),
        axis=1,
    )
    return frame[
        [
            "algorithm",
            "repeat",
            "outer_fold",
            "task_id",
            "candidate",
            "parameters_json",
            "inner_macro_f1",
            "inner_fold_scores",
            "selected",
        ]
    ]


def save_figures(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    ranking: pd.DataFrame,
    output_root: Path,
) -> None:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colours = {"svm": "#222222", "random_forest": "#2878B5", "logistic_regression": "#D95319"}

    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    for algorithm, frame in summary.groupby("algorithm", sort=False):
        ordered = frame.sort_values("task_id")
        axis.plot(
            ordered["task_id"],
            ordered["macro_f1_mean"],
            marker="o",
            label=LABELS[algorithm],
            color=colours[algorithm],
        )
    axis.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    axis.set(xlabel="PaHaW task", ylabel="Macro F1", ylim=(0.25, 1.0), xticks=range(1, 9))
    axis.set_title("Classical classifiers on the frozen RGB encoding")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "per_task_classifiers.png", dpi=220)
    plt.close(fig)

    matrix = comparisons.pivot(index="algorithm", columns="task_id", values="delta_macro_f1")
    matrix = matrix.loc[["random_forest", "logistic_regression"]]
    fig, axis = plt.subplots(figsize=(9.2, 2.8))
    image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
    axis.set_xticks(range(8), [str(value) for value in matrix.columns])
    axis.set_yticks(range(2), [LABELS[value] for value in matrix.index])
    axis.set(xlabel="PaHaW task", title="Delta macro F1 relative to SVM")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix.iloc[row, column]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Classifier − SVM")
    fig.tight_layout()
    fig.savefig(figure_dir / "classifier_delta_vs_svm.png", dpi=220)
    plt.close(fig)

    ordered = ranking.sort_values("task_mean_macro_f1")
    fig, axis = plt.subplots(figsize=(7.5, 3.8))
    axis.barh(
        [LABELS[value] for value in ordered["algorithm"]],
        ordered["task_mean_macro_f1"],
        color=[colours[value] for value in ordered["algorithm"]],
    )
    axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    axis.set(xlabel="Unweighted mean of task-specific macro F1", xlim=(0.45, 0.65))
    axis.set_title("Classical-classifier ranking")
    for index, value in enumerate(ordered["task_mean_macro_f1"]):
        axis.text(value + 0.002, index, f"{value:.4f}", va="center")
    fig.tight_layout()
    fig.savefig(figure_dir / "classical_classifier_ranking.png", dpi=220)
    plt.close(fig)


def write_report(
    config: dict,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    global_pairs: pd.DataFrame,
    ranking: pd.DataFrame,
    output_root: Path,
) -> None:
    compact_summary = summary[
        [
            "algorithm",
            "task_id",
            "n_subjects",
            "macro_f1_mean",
            "macro_f1_ci_low",
            "macro_f1_ci_high",
            "balanced_accuracy_mean",
            "roc_auc_mean",
        ]
    ].copy()
    compact_comparisons = comparisons.copy()
    compact_pairs = global_pairs.copy()
    compact_ranking = ranking.copy()
    for frame in (compact_summary, compact_comparisons, compact_pairs, compact_ranking):
        excluded = {
            "algorithm",
            "candidate",
            "reference",
            "task_id",
            "n_subjects",
            "rank",
            "supported_holm_0_05_global_16",
            "supported_holm_0_05_global_3",
        }
        numeric = [column for column in frame.columns if column not in excluded]
        frame[numeric] = frame[numeric].map(lambda value: f"{value:.4f}")
    winner = ranking.iloc[0]
    report = f"""# Classical classifier comparison

Experiment: `{config['experiment_id']}`. Every classifier uses the frozen `speed_altitude_azimuth` RGB image and its 768-dimensional RGB-LPQ descriptor. The subject-level outer and inner splits are identical to the preceding SVM experiments.

## Models

The existing nested-CV SVM predictions are reused without retraining. Random Forest uses {config['algorithms']['random_forest']['n_estimators']} trees and tunes `max_features` and `min_samples_leaf`. Logistic Regression standardizes LPQ features inside each training fold and tunes L1/L2 regularization and C. Both new models use balanced class weights. Hyperparameters are selected by inner-fold macro F1.

## Global ranking

The declared endpoint is the unweighted mean of task-specific macro F1 over eight tasks and five repetitions. The leading classical classifier is `{winner['algorithm']}` at {winner['task_mean_macro_f1']:.4f}.

{markdown_table(compact_ranking)}

## Per-task results

{markdown_table(compact_summary)}

## Per-task paired contrasts against SVM

Delta is candidate minus SVM. Participant bootstrap confidence intervals preserve the same draw across repetitions. Randomization swaps classifier labels per participant across repetitions. Holm correction covers all 16 classifier-by-task contrasts.

{markdown_table(compact_comparisons)}

## Global paired comparisons

The global endpoint averages task-specific macro F1. Participant clustering is preserved across all tasks and repetitions. Holm correction covers the three classifier pairs; delta is candidate minus reference.

{markdown_table(compact_pairs)}

The CNN remains outside this experiment because it requires raw-image training choices and a separate XAI protocol. It will be evaluated next on the same frozen RGB representation and outer participant splits.
"""
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "classical_classifier_comparison_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = (project_root / config["output_root"]).resolve()
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise FileExistsError(f"Refusing to resume or overwrite non-empty run: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    feature_run = (project_root / config["feature_run"]).resolve()
    encoding = config["selected_encoding"]
    feature_path = feature_run / "features" / f"{encoding}_lpq_rgb.npz"
    metadata_path = feature_run / "features" / f"{encoding}_metadata.csv"
    with np.load(feature_path) as payload:
        features = np.asarray(payload["features"], dtype=np.float32)
    metadata = pd.read_csv(metadata_path, dtype={"subject_id": str})
    if len(metadata) != len(features):
        raise ValueError("Feature and metadata row counts differ.")

    static_run = (project_root / config["static_run"]).resolve()
    split_path = static_run / "metrics" / "cv_splits.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    svm_predictions_path = feature_run / "metrics" / "outer_predictions.csv"
    svm_predictions = pd.read_csv(svm_predictions_path, dtype={"subject_id": str})
    svm_predictions = svm_predictions[svm_predictions["encoding"] == encoding].copy()
    svm_predictions = svm_predictions.drop(columns="encoding")
    svm_predictions.insert(0, "algorithm", "svm")

    predictions_by_algorithm = {"svm": svm_predictions}
    search_frames = []
    fold_frames = []
    for algorithm in ("random_forest", "logistic_regression"):
        candidates = candidates_from_config(algorithm, config["algorithms"][algorithm])
        predictions, search, fold_metrics = run_nested_classical_cv(
            metadata,
            features,
            split_payload,
            candidates,
            master_seed=int(config["algorithms"][algorithm]["master_seed"]),
        )
        predictions_by_algorithm[algorithm] = predictions
        search_frames.append(search)
        fold_frames.append(fold_metrics)

    svm_search_path = feature_run / "metrics" / "inner_search.csv"
    svm_search = pd.read_csv(svm_search_path)
    svm_search = svm_search[svm_search["encoding"] == encoding].drop(columns="encoding")
    search_frames.insert(0, normalize_svm_search(svm_search))
    svm_fold_path = feature_run / "metrics" / "outer_fold_metrics.csv"
    svm_fold = pd.read_csv(svm_fold_path)
    svm_fold = svm_fold[svm_fold["encoding"] == encoding].drop(columns="encoding")
    svm_fold.insert(0, "algorithm", "svm")
    fold_frames.insert(0, svm_fold)

    predictions = pd.concat(predictions_by_algorithm.values(), ignore_index=True)
    search = pd.concat(search_frames, ignore_index=True)
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    repeat_frames = []
    summary_frames = []
    for algorithm, frame in predictions_by_algorithm.items():
        repeat_metrics, summary = summarize_predictions(
            frame,
            bootstrap_resamples=int(config["validation"]["bootstrap_resamples"]),
            bootstrap_seed=int(config["validation"]["bootstrap_seed"]),
        )
        repeat_metrics.insert(0, "algorithm", algorithm)
        summary.insert(0, "algorithm", algorithm)
        repeat_frames.append(repeat_metrics)
        summary_frames.append(summary)
    repeat_metrics = pd.concat(repeat_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)

    predictions.to_csv(metrics_dir / "outer_predictions.csv", index=False)
    search.to_csv(metrics_dir / "inner_search.csv", index=False)
    fold_metrics.to_csv(metrics_dir / "outer_fold_metrics.csv", index=False)
    repeat_metrics.to_csv(metrics_dir / "repeat_metrics.csv", index=False)
    summary.to_csv(metrics_dir / "task_summary.csv", index=False)
    comparisons = compare_all_to_reference(
        predictions[predictions["algorithm"] != "svm"],
        svm_predictions,
        group_column="algorithm",
        resamples=int(config["validation"]["paired_randomization_resamples"]),
        seed=int(config["validation"]["paired_randomization_seed"]),
    )
    comparisons.to_csv(metrics_dir / "paired_contrasts_vs_svm.csv", index=False)
    global_pairs = compare_global_method_pairs(
        predictions_by_algorithm,
        ["svm", "random_forest", "logistic_regression"],
        resamples=int(config["validation"]["paired_randomization_resamples"]),
        seed=int(config["validation"]["paired_randomization_seed"]) + 1,
    )
    global_pairs.to_csv(metrics_dir / "global_pairwise_comparisons.csv", index=False)
    ranking = pd.DataFrame(
        [
            {"algorithm": algorithm, "task_mean_macro_f1": task_mean_macro_f1(frame)}
            for algorithm, frame in predictions_by_algorithm.items()
        ]
    ).sort_values(["task_mean_macro_f1", "algorithm"], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(metrics_dir / "classifier_ranking.csv", index=False)
    source_rows = [
        {"role": "features", "path": str(feature_path), "sha256": sha256_file(feature_path)},
        {"role": "metadata", "path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        {"role": "splits", "path": str(split_path), "sha256": sha256_file(split_path)},
        {"role": "svm_predictions", "path": str(svm_predictions_path), "sha256": sha256_file(svm_predictions_path)},
        {"role": "svm_search", "path": str(svm_search_path), "sha256": sha256_file(svm_search_path)},
        {"role": "svm_fold_metrics", "path": str(svm_fold_path), "sha256": sha256_file(svm_fold_path)},
    ]
    pd.DataFrame(source_rows).to_csv(metrics_dir / "source_artifacts.csv", index=False)

    save_figures(summary, comparisons, ranking, output_root)
    write_report(config, summary, comparisons, global_pairs, ranking, output_root)
    git_commit = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "fresh_run_no_resume": True,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_git_commit": git_commit,
        "selected_encoding": encoding,
        "reused_cv_split_checksum": split_payload["checksum"],
        "feature_shape": list(features.shape),
        "algorithms": list(predictions_by_algorithm),
        "outer_predictions": len(predictions),
        "selected_models": int(search["selected"].astype(str).str.lower().eq("true").sum()),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            package: version(package)
            for package in (
                "joblib",
                "matplotlib",
                "numpy",
                "pandas",
                "pillow",
                "scikit-learn",
                "scipy",
            )
        },
    }
    save_json(manifest, output_root / "manifests" / "run_manifest.json")
    save_json(artifact_manifest(output_root), output_root / "manifests" / "artifact_manifest.json")
    print(ranking.to_string(index=False))
    print(global_pairs.to_string(index=False))
    print(f"complete in {manifest['duration_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
