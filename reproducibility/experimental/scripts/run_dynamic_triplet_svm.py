"""Run all leave-one-signal-out RGB triplets with the frozen SVM protocol."""

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
from PIL import Image

from pdhms_restart.comparison import compare_all_to_static, compare_global_method_pairs
from pdhms_restart.data import audit_samples, index_samples, sha256_file
from pdhms_restart.dynamic_image import DynamicRenderConfig, render_triplet_sample
from pdhms_restart.evaluation import candidates_from_config, run_nested_cv, summarize_predictions
from pdhms_restart.lpq import extract_path, feature_names


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


def render_encoding(
    samples,
    encoding: str,
    signals: tuple[str, str, str],
    config: dict,
    output_root: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    render_values = dict(config["render"])
    for key in (
        "normalization",
        "speed_transform",
        "azimuth_transform",
        "include_air",
        "coordinate_frame",
        "channel_order_rationale",
    ):
        render_values.pop(key, None)
    render_values["constant_channels"] = tuple(render_values["constant_channels"])
    render_values["background_rgb"] = tuple(render_values["background_rgb"])
    render_config = DynamicRenderConfig(**render_values)
    rows = []
    for index, sample in enumerate(samples, start=1):
        output_path = (
            output_root
            / "images"
            / encoding
            / f"task_{sample.task_id:02d}"
            / f"subject_{sample.subject_id}_task_{sample.task_id:02d}_{encoding}.png"
        )
        row = render_triplet_sample(sample, encoding, signals, output_path, render_config)
        row["source_path"] = str(sample.path)
        row["image_sha256"] = sha256_file(output_path)
        rows.append(row)
        if index % 100 == 0 or index == len(samples):
            print(f"{encoding}: rendered {index}/{len(samples)}", flush=True)
    metadata = pd.DataFrame(rows).sort_values(["task_id", "subject_id"]).reset_index(drop=True)
    features = np.vstack([extract_path(Path(path)) for path in metadata["image_path"]]).astype(np.float32)
    feature_dir = output_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(feature_dir / f"{encoding}_metadata.csv", index=False)
    np.savez_compressed(
        feature_dir / f"{encoding}_lpq_rgb.npz",
        features=features,
        feature_names=np.asarray(feature_names()),
    )
    return metadata, features


def encoding_ranking(summary: pd.DataFrame, static_summary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "encoding": encoding,
            "task_mean_macro_f1": float(frame["macro_f1_mean"].mean()),
        }
        for encoding, frame in summary.groupby("encoding", sort=False)
    ]
    rows.append(
        {
            "encoding": "static",
            "task_mean_macro_f1": float(static_summary["macro_f1_mean"].mean()),
        }
    )
    ranking = pd.DataFrame(rows).sort_values(
        ["task_mean_macro_f1", "encoding"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def save_figures(
    metadata: pd.DataFrame,
    summary: pd.DataFrame,
    static_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    ranking: pd.DataFrame,
    encoding_specs: dict,
    output_root: Path,
) -> None:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "speed_pressure_altitude": "speed + pressure + altitude",
        "speed_pressure_azimuth": "speed + pressure + azimuth",
        "speed_altitude_azimuth": "speed + altitude + azimuth",
        "pressure_altitude_azimuth": "pressure + altitude + azimuth",
        "static": "static",
    }

    fig, axis = plt.subplots(figsize=(10.5, 5.5))
    for encoding, frame in summary.groupby("encoding", sort=False):
        ordered = frame.sort_values("task_id")
        axis.plot(
            ordered["task_id"],
            ordered["macro_f1_mean"],
            marker="o",
            label=labels[encoding],
        )
    axis.plot(
        static_summary["task_id"],
        static_summary["macro_f1_mean"],
        color="#222222",
        linestyle="--",
        marker="s",
        label="static",
    )
    axis.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    axis.set(xlabel="PaHaW task", ylabel="Macro F1", ylim=(0.25, 1.0), xticks=range(1, 9))
    axis.set_title("Three-signal RGB encodings with nested SVM")
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_task_triplets_and_static.png", dpi=220)
    plt.close(fig)

    matrix = contrasts.pivot(index="encoding", columns="task_id", values="delta_macro_f1")
    matrix = matrix.loc[list(encoding_specs)]
    fig, axis = plt.subplots(figsize=(10.0, 4.0))
    image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
    axis.set_xticks(range(8), [str(value) for value in matrix.columns])
    axis.set_yticks(range(len(matrix)), [labels[value] for value in matrix.index])
    axis.set(xlabel="PaHaW task", title="Delta macro F1 relative to static")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix.iloc[row, column]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Triplet − static")
    fig.tight_layout()
    fig.savefig(figure_dir / "triplet_delta_vs_static_heatmap.png", dpi=220)
    plt.close(fig)

    ordered = ranking.sort_values("task_mean_macro_f1")
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.barh([labels[value] for value in ordered["encoding"]], ordered["task_mean_macro_f1"])
    axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    axis.set(xlabel="Unweighted mean of task-specific macro F1", xlim=(0.45, 0.65))
    axis.set_title("Descriptive encoding ranking across eight tasks")
    for index, value in enumerate(ordered["task_mean_macro_f1"]):
        axis.text(value + 0.002, index, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(figure_dir / "global_encoding_ranking.png", dpi=220)
    plt.close(fig)

    chosen = metadata[(metadata["task_id"] == 4) & (metadata["label"] == "PD")]
    subject = chosen["subject_id"].sort_values().iloc[0]
    chosen = chosen[chosen["subject_id"] == subject].set_index("encoding")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for axis, encoding in zip(axes, encoding_specs, strict=True):
        with Image.open(chosen.loc[encoding, "image_path"]) as image_value:
            axis.imshow(image_value)
        axis.set_title(labels[encoding], fontsize=8)
        axis.axis("off")
    fig.suptitle(f"Task 4, PD subject {subject}: three-signal RGB encodings")
    fig.tight_layout()
    fig.savefig(figure_dir / "triplet_examples.png", dpi=180)
    plt.close(fig)


def write_report(
    config: dict,
    summary: pd.DataFrame,
    static_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    global_pairs: pd.DataFrame,
    ranking: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_root: Path,
) -> None:
    compact_summary = summary[
        [
            "encoding",
            "task_id",
            "n_subjects",
            "macro_f1_mean",
            "macro_f1_ci_low",
            "macro_f1_ci_high",
            "balanced_accuracy_mean",
            "roc_auc_mean",
        ]
    ].copy()
    compact_contrasts = contrasts[
        [
            "encoding",
            "task_id",
            "delta_macro_f1",
            "ci_low",
            "ci_high",
            "permutation_p",
            "holm_p_global_32",
            "supported_holm_0_05",
        ]
    ].copy()
    compact_pairs = global_pairs.copy()
    compact_ranking = ranking.copy()
    for frame in (compact_summary, compact_contrasts, compact_pairs, compact_ranking):
        excluded = {
            "encoding",
            "candidate",
            "reference",
            "task_id",
            "n_subjects",
            "rank",
            "supported_holm_0_05",
            "supported_holm_0_05_global_10",
        }
        numeric = [column for column in frame.columns if column not in excluded]
        frame[numeric] = frame[numeric].map(lambda value: f"{value:.4f}")

    degenerate_columns = [column for column in diagnostics if column.endswith("_degenerate_range")]
    invalid_columns = [column for column in diagnostics if column.endswith("_invalid_surface_time_steps")]
    degenerate_count = int(diagnostics[degenerate_columns].fillna(False).astype(bool).to_numpy().sum())
    invalid_count = int(diagnostics[invalid_columns].fillna(0).to_numpy().sum())
    best = ranking.iloc[0]
    static_value = float(static_summary["macro_f1_mean"].mean())
    report = f"""# Three-signal RGB encodings with SVM

Experiment: `{config['experiment_id']}`. The four leave-one-signal-out triplets use speed, pressure, altitude, and azimuth. Geometry, width, background, LPQ descriptor, participant splits, and SVM search are frozen from the preceding experiments.

## Channel-order control

LPQ is extracted independently from R, G, and B into three 256-feature blocks. Reassigning a triplet among RGB therefore only permutes feature columns. StandardScaler and both linear and RBF SVM are invariant to this permutation. Automated tests verify the rendered-channel permutation and classifier decision scores. One declared RGB order per triplet is consequently sufficient.

## Signal processing and diagnostics

Each signal is computed only on surface-to-surface segments, clipped to its recording-level p05/p95, and mapped to [30, 220]. Speed uses `log1p`; azimuth is circularly centered. No in-air path is drawn and stroke width remains 4 px.

Rendered images: {len(diagnostics)}. Degenerate signal ranges: {degenerate_count}. Invalid on-surface timestamp steps: {invalid_count}.

## Descriptive ranking across tasks

The ranking uses the unweighted mean of the eight task-specific macro F1 estimates. The leading encoding is `{best['encoding']}` at {best['task_mean_macro_f1']:.4f}; the static value is {static_value:.4f}. Selection should also consider the paired uncertainty below.

{markdown_table(compact_ranking)}

## Per-task results

{markdown_table(compact_summary)}

## Per-task paired contrasts against static

The same participants and outer splits are paired. Confidence intervals use 10,000 stratified participant bootstrap samples. Randomization swaps the methods per participant across all repeats. Holm correction covers all 32 triplet-by-task contrasts.

{markdown_table(compact_contrasts)}

## Global pairwise comparisons

The global endpoint is the unweighted mean of macro F1 over eight tasks and five repeats. Bootstrap sampling and method-label swaps operate at participant level and are shared across tasks and repeats. Holm correction covers all ten pairs among the four triplets and static encoding. Delta is candidate minus reference.

{markdown_table(compact_pairs)}
"""
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dynamic_triplet_svm_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = (project_root / config["output_root"]).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to resume or overwrite non-empty run: {output_root}")
    output_root.mkdir(parents=True)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True)
    static_run = (project_root / config["static_run"]).resolve()
    split_path = static_run / "metrics" / "cv_splits.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    samples = [
        sample
        for sample in index_samples(Path(config["dataset_root"]))
        if sample.task_id in set(config["tasks"])
    ]
    audit = audit_samples(samples, Path(config["dataset_root"]))
    audit.to_csv(metrics_dir / "dataset_audit.csv", index=False)
    static_predictions = pd.read_csv(
        static_run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str}
    )
    static_summary = pd.read_csv(static_run / "metrics" / "task_summary.csv")
    all_metadata = []
    all_predictions = []
    all_search = []
    all_fold_metrics = []
    all_repeat_metrics = []
    all_summaries = []
    candidates = candidates_from_config(config["svm_search"])
    for encoding, specification in config["encodings"].items():
        signals = tuple(specification["rgb"])
        metadata, features = render_encoding(samples, encoding, signals, config, output_root)
        all_metadata.append(metadata)
        predictions, search, fold_metrics = run_nested_cv(metadata, features, split_payload, candidates)
        predictions.insert(0, "encoding", encoding)
        search.insert(0, "encoding", encoding)
        fold_metrics.insert(0, "encoding", encoding)
        repeat_metrics, summary = summarize_predictions(
            predictions,
            bootstrap_resamples=int(config["validation"]["bootstrap_resamples"]),
            bootstrap_seed=int(config["validation"]["bootstrap_seed"]),
        )
        repeat_metrics.insert(0, "encoding", encoding)
        summary.insert(0, "encoding", encoding)
        all_predictions.append(predictions)
        all_search.append(search)
        all_fold_metrics.append(fold_metrics)
        all_repeat_metrics.append(repeat_metrics)
        all_summaries.append(summary)
        print(f"{encoding}: evaluation complete", flush=True)

    metadata = pd.concat(all_metadata, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    search = pd.concat(all_search, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    repeat_metrics = pd.concat(all_repeat_metrics, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    metadata.to_csv(metrics_dir / "image_manifest.csv", index=False)
    predictions.to_csv(metrics_dir / "outer_predictions.csv", index=False)
    search.to_csv(metrics_dir / "inner_search.csv", index=False)
    fold_metrics.to_csv(metrics_dir / "outer_fold_metrics.csv", index=False)
    repeat_metrics.to_csv(metrics_dir / "repeat_metrics.csv", index=False)
    summary.to_csv(metrics_dir / "task_summary.csv", index=False)

    contrasts = compare_all_to_static(
        predictions,
        static_predictions,
        resamples=int(config["validation"]["paired_randomization_resamples"]),
        seed=int(config["validation"]["paired_randomization_seed"]),
    )
    contrasts.to_csv(metrics_dir / "paired_contrasts_vs_static.csv", index=False)
    method_frames = {
        encoding: predictions[predictions["encoding"] == encoding].copy()
        for encoding in config["encodings"]
    }
    method_frames["static"] = static_predictions
    global_pairs = compare_global_method_pairs(
        method_frames,
        [*config["encodings"], "static"],
        resamples=int(config["validation"]["paired_randomization_resamples"]),
        seed=int(config["validation"]["paired_randomization_seed"]) + 1,
    )
    global_pairs.to_csv(metrics_dir / "global_pairwise_comparisons.csv", index=False)
    ranking = encoding_ranking(summary, static_summary)
    ranking.to_csv(metrics_dir / "encoding_ranking.csv", index=False)
    save_figures(
        metadata,
        summary,
        static_summary,
        contrasts,
        ranking,
        config["encodings"],
        output_root,
    )
    write_report(
        config,
        summary,
        static_summary,
        contrasts,
        global_pairs,
        ranking,
        metadata,
        output_root,
    )

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
        "static_run": str(static_run),
        "single_signal_run": str((project_root / config["single_signal_run"]).resolve()),
        "reused_cv_split_checksum": split_payload["checksum"],
        "dataset_recordings": len(samples),
        "encodings": config["encodings"],
        "rendered_images": len(metadata),
        "feature_shape_per_encoding": [len(samples), 768],
        "outer_predictions": len(predictions),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            package: version(package)
            for package in (
                "joblib",
                "matplotlib",
                "numpy",
                "openpyxl",
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
