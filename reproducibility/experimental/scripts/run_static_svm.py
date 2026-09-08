"""Run the first clean experiment: static reconstruction plus SVM on all tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

from pdhms_restart.data import audit_samples, index_samples, sha256_file
from pdhms_restart.evaluation import (
    build_splits,
    candidates_from_config,
    run_nested_cv,
    summarize_predictions,
)
from pdhms_restart.lpq import extract_path, feature_names
from pdhms_restart.static_image import StaticRenderConfig, render_sample


def read_config(path: Path) -> dict:
    """Read the immutable run configuration."""

    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: object, path: Path) -> None:
    """Write stable, readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without an optional tabulate dependency."""

    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_and_extract(config: dict, project_root: Path, output_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Audit, render, and extract all selected recordings from scratch."""

    dataset_root = Path(config["dataset_root"])
    task_ids = set(int(value) for value in config["tasks"])
    samples = [sample for sample in index_samples(dataset_root) if sample.task_id in task_ids]
    audit = audit_samples(samples, dataset_root)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(metrics_dir / "dataset_audit.csv", index=False)
    invalid = audit[
        (~audit["finite"])
        | (~audit["state_values_valid"])
        | (audit["surface_segments"] <= 0)
    ]
    if not invalid.empty:
        raise ValueError(f"Dataset audit found {len(invalid)} unusable recordings.")
    render_values = dict(config["render"])
    render_values.pop("coordinate_frame", None)
    render_values["foreground_rgb"] = tuple(render_values["foreground_rgb"])
    render_values["background_rgb"] = tuple(render_values["background_rgb"])
    render_config = StaticRenderConfig(**render_values)
    image_rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples, start=1):
        image_path = (
            output_root
            / "images"
            / "static"
            / f"task_{sample.task_id:02d}"
            / f"subject_{sample.subject_id}_task_{sample.task_id:02d}_static.png"
        )
        row = render_sample(sample, image_path, render_config)
        row["source_path"] = str(sample.path)
        row["image_sha256"] = sha256_file(image_path)
        image_rows.append(row)
        if index % 50 == 0 or index == len(samples):
            print(f"rendered {index}/{len(samples)}", flush=True)
    metadata = pd.DataFrame(image_rows).sort_values(["task_id", "subject_id"]).reset_index(drop=True)
    metadata.to_csv(metrics_dir / "image_manifest.csv", index=False)
    matrix = np.vstack([extract_path(Path(path)) for path in metadata["image_path"]]).astype(np.float32)
    feature_dir = output_root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_dir / "static_lpq_rgb.npz",
        features=matrix,
        feature_names=np.asarray(feature_names()),
    )
    metadata.to_csv(feature_dir / "metadata.csv", index=False)
    return metadata, matrix


def save_figures(metadata: pd.DataFrame, summary: pd.DataFrame, output_root: Path) -> None:
    """Create a task result plot and an H/PD example sheet."""

    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("task_id")
    lower = ordered["macro_f1_mean"] - ordered["macro_f1_ci_low"]
    upper = ordered["macro_f1_ci_high"] - ordered["macro_f1_mean"]
    fig, axis = plt.subplots(figsize=(8.4, 4.6))
    axis.bar(ordered["task_id"].astype(str), ordered["macro_f1_mean"], color="#336699")
    axis.errorbar(
        ordered["task_id"].astype(str),
        ordered["macro_f1_mean"],
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#1d2630",
        capsize=4,
    )
    axis.axhline(0.5, color="#9a3f3f", linestyle="--", linewidth=1)
    axis.set(xlabel="PaHaW task", ylabel="Macro F1", ylim=(0.25, 1.0))
    axis.set_title("Static reconstruction + nested SVM")
    fig.tight_layout()
    fig.savefig(figure_dir / "per_task_macro_f1.png", dpi=220)
    plt.close(fig)

    examples = []
    for task_id in sorted(metadata["task_id"].unique()):
        for label in ("H", "PD"):
            examples.append(metadata[(metadata["task_id"] == task_id) & (metadata["label"] == label)].iloc[0])
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
    for axis, row in zip(axes.ravel(), examples, strict=True):
        with Image.open(row["image_path"]) as image:
            axis.imshow(image)
        axis.set_title(f"T{int(row['task_id'])} · {row['label']}")
        axis.axis("off")
    fig.suptitle("Static reconstruction examples (first subject by task and class)")
    fig.tight_layout()
    fig.savefig(figure_dir / "static_examples.png", dpi=180)
    plt.close(fig)


def write_report(config: dict, audit: pd.DataFrame, summary: pd.DataFrame, output_root: Path) -> None:
    """Write a concise, source-linked Markdown report."""

    table = summary.copy()
    numeric = [column for column in table.columns if column not in {"task_id", "n_subjects", "selected_svm_counts"}]
    table[numeric] = table[numeric].map(lambda value: f"{value:.4f}")
    audit_counts = audit.groupby(["task_id", "label"]).size().unstack(fill_value=0).reset_index()
    report = f"""# Static reconstruction + SVM across PaHaW tasks

Experiment: `{config['experiment_id']}`. Generated locally from raw PaHaW `.svc` files.

## Question

How much class signal is present in the visible, static geometry of each handwriting task when all dynamic attributes are omitted?

## Input and descriptor

Only consecutive pen-down points are drawn. The image is grayscale RGB, uses a constant {config['render']['line_width_px']} px width, and contains no in-air trajectory, pressure, altitude, azimuth, timestamps, speed, duration, or temporal colour. Bounds are calculated from pen-down points only. Each sample is scaled independently while preserving aspect ratio; absolute writing size is therefore not retained.

The classifier receives a {config['features']['dimension']}-dimensional channel-wise LPQ descriptor. The three blocks are identical for this grayscale baseline and retain the same interface intended for later RGB encodings.

## Evaluation

Nested validation uses {config['validation']['repeats']} repeats of {config['validation']['outer_folds']} subject-level outer folds and {config['validation']['inner_folds']} inner folds. All samples derived from a participant remain together. Standardization and SVM selection are fitted inside the corresponding training fold. Kernel and hyperparameters are selected by inner macro F1. The primary estimate is mean outer macro F1 over repeats; its interval is a stratified participant bootstrap with {config['validation']['bootstrap_resamples']:,} resamples.

## Dataset audit

{markdown_table(audit_counts)}

Recordings: {len(audit)}. Declared/observed point-count mismatches: {int((~audit['count_matches']).sum())}. Non-finite recordings: {int((~audit['finite']).sum())}. Invalid state recordings: {int((~audit['state_values_valid']).sum())}. Recordings without surface segments: {int((audit['surface_segments'] <= 0).sum())}. Negative timestamp steps: {int(audit['negative_timestamp_steps'].sum())} across {int((audit['negative_timestamp_steps'] > 0).sum())} recordings. The static reconstruction does not use timestamps; these anomalies must be handled before the dynamic experiments.

## Results

{markdown_table(table)}

Macro F1 and balanced accuracy use both classes symmetrically. `f1_pd_mean` is included for comparison with studies that report F1 only for PD. These are internal, repeated nested-CV estimates on PaHaW, not external clinical validation. Task differences are descriptive in this first experiment; no task was selected for a headline result.
"""
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "static_svm_report.md").write_text(report, encoding="utf-8")


def artifact_manifest(output_root: Path) -> dict:
    """Hash every result artifact except the manifest itself."""

    rows = []
    target = output_root / "manifests" / "artifact_manifest.json"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = read_config(config_path)
    output_root = (project_root / config["output_root"]).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to resume or overwrite non-empty run: {output_root}")
    output_root.mkdir(parents=True)
    print(f"experiment={config['experiment_id']} output={output_root}")
    metadata, features = render_and_extract(config, project_root, output_root)
    split_payload = build_splits(
        metadata,
        repeats=int(config["validation"]["repeats"]),
        outer_folds=int(config["validation"]["outer_folds"]),
        inner_folds=int(config["validation"]["inner_folds"]),
        master_seed=int(config["validation"]["master_seed"]),
    )
    split_text = json.dumps(split_payload, sort_keys=True, separators=(",", ":"))
    split_payload["checksum"] = hashlib.sha256(split_text.encode("utf-8")).hexdigest()
    save_json(split_payload, output_root / "metrics" / "cv_splits.json")
    candidates = candidates_from_config(config["svm_search"])
    predictions, selections, fold_metrics = run_nested_cv(metadata, features, split_payload, candidates)
    predictions.to_csv(output_root / "metrics" / "outer_predictions.csv", index=False)
    selections.to_csv(output_root / "metrics" / "inner_search.csv", index=False)
    fold_metrics.to_csv(output_root / "metrics" / "outer_fold_metrics.csv", index=False)
    repeat_metrics, summary = summarize_predictions(
        predictions,
        bootstrap_resamples=int(config["validation"]["bootstrap_resamples"]),
        bootstrap_seed=int(config["validation"]["bootstrap_seed"]),
    )
    repeat_metrics.to_csv(output_root / "metrics" / "repeat_metrics.csv", index=False)
    summary.to_csv(output_root / "metrics" / "task_summary.csv", index=False)
    audit = pd.read_csv(output_root / "metrics" / "dataset_audit.csv")
    save_figures(metadata, summary, output_root)
    write_report(config, audit, summary, output_root)
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        git_commit = None
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "fresh_run_no_resume": True,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_root": config["dataset_root"],
        "dataset_recordings": len(metadata),
        "dataset_subjects": int(metadata["subject_id"].nunique()),
        "feature_shape": list(features.shape),
        "cv_split_checksum": split_payload["checksum"],
        "git_commit": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
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
        "command": f"python scripts/run_static_svm.py --config {config_path}",
    }
    save_json(manifest, output_root / "manifests" / "run_manifest.json")
    save_json(artifact_manifest(output_root), output_root / "manifests" / "artifact_manifest.json")
    print(summary.to_string(index=False), flush=True)
    print(f"complete in {manifest['duration_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
