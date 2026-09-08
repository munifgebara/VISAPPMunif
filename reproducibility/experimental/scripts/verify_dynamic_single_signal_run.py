"""Verify a completed isolated dynamic-signal SVM run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pdhms_restart.data import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--static-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    static_run = args.static_run.resolve()
    errors: list[str] = []
    encodings = ["speed", "pressure", "altitude", "azimuth"]

    manifest = json.loads((run / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads(
        (run / "manifests" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    image_manifest = pd.read_csv(run / "metrics" / "image_manifest.csv", dtype={"subject_id": str})
    predictions = pd.read_csv(run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str})
    static_predictions = pd.read_csv(
        static_run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str}
    )
    search = pd.read_csv(run / "metrics" / "inner_search.csv")
    summary = pd.read_csv(run / "metrics" / "task_summary.csv")
    contrasts = pd.read_csv(run / "metrics" / "paired_contrasts_vs_static.csv")

    expected_per_task = {1: 72, **{task: 75 for task in range(2, 9)}}
    expected_per_encoding = sum(expected_per_task.values())
    if set(image_manifest["encoding"]) != set(encodings):
        errors.append("Unexpected image encodings.")
    for encoding in encodings:
        current = image_manifest[image_manifest["encoding"] == encoding]
        observed = current.groupby("task_id").size().to_dict()
        if observed != expected_per_task:
            errors.append(f"Unexpected image counts for {encoding}: {observed}.")
        feature_path = run / "features" / f"{encoding}_lpq_rgb.npz"
        with np.load(feature_path) as payload:
            shape = tuple(payload["features"].shape)
            finite = bool(np.all(np.isfinite(payload["features"])))
        if shape != (expected_per_encoding, 768) or not finite:
            errors.append(f"Invalid {encoding} feature matrix: shape={shape}, finite={finite}.")

    missing_images = [path for path in image_manifest["image_path"] if not Path(path).is_file()]
    if missing_images:
        errors.append(f"Missing {len(missing_images)} rendered images.")
    if image_manifest["invalid_surface_time_steps"].sum() != 0:
        errors.append("Dynamic image manifest contains invalid on-surface time steps.")
    if len(predictions) != len(static_predictions) * len(encodings):
        errors.append(f"Unexpected prediction count: {len(predictions)}.")
    if len(summary) != 32 or len(contrasts) != 32:
        errors.append(f"Expected 32 summary and contrast rows, found {len(summary)} and {len(contrasts)}.")
    selected = search[search["selected"].astype(str).str.lower() == "true"]
    if len(selected) != 800:
        errors.append(f"Expected 800 selected task/fold models, found {len(selected)}.")

    keys = ["repeat", "outer_fold", "task_id", "subject_id"]
    reference = static_predictions[keys + ["y_true"]].sort_values(keys).reset_index(drop=True)
    for encoding in encodings:
        current = (
            predictions[predictions["encoding"] == encoding][keys + ["y_true"]]
            .sort_values(keys)
            .reset_index(drop=True)
        )
        if not current.equals(reference):
            errors.append(f"Prediction pairing differs from static for {encoding}.")

    static_splits = json.loads(
        (static_run / "metrics" / "cv_splits.json").read_text(encoding="utf-8")
    )
    if manifest["reused_cv_split_checksum"] != static_splits["checksum"]:
        errors.append("Recorded CV split checksum differs from the static run.")

    bad_hashes: list[str] = []
    for artifact in artifacts["artifacts"]:
        path = run / artifact["relative_path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            bad_hashes.append(artifact["relative_path"])
    if bad_hashes:
        errors.append(f"Missing or changed artifact hashes: {bad_hashes}.")

    result = {
        "status": "passed" if not errors else "failed",
        "run": str(run),
        "encodings": encodings,
        "rendered_images": len(image_manifest),
        "outer_predictions": len(predictions),
        "selected_task_fold_models": len(selected),
        "contrasts": len(contrasts),
        "artifact_hashes_checked": len(artifacts["artifacts"]),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
