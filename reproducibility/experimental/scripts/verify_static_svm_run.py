"""Verify the structural integrity of a completed static-SVM run."""

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
    args = parser.parse_args()
    run = args.run.resolve()
    errors: list[str] = []

    image_manifest = pd.read_csv(run / "metrics" / "image_manifest.csv", dtype={"subject_id": str})
    predictions = pd.read_csv(run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str})
    search = pd.read_csv(run / "metrics" / "inner_search.csv")
    with np.load(run / "features" / "static_lpq_rgb.npz") as payload:
        feature_shape = tuple(payload["features"].shape)
        finite_features = bool(np.all(np.isfinite(payload["features"])))
    splits = json.loads((run / "metrics" / "cv_splits.json").read_text(encoding="utf-8"))
    artifact_payload = json.loads(
        (run / "manifests" / "artifact_manifest.json").read_text(encoding="utf-8")
    )

    if len(image_manifest) != 597:
        errors.append(f"Expected 597 images, found {len(image_manifest)}.")
    expected_per_task = {1: 72, **{task: 75 for task in range(2, 9)}}
    observed_per_task = image_manifest.groupby("task_id").size().to_dict()
    if observed_per_task != expected_per_task:
        errors.append(f"Unexpected task counts: {observed_per_task}.")
    missing_images = [path for path in image_manifest["image_path"] if not Path(path).is_file()]
    if missing_images:
        errors.append(f"Missing {len(missing_images)} rendered images.")
    if feature_shape != (597, 768) or not finite_features:
        errors.append(f"Invalid feature matrix: shape={feature_shape}, finite={finite_features}.")
    if len(predictions) != 2_985:
        errors.append(f"Expected 2,985 predictions, found {len(predictions)}.")
    prediction_keys = predictions[["repeat", "task_id", "subject_id"]].drop_duplicates()
    if len(prediction_keys) != len(predictions):
        errors.append("Outer prediction keys are not unique.")
    selected = search[search["selected"].astype(str).str.lower() == "true"]
    if len(selected) != 200:
        errors.append(f"Expected 200 selected task/fold models, found {len(selected)}.")

    split_lookup: dict[tuple[int, int], dict] = {}
    for split in splits["outer_splits"]:
        key = (int(split["repeat"]), int(split["outer_fold"]))
        split_lookup[key] = split
        train = set(str(value).zfill(5) for value in split["train_subject_ids"])
        test = set(str(value).zfill(5) for value in split["test_subject_ids"])
        if train & test:
            errors.append(f"Outer subject leakage in {key}.")
        for task_id, task in split["tasks"].items():
            for inner in task["splits"]:
                inner_train = set(str(value).zfill(5) for value in inner["train_subject_ids"])
                validation = set(str(value).zfill(5) for value in inner["validation_subject_ids"])
                if inner_train & validation:
                    errors.append(f"Inner subject leakage in {key}, task {task_id}.")
    for (repeat, fold), frame in predictions.groupby(["repeat", "outer_fold"]):
        allowed = set(str(value).zfill(5) for value in split_lookup[(int(repeat), int(fold))]["test_subject_ids"])
        predicted_subjects = set(frame["subject_id"].astype(str).str.zfill(5))
        if not predicted_subjects <= allowed:
            errors.append(f"Predictions outside outer test subjects in {(repeat, fold)}.")

    bad_hashes: list[str] = []
    for artifact in artifact_payload["artifacts"]:
        path = run / artifact["relative_path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            bad_hashes.append(artifact["relative_path"])
    if bad_hashes:
        errors.append(f"Missing or changed artifact hashes: {bad_hashes}.")

    result = {
        "status": "passed" if not errors else "failed",
        "run": str(run),
        "images": len(image_manifest),
        "images_per_task": observed_per_task,
        "feature_shape": feature_shape,
        "outer_predictions": len(predictions),
        "selected_task_fold_models": len(selected),
        "outer_splits": len(splits["outer_splits"]),
        "artifact_hashes_checked": len(artifact_payload["artifacts"]),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
