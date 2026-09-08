"""Verify a completed eight-task participant-fusion run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pdhms_restart.data import sha256_file
from pdhms_restart.task_fusion import FUSION_METHODS, fuse_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--static-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    source_run = args.source_run.resolve()
    static_run = args.static_run.resolve()
    errors: list[str] = []

    manifest = json.loads((run / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads(
        (run / "manifests" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    base = pd.read_csv(run / "metrics" / "calibrated_task_predictions.csv", dtype={"subject_id": str})
    fused = pd.read_csv(run / "metrics" / "fused_predictions.csv", dtype={"subject_id": str})
    repeat_metrics = pd.read_csv(run / "metrics" / "repeat_metrics.csv")
    summary = pd.read_csv(run / "metrics" / "method_summary.csv")
    ranking = pd.read_csv(run / "metrics" / "fusion_ranking.csv")
    pairwise = pd.read_csv(run / "metrics" / "global_pairwise_comparisons.csv")

    if len(base) != 2985:
        errors.append(f"Expected 2985 calibrated task predictions, found {len(base)}.")
    if not np.all(np.isfinite(base["calibrated_probability_pd"])):
        errors.append("Non-finite calibrated probabilities.")
    if not base["calibrated_probability_pd"].between(0, 1).all():
        errors.append("Calibrated probability outside [0, 1].")
    if not np.all(np.isfinite(base["reliability_weight"])) or (base["reliability_weight"] <= 0).any():
        errors.append("Invalid inner reliability weight.")

    source = pd.read_csv(
        source_run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str}
    )
    source = source[source["encoding"] == manifest["selected_encoding"]].copy()
    keys = ["repeat", "outer_fold", "task_id", "subject_id"]
    current = base[keys + ["y_true", "base_y_pred"]].sort_values(keys).reset_index(drop=True)
    frozen = source[keys + ["y_true", "y_pred"]].rename(columns={"y_pred": "base_y_pred"})
    frozen = frozen.sort_values(keys).reset_index(drop=True)
    if not current.equals(frozen):
        errors.append("Base predictions do not match the frozen SAZ SVM run.")

    expected_fused = []
    for _, frame in base.groupby(["repeat", "outer_fold"], sort=True):
        expected_fused.append(fuse_fold(frame))
    expected = pd.concat(expected_fused, ignore_index=True).sort_values(
        ["method", *keys]
    ).reset_index(drop=True)
    observed = fused.sort_values(["method", *keys]).reset_index(drop=True)
    compare_columns = [
        "method",
        *keys,
        "y_true",
        "y_pred",
        "decision_score_pd",
        "available_task_count",
        "available_tasks",
    ]
    try:
        pd.testing.assert_frame_equal(
            observed[compare_columns], expected[compare_columns], check_exact=False, atol=1e-14
        )
    except AssertionError:
        errors.append("Stored fused predictions do not reproduce from calibrated task predictions.")

    if set(fused["method"]) != set(FUSION_METHODS):
        errors.append("Unexpected fusion method set.")
    if len(fused) != 1125 or not (fused.groupby("method").size() == 375).all():
        errors.append("Unexpected fused prediction count.")
    per_subject = fused.groupby(["method", "repeat", "subject_id"]).size()
    if not (per_subject == 1).all():
        errors.append("A fusion method has duplicate participant predictions within a repeat.")
    if set(fused["available_task_count"]) != {7, 8}:
        errors.append("Unexpected available-task counts.")
    if len(repeat_metrics) != 15 or len(summary) != 3 or len(ranking) != 3 or len(pairwise) != 3:
        errors.append("Unexpected result table dimensions.")

    static_splits = json.loads(
        (static_run / "metrics" / "cv_splits.json").read_text(encoding="utf-8")
    )
    if manifest["reused_cv_split_checksum"] != static_splits["checksum"]:
        errors.append("Run manifest does not match the frozen participant splits.")
    bad_hashes = []
    for artifact in artifacts["artifacts"]:
        path = run / artifact["relative_path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            bad_hashes.append(artifact["relative_path"])
    if bad_hashes:
        errors.append(f"Missing or changed artifacts: {bad_hashes}.")

    result = {
        "status": "passed" if not errors else "failed",
        "run": str(run),
        "calibrated_task_predictions": len(base),
        "fused_predictions": len(fused),
        "methods": sorted(fused["method"].unique()),
        "participant_predictions_per_method": fused.groupby("method").size().to_dict(),
        "artifact_hashes_checked": len(artifacts["artifacts"]),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
