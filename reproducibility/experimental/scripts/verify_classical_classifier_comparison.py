"""Verify a completed classical-classifier comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pdhms_restart.data import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    errors: list[str] = []
    algorithms = ["svm", "random_forest", "logistic_regression"]
    predictions = pd.read_csv(run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str})
    search = pd.read_csv(run / "metrics" / "inner_search.csv")
    summary = pd.read_csv(run / "metrics" / "task_summary.csv")
    comparisons = pd.read_csv(run / "metrics" / "paired_contrasts_vs_svm.csv")
    global_pairs = pd.read_csv(run / "metrics" / "global_pairwise_comparisons.csv")
    ranking = pd.read_csv(run / "metrics" / "classifier_ranking.csv")
    sources = pd.read_csv(run / "metrics" / "source_artifacts.csv")
    manifest = json.loads((run / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads(
        (run / "manifests" / "artifact_manifest.json").read_text(encoding="utf-8")
    )

    if set(predictions["algorithm"]) != set(algorithms):
        errors.append("Unexpected algorithms in predictions.")
    for algorithm in algorithms:
        if len(predictions[predictions["algorithm"] == algorithm]) != 2_985:
            errors.append(f"Unexpected prediction count for {algorithm}.")
    keys = ["repeat", "outer_fold", "task_id", "subject_id", "y_true"]
    reference = predictions[predictions["algorithm"] == "svm"][keys].sort_values(keys).reset_index(drop=True)
    for algorithm in algorithms[1:]:
        current = predictions[predictions["algorithm"] == algorithm][keys].sort_values(keys).reset_index(drop=True)
        if not current.equals(reference):
            errors.append(f"Prediction pairing differs from SVM for {algorithm}.")
    selected = search[search["selected"].astype(str).str.lower() == "true"]
    if len(selected) != 600:
        errors.append(f"Expected 600 selected outer models, found {len(selected)}.")
    if len(summary) != 24 or len(comparisons) != 16 or len(global_pairs) != 3 or len(ranking) != 3:
        errors.append("Unexpected metric table dimensions.")
    if not ranking["task_mean_macro_f1"].is_monotonic_decreasing:
        errors.append("Classifier ranking is not descending.")
    for row in sources.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or sha256_file(path) != row.sha256:
            errors.append(f"Source artifact missing or changed: {row.role}.")
    bad_hashes = []
    for artifact in artifacts["artifacts"]:
        path = run / artifact["relative_path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            bad_hashes.append(artifact["relative_path"])
    if bad_hashes:
        errors.append(f"Missing or changed artifact hashes: {bad_hashes}.")
    result = {
        "status": "passed" if not errors else "failed",
        "run": str(run),
        "algorithms": algorithms,
        "outer_predictions": len(predictions),
        "selected_models": len(selected),
        "per_task_comparisons": len(comparisons),
        "global_comparisons": len(global_pairs),
        "artifact_hashes_checked": len(artifacts["artifacts"]),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
