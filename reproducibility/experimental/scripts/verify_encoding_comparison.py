"""Verify the consolidated comparison of all image encodings."""

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
    ranking = pd.read_csv(run / "metrics" / "all_encoding_ranking.csv")
    comparisons = pd.read_csv(run / "metrics" / "all_encoding_global_pairwise.csv")
    sources = pd.read_csv(run / "metrics" / "prediction_sources.csv")
    manifest = json.loads((run / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads(
        (run / "manifests" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if len(ranking) != 9 or ranking["encoding"].nunique() != 9:
        errors.append("Expected nine unique encoding ranks.")
    if ranking["rank"].tolist() != list(range(1, 10)):
        errors.append("Ranking positions are not consecutive.")
    if not ranking["task_mean_macro_f1"].is_monotonic_decreasing:
        errors.append("Ranking values are not descending.")
    if len(comparisons) != 36:
        errors.append(f"Expected 36 pairwise comparisons, found {len(comparisons)}.")
    observed_pairs = {
        frozenset(pair)
        for pair in comparisons[["candidate", "reference"]].itertuples(index=False, name=None)
    }
    if len(observed_pairs) != 36:
        errors.append("Global method pairs are duplicated or incomplete.")
    if len(sources) != 9:
        errors.append(f"Expected nine prediction sources, found {len(sources)}.")
    for row in sources.itertuples(index=False):
        path = Path(row.prediction_path)
        if not path.is_file() or sha256_file(path) != row.prediction_sha256:
            errors.append(f"Prediction source missing or changed: {row.encoding}.")
    if manifest["selected_encoding"] != ranking.iloc[0]["encoding"]:
        errors.append("Manifest selection differs from the ranking winner.")
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
        "encodings": len(ranking),
        "comparisons": len(comparisons),
        "selected_encoding": manifest["selected_encoding"],
        "artifact_hashes_checked": len(artifacts["artifacts"]),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
