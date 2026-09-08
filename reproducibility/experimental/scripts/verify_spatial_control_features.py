"""Audit the frozen spatial-control features; does not render or fit models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pdhms_restart.data import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest_path = run / "manifests/feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["status"] == "complete", "Feature generation is not complete.")
    require(manifest["model_fitting"] is False and manifest["predictions_or_metrics_computed"] is False,
            "Manifest does not preserve the feature-only scope.")
    sources = manifest["frozen_sources"]
    for relative, expected in sources["code_sha256"].items():
        require(sha256_file(ROOT / relative) == expected, f"Source changed: {relative}")
        require(sha256_file(run / "manifests/source_snapshot" / relative) == expected,
                f"Source snapshot changed: {relative}")
    for path, expected in sources["reference_sha256"].items():
        require(sha256_file(Path(path)) == expected, f"Original reference changed: {path}")
    config = json.loads((run / "manifests/source_snapshot/campaign_config.json").read_text(encoding="utf-8"))
    dataset_root = Path(config["dataset_root"])
    for relative, expected in sources["recording_sha256"].items():
        require(sha256_file(dataset_root / relative) == expected, f"Original recording changed: {relative}")
    expected_units = {f"{encoding}__{control}__seed{seed}"
                      for encoding in manifest["encoding_order"]
                      for control in manifest["controls"] for seed in manifest["seeds"]}
    require(set(manifest["units"]) == expected_units, "Control/seed/encoding unit inventory differs.")
    reference_keys = None
    geometry = {}
    normalized = {}
    permutations = {}
    diagnostic_rows = 0
    artifact_count = 0
    identity_count = 0
    image_count = 0
    for unit, details in manifest["units"].items():
        for relative, expected in details["artifact_sha256"].items():
            require(sha256_file(run / relative) == expected, f"Derived artifact changed: {relative}")
            artifact_count += 1
            image_count += Path(relative).suffix == ".png"
        with np.load(run / "features" / f"{unit}_lpq_rgb.npz", allow_pickle=False) as archive:
            require(set(archive.files) == {"features"}, f"Unexpected NPZ payload: {unit}")
            features = archive["features"]
        require(features.shape == (manifest["records"], 768), f"Unexpected feature shape: {unit}")
        require(features.dtype == np.float32 and bool(np.all(np.isfinite(features))), f"Invalid features: {unit}")
        metadata = pd.read_csv(run / "features" / f"{unit}_metadata.csv", dtype={"subject_id": str})
        keys = list(metadata[["subject_id", "task_id", "repetition_id", "label"]].itertuples(index=False, name=None))
        require(len(keys) == manifest["records"] and len(set(keys)) == len(keys), f"Row inventory differs: {unit}")
        if reference_keys is None:
            reference_keys = keys
        require(keys == reference_keys, f"Row order or labels differ across controls: {unit}")
        require(keys == sorted(keys, key=lambda row: (row[1], row[0], row[2])), f"Unsorted metadata: {unit}")
        identity_count += int(metadata["identity_verified_against_original"].sum())
        lines = (run / "diagnostics" / f"{unit}.jsonl").read_text(encoding="utf-8").splitlines()
        require(len(lines) == len(keys), f"Diagnostic count differs: {unit}")
        for row, line in zip(metadata.itertuples(index=False), lines):
            diagnostic = json.loads(line)
            require(diagnostic["record_key"] == row.record_key, f"Diagnostic row order differs: {unit}")
            require("label" not in diagnostic, f"Diagnosis entered renderer diagnostics: {unit}")
            require(all(diagnostic[name] for name in
                        ("surface_mask_preserved", "non_surface_rows_unchanged", "joint_vectors_preserved_per_stroke")),
                    f"A segment invariant failed: {unit}, {row.record_key}")
            moved = diagnostic["moved_segment_fraction"]
            changed = diagnostic["changed_signal_vector_fraction"]
            colour = diagnostic["changed_colour_vector_fraction"]
            require(0 <= colour <= changed <= moved <= 1, f"Inconsistent changed fractions: {unit}")
            record_key = row.record_key
            normalized_key = (row.encoding, record_key)
            permutation_key = (row.control, int(row.control_seed), record_key)
            for cache, key, value in (
                (geometry, record_key, diagnostic["geometry_sha256"]),
                (normalized, normalized_key, diagnostic["original_normalized_sha256"]),
                (permutations, permutation_key, diagnostic["source_indices_sha256"]),
            ):
                require(cache.setdefault(key, value) == value, f"Control changed an invariant hash: {unit}")
            require(sum(stroke["length"] for stroke in diagnostic["strokes"]) == diagnostic["surface_segments"],
                    f"Stroke lengths do not cover the surface mask: {unit}")
            last_stop = -1
            for stroke in diagnostic["strokes"]:
                length = stroke["length"]
                require(stroke["start"] > last_stop and length == stroke["stop"] - stroke["start"] and length > 0,
                        f"Stroke intervals overlap or join pen-up gaps: {unit}")
                last_stop = stroke["stop"]
                require(stroke["joint_vectors_preserved"], f"Joint vectors not preserved: {unit}")
                if row.control == "circular_shift":
                    require(stroke["shift"] == 0 if length == 1 else
                            max(1, (length + 3) // 4) <= stroke["shift"] <= (3 * length) // 4,
                            f"Shift outside the declared range: {unit}")
                else:
                    require(stroke["shift"] is None, f"A permutation is mislabeled as a circular shift: {unit}")
            diagnostic_rows += 1
    require(identity_count == len(manifest["encoding_order"]) * manifest["records"], "Identity check coverage differs.")
    require(diagnostic_rows == manifest["feature_vectors"], "Feature and diagnostic totals differ.")
    require(image_count == len(expected_units) * len(manifest["panel_selection"]["recording_keys"]),
            "The preview image inventory differs from the fixed selection.")
    report = {
        "status": "passed", "feature_units": len(expected_units), "feature_vectors": diagnostic_rows,
        "identity_image_and_lpq_checks": identity_count, "artifacts_sha256_checked": artifact_count,
        "preview_images": image_count, "geometry_invariants_across_all_conditions": len(geometry),
        "shared_permutation_invariants_across_encodings": len(permutations),
        "model_fitting_or_inference": False,
        "manifest_sha256": sha256_file(manifest_path), "verifier_sha256": sha256_file(Path(__file__)),
    }
    (run / "manifests/verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
