"""Extract the fixed 83-dimensional baseline in the existing static row order."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

from pdhms_restart.data import load_svc, sha256_file
from pdhms_restart.kinematic_features import (
    DISTRIBUTION_NAMES, FEATURE_NAMES, GLOBAL_FEATURE_NAMES, MAX_VALID_INTERVAL_MS,
    SUMMARY_NAMES, TIMESTAMP_SECONDS_PER_UNIT, extract_kinematic_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("C:/Users/munif/PycharmProjects/PseudodynamicHandwritingMinimapSignatures/dataset/PaHaW")
DEFAULT_METADATA = PROJECT_ROOT / "experiments/2026-09-restart/runs/static-svm-v1/features/metadata.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/2026-09-restart/runs/competitive-task-subset-v1/kinematic_features"
DEFAULT_DESCRIPTOR = PROJECT_ROOT / "experiments/2026-09-restart/competitive-task-subset-v1/kinematic_descriptor.json"


def write_json(value: object, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_descriptor(descriptor: dict) -> None:
    expected = {
        "dimension": len(FEATURE_NAMES), "timestamp_seconds_per_unit": TIMESTAMP_SECONDS_PER_UNIT,
        "max_valid_interval_ms": MAX_VALID_INTERVAL_MS,
        "global_features": list(GLOBAL_FEATURE_NAMES), "distribution_order": list(DISTRIBUTION_NAMES),
        "summary_order": list(SUMMARY_NAMES), "standard_deviation_ddof": 0,
    }
    for key, value in expected.items():
        if descriptor[key] != value:
            raise ValueError(f"Descriptor/source mismatch for {key}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--descriptor", type=Path, default=DEFAULT_DESCRIPTOR)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing feature bundle: {output}")
    descriptor = json.loads(args.descriptor.read_text(encoding="utf-8"))
    validate_descriptor(descriptor)
    metadata = pd.read_csv(args.metadata, dtype={"subject_id": str})
    if len(metadata) != 597:
        raise ValueError(f"Expected the full 597-row static metadata, got {len(metadata)}.")
    if metadata.duplicated(["subject_id", "task_id", "repetition_id"]).any():
        raise ValueError("Duplicate subject/task/repetition keys.")
    dataset = args.dataset_root.resolve()
    feature_rows, diagnostic_rows, output_rows = [], [], []
    source_hashes = {}
    for index, row in enumerate(metadata.itertuples(index=False)):
        # Resolve solely from the explicit record keys, not class directories.
        relative = Path("PaHaW_public") / row.subject_id / f"{row.subject_id}__{int(row.task_id)}_{int(row.repetition_id)}.svc"
        source = dataset / relative
        declared, points = load_svc(source)
        extracted = extract_kinematic_features(points)
        feature_rows.append(extracted.values)
        digest = sha256_file(source)
        source_hashes[relative.as_posix()] = digest
        keys = {"row_index": index, "subject_id": row.subject_id, "task_id": int(row.task_id), "repetition_id": int(row.repetition_id)}
        output_rows.append({**keys, "label": row.label, "source_relative_path": relative.as_posix(), "source_sha256": digest})
        diagnostics = dict(extracted.diagnostics)
        counts = diagnostics.pop("distribution_counts")
        diagnostics["empty_distributions"] = "|".join(diagnostics["empty_distributions"])
        diagnostic_rows.append({**keys, "declared_points": declared, "count_matches": declared == len(points),
                                **diagnostics, **{f"{name}__count": count for name, count in counts.items()}})
        if (index + 1) % 100 == 0:
            print(f"Extracted {index + 1}/{len(metadata)} records", flush=True)
    features = np.vstack(feature_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    if features.shape != (597, 83) or not np.isfinite(features).all():
        raise ValueError("Expected a finite 597x83 feature matrix.")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "features.npy", features, allow_pickle=False)
    write_json(list(FEATURE_NAMES), output / "feature_names.json")
    pd.DataFrame(output_rows).to_csv(output / "metadata.csv", index=False)
    diagnostics.to_csv(output / "diagnostics.csv", index=False)
    shutil.copyfile(args.descriptor, output / "descriptor.json")
    totals = {name: int(diagnostics[name].sum()) for name in (
        "observed_points", "intervals", "negative_intervals", "zero_intervals",
        "over_60s_intervals", "excluded_intervals", "excluded_contact_intervals",
        "excluded_air_intervals", "excluded_transition_intervals", "valid_over_1s_intervals",
    )}
    summary = {
        "records": len(features), "dimension": features.shape[1], "finite": bool(np.isfinite(features).all()),
        "timestamp_diagnostics_totals": totals,
        "records_with_excluded_intervals": int((diagnostics.excluded_intervals > 0).sum()),
        "excluded_interval_fraction_total": totals["excluded_intervals"] / totals["intervals"],
        "zero_fraction_by_feature": dict(zip(FEATURE_NAMES, (features == 0).mean(axis=0).tolist())),
        "constant_features": [FEATURE_NAMES[i] for i in np.flatnonzero(np.ptp(features, axis=0) == 0)],
        "empty_distribution_records": {name: int((diagnostics[f"{name}__count"] == 0).sum()) for name in DISTRIBUTION_NAMES},
        "valid_acquisition_time_s_percentiles": dict(zip(["min", "p05", "median", "p95", "max"], np.percentile(features[:, 0], [0, 5, 50, 95, 100]).tolist())),
        "count_mismatches": int((~diagnostics.count_matches).sum()),
        "models_fitted": 0,
    }
    write_json(summary, output / "summary.json")
    artifact_names = ("features.npy", "feature_names.json", "metadata.csv", "diagnostics.csv", "descriptor.json", "summary.json")
    source_names = ("scripts/prepare_kinematic_features.py", "src/pdhms_restart/kinematic_features.py", "src/pdhms_restart/data.py")
    manifest = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "descriptor": descriptor["descriptor"], "shape": list(features.shape), "dtype": str(features.dtype),
        "dataset_root": str(dataset), "source_metadata": str(args.metadata.resolve()),
        "source_metadata_sha256": sha256_file(args.metadata),
        "descriptor_source_sha256": sha256_file(args.descriptor),
        "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in source_names},
        "input_sha256": source_hashes,
        "artifact_sha256": {name: sha256_file(output / name) for name in artifact_names},
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas")},
        "models_fitted": 0,
        "labels": "copied unchanged into metadata only; extraction accepts only seven-column signal arrays",
        "units": descriptor["unit_evidence"],
        "summary": summary,
    }
    write_json(manifest, output / "manifest.json")
    print(json.dumps({"status": "complete", "shape": list(features.shape), "output": str(output), "totals": totals}, indent=2), flush=True)


if __name__ == "__main__":
    main()
