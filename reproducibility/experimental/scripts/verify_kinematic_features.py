"""Audit the descriptor bundle without fitting or calling its feature extractor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pdhms_restart.data import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "experiments/2026-09-restart/runs/competitive-task-subset-v1/kinematic_features")
    args = parser.parse_args()
    output = args.output_root.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    checks = 0

    def require(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    require(manifest["status"] == "complete", "Feature bundle incomplete.")
    for name, expected in manifest["artifact_sha256"].items():
        require(sha256_file(output / name) == expected, f"Artifact hash: {name}")
    for name, expected in manifest["source_sha256"].items():
        require(sha256_file(ROOT / name) == expected, f"Source hash: {name}")
    source_metadata = Path(manifest["source_metadata"])
    require(sha256_file(source_metadata) == manifest["source_metadata_sha256"], "Static metadata hash.")
    metadata = pd.read_csv(output / "metadata.csv", dtype={"subject_id": str})
    reference = pd.read_csv(source_metadata, dtype={"subject_id": str})
    keys = ["subject_id", "task_id", "repetition_id", "label"]
    require(metadata[keys].equals(reference[keys]), "Metadata row alignment and labels.")
    require(metadata.row_index.tolist() == list(range(597)), "Explicit consecutive row index.")
    require(metadata.subject_id.nunique() == 75, "Participant count.")
    require(metadata.groupby("task_id").size().to_dict() == {1: 72, **{task: 75 for task in range(2, 9)}}, "Task availability.")
    features = np.load(output / "features.npy", allow_pickle=False)
    names = json.loads((output / "feature_names.json").read_text(encoding="utf-8"))
    lookup = {name: index for index, name in enumerate(names)}
    require(features.shape == (597, 83) and features.dtype == np.dtype("float64"), "Feature shape/dtype.")
    require(bool(np.isfinite(features).all()), "Feature finiteness.")
    require(len(names) == len(set(names)) == 83, "Unique descriptor names.")
    diagnostics = pd.read_csv(output / "diagnostics.csv", dtype={"subject_id": str})
    require(metadata[keys[:-1]].equals(diagnostics[keys[:-1]]), "Diagnostic row alignment.")
    max_duration_error, max_summary_error = 0.0, 0.0
    for row in metadata.itertuples(index=False):
        source = Path(manifest["dataset_root"]) / row.source_relative_path
        require(sha256_file(source) == manifest["input_sha256"][row.source_relative_path] == row.source_sha256,
                f"Input hash: {row.source_relative_path}")
        points = np.loadtxt(source, skiprows=1, ndmin=2)
        intervals = points[1:, 2] - points[:-1, 2]
        valid = (intervals > 0) & (intervals <= 60_000)
        contact = points[:, 3] == 1
        transition = contact[:-1] != contact[1:]
        contact_edges = valid & contact[:-1] & contact[1:]
        air_edges = valid & ~contact[:-1] & ~contact[1:]
        t_total = intervals[valid].sum() / 1000.0
        t_transition = intervals[valid & transition].sum() / 1000.0
        expected_times = {
            "valid_acquisition_time_s": t_total,
            "contact_acquisition_time_s": intervals[contact_edges].sum() / 1000.0 + t_transition / 2,
            "air_acquisition_time_s": intervals[air_edges].sum() / 1000.0 + t_transition / 2,
            "transition_acquisition_time_s": t_transition,
        }
        for name, expected in expected_times.items():
            error = abs(features[row.row_index, lookup[name]] - expected)
            max_duration_error = max(max_duration_error, error)
            require(bool(np.isclose(features[row.row_index, lookup[name]], expected, rtol=1e-12, atol=1e-12)),
                    f"Independent duration: row{row.row_index} {name}")
        distances = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
        raw_distributions = {
            "contact_speed_native_per_s": distances[contact_edges] / (intervals[contact_edges] / 1000.0),
            "air_speed_native_per_s": distances[air_edges] / (intervals[air_edges] / 1000.0),
            "contact_pressure_native": points[contact, 6],
            "contact_absolute_pressure_rate_native_per_s": np.abs(np.diff(points[:, 6])[contact_edges]) / (intervals[contact_edges] / 1000.0),
        }
        for distribution, values in raw_distributions.items():
            if len(values):
                p05, p25, median, p75, p95 = np.quantile(values, [.05, .25, .5, .75, .95])
                expected = [values.mean(), values.std(ddof=0), median, p05, p95, p75 - p25]
            else:
                expected = [0.0] * 6
            observed = [features[row.row_index, lookup[f"{distribution}__{stat}"]]
                        for stat in ("mean", "std", "median", "p05", "p95", "iqr")]
            max_summary_error = max(max_summary_error, float(np.max(np.abs(np.array(observed) - expected))))
            require(bool(np.allclose(observed, expected, rtol=1e-12, atol=1e-10)),
                    f"Independent raw summary: row{row.row_index} {distribution}")
        require(int(np.count_nonzero(~valid)) == int(diagnostics.loc[row.row_index, "excluded_intervals"]),
                f"Excluded interval count: row{row.row_index}")
    summary = manifest["summary"]
    require(summary["timestamp_diagnostics_totals"]["excluded_intervals"] == 255, "Observed pre-fit anomaly count.")
    require(summary["timestamp_diagnostics_totals"]["excluded_contact_intervals"] == 0, "No excluded contact intervals.")
    report = {
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "input_hashes_checked": len(metadata), "records_audited": len(metadata),
        "independent_features_checked_per_record": 28,
        "maximum_duration_absolute_error": max_duration_error,
        "maximum_raw_summary_absolute_error": max_summary_error,
        "new_model_fits": 0, "extractor_called": False,
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "verifier_source_sha256": sha256_file(Path(__file__)),
        "limits": "Acceleration and jerk are checked by analytical synthetic tests; the raw-record audit independently recomputes durations, speed, pressure and pressure rate.",
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
