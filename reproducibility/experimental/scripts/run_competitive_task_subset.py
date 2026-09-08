"""Fit the prespecified competitive pipelines on T2/T3/T4 and all eight tasks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time

from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd

from pdhms_restart.competitive_selection import run_competitive_unit
from pdhms_restart.data import sha256_file
from run_nested_selection_controls import load_features, save_json, KEYS


ROOT = Path(__file__).resolve().parents[1]
FIT_SOURCES = ["scripts/run_competitive_task_subset.py", "scripts/run_nested_selection_controls.py",
               "src/pdhms_restart/competitive_selection.py", "src/pdhms_restart/nested_selection.py",
               "src/pdhms_restart/evaluation.py", "src/pdhms_restart/data.py",
               "src/pdhms_restart/task_fusion.py"]


def align_matrix(master, metadata_path, matrix_path, inputs):
    metadata = pd.read_csv(metadata_path, dtype={"subject_id": str})[KEYS]
    metadata.subject_id = metadata.subject_id.str.zfill(5)
    alignment = master.merge(metadata.reset_index(names="position"), on=KEYS, validate="one_to_one")
    if len(alignment) != len(master) or len(metadata) != len(master):
        raise ValueError(f"Mismatched recording keys: {metadata_path}")
    if matrix_path.suffix == ".npy":
        matrix = np.load(matrix_path, allow_pickle=False)
    else:
        with np.load(matrix_path, allow_pickle=False) as archive:
            key = "X" if "X" in archive else "features"
            matrix = archive[key]
    if len(matrix) != len(metadata) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid feature matrix: {matrix_path}")
    for path in (metadata_path, matrix_path):
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return matrix[alignment.position.to_numpy()].astype(np.float64)


def load_competitive_inputs(config):
    metadata, lpq, inputs = load_features(config)
    kinematic = ROOT / config["kinematic_directory"]
    transfer = ROOT / config["transfer_directory"]
    matrix = align_matrix(metadata, kinematic / "metadata.csv", kinematic / "features.npy", inputs)
    if matrix.shape != (597, 83):
        raise ValueError(f"Expected 597x83 kinematics, got {matrix.shape}.")
    transferred = {}
    for encoding in config["encoding_order"]:
        path = transfer / "features" / f"{encoding}_resnet18_imagenet1k_v1.npz"
        transferred[encoding] = align_matrix(metadata, transfer / "metadata.csv", path, inputs)
        if transferred[encoding].shape != (597, 512):
            raise ValueError("Expected 597x512 CNN features.")
    for path in (kinematic / "manifest.json", transfer / "manifests/feature_manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return metadata, {"lpq": lpq, "kinematic": {"kinematic": matrix}, "transfer": transferred}, inputs


def establish_lock(config_path, config, metadata, inputs, output):
    plan_path = ROOT / config["validation"]["selection_plan_from"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_subjects = metadata[["subject_id", "label"]].drop_duplicates().sort_values("subject_id")
    actual_subjects = pd.DataFrame(plan["subjects"]).sort_values("subject_id")
    if expected_subjects.to_dict("records") != actual_subjects.to_dict("records"):
        raise ValueError("Participant plan differs from feature metadata.")
    if len(metadata) != 597 or len(expected_subjects) != 75 or len(plan["outer_splits"]) != 25:
        raise ValueError("Unexpected cohort or split counts.")
    protocol = config_path.with_name("DECISION.md")
    inputs[plan_path.relative_to(ROOT).as_posix()] = sha256_file(plan_path)
    split_path = ROOT / config["validation"]["outer_splits_from"]
    inputs[split_path.relative_to(ROOT).as_posix()] = sha256_file(split_path)
    lock = {"config_sha256": sha256_file(config_path), "protocol_sha256": sha256_file(protocol),
            "source_sha256": {name: sha256_file(ROOT / name) for name in FIT_SOURCES},
            "input_sha256": inputs}
    lock_path = output / "manifests/run_lock.json"
    if lock_path.exists():
        old = json.loads(lock_path.read_text(encoding="utf-8"))
        for name, value in lock.items():
            if old[name] != value:
                raise ValueError(f"Refusing to resume with changed {name}.")
    else:
        lock.update(started_at_utc=datetime.now(timezone.utc).isoformat(),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    python=platform.python_version(),
                    versions={name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")})
        save_json(lock, lock_path)
        save_json(config, output / "manifests/config.json")
        save_json(plan, output / "manifests/selection_plan.json")
        for name in FIT_SOURCES:
            path = output / "manifests/source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        (output / "manifests/DECISION.md").write_bytes(protocol.read_bytes())
    return plan


def aggregate(output):
    units = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "units").glob("*/result.json"))]
    destination = output / "metrics"
    destination.mkdir(parents=True, exist_ok=True)
    for key, name in (("task_predictions", "outer_task_predictions.csv"),
                       ("fusion_predictions", "outer_fusion_predictions.csv")):
        pd.DataFrame([row for unit in units for row in unit[key]]).to_csv(destination / name, index=False)
    selections = []
    for unit in units:
        for row in unit["selections"]:
            selections.append({"repeat": unit["repeat"], "outer_fold": unit["outer_fold"],
                               **{key: value for key, value in row.items() if key not in ("tasks", "encoding_scores")}})
    pd.DataFrame(selections).to_csv(destination / "selections.csv", index=False)
    save_json({"status": "fits_complete" if len(units) == 25 else "fitting",
               "completed_outer_units": len(units), "expected_outer_units": 25,
               "tuning_fits": sum(unit["tuning_fits"] for unit in units),
               "final_fits": sum(unit["final_fits"] for unit in units),
               "task_prediction_rows": sum(len(unit["task_predictions"]) for unit in units),
               "fusion_prediction_rows": sum(len(unit["fusion_predictions"]) for unit in units),
               "updated_at_utc": datetime.now(timezone.utc).isoformat()}, output / "manifests/run_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    metadata, features, inputs = load_competitive_inputs(config)
    plan = establish_lock(config_path, config, metadata, inputs, output)
    jobs = [(outer, output / "units" / f"r{outer['repeat']:02}_f{outer['outer_fold']:02}")
            for outer in plan["outer_splits"]]
    pending = [(outer, directory) for outer, directory in jobs if not (directory / "result.json").exists()]
    started = time.perf_counter()
    print(f"{len(pending)}/25 outer units pending", flush=True)
    temporary = ROOT / "tmp/competitive-joblib"
    temporary.mkdir(parents=True, exist_ok=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=config["execution"]["n_jobs"], return_as="generator_unordered", temp_folder=str(temporary))(
            delayed(run_competitive_unit)(metadata, features, outer, config, str(directory)) for outer, directory in pending)
        for count, path in enumerate(results, 1):
            print(f"Completed {Path(path).parent.name}; {count}/{len(pending)} new units; {time.perf_counter()-started:.1f}s", flush=True)
    aggregate(output)
    print("All declared fitting units complete; analysis and independent verification remain.", flush=True)


if __name__ == "__main__":
    main()
