"""Run the two authorized participant-nested SVM experiments locally."""

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

from pdhms_restart.data import sha256_file
from pdhms_restart.nested_selection import build_selection_plan, run_unit


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["subject_id", "task_id", "repetition_id", "label"]
FIT_SOURCES = ["scripts/run_nested_selection_controls.py", "src/pdhms_restart/nested_selection.py",
               "src/pdhms_restart/evaluation.py", "src/pdhms_restart/task_fusion.py",
               "src/pdhms_restart/data.py"]


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_features(config: dict, condition: str = "authentic", seed: int = 0) -> tuple[pd.DataFrame, dict, dict]:
    master, matrices, files = None, {}, {}
    for encoding in config["encoding_order"]:
        if condition != "authentic" and encoding == "static":
            continue
        if condition != "authentic":
            directory = ROOT / config["controls_output_root"] / "features"
            stem = f"{encoding}__{condition}__seed{seed}"
            metadata_path = directory / f"{stem}_metadata.csv"
        else:
            source = "static_run" if encoding == "static" else "single_signal_run" if encoding in ("speed", "pressure", "altitude", "azimuth") else "triplet_run"
            directory = ROOT / config[source] / "features"
            stem = encoding
            metadata_path = directory / ("metadata.csv" if encoding == "static" else f"{stem}_metadata.csv")
        features_path = directory / f"{stem}_lpq_rgb.npz"
        metadata = pd.read_csv(metadata_path, dtype={"subject_id": str})[KEYS]
        metadata.subject_id = metadata.subject_id.str.zfill(5)
        if metadata.duplicated(KEYS[:-1]).any():
            raise ValueError("Expected at most one recording per participant and task.")
        with np.load(features_path) as archive:
            matrix = archive["features"].astype(np.float32)
        if matrix.shape != (len(metadata), config["features"]["dimension"]) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid features: {features_path}")
        if master is None:
            order = metadata.sort_values(KEYS[:3]).index.to_numpy()
            master = metadata.iloc[order].reset_index(drop=True)
        else:
            alignment = master.merge(metadata.reset_index(names="position"), on=KEYS, validate="one_to_one")
            if len(alignment) != len(master) or len(metadata) != len(master):
                raise ValueError("Feature matrices refer to different recordings.")
            order = alignment.position.to_numpy()
        matrices[encoding] = matrix[order]
        for path in (metadata_path, features_path):
            files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return master, matrices, files


def establish_lock(config_path: Path, config: dict, metadata: pd.DataFrame, inputs: dict, output: Path) -> dict:
    splits_path = ROOT / config["validation"]["outer_splits_from"]
    decision_path = config_path.with_name("DECISION.md")
    lock = {"config_sha256": sha256_file(config_path), "protocol_sha256": sha256_file(decision_path),
            "source_sha256": {name: sha256_file(ROOT / name) for name in FIT_SOURCES},
            "input_sha256": {**inputs, config["validation"]["outer_splits_from"]: sha256_file(splits_path)}}
    lock_path = output / "manifests/run_lock.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        for key, value in lock.items():
            if previous[key] != value:
                raise ValueError(f"Refusing resume after changing {key}. Start a new version.")
    else:
        lock.update(started_at_utc=datetime.now(timezone.utc).isoformat(),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    python=platform.python_version(), platform=platform.platform(),
                    versions={name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")})
        save_json(lock, lock_path)
        snapshot = output / "manifests/source_snapshot"
        for name in FIT_SOURCES:
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        save_json(config, output / "manifests/config.json")
        original = json.loads(splits_path.read_text(encoding="utf-8"))
        plan = build_selection_plan(metadata, original, config)
        save_json(plan, output / "manifests/selection_plan.json")
    return json.loads((output / "manifests/selection_plan.json").read_text(encoding="utf-8"))


def aggregate_units(output: Path, config: dict) -> None:
    task, fusion, selections, searches, counts = [], [], [], [], {}
    for path in sorted((output / "units").glob("*/*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        task.extend(result["task_predictions"])
        fusion.extend(result["fusion_predictions"])
        context = {key: result[key] for key in ("repeat", "outer_fold", "condition", "control_seed")}
        selections.extend({**row, **context} for row in result["selections"])
        searches.extend({**row, **context} for row in result["fusion_search"])
        group = path.parent.parent.name
        counts[group] = counts.get(group, 0) + 1
    metrics = output / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(task).to_csv(metrics / "outer_task_predictions.csv", index=False)
    pd.DataFrame(fusion).to_csv(metrics / "outer_fusion_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(metrics / "selections.csv", index=False)
    pd.DataFrame(searches).to_csv(metrics / "fusion_search.csv", index=False)
    expected_groups = ["authentic"] + [f"{method}_seed{seed}" for method in config["controls"]["methods"] for seed in config["controls"]["seeds"]]
    complete = counts == {group: 25 for group in expected_groups}
    save_json({"status": "fits_complete" if complete else "fitting", "unit_counts": counts,
               "completed_outer_units": sum(counts.values()), "expected_outer_units": 175,
               "task_prediction_rows": len(task), "fusion_prediction_rows": len(fusion),
               "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "note": "Selection and predictions complete only when all 175 outer units exist; final analysis and verification are separate."}, output / "manifests/run_manifest.json")


def execute_group(metadata, features, plan, config, output, condition, seed):
    group = "authentic" if condition == "authentic" else f"{condition}_seed{seed}"
    jobs = []
    for outer in plan["outer_splits"]:
        unit = output / "units" / group / f"r{outer['repeat']:02}_f{outer['outer_fold']:02}"
        if not (unit / "result.json").exists():
            jobs.append((outer, unit))
    print(f"{group}: {len(jobs)} of 25 outer units pending", flush=True)
    temporary = ROOT / "tmp/nested-selection-joblib"
    temporary.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        generator = Parallel(n_jobs=config["execution"]["n_jobs"], return_as="generator_unordered", temp_folder=str(temporary))(
            delayed(run_unit)(metadata, features, outer, config, str(unit), condition, seed) for outer, unit in jobs)
        for completed, path in enumerate(generator, 1):
            print(f"{group}: completed {Path(path).parent.name}; {completed}/{len(jobs)} new units; {time.perf_counter()-started:.1f}s", flush=True)
    aggregate_units(output, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("authentic", "controls", "all"), default="all")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    metadata, features, inputs = load_features(config)
    if len(metadata) != 597 or metadata.subject_id.nunique() != 75:
        raise ValueError("The declared experiment requires the audited 597 records / 75 participants.")
    plan = establish_lock(config_path, config, metadata, inputs, output)
    if args.phase in ("authentic", "all"):
        execute_group(metadata, features, plan, config, output, "authentic", 0)
    if args.phase in ("controls", "all"):
        for condition in config["controls"]["methods"]:
            for seed in config["controls"]["seeds"]:
                control_metadata, controls, control_inputs = load_features(config, condition, seed)
                if not metadata.equals(control_metadata):
                    raise ValueError("Authentic/control records differ.")
                path = output / f"manifests/{condition}_seed{seed}_inputs.json"
                if path.exists() and json.loads(path.read_text(encoding="utf-8")) != control_inputs:
                    raise ValueError("Control inputs changed after fitting started.")
                save_json(control_inputs, path)
                execute_group(metadata, controls, plan, config, output, condition, seed)
    print("Requested fitting phase complete.", flush=True)


if __name__ == "__main__":
    main()
