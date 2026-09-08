"""Prepare label-independent spatial controls and LPQ features without fitting models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
from PIL import Image

from pdhms_restart.data import Sample, index_samples, load_svc, sha256_file
from pdhms_restart.dynamic_image import DynamicRenderConfig
from pdhms_restart.lpq import extract
from pdhms_restart.spatial_controls import render_spatial_control_array


ROOT = Path(__file__).resolve().parents[1]
KEYS = ["subject_id", "task_id", "repetition_id"]
CODE_SOURCES = [
    ROOT / "scripts/prepare_spatial_control_features.py",
    *[ROOT / "src/pdhms_restart" / name for name in
      ("__init__.py", "spatial_controls.py", "dynamic_image.py", "static_image.py", "lpq.py", "data.py")],
]


def save_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def render_config_from(source: dict) -> DynamicRenderConfig:
    values = dict(source["render"])
    accepted = {field.name for field in fields(DynamicRenderConfig)}
    descriptive = {"channel", "normalization", "speed_transform", "azimuth_transform", "include_air",
                   "coordinate_frame", "channel_order_rationale"}
    unknown = set(values) - accepted - descriptive
    if unknown:
        raise ValueError(f"Unrecognized rendering settings: {unknown}")
    values = {key: value for key, value in values.items() if key in accepted}
    for name in ("constant_channels", "background_rgb"):
        if name in values:
            values[name] = tuple(values[name])
    return DynamicRenderConfig(**values)


def sample_key(sample: Sample) -> tuple[str, int, int]:
    return sample.subject_id, sample.task_id, sample.repetition_id


def prepare_record(
    sample: Sample,
    dataset_root: Path,
    encoding: str,
    signals: tuple[str, ...],
    rendering: DynamicRenderConfig,
    control: str,
    seed: int,
    unit: str,
    output: Path,
    save_example: bool,
    original_image_path: str,
    original_feature: np.ndarray | None,
) -> tuple[np.ndarray, dict, dict]:
    _, points = load_svc(sample.path)
    # This key is derived from the recording path, without using Sample.label.
    record_key = sample.path.relative_to(dataset_root).as_posix()
    identity_verified = False
    if original_feature is not None:
        identity, identity_diagnostics = render_spatial_control_array(
            points, signals, rendering, control="identity", record_key=record_key, seed=seed
        )
        with Image.open(original_image_path) as image:
            expected_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(identity, expected_image):
            raise AssertionError(f"Original image differs from identity: {encoding}, {record_key}")
        if not np.array_equal(extract(identity), original_feature):
            raise AssertionError(f"Original LPQ differs from identity: {encoding}, {record_key}")
        identity_geometry = identity_diagnostics.geometry_sha256
        identity_verified = True
    else:
        identity_geometry = None

    rgb, diagnostics = render_spatial_control_array(
        points, signals, rendering, control=control, record_key=record_key, seed=seed
    )
    summary = diagnostics.summary()
    if not all(summary[name] for name in ("surface_mask_preserved", "non_surface_rows_unchanged",
                                         "joint_vectors_preserved_per_stroke")):
        raise AssertionError(f"Spatial invariant failed: {unit}, {record_key}")
    if identity_geometry is not None and diagnostics.geometry_sha256 != identity_geometry:
        raise AssertionError(f"Drawing geometry changed: {unit}, {record_key}")
    feature = extract(rgb)
    image_path = ""
    image_sha256 = ""
    if save_example:
        path = output / "images" / unit / f"task_{sample.task_id:02d}" / (sample.path.stem + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(path, format="PNG", optimize=True)
        image_path = str(path)
        image_sha256 = sha256_file(path)
    row = {
        "encoding": encoding, "control": control, "control_seed": seed,
        "subject_id": sample.subject_id, "task_id": sample.task_id, "repetition_id": sample.repetition_id,
        "label": sample.label, "source_path": str(sample.path), "record_key": record_key,
        "image_path": image_path, "image_sha256": image_sha256,
        "identity_verified_against_original": identity_verified,
    }
    summary.update({key: row[key] for key in ("encoding", "control_seed", *KEYS)})
    summary["identity_verified_against_original"] = identity_verified
    return feature, row, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = (ROOT / config["controls_output_root"]).resolve()
    dataset_root = Path(config["dataset_root"]).resolve()
    methods = config["controls"]["methods"]
    seeds = config["controls"]["seeds"]
    signals_by_encoding = {key: tuple(value) for key, value in config["signal_channels"].items()}
    encoding_order = [key for key in config["encoding_order"] if key != "static"]
    if set(encoding_order) != set(signals_by_encoding) or len(encoding_order) != 8:
        raise ValueError("The declared campaign must contain eight dynamic encodings.")
    if methods != ["circular_shift", "joint_permutation"] or len(seeds) != len(set(seeds)):
        raise ValueError("Expected the two declared controls and distinct seeds.")
    if config["features"] != {"descriptor": "LPQ_RGB", "dimension": 768, "preprocess_size": 64, "window_size": 5}:
        raise ValueError("The original LPQ settings must remain fixed.")
    render_path = (ROOT / config["render_config"]).resolve()
    rendering = render_config_from(json.loads(render_path.read_text(encoding="utf-8")))
    originals = {"single": (ROOT / config["single_signal_run"]).resolve(),
                 "triplet": (ROOT / config["triplet_run"]).resolve()}
    original_config_paths = {
        name: ROOT / "experiments/2026-09-restart" / path.name / "config.json"
        for name, path in originals.items()
    }
    for path in original_config_paths.values():
        original_render = render_config_from(json.loads(path.read_text(encoding="utf-8")))
        if asdict(original_render) != asdict(rendering):
            raise ValueError(f"The campaign rendering differs from {path}.")

    samples = [sample for sample in index_samples(dataset_root) if sample.task_id in config["tasks"]]
    samples = sorted(samples, key=lambda sample: (sample.task_id, sample.subject_id, sample.repetition_id))
    if len({sample_key(sample) for sample in samples}) != len(samples):
        raise ValueError("Duplicate recording keys.")
    expected_rows = [(sample.subject_id, sample.task_id, sample.repetition_id, sample.label) for sample in samples]
    original_metadata = {}
    original_feature_paths = {}
    reference_paths = [config_path, render_path, *original_config_paths.values(),
                       dataset_root / "PaHaW_files/corpus_PaHaW.xlsx"]
    for encoding in encoding_order:
        run = originals["single" if len(signals_by_encoding[encoding]) == 1 else "triplet"]
        metadata_path = run / "features" / f"{encoding}_metadata.csv"
        feature_path = run / "features" / f"{encoding}_lpq_rgb.npz"
        metadata = pd.read_csv(metadata_path, dtype={"subject_id": str})
        rows = list(metadata[[*KEYS, "label"]].itertuples(index=False, name=None))
        if rows != expected_rows:
            raise ValueError(f"Original row order/inventory differs for {encoding}.")
        original_metadata[encoding] = metadata
        original_feature_paths[encoding] = feature_path
        reference_paths.extend([metadata_path, feature_path])

    # Select four identifiers having both panel tasks, without diagnosis or scores.
    by_task = {task: {sample.subject_id for sample in samples if sample.task_id == task} for task in (1, 4)}
    selected_subjects = sorted(by_task[1] & by_task[4])[:4]
    panel_keys = {sample_key(sample) for sample in samples
                  if sample.subject_id in selected_subjects and sample.task_id in (1, 4)}
    signature = {
        "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in CODE_SOURCES},
        "reference_sha256": {str(path): sha256_file(path) for path in sorted(set(reference_paths))},
        "recording_sha256": {sample.path.relative_to(dataset_root).as_posix(): sha256_file(sample.path) for sample in samples},
    }
    manifest_path = output / "manifests/feature_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not config["execution"]["resume_completed_units_only_with_identical_config_and_source"]:
            raise FileExistsError(f"Output already exists: {output}")
        if manifest["frozen_sources"] != signature:
            raise ValueError("Cannot resume: source, configuration or input hashes changed.")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to use a non-empty output without its manifest: {output}")
        output.mkdir(parents=True, exist_ok=True)
        git_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        manifest = {
            "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": "spatial-controls-v1", "parent_experiment_id": config["experiment_id"],
            "source_git_commit": git_commit, "frozen_sources": signature,
            "model_fitting": False, "predictions_or_metrics_computed": False,
            "records": len(samples), "participants": len({sample.subject_id for sample in samples}),
            "encoding_order": encoding_order, "controls": methods, "seeds": seeds,
            "panel_selection": {
                "rule": "lexicographically first four identifiers with records in both T1 and T4; no diagnosis or result used",
                "subject_ids": selected_subjects, "recording_keys": [list(key) for key in sorted(panel_keys)],
            },
            "rendering": asdict(rendering), "units": {},
            "permutation_note": "Joint permutation can retain fixed points; observed moved/changed fractions are reported.",
            "pixel_note": "Geometry and segment-vector multisets are preserved; raster histograms and thresholded masks need not be.",
        }
        snapshots = output / "manifests/source_snapshot"
        for path in CODE_SOURCES:
            target = snapshots / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for name, path in [("campaign_config.json", config_path), ("render_config.json", render_path)]:
            shutil.copyfile(path, snapshots / name)
        save_json(manifest, manifest_path)
    feature_dir = output / "features"
    diagnostic_dir = output / "diagnostics"
    feature_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = int(config["execution"]["n_jobs"])
    started = time.perf_counter()
    with parallel_config(backend="loky", n_jobs=n_jobs,
                         inner_max_num_threads=int(config["execution"]["inner_threads"])):
        with Parallel() as parallel:
            for encoding in encoding_order:
                with np.load(original_feature_paths[encoding], allow_pickle=False) as archive:
                    original_features = archive["features"]
                if original_features.shape != (len(samples), 768):
                    raise ValueError(f"Unexpected original features for {encoding}: {original_features.shape}")
                for control in methods:
                    for seed in seeds:
                        unit = f"{encoding}__{control}__seed{seed}"
                        if unit in manifest["units"]:
                            for relative, digest in manifest["units"][unit]["artifact_sha256"].items():
                                if sha256_file(output / relative) != digest:
                                    raise ValueError(f"Completed unit artifact changed: {unit}, {relative}")
                            print(f"verified existing {unit}", flush=True)
                            continue
                        check_identity = control == methods[0] and seed == seeds[0]
                        results = parallel(
                            delayed(prepare_record)(
                                sample, dataset_root, encoding, signals_by_encoding[encoding], rendering,
                                control, seed, unit, output, sample_key(sample) in panel_keys,
                                str(original_metadata[encoding].iloc[index]["image_path"]),
                                original_features[index] if check_identity else None,
                            )
                            for index, sample in enumerate(samples)
                        )
                        features = np.vstack([result[0] for result in results]).astype(np.float32)
                        if features.shape != (len(samples), 768) or not np.all(np.isfinite(features)):
                            raise AssertionError(f"Invalid features: {unit}")
                        metadata = pd.DataFrame([result[1] for result in results])
                        if list(metadata[[*KEYS, "label"]].itertuples(index=False, name=None)) != expected_rows:
                            raise AssertionError(f"Parallel execution changed row order: {unit}")
                        feature_path = feature_dir / f"{unit}_lpq_rgb.npz"
                        metadata_path = feature_dir / f"{unit}_metadata.csv"
                        diagnostic_path = diagnostic_dir / f"{unit}.jsonl"
                        with feature_path.with_suffix(".npz.tmp").open("wb") as stream:
                            np.savez_compressed(stream, features=features)
                        feature_path.with_suffix(".npz.tmp").replace(feature_path)
                        metadata.to_csv(metadata_path.with_suffix(".csv.tmp"), index=False)
                        metadata_path.with_suffix(".csv.tmp").replace(metadata_path)
                        with diagnostic_path.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as stream:
                            for _, _, diagnostic in results:
                                stream.write(json.dumps(diagnostic, ensure_ascii=False, allow_nan=False) + "\n")
                        diagnostic_path.with_suffix(".jsonl.tmp").replace(diagnostic_path)
                        artifacts = [feature_path, metadata_path, diagnostic_path,
                                     *[Path(row["image_path"]) for _, row, _ in results if row["image_path"]]]
                        manifest["units"][unit] = {
                            "records": len(samples), "feature_shape": list(features.shape),
                            "identity_checks": len(samples) if check_identity else 0,
                            "all_segment_invariants_passed": True,
                            "example_images": len(panel_keys),
                            "artifact_sha256": {path.relative_to(output).as_posix(): sha256_file(path) for path in artifacts},
                        }
                        save_json(manifest, manifest_path)
                        print(f"completed {len(manifest['units'])}/48 {unit}: {features.shape}, "
                              f"identity_checks={len(samples) if check_identity else 0}, "
                              f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
                        del results

    for relative, expected in signature["code_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Source changed during generation: {relative}")
    for path, expected in signature["reference_sha256"].items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"Original reference changed during generation: {path}")
    for relative, expected in signature["recording_sha256"].items():
        if sha256_file(dataset_root / relative) != expected:
            raise RuntimeError(f"Recording changed during generation: {relative}")
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["identity_image_and_lpq_checks"] = sum(unit["identity_checks"] for unit in manifest["units"].values())
    manifest["feature_vectors"] = sum(unit["records"] for unit in manifest["units"].values())
    save_json(manifest, manifest_path)
    print(json.dumps({key: manifest[key] for key in
                      ("status", "records", "participants", "identity_image_and_lpq_checks", "feature_vectors")}), flush=True)


if __name__ == "__main__":
    main()
