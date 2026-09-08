"""Freeze and extract label-independent ResNet-18 features for nine existing encodings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import PIL
import torch
import torchvision

from pdhms_restart.data import sha256_file
from pdhms_restart.transfer_features import (
    FEATURE_DIMENSION, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    OFFICIAL_HASH_PREFIX, WEIGHTS_NAME, WEIGHTS_URL, array_sha256,
    configure_determinism, extract_features, load_frozen_resnet18, model_state_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/2026-09-restart/runs"
DEFAULT_OUTPUT = RUNS / "competitive-task-subset-v1/transfer_features"
KEYS = ["subject_id", "task_id", "repetition_id"]
ENCODINGS = ["static", "speed", "pressure", "altitude", "azimuth",
             "speed_pressure_altitude", "speed_pressure_azimuth",
             "speed_altitude_azimuth", "pressure_altitude_azimuth"]
CODE_SOURCES = [ROOT / "scripts/prepare_transfer_features.py",
                ROOT / "src/pdhms_restart/transfer_features.py",
                ROOT / "src/pdhms_restart/lpq.py",
                ROOT / "src/pdhms_restart/data.py",
                ROOT / "tests/test_transfer_features.py"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def default_config(output: Path) -> dict:
    checkpoint = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    weight_hash = sha256_file(checkpoint)
    if not weight_hash.startswith(OFFICIAL_HASH_PREFIX):
        raise ValueError("The local checkpoint does not match the official filename hash prefix.")
    return {
        "schema_version": 1, "experiment_id": "competitive-task-subset-v1",
        "stage": "label_independent_frozen_transfer_feature_extraction",
        "output_root": str(output.resolve()),
        "encoding_order": ENCODINGS, "expected_rows_per_encoding": 597,
        "row_order": "exactly static-svm-v1/features/metadata.csv; subject_id retained as text",
        "model": {"architecture": "torchvision.models.resnet18", "weights": WEIGHTS_NAME,
                  "weights_path": str(checkpoint), "weights_url": WEIGHTS_URL,
                  "weights_sha256": weight_hash, "output": "global average pool; fc replaced by Identity",
                  "feature_dimension": FEATURE_DIMENSION, "trainable_backbone_parameters": 0,
                  "mode": "eval", "batchnorm_updates": False, "fine_tuning": False},
        "preprocess": {"implementation": "pdhms_restart.lpq.preprocess(rgb, output_size=224)",
                       "foreground": "any RGB channel < 250", "bounding_box_margin_fraction": 0.05,
                       "padding": "white square, centred, integer floor offsets",
                       "resize_size": IMAGE_SIZE, "resize": "PIL.Image.Resampling.BILINEAR",
                       "centre_crop": False, "augmentation": False, "input_mode": "RGB uint8",
                       "rescale": "float32 / 255", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
                       "cohort_fitted_statistics": False},
        "execution": {"device": "cuda", "batch_size": 32, "cpu_threads": 4,
                      "seed": 20260908, "dtype": "float32", "autocast": False, "tf32": False,
                      "deterministic_algorithms": True, "cudnn_benchmark": False,
                      "cudnn_deterministic": True, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                      "repeat_first_batch_per_encoding_bitwise": True,
                      "require_model_state_unchanged_after_each_encoding": True},
        "scope": "Frozen ImageNet transfer. No PaHaW backbone optimization, labels, classifier fitting, "
                 "feature standardization or encoding selection enter extraction. A separately nested "
                 "training-only logistic head and encoding selection belong to the downstream experiment.",
        "sources": ["https://docs.pytorch.org/vision/0.26/models/generated/torchvision.models.resnet18.html",
                    "https://docs.pytorch.org/docs/2.11/notes/randomness.html", WEIGHTS_URL],
    }


def metadata_inventory() -> tuple[Path, pd.DataFrame, dict[str, tuple[Path, pd.DataFrame]]]:
    static_path = RUNS / "static-svm-v1/features/metadata.csv"
    static = pd.read_csv(static_path, dtype={"subject_id": str})
    if len(static) != 597 or static[KEYS].duplicated().any():
        raise ValueError("The static inventory must contain 597 unique recording keys.")
    expected = list(static[[*KEYS, "label"]].itertuples(index=False, name=None))
    inventory = {}
    for encoding in ENCODINGS:
        if encoding == "static":
            path = static_path
        else:
            run = "dynamic-single-signal-v1" if encoding in ENCODINGS[1:5] else "dynamic-triplet-svm-v1"
            path = RUNS / run / "features" / f"{encoding}_metadata.csv"
        metadata = pd.read_csv(path, dtype={"subject_id": str})
        if list(metadata[[*KEYS, "label"]].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"Inventory or row order differs from static metadata: {encoding}")
        if metadata.image_sha256.isna().any() or not metadata.image_sha256.str.fullmatch(r"[a-f0-9]{64}").all():
            raise ValueError(f"Missing or invalid original image hash: {encoding}")
        if "encoding" in metadata and not metadata.encoding.eq(encoding).all():
            raise ValueError(f"Unexpected encoding in {path}")
        inventory[encoding] = path, metadata
    return static_path, static, inventory


def runtime_manifest() -> dict:
    source = Path(inspect.getsourcefile(inspect.unwrap(torchvision.models.resnet18)))
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "pillow": PIL.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0), "device_capability": torch.cuda.get_device_capability(0),
            "torchvision_resnet_source": str(source), "torchvision_resnet_source_sha256": sha256_file(source),
            "torch_build_configuration": torch.__config__.show(),
            "determinism_scope": "Identical inputs and batches on this recorded hardware/software stack; "
                                 "no claim of bitwise equality across devices or library releases."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, help="An already frozen extraction config; settings must match this protocol.")
    parser.add_argument("--benchmark-only", action="store_true", help="Infer twice on the first static batch; write no artifacts.")
    args = parser.parse_args()
    output = args.output_root.resolve()
    expected = default_config(output)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else expected
    if config != expected:
        raise ValueError("Configuration differs from the declared extraction protocol or current checkpoint hash.")
    configure_determinism(config["execution"]["seed"], config["execution"]["cpu_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen run requires its declared CUDA device; no silent CPU fallback.")
    static_path, static, inventory = metadata_inventory()
    model = load_frozen_resnet18(Path(config["model"]["weights_path"]), config["model"]["weights_sha256"])
    if args.benchmark_only:
        count = config["execution"]["batch_size"]
        start = time.perf_counter()
        result = extract_features(model, [Path(path) for path in static.image_path.iloc[:count]],
                                  device="cuda", batch_size=count,
                                  expected_image_hashes=static.image_sha256.iloc[:count].tolist())
        print(json.dumps({"benchmark_only": True, "images": count, "shape": list(result.features.shape),
                          "elapsed_seconds": time.perf_counter() - start,
                          "first_batch_bitwise_repeatable": result.first_batch_bitwise_repeatable,
                          "model_state_unchanged": result.model_state_sha256_before == result.model_state_sha256_after}), flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty extraction directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "manifests/config.json"
    save_json(config, config_path)
    shutil.copy2(static_path, output / "metadata.csv")
    snapshot = output / "manifests/source_snapshot"
    for source in CODE_SOURCES:
        destination = snapshot / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = {
        "schema_version": 1, "status": "running", "created_at_utc": utc_now(),
        "config_sha256": sha256_file(config_path),
        "source_git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {source.relative_to(ROOT).as_posix(): sha256_file(source) for source in CODE_SOURCES},
        "input_metadata_sha256": {str(path): sha256_file(path) for path, _ in inventory.values()},
        "shared_metadata_sha256": sha256_file(output / "metadata.csv"),
        "weight_file_sha256": config["model"]["weights_sha256"],
        "backbone_state_sha256": model_state_sha256(model),
        "backbone_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_backbone_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "runtime": runtime_manifest(), "encoding_order": ENCODINGS,
        "feature_schema": {"file": "features/{encoding}_resnet18_imagenet1k_v1.npz",
                           "X": {"shape": [597, 512], "dtype": "float32"},
                           "record_keys": "Unicode vector of subject_id|task_id|repetition_id; row order shared with metadata.csv",
                           "feature_names": "Unicode vector, resnet18_gap_000 through resnet18_gap_511"},
        "completed_encodings": {},
    }
    manifest["feature_schema_sha256"] = hashlib.sha256(
        json.dumps(manifest["feature_schema"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = output / "manifests/feature_manifest.json"
    save_json(manifest, manifest_path)
    print(json.dumps({"status": "frozen_before_extraction", "config_sha256": manifest["config_sha256"],
                      "feature_schema_sha256": manifest["feature_schema_sha256"],
                      "weight_file_sha256": manifest["weight_file_sha256"],
                      "backbone_state_sha256": manifest["backbone_state_sha256"],
                      "feature_schema": manifest["feature_schema"]}), flush=True)
    rows = []
    record_keys = np.asarray([f"{subject}|{task}|{repetition}" for subject, task, repetition
                              in static[KEYS].itertuples(index=False, name=None)])
    names = np.asarray([f"resnet18_gap_{i:03d}" for i in range(FEATURE_DIMENSION)])
    (output / "features").mkdir()
    try:
        for encoding, (_, metadata) in inventory.items():
            start = time.perf_counter()
            result = extract_features(model, [Path(path) for path in metadata.image_path], device="cuda",
                                      batch_size=config["execution"]["batch_size"],
                                      expected_image_hashes=metadata.image_sha256.tolist())
            if result.features.shape != (597, 512) or result.features.dtype != np.float32:
                raise AssertionError(f"Unexpected extracted feature schema for {encoding}.")
            if result.model_state_sha256_after != manifest["backbone_state_sha256"]:
                raise AssertionError("Backbone changed between encodings.")
            path = output / "features" / f"{encoding}_resnet18_imagenet1k_v1.npz"
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, X=result.features, record_keys=record_keys, feature_names=names)
            temporary.replace(path)
            for record in result.image_inputs:
                row_index = record["row_index"]
                rows.append({"encoding": encoding, "record_key": record_keys[row_index],
                             **{key: metadata.iloc[row_index][key] for key in KEYS}, **record})
            manifest["completed_encodings"][encoding] = {
                "path": path.relative_to(output).as_posix(), "sha256": sha256_file(path),
                "array_sha256": array_sha256(result.features), "shape": list(result.features.shape),
                "dtype": str(result.features.dtype), "elapsed_seconds": time.perf_counter() - start,
                "first_batch_bitwise_repeatable": result.first_batch_bitwise_repeatable,
                "model_state_sha256_before": result.model_state_sha256_before,
                "model_state_sha256_after": result.model_state_sha256_after,
            }
            save_json(manifest, manifest_path)
            print(json.dumps({"encoding_completed": encoding, **manifest["completed_encodings"][encoding]}), flush=True)
        for source in CODE_SOURCES:
            if sha256_file(source) != manifest["code_sha256"][source.relative_to(ROOT).as_posix()]:
                raise RuntimeError(f"Source changed during extraction: {source}")
        for path, _ in inventory.values():
            if sha256_file(path) != manifest["input_metadata_sha256"][str(path)]:
                raise RuntimeError(f"Metadata changed during extraction: {path}")
        image_manifest_path = output / "manifests/image_inputs.csv"
        pd.DataFrame(rows).to_csv(image_manifest_path, index=False)
        manifest.update({"status": "complete", "completed_at_utc": utc_now(),
                         "image_input_count": len(rows), "image_inputs_sha256": sha256_file(image_manifest_path)})
        save_json(manifest, manifest_path)
    except Exception as error:
        manifest.update({"status": "failed", "failed_at_utc": utc_now(), "error": repr(error)})
        save_json(manifest, manifest_path)
        raise
    print(json.dumps({"status": "complete", "encodings": len(ENCODINGS), "image_input_count": len(rows),
                      "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
