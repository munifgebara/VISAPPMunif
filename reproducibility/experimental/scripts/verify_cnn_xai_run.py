"""Verify maps, sample linkage, and hashes for the frozen CNN XAI run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pdhms_restart.cnn_model import SmallHandwritingCNN, normalized_tensor, prepared_rgb
from pdhms_restart.data import sha256_file
from pdhms_restart.xai import gradcam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cnn-run", type=Path, required=True)
    parser.add_argument("--image-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    cnn_run = args.cnn_run.resolve()
    image_run = args.image_run.resolve()
    errors: list[str] = []

    manifest = json.loads((run / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads(
        (run / "manifests" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(run / "metrics" / "prediction_xai_metrics.csv", dtype={"subject_id": str})
    participant = pd.read_csv(
        run / "metrics" / "participant_task_xai_metrics.csv", dtype={"subject_id": str}
    )
    subgroup = pd.read_csv(run / "metrics" / "task_class_xai_summary.csv")
    occlusion = pd.read_csv(run / "metrics" / "occlusion_metrics.csv", dtype={"subject_id": str})
    sanity = pd.read_csv(run / "metrics" / "weight_randomization_sanity.csv", dtype={"subject_id": str})
    bootstrap = pd.read_csv(run / "metrics" / "global_participant_bootstrap.csv")
    source_artifacts = pd.read_csv(run / "metrics" / "source_artifacts.csv")
    with np.load(run / "maps" / "spatial_maps.npz") as payload:
        gradcam_maps = payload["gradcam_maps"]
        gradcam_index = payload["gradcam_prediction_index"]
        occlusion_maps = payload["occlusion_maps"]
        occlusion_index = payload["occlusion_prediction_index"]

    predictions = pd.read_csv(
        cnn_run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str}
    ).sort_values(["repeat", "outer_fold", "task_id", "subject_id"]).reset_index(drop=True)
    keys = ["repeat", "outer_fold", "task_id", "subject_id", "y_true", "y_pred"]
    current = metrics[keys].copy()
    expected = predictions[keys].copy()
    current["subject_id"] = current["subject_id"].astype(str).str.zfill(5)
    expected["subject_id"] = expected["subject_id"].astype(str).str.zfill(5)
    if not current.equals(expected):
        errors.append("XAI sample linkage differs from frozen external predictions.")
    if len(metrics) != 2985 or metrics["prediction_index"].nunique() != 2985:
        errors.append("Expected one Grad-CAM metric row for every external prediction.")
    if gradcam_maps.shape != (2985, 128, 128) or not np.array_equal(
        gradcam_index, np.arange(2985)
    ):
        errors.append(f"Unexpected Grad-CAM map shape/index: {gradcam_maps.shape}.")
    if not np.all(np.isfinite(gradcam_maps)) or np.any((gradcam_maps < 0) | (gradcam_maps > 1)):
        errors.append("Grad-CAM maps contain invalid values.")
    if len(occlusion) != 597 or occlusion_maps.shape != (597, 128, 128):
        errors.append("Expected 597 repeat-1 occlusion maps.")
    if set(occlusion["repeat"]) != {1} or not np.array_equal(
        occlusion_index, occlusion["prediction_index"].to_numpy()
    ):
        errors.append("Occlusion map linkage is invalid.")
    if not np.all(np.isfinite(occlusion_maps)) or np.any(
        (occlusion_maps < 0) | (occlusion_maps > 1)
    ):
        errors.append("Occlusion maps contain invalid values.")
    if len(sanity) != 200 or sanity.duplicated(["repeat", "outer_fold", "task_id"]).any():
        errors.append("Expected one randomization sanity check per checkpoint.")
    if len(participant) != 597 or len(subgroup) != 16 or len(bootstrap) != 7:
        errors.append("Unexpected aggregate XAI table dimensions.")

    bad_source_hashes = []
    for row in source_artifacts.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or sha256_file(path) != row.sha256:
            bad_source_hashes.append(str(path))
    if bad_source_hashes:
        errors.append(f"Missing or changed XAI sources: {bad_source_hashes}.")

    selections = pd.read_csv(cnn_run / "metrics" / "selected_epochs.csv")
    first_selection = selections.sort_values(["repeat", "outer_fold", "task_id"]).iloc[0]
    checkpoint = torch.load(first_selection["checkpoint_path"], map_location="cpu", weights_only=True)
    model = SmallHandwritingCNN(dropout=float(checkpoint["architecture"]["dropout"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    first_prediction = predictions[
        (predictions["repeat"] == int(first_selection["repeat"]))
        & (predictions["outer_fold"] == int(first_selection["outer_fold"]))
        & (predictions["task_id"] == int(first_selection["task_id"]))
    ].sort_values("subject_id").iloc[0]
    metadata = pd.read_csv(
        image_run / "features" / f"{manifest['selected_encoding']}_metadata.csv",
        dtype={"subject_id": str},
    )
    metadata["subject_id"] = metadata["subject_id"].astype(str).str.zfill(5)
    image_row = metadata[
        (metadata["task_id"] == int(first_prediction["task_id"]))
        & (metadata["subject_id"] == str(first_prediction["subject_id"]).zfill(5))
    ].iloc[0]
    tensor = normalized_tensor(prepared_rgb(Path(image_row["image_path"]), 128))[None]
    predicted = torch.tensor([int(first_prediction["y_pred"])], dtype=torch.int64)
    reproduced, probability = gradcam(model, tensor, predicted, model.gradcam_layer)
    stored_index = int(
        metrics[
            (metrics["repeat"] == int(first_prediction["repeat"]))
            & (metrics["outer_fold"] == int(first_prediction["outer_fold"]))
            & (metrics["task_id"] == int(first_prediction["task_id"]))
            & (metrics["subject_id"].astype(str).str.zfill(5) == str(first_prediction["subject_id"]).zfill(5))
        ]["prediction_index"].iloc[0]
    )
    if not np.allclose(reproduced[0], gradcam_maps[stored_index].astype(np.float32), atol=6e-4):
        errors.append("Saved Grad-CAM map does not reproduce from its checkpoint.")
    if not np.isclose(probability[0], float(first_prediction["probability_pd"]), atol=2e-5):
        errors.append("XAI model does not reproduce the frozen prediction probability.")

    bad_artifacts = []
    for artifact in artifacts["artifacts"]:
        path = run / artifact["relative_path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            bad_artifacts.append(artifact["relative_path"])
    if bad_artifacts:
        errors.append(f"Missing or changed XAI artifacts: {bad_artifacts}.")

    result = {
        "status": "passed" if not errors else "failed",
        "run": str(run),
        "gradcam_maps": len(gradcam_maps),
        "occlusion_maps": len(occlusion_maps),
        "randomization_checks": len(sanity),
        "participant_task_rows": len(participant),
        "artifact_hashes_checked": len(artifacts["artifacts"]),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
