"""Verify the frozen CNN run and its saved external-fold models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pdhms_restart.cnn_model import SmallHandwritingCNN, normalized_tensor, prepared_rgb
from pdhms_restart.data import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--image-run", type=Path, required=True)
    parser.add_argument("--classical-run", type=Path, required=True)
    parser.add_argument("--static-run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    image_run = args.image_run.resolve()
    classical_run = args.classical_run.resolve()
    static_run = args.static_run.resolve()
    errors: list[str] = []

    manifest = json.loads((run / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
    artifacts = json.loads(
        (run / "manifests" / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_csv(run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str})
    selections = pd.read_csv(run / "metrics" / "selected_epochs.csv")
    history = pd.read_csv(run / "metrics" / "training_history.csv")
    summary = pd.read_csv(run / "metrics" / "task_summary.csv")
    combined_summary = pd.read_csv(run / "metrics" / "combined_task_summary.csv")
    task_ranking = pd.read_csv(run / "metrics" / "combined_task_ranking.csv")
    task_pairwise = pd.read_csv(run / "metrics" / "combined_task_pairwise.csv")
    per_task = pd.read_csv(run / "metrics" / "paired_contrasts_vs_svm.csv")
    fused = pd.read_csv(run / "metrics" / "fused_majority_predictions.csv", dtype={"subject_id": str})
    fusion_summary = pd.read_csv(run / "metrics" / "fused_majority_summary.csv")
    fusion_ranking = pd.read_csv(run / "metrics" / "fused_majority_ranking.csv")
    fusion_pairwise = pd.read_csv(run / "metrics" / "fused_majority_pairwise.csv")

    if len(predictions) != 2985:
        errors.append(f"Expected 2985 CNN predictions, found {len(predictions)}.")
    if set(predictions["algorithm"]) != {"cnn"}:
        errors.append("Unexpected algorithm in CNN predictions.")
    if not predictions["probability_pd"].between(0, 1).all():
        errors.append("CNN probability outside [0, 1].")
    keys = ["repeat", "outer_fold", "task_id", "subject_id"]
    if predictions.duplicated(keys).any():
        errors.append("Duplicate CNN external prediction.")
    svm = pd.read_csv(
        classical_run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str}
    )
    svm = svm[svm["algorithm"] == "svm"]
    current_keys = predictions[keys + ["y_true"]].sort_values(keys).reset_index(drop=True)
    svm_keys = svm[keys + ["y_true"]].sort_values(keys).reset_index(drop=True)
    if not current_keys.equals(svm_keys):
        errors.append("CNN and SVM external participant folds differ.")

    if len(selections) != 200 or selections.duplicated(["repeat", "outer_fold", "task_id"]).any():
        errors.append("Expected exactly 200 selected epoch/model rows.")
    if not selections["selected_epoch"].between(1, 60).all():
        errors.append("Selected epoch outside declared bounds.")
    if len(summary) != 8 or len(combined_summary) != 32:
        errors.append("Unexpected task summary dimensions.")
    if len(task_ranking) != 4 or len(task_pairwise) != 6 or len(per_task) != 8:
        errors.append("Unexpected classifier comparison dimensions.")
    if history.empty or set(history["phase"]) != {"epoch_selection", "final_fit"}:
        errors.append("Training history is incomplete.")
    selected_history = history[
        (history["phase"] == "epoch_selection")
        & history["is_selected_epoch"].astype(str).str.lower().eq("true")
    ]
    if len(selected_history) != 200:
        errors.append("Training history lacks one selected epoch per model.")

    missing_models = []
    bad_model_hashes = []
    for row in selections.itertuples(index=False):
        path = Path(row.checkpoint_path)
        if not path.is_file():
            missing_models.append(str(path))
        elif sha256_file(path) != row.checkpoint_sha256:
            bad_model_hashes.append(str(path))
    if missing_models:
        errors.append(f"Missing {len(missing_models)} model checkpoints.")
    if bad_model_hashes:
        errors.append(f"Changed {len(bad_model_hashes)} model checkpoints.")

    if len(fused) != 1500 or not (fused.groupby("algorithm").size() == 375).all():
        errors.append("Unexpected fused majority prediction count.")
    if set(fused["available_task_count"]) != {7, 8}:
        errors.append("Unexpected number of tasks in participant fusion.")
    if len(fusion_summary) != 4 or len(fusion_ranking) != 4 or len(fusion_pairwise) != 6:
        errors.append("Unexpected fusion comparison dimensions.")

    if not missing_models and not bad_model_hashes:
        first = selections.sort_values(["repeat", "outer_fold", "task_id"]).iloc[0]
        checkpoint = torch.load(first["checkpoint_path"], map_location="cpu", weights_only=True)
        model = SmallHandwritingCNN(dropout=float(checkpoint["architecture"]["dropout"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        metadata = pd.read_csv(
            image_run / "features" / f"{manifest['selected_encoding']}_metadata.csv",
            dtype={"subject_id": str},
        )
        external = predictions[
            (predictions["repeat"] == int(first["repeat"]))
            & (predictions["outer_fold"] == int(first["outer_fold"]))
            & (predictions["task_id"] == int(first["task_id"]))
        ].sort_values("subject_id")
        task_metadata = metadata[metadata["task_id"] == int(first["task_id"])].copy()
        task_metadata["subject_id"] = task_metadata["subject_id"].astype(str).str.zfill(5)
        task_metadata = task_metadata.set_index("subject_id").loc[external["subject_id"]]
        tensors = torch.stack(
            [
                normalized_tensor(
                    prepared_rgb(Path(path), int(checkpoint["input"]["image_size"]))
                )
                for path in task_metadata["image_path"]
            ]
        )
        with torch.no_grad():
            reproduced = torch.sigmoid(model(tensors)).numpy()
        if not np.allclose(reproduced, external["probability_pd"].to_numpy(), atol=2e-5):
            errors.append("Saved model does not reproduce its stored external probabilities.")

    static_splits = json.loads(
        (static_run / "metrics" / "cv_splits.json").read_text(encoding="utf-8")
    )
    if manifest["reused_cv_split_checksum"] != static_splits["checksum"]:
        errors.append("CNN run does not match frozen participant splits.")
    bad_artifacts = []
    for artifact in artifacts["artifacts"]:
        path = run / artifact["relative_path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            bad_artifacts.append(artifact["relative_path"])
    if bad_artifacts:
        errors.append(f"Missing or changed run artifacts: {bad_artifacts}.")

    result = {
        "status": "passed" if not errors else "failed",
        "run": str(run),
        "outer_predictions": len(predictions),
        "saved_models": len(selections),
        "training_history_rows": len(history),
        "combined_task_methods": len(task_ranking),
        "fused_predictions": len(fused),
        "artifact_hashes_checked": len(artifacts["artifacts"]),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
