"""Explain every frozen external CNN prediction with spatial and channel checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from pdhms_restart.cnn_model import SmallHandwritingCNN, normalized_tensor, prepared_rgb
from pdhms_restart.data import sha256_file
from pdhms_restart.xai import (
    channel_ablation_drops,
    deletion_drops,
    gradcam,
    occlusion_map,
    top_fraction_iou,
)


SIGNALS = ("speed", "altitude", "azimuth")


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def artifact_manifest(output_root: Path) -> dict[str, object]:
    target = output_root / "manifests" / "artifact_manifest.json"
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != target:
            artifacts.append(
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"artifact_count": len(artifacts), "artifacts": artifacts}


def cache_images(metadata: pd.DataFrame, image_size: int) -> np.ndarray:
    images = []
    for index, path in enumerate(metadata["image_path"], start=1):
        images.append(prepared_rgb(Path(path), image_size))
        if index % 100 == 0 or index == len(metadata):
            print(f"prepared images {index}/{len(metadata)}", flush=True)
    return np.stack(images).astype(np.uint8)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SmallHandwritingCNN, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SmallHandwritingCNN(dropout=float(checkpoint["architecture"]["dropout"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def attention_metrics(attribution: np.ndarray, foreground: np.ndarray) -> tuple[float, float, float]:
    area_fraction = float(foreground.mean())
    total = float(attribution.sum())
    attention_fraction = float(attribution[foreground].sum() / total) if total > 0 else float("nan")
    enrichment = attention_fraction / area_fraction if area_fraction > 0 else float("nan")
    return area_fraction, attention_fraction, enrichment


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return float("nan")
    return float(spearmanr(first.ravel(), second.ravel()).statistic)


def bootstrap_means(
    participant_metrics: pd.DataFrame,
    metrics: list[str],
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    subject = participant_metrics.groupby(["subject_id", "y_true"], as_index=False)[metrics].mean()
    healthy = subject[subject["y_true"] == 0].reset_index(drop=True)
    pd_group = subject[subject["y_true"] == 1].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        values = subject[metric].dropna().to_numpy(dtype=float)
        boot = []
        for _ in range(resamples):
            h_values = healthy[metric].dropna().to_numpy(dtype=float)
            pd_values = pd_group[metric].dropna().to_numpy(dtype=float)
            draw = np.concatenate(
                (
                    rng.choice(h_values, size=len(h_values), replace=True),
                    rng.choice(pd_values, size=len(pd_values), replace=True),
                )
            )
            boot.append(float(np.mean(draw)))
        rows.append(
            {
                "metric": metric,
                "participant_mean": float(np.mean(values)),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def explain_predictions(
    predictions: pd.DataFrame,
    images: np.ndarray,
    selections: pd.DataFrame,
    config: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    image_size = int(config["input_size"])
    maps = np.zeros((len(predictions), image_size, image_size), dtype=np.float16)
    metrics_rows = []
    occlusion_rows = []
    occlusion_maps = []
    sanity_rows = []
    selection_index = selections.set_index(["repeat", "outer_fold", "task_id"])
    deletion_config = config["deletion"]
    occlusion_config = config["occlusion"]
    random_config = config["weight_randomization_sanity"]
    processed = 0
    for (repeat, outer_fold, task_id), frame in predictions.groupby(
        ["repeat", "outer_fold", "task_id"], sort=True
    ):
        frame = frame.sort_values("subject_id")
        selected = selection_index.loc[(repeat, outer_fold, task_id)]
        model, checkpoint = load_model(Path(selected["checkpoint_path"]), device)
        checkpoint_test = {str(value).zfill(5) for value in checkpoint["outer_test_subject_ids"]}
        if set(frame["subject_id"]) != checkpoint_test:
            raise ValueError("Prediction subjects do not match the checkpoint external test set.")
        batch = torch.stack(
            [normalized_tensor(images[int(index)]) for index in frame["image_index"]]
        ).to(device)
        predicted = torch.as_tensor(frame["y_pred"].to_numpy(), dtype=torch.int64, device=device)
        cams, reproduced = gradcam(model, batch, predicted, model.gradcam_layer)
        maximum_error = float(
            np.max(np.abs(reproduced - frame["probability_pd"].to_numpy(dtype=float)))
        )
        if maximum_error > 2e-5:
            raise ValueError(f"Checkpoint probability reproduction error: {maximum_error}.")
        channel_drops = channel_ablation_drops(model, batch, predicted)
        targeted_drop, random_drop = deletion_drops(
            model,
            batch,
            predicted,
            cams,
            fraction=float(deletion_config["fraction"]),
            seed=int(deletion_config["seed"]) + processed,
        )
        for local_index, (prediction_index, row) in enumerate(frame.iterrows()):
            maps[int(prediction_index)] = cams[local_index].astype(np.float16)
            foreground = np.any(images[int(row["image_index"])] < 250, axis=2)
            area, fraction, enrichment = attention_metrics(cams[local_index], foreground)
            predicted_confidence = (
                float(row["probability_pd"])
                if int(row["y_pred"]) == 1
                else 1.0 - float(row["probability_pd"])
            )
            metrics_rows.append(
                {
                    "prediction_index": int(prediction_index),
                    "repeat": int(repeat),
                    "outer_fold": int(outer_fold),
                    "task_id": int(task_id),
                    "subject_id": row["subject_id"],
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                    "correct": bool(int(row["y_true"]) == int(row["y_pred"])),
                    "probability_pd": float(row["probability_pd"]),
                    "predicted_confidence": predicted_confidence,
                    "foreground_area_fraction": area,
                    "gradcam_foreground_attention_fraction": fraction,
                    "gradcam_foreground_enrichment": enrichment,
                    "gradcam_degenerate": bool(float(cams[local_index].max()) <= 0),
                    "gradcam_deletion_drop": float(targeted_drop[local_index]),
                    "random_deletion_drop": float(random_drop[local_index]),
                    "deletion_advantage": float(targeted_drop[local_index] - random_drop[local_index]),
                    "speed_channel_drop": float(channel_drops[local_index, 0]),
                    "altitude_channel_drop": float(channel_drops[local_index, 1]),
                    "azimuth_channel_drop": float(channel_drops[local_index, 2]),
                }
            )
            if int(repeat) == 1:
                current_occlusion = occlusion_map(
                    model,
                    batch[local_index],
                    int(row["y_pred"]),
                    patch_size=int(occlusion_config["patch_size"]),
                    stride=int(occlusion_config["stride"]),
                    batch_size=int(occlusion_config["batch_size"]),
                )
                occlusion_position = len(occlusion_maps)
                occlusion_maps.append(current_occlusion.astype(np.float16))
                _, occlusion_fraction, occlusion_enrichment = attention_metrics(
                    current_occlusion, foreground
                )
                occlusion_rows.append(
                    {
                        "occlusion_index": occlusion_position,
                        "prediction_index": int(prediction_index),
                        "repeat": int(repeat),
                        "outer_fold": int(outer_fold),
                        "task_id": int(task_id),
                        "subject_id": row["subject_id"],
                        "y_true": int(row["y_true"]),
                        "y_pred": int(row["y_pred"]),
                        "correct": bool(int(row["y_true"]) == int(row["y_pred"])),
                        "gradcam_occlusion_spearman": correlation(
                            cams[local_index], current_occlusion
                        ),
                        "gradcam_occlusion_top10_iou": top_fraction_iou(
                            cams[local_index], current_occlusion, fraction=0.10
                        ),
                        "occlusion_foreground_attention_fraction": occlusion_fraction,
                        "occlusion_foreground_enrichment": occlusion_enrichment,
                        "occlusion_degenerate": bool(float(current_occlusion.max()) <= 0),
                    }
                )

        first_index = int(frame.index[0])
        torch.manual_seed(int(random_config["seed"]) + processed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(random_config["seed"]) + processed)
        randomized = SmallHandwritingCNN(
            dropout=float(checkpoint["architecture"]["dropout"])
        ).to(device).eval()
        random_cam, _ = gradcam(
            randomized,
            batch[:1],
            predicted[:1],
            randomized.gradcam_layer,
        )
        sanity_rows.append(
            {
                "repeat": int(repeat),
                "outer_fold": int(outer_fold),
                "task_id": int(task_id),
                "subject_id": frame.iloc[0]["subject_id"],
                "prediction_index": first_index,
                "trained_random_spearman": correlation(cams[0], random_cam[0]),
                "trained_random_top10_iou": top_fraction_iou(
                    cams[0], random_cam[0], fraction=0.10
                ),
                "random_gradcam_degenerate": bool(float(random_cam[0].max()) <= 0),
            }
        )
        processed += 1
        if processed % 20 == 0 or processed == len(selections):
            print(f"explained checkpoints {processed}/{len(selections)}", flush=True)
        del model, randomized, batch
    return (
        pd.DataFrame(metrics_rows).sort_values("prediction_index").reset_index(drop=True),
        maps,
        pd.DataFrame(occlusion_rows),
        np.stack(occlusion_maps),
        pd.DataFrame(sanity_rows),
    )


def participant_and_subgroup_summaries(sample_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [
        "correct",
        "predicted_confidence",
        "foreground_area_fraction",
        "gradcam_foreground_attention_fraction",
        "gradcam_foreground_enrichment",
        "gradcam_deletion_drop",
        "random_deletion_drop",
        "deletion_advantage",
        "speed_channel_drop",
        "altitude_channel_drop",
        "azimuth_channel_drop",
    ]
    participant = (
        sample_metrics.groupby(["task_id", "subject_id", "y_true"], as_index=False)[numeric]
        .mean()
        .rename(columns={"correct": "correct_repeat_fraction"})
    )
    subgroup = (
        participant.groupby(["task_id", "y_true"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            correct_repeat_fraction=("correct_repeat_fraction", "mean"),
            gradcam_foreground_enrichment=("gradcam_foreground_enrichment", "mean"),
            deletion_advantage=("deletion_advantage", "mean"),
            speed_channel_drop=("speed_channel_drop", "mean"),
            altitude_channel_drop=("altitude_channel_drop", "mean"),
            azimuth_channel_drop=("azimuth_channel_drop", "mean"),
        )
    )
    return participant, subgroup


def save_figures(
    predictions: pd.DataFrame,
    images: np.ndarray,
    sample_metrics: pd.DataFrame,
    gradcam_maps: np.ndarray,
    occlusion_metrics: pd.DataFrame,
    occlusion_maps: np.ndarray,
    participant_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    sanity: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    label_names = {0: "Healthy", 1: "PD"}

    fig, axes = plt.subplots(2, 8, figsize=(20, 5.4))
    for row, truth in enumerate((0, 1)):
        for column, task_id in enumerate(range(1, 9)):
            chosen = sample_metrics[
                (sample_metrics["task_id"] == task_id) & (sample_metrics["y_true"] == truth)
            ]
            indices = chosen["prediction_index"].to_numpy(dtype=int)
            image_indices = predictions.loc[indices, "image_index"].drop_duplicates().to_numpy(dtype=int)
            mean_image = images[image_indices].mean(axis=0).astype(np.uint8)
            mean_map = gradcam_maps[indices].astype(np.float32).mean(axis=0)
            axis = axes[row, column]
            axis.imshow(mean_image)
            axis.imshow(mean_map, cmap="jet", alpha=0.48, vmin=0, vmax=max(float(mean_map.max()), 1e-6))
            axis.set_title(f"T{task_id} {label_names[truth]}", fontsize=9)
            axis.axis("off")
    fig.suptitle("Mean Grad-CAM for external predictions; participant-balanced across five repeats")
    fig.tight_layout()
    fig.savefig(figure_dir / "mean_gradcam_by_task_and_class.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 8, figsize=(20, 5.4))
    for row, truth in enumerate((0, 1)):
        for column, task_id in enumerate(range(1, 9)):
            chosen = occlusion_metrics[
                (occlusion_metrics["task_id"] == task_id) & (occlusion_metrics["y_true"] == truth)
            ]
            occ_indices = chosen["occlusion_index"].to_numpy(dtype=int)
            pred_indices = chosen["prediction_index"].to_numpy(dtype=int)
            image_indices = predictions.loc[pred_indices, "image_index"].to_numpy(dtype=int)
            mean_image = images[image_indices].mean(axis=0).astype(np.uint8)
            mean_map = occlusion_maps[occ_indices].astype(np.float32).mean(axis=0)
            axis = axes[row, column]
            axis.imshow(mean_image)
            axis.imshow(mean_map, cmap="magma", alpha=0.50, vmin=0, vmax=max(float(mean_map.max()), 1e-6))
            axis.set_title(f"T{task_id} {label_names[truth]}", fontsize=9)
            axis.axis("off")
    fig.suptitle("Mean occlusion sensitivity for repeat-1 external predictions")
    fig.tight_layout()
    fig.savefig(figure_dir / "mean_occlusion_by_task_and_class.png", dpi=200)
    plt.close(fig)

    channel_summary = participant_metrics.groupby("task_id")[
        [f"{signal}_channel_drop" for signal in SIGNALS]
    ].mean()
    fig, axis = plt.subplots(figsize=(8.4, 5.0))
    image = axis.imshow(channel_summary.to_numpy(), cmap="RdBu_r", aspect="auto")
    axis.set_xticks(range(3), SIGNALS)
    axis.set_yticks(range(8), [str(value) for value in channel_summary.index])
    axis.set(xlabel="Ablated RGB signal", ylabel="PaHaW task", title="Predicted-class confidence drop after channel ablation")
    for row in range(8):
        for column in range(3):
            axis.text(column, row, f"{channel_summary.iloc[row, column]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Confidence drop")
    fig.tight_layout()
    fig.savefig(figure_dir / "channel_ablation_by_task.png", dpi=220)
    plt.close(fig)

    repeat_one = sample_metrics[sample_metrics["repeat"] == 1].copy()
    example_rows = []
    for truth in (0, 1):
        for correct in (False, True):
            candidates = repeat_one[
                (repeat_one["task_id"] == 4)
                & (repeat_one["y_true"] == truth)
                & (repeat_one["correct"] == correct)
            ].sort_values(["predicted_confidence", "subject_id"])
            if candidates.empty:
                continue
            selected = candidates.iloc[len(candidates) // 2]
            example_rows.append(selected)
    examples = pd.DataFrame(example_rows).reset_index(drop=True)
    fig, axes = plt.subplots(len(examples), 3, figsize=(10.2, 3.1 * len(examples)), squeeze=False)
    occ_lookup = occlusion_metrics.set_index("prediction_index")
    for row_index, example in examples.iterrows():
        prediction_index = int(example["prediction_index"])
        image_index = int(predictions.loc[prediction_index, "image_index"])
        occlusion_index = int(occ_lookup.loc[prediction_index, "occlusion_index"])
        axes[row_index, 0].imshow(images[image_index])
        axes[row_index, 1].imshow(images[image_index])
        axes[row_index, 1].imshow(gradcam_maps[prediction_index], cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[row_index, 2].imshow(images[image_index])
        axes[row_index, 2].imshow(occlusion_maps[occlusion_index], cmap="magma", alpha=0.5, vmin=0, vmax=1)
        label = label_names[int(example["y_true"])]
        outcome = "correct" if bool(example["correct"]) else "error"
        axes[row_index, 0].set_ylabel(f"{label}, {outcome}\nsubject {example['subject_id']}")
        for column, title in enumerate(("Input", "Grad-CAM", "Occlusion")):
            axes[row_index, column].set_title(title)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
    fig.suptitle("Task 4 external examples selected at median confidence within each stratum")
    fig.tight_layout()
    fig.savefig(figure_dir / "task4_xai_examples.png", dpi=200)
    plt.close(fig)
    examples.to_csv(output_root / "metrics" / "figure_example_selection.csv", index=False)

    validation = bootstrap.set_index("metric")
    names = ["gradcam_deletion_drop", "random_deletion_drop", "deletion_advantage"]
    values = validation.loc[names, "participant_mean"].to_numpy()
    lower = values - validation.loc[names, "ci_low"].to_numpy()
    upper = validation.loc[names, "ci_high"].to_numpy() - values
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].bar(range(3), values, yerr=np.vstack((lower, upper)), capsize=4)
    axes[0].set_xticks(range(3), ["Grad-CAM deletion", "Random deletion", "Advantage"], rotation=12)
    axes[0].axhline(0, color="#333333", linewidth=1)
    axes[0].set(ylabel="Predicted-class confidence drop", title="Deletion faithfulness")
    agreement_values = [
        occlusion_metrics["gradcam_occlusion_spearman"].median(),
        sanity["trained_random_spearman"].median(),
    ]
    axes[1].bar(range(2), agreement_values, color=["#2c6e9b", "#b04a5a"])
    axes[1].set_xticks(range(2), ["Grad-CAM vs occlusion", "Trained vs random"], rotation=12)
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set(ylabel="Median Spearman correlation", title="Agreement and model-dependence checks")
    fig.tight_layout()
    fig.savefig(figure_dir / "xai_validation.png", dpi=220)
    plt.close(fig)
    return examples


def write_report(
    sample_metrics: pd.DataFrame,
    participant_metrics: pd.DataFrame,
    subgroup: pd.DataFrame,
    occlusion_metrics: pd.DataFrame,
    sanity: pd.DataFrame,
    bootstrap: pd.DataFrame,
    output_root: Path,
) -> None:
    bootstrap_index = bootstrap.set_index("metric")
    enrichment = bootstrap_index.loc["gradcam_foreground_enrichment"]
    deletion = bootstrap_index.loc["gradcam_deletion_drop"]
    random_deletion = bootstrap_index.loc["random_deletion_drop"]
    advantage = bootstrap_index.loc["deletion_advantage"]
    channel_rows = []
    for signal in SIGNALS:
        row = bootstrap_index.loc[f"{signal}_channel_drop"]
        channel_rows.append(
            {
                "signal": signal,
                "mean_confidence_drop": row["participant_mean"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
            }
        )
    channels = pd.DataFrame(channel_rows)
    compact_channels = channels.copy()
    compact_subgroup = subgroup.copy()
    for frame in (compact_channels, compact_subgroup):
        for column in frame.select_dtypes(include=[np.number]).columns:
            if column not in {"task_id", "y_true", "n_subjects"}:
                frame[column] = frame[column].map(lambda value: f"{value:.4f}")
    report = f"""# XAI for the frozen external CNN models

Experiment: `cnn-xai-v1`. Explanations use the exact checkpoint that produced each external prediction. The CNN has low predictive performance (task-mean macro F1 0.5105), so these results diagnose model behaviour and are not evidence of clinical biomarkers.

## Scope

- Grad-CAM, 10% targeted deletion, equal-area random deletion, and RGB channel ablation: all {len(sample_metrics)} external predictions.
- Sliding 16×16 white occlusion: all {len(occlusion_metrics)} external predictions in repeat 1.
- Weight-randomization sanity check: one external image for each of {len(sanity)} checkpoints.
- Statistical summaries first average the five repeated predictions within participant and task; participants are the analysis unit.

## Does the model attend to handwriting?

Grad-CAM foreground enrichment was {enrichment['participant_mean']:.3f}, 95% participant-bootstrap CI [{enrichment['ci_low']:.3f}, {enrichment['ci_high']:.3f}]. A value of one means attribution follows foreground area by chance; values above one indicate concentration on rendered strokes.

Degenerate all-zero Grad-CAM maps: {int(sample_metrics['gradcam_degenerate'].sum())}/{len(sample_metrics)}. Degenerate occlusion maps: {int(occlusion_metrics['occlusion_degenerate'].sum())}/{len(occlusion_metrics)}.

## Perturbation faithfulness

Deleting the top 10% Grad-CAM pixels changed predicted-class confidence by {deletion['participant_mean']:+.4f} [{deletion['ci_low']:+.4f}, {deletion['ci_high']:+.4f}]. Equal-area random deletion changed it by {random_deletion['participant_mean']:+.4f} [{random_deletion['ci_low']:+.4f}, {random_deletion['ci_high']:+.4f}]. The targeted-minus-random advantage was {advantage['participant_mean']:+.4f} [{advantage['ci_low']:+.4f}, {advantage['ci_high']:+.4f}].

Grad-CAM and occlusion had median pixelwise Spearman correlation {occlusion_metrics['gradcam_occlusion_spearman'].median():.3f} and mean top-10% IoU {occlusion_metrics['gradcam_occlusion_top10_iou'].mean():.3f}. Trained versus fully randomized networks had median Grad-CAM correlation {sanity['trained_random_spearman'].median():.3f} and mean top-10% IoU {sanity['trained_random_top10_iou'].mean():.3f}. Lower trained-random agreement supports dependence on learned weights.

## Dynamic channels

The values below are changes in confidence for the originally predicted class after replacing one complete colour channel by white. Positive values mean that the channel supported the prediction. Channel removal also changes colour contrast and therefore cannot be interpreted as an isolated physiological effect.

{markdown_table(compact_channels)}

## Task and diagnosis summaries

`y_true=0` is healthy and `y_true=1` is Parkinson's disease. Correctness is the fraction of the five external repetitions classified correctly.

{markdown_table(compact_subgroup)}

## Interpretation boundary

Grad-CAM localizes gradient-weighted activations; occlusion measures sensitivity to a white patch. Their agreement is a faithfulness check, not proof of causal handwriting pathology. Because the classifier is close to chance on several tasks, visually plausible heatmaps must not be used to rescue its predictive claim. These maps are suitable for explaining why this CNN was rejected in favour of the classical models.
"""
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cnn_xai_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = (project_root / config["output_root"]).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {output_root}")
    output_root.mkdir(parents=True)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True)
    maps_dir = output_root / "maps"
    maps_dir.mkdir(parents=True)
    cnn_run = (project_root / config["cnn_run"]).resolve()
    image_run = (project_root / config["image_run"]).resolve()
    predictions = pd.read_csv(
        cnn_run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str}
    ).sort_values(["repeat", "outer_fold", "task_id", "subject_id"]).reset_index(drop=True)
    predictions.index.name = "prediction_index"
    selections = pd.read_csv(cnn_run / "metrics" / "selected_epochs.csv")
    metadata = pd.read_csv(
        image_run / "features" / f"{config['selected_encoding']}_metadata.csv",
        dtype={"subject_id": str},
    ).sort_values(["task_id", "subject_id"]).reset_index(drop=True)
    metadata["subject_id"] = metadata["subject_id"].astype(str).str.zfill(5)
    metadata["image_index"] = np.arange(len(metadata))
    predictions["subject_id"] = predictions["subject_id"].astype(str).str.zfill(5)
    predictions = predictions.merge(
        metadata[["task_id", "subject_id", "image_index", "image_path"]],
        on=["task_id", "subject_id"],
        how="left",
        validate="many_to_one",
    )
    predictions.index.name = "prediction_index"
    if predictions["image_index"].isna().any():
        raise ValueError("An external prediction has no source image.")
    images = cache_images(metadata, int(config["input_size"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_metrics, gradcam_maps, occlusion_metrics, occlusion_maps, sanity = explain_predictions(
        predictions, images, selections, config, device
    )
    participant_metrics, subgroup = participant_and_subgroup_summaries(sample_metrics)
    bootstrap_metrics = [
        "gradcam_foreground_enrichment",
        "gradcam_deletion_drop",
        "random_deletion_drop",
        "deletion_advantage",
        "speed_channel_drop",
        "altitude_channel_drop",
        "azimuth_channel_drop",
    ]
    bootstrap = bootstrap_means(
        participant_metrics,
        bootstrap_metrics,
        resamples=int(config["aggregation"]["bootstrap_resamples"]),
        seed=int(config["aggregation"]["bootstrap_seed"]),
    )
    sample_metrics.to_csv(metrics_dir / "prediction_xai_metrics.csv", index=False)
    participant_metrics.to_csv(metrics_dir / "participant_task_xai_metrics.csv", index=False)
    subgroup.to_csv(metrics_dir / "task_class_xai_summary.csv", index=False)
    occlusion_metrics.to_csv(metrics_dir / "occlusion_metrics.csv", index=False)
    sanity.to_csv(metrics_dir / "weight_randomization_sanity.csv", index=False)
    bootstrap.to_csv(metrics_dir / "global_participant_bootstrap.csv", index=False)
    np.savez_compressed(
        maps_dir / "spatial_maps.npz",
        gradcam_maps=gradcam_maps,
        gradcam_prediction_index=np.arange(len(gradcam_maps)),
        occlusion_maps=occlusion_maps,
        occlusion_prediction_index=occlusion_metrics["prediction_index"].to_numpy(dtype=int),
    )
    save_figures(
        predictions,
        images,
        sample_metrics,
        gradcam_maps,
        occlusion_metrics,
        occlusion_maps,
        participant_metrics,
        bootstrap,
        sanity,
        output_root,
    )
    write_report(
        sample_metrics,
        participant_metrics,
        subgroup,
        occlusion_metrics,
        sanity,
        bootstrap,
        output_root,
    )
    source_paths = [
        cnn_run / "metrics" / "outer_predictions.csv",
        cnn_run / "metrics" / "selected_epochs.csv",
        cnn_run / "manifests" / "artifact_manifest.json",
        image_run / "features" / f"{config['selected_encoding']}_metadata.csv",
    ]
    pd.DataFrame(
        [{"path": str(path), "sha256": sha256_file(path)} for path in source_paths]
    ).to_csv(metrics_dir / "source_artifacts.csv", index=False)

    git_commit = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "fresh_run_no_resume": True,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_git_commit": git_commit,
        "cnn_run": str(cnn_run),
        "selected_encoding": config["selected_encoding"],
        "device": str(device),
        "external_gradcam_maps": len(gradcam_maps),
        "repeat1_occlusion_maps": len(occlusion_maps),
        "weight_randomization_checks": len(sanity),
        "spatial_map_shape": list(gradcam_maps.shape[1:]),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            package: version(package)
            for package in (
                "matplotlib",
                "numpy",
                "pandas",
                "pillow",
                "scipy",
                "torch",
            )
        },
    }
    save_json(manifest, output_root / "manifests" / "run_manifest.json")
    save_json(artifact_manifest(output_root), output_root / "manifests" / "artifact_manifest.json")
    print(bootstrap.to_string(index=False))
    print(
        {
            "median_gradcam_occlusion_spearman": float(
                occlusion_metrics["gradcam_occlusion_spearman"].median()
            ),
            "median_trained_random_spearman": float(
                sanity["trained_random_spearman"].median()
            ),
        }
    )
    print(f"complete in {manifest['duration_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
