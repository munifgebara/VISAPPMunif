"""Train and externally evaluate the frozen small CNN on SAZ images."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomAffine

from pdhms_restart.cnn_model import SmallHandwritingCNN, normalized_tensor, parameter_count, prepared_rgb
from pdhms_restart.comparison import (
    compare_all_to_reference,
    compare_global_method_pairs,
)
from pdhms_restart.data import sha256_file
from pdhms_restart.evaluation import metric_values, summarize_predictions


ALGORITHM_ORDER = ("svm", "random_forest", "logistic_regression", "cnn")
ALGORITHM_LABELS = {
    "svm": "SVM",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "cnn": "CNN",
}


class CachedImageDataset(Dataset):
    """In-memory RGB data with optional on-the-fly affine augmentation."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        augment: bool,
        augmentation_config: dict,
    ) -> None:
        self.images = images
        self.labels = labels.astype(np.float32)
        self.transform = None
        if augment:
            self.transform = RandomAffine(
                degrees=float(augmentation_config["random_rotation_degrees"]),
                translate=(
                    float(augmentation_config["random_translation_fraction"]),
                    float(augmentation_config["random_translation_fraction"]),
                ),
                scale=tuple(float(value) for value in augmentation_config["random_scale"]),
                fill=float(augmentation_config["fill_normalized"]),
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.from_numpy(
            np.ascontiguousarray(self.images[index].transpose(2, 0, 1))
        ).float() / 255.0
        if self.transform is not None:
            tensor = self.transform(tensor)
        tensor = (tensor - 0.5) / 0.5
        return tensor, torch.tensor(self.labels[index], dtype=torch.float32)


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


def seed_value(master_seed: int, repeat: int, outer_fold: int, task_id: int, phase: int) -> int:
    return int(
        np.random.SeedSequence([master_seed, repeat, outer_fold, task_id, phase])
        .generate_state(1, dtype=np.uint32)[0]
    )


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    augmentation_config: dict,
    seed: int,
) -> DataLoader:
    dataset = CachedImageDataset(
        images[indices], labels[indices], augment=augment, augmentation_config=augmentation_config
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def criterion_for(labels: np.ndarray, device: torch.device) -> nn.Module:
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("CNN training split must contain both diagnoses.")
    positive_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=positive_weight)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    truths = []
    probabilities = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            losses.append(float(criterion(logits, labels).detach().cpu()))
            truths.append(labels.cpu().numpy().astype(np.int8))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    truth = np.concatenate(truths)
    probability = np.concatenate(probabilities).astype(float)
    predicted = (probability >= 0.5).astype(np.int8)
    macro_f1 = float(
        f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)
    )
    return float(np.mean(losses)), macro_f1, truth, probability


def select_epoch(
    images: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: dict,
    device: torch.device,
    seed: int,
) -> tuple[int, list[dict[str, object]]]:
    set_determinism(seed)
    model = SmallHandwritingCNN(dropout=float(config["architecture"]["dropout"])).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = criterion_for(labels[train_indices], device)
    train_loader = make_loader(
        images,
        labels,
        train_indices,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        augment=True,
        augmentation_config=config["augmentation"],
        seed=seed,
    )
    validation_loader = make_loader(
        images,
        labels,
        validation_indices,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        augment=False,
        augmentation_config=config["augmentation"],
        seed=seed,
    )
    best_epoch = 1
    best_macro_f1 = -1.0
    best_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    maximum = int(training["maximum_epoch_selection_epochs"])
    minimum = int(training["minimum_epochs_before_stopping"])
    patience = int(training["early_stopping_patience"])
    for epoch in range(1, maximum + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        validation_loss, validation_f1, _, _ = evaluate(
            model, validation_loader, criterion, device
        )
        improved = validation_f1 > best_macro_f1 + 1e-12 or (
            abs(validation_f1 - best_macro_f1) <= 1e-12 and validation_loss < best_loss - 1e-6
        )
        if improved:
            best_epoch = epoch
            best_macro_f1 = validation_f1
            best_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_f1,
                "is_selected_epoch": False,
            }
        )
        if epoch >= minimum and epochs_without_improvement >= patience:
            break
    history[best_epoch - 1]["is_selected_epoch"] = True
    return best_epoch, history


def fit_final_model(
    images: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    epochs: int,
    config: dict,
    device: torch.device,
    seed: int,
) -> tuple[SmallHandwritingCNN, list[float]]:
    set_determinism(seed)
    model = SmallHandwritingCNN(dropout=float(config["architecture"]["dropout"])).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = criterion_for(labels[train_indices], device)
    loader = make_loader(
        images,
        labels,
        train_indices,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        augment=True,
        augmentation_config=config["augmentation"],
        seed=seed,
    )
    losses = [train_epoch(model, loader, optimizer, criterion, device) for _ in range(epochs)]
    return model, losses


def cache_images(metadata: pd.DataFrame, image_size: int) -> np.ndarray:
    images = []
    for index, path in enumerate(metadata["image_path"], start=1):
        images.append(prepared_rgb(Path(path), image_size))
        if index % 100 == 0 or index == len(metadata):
            print(f"prepared images {index}/{len(metadata)}", flush=True)
    return np.stack(images).astype(np.uint8)


def run_external_cv(
    metadata: pd.DataFrame,
    images: np.ndarray,
    split_payload: dict,
    config: dict,
    output_root: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = metadata.reset_index(drop=True).copy()
    metadata["subject_id"] = metadata["subject_id"].astype(str).str.zfill(5)
    labels = (metadata["label"].to_numpy() == "PD").astype(np.int8)
    predictions = []
    selections = []
    history_rows = []
    fold_rows = []
    master_seed = int(config["training"]["master_seed"])
    model_dir = output_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for outer in split_payload["outer_splits"]:
        repeat = int(outer["repeat"])
        outer_fold = int(outer["outer_fold"])
        outer_train_subjects = {str(value).zfill(5) for value in outer["train_subject_ids"]}
        outer_test_subjects = {str(value).zfill(5) for value in outer["test_subject_ids"]}
        for task_id in sorted(int(value) for value in metadata["task_id"].unique()):
            task_mask = metadata["task_id"].to_numpy() == task_id
            outer_train = task_mask & metadata["subject_id"].isin(outer_train_subjects).to_numpy()
            outer_test = task_mask & metadata["subject_id"].isin(outer_test_subjects).to_numpy()
            inner = outer["tasks"][str(task_id)]["splits"][0]
            inner_train_subjects = {str(value).zfill(5) for value in inner["train_subject_ids"]}
            validation_subjects = {
                str(value).zfill(5) for value in inner["validation_subject_ids"]
            }
            inner_train = outer_train & metadata["subject_id"].isin(inner_train_subjects).to_numpy()
            validation = outer_train & metadata["subject_id"].isin(validation_subjects).to_numpy()
            outer_train_indices = np.flatnonzero(outer_train)
            outer_test_indices = np.flatnonzero(outer_test)
            selection_seed = seed_value(master_seed, repeat, outer_fold, task_id, 1)
            final_seed = seed_value(master_seed, repeat, outer_fold, task_id, 2)
            selected_epoch, history = select_epoch(
                images,
                labels,
                np.flatnonzero(inner_train),
                np.flatnonzero(validation),
                config,
                device,
                selection_seed,
            )
            for row in history:
                history_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "task_id": task_id,
                        "phase": "epoch_selection",
                        **row,
                    }
                )
            model, final_losses = fit_final_model(
                images,
                labels,
                outer_train_indices,
                selected_epoch,
                config,
                device,
                final_seed,
            )
            for epoch, loss in enumerate(final_losses, start=1):
                history_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "task_id": task_id,
                        "phase": "final_fit",
                        "epoch": epoch,
                        "train_loss": loss,
                        "validation_loss": np.nan,
                        "validation_macro_f1": np.nan,
                        "is_selected_epoch": epoch == selected_epoch,
                    }
                )
            test_loader = make_loader(
                images,
                labels,
                outer_test_indices,
                batch_size=int(config["training"]["batch_size"]),
                shuffle=False,
                augment=False,
                augmentation_config=config["augmentation"],
                seed=final_seed,
            )
            test_criterion = criterion_for(labels[outer_train_indices], device)
            test_loss, _, test_truth, probabilities = evaluate(
                model, test_loader, test_criterion, device
            )
            predicted = (probabilities >= 0.5).astype(np.int8)
            scores = np.log(np.clip(probabilities, 1e-7, 1 - 1e-7) / np.clip(1 - probabilities, 1e-7, 1))
            candidate_name = f"SmallHandwritingCNN_epochs={selected_epoch}"
            for index, truth, prediction, probability, score in zip(
                outer_test_indices, test_truth, predicted, probabilities, scores, strict=True
            ):
                sample = metadata.loc[index]
                predictions.append(
                    {
                        "algorithm": "cnn",
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "task_id": task_id,
                        "subject_id": sample["subject_id"],
                        "y_true": int(truth),
                        "label": sample["label"],
                        "y_pred": int(prediction),
                        "decision_score_pd": float(score),
                        "probability_pd": float(probability),
                        "selected_candidate": candidate_name,
                    }
                )
            values = metric_values(test_truth, predicted, scores)
            fold_rows.append(
                {
                    "algorithm": "cnn",
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "task_id": task_id,
                    "n_train": len(outer_train_indices),
                    "n_test": len(outer_test_indices),
                    "selected_epoch": selected_epoch,
                    "test_bce_loss": test_loss,
                    **values,
                }
            )
            checkpoint_path = model_dir / f"repeat_{repeat}_fold_{outer_fold}_task_{task_id}.pt"
            state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": state,
                    "architecture": config["architecture"],
                    "input": config["input"],
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "task_id": task_id,
                    "selected_epoch": selected_epoch,
                    "selection_seed": selection_seed,
                    "final_seed": final_seed,
                    "outer_train_subject_ids": sorted(
                        metadata.loc[outer_train, "subject_id"].astype(str).tolist()
                    ),
                    "outer_test_subject_ids": sorted(
                        metadata.loc[outer_test, "subject_id"].astype(str).tolist()
                    ),
                },
                checkpoint_path,
            )
            selections.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "task_id": task_id,
                    "selected_epoch": selected_epoch,
                    "epochs_evaluated": len(history),
                    "selected_validation_macro_f1": history[selected_epoch - 1][
                        "validation_macro_f1"
                    ],
                    "selected_validation_loss": history[selected_epoch - 1]["validation_loss"],
                    "selection_seed": selection_seed,
                    "final_seed": final_seed,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
            )
        print(
            f"trained repeat {repeat}/{split_payload['repeats']} "
            f"outer fold {outer_fold}/{split_payload['outer_folds']}",
            flush=True,
        )
    return (
        pd.DataFrame(predictions),
        pd.DataFrame(selections),
        pd.DataFrame(history_rows),
        pd.DataFrame(fold_rows),
    )


def majority_fusion(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algorithm, repeat, outer_fold, subject_id), frame in predictions.groupby(
        ["algorithm", "repeat", "outer_fold", "subject_id"], sort=True
    ):
        truth_values = frame["y_true"].unique()
        if len(truth_values) != 1:
            raise ValueError("A participant has conflicting task labels.")
        vote = float(frame["y_pred"].mean())
        rows.append(
            {
                "algorithm": algorithm,
                "repeat": int(repeat),
                "outer_fold": int(outer_fold),
                "task_id": 0,
                "subject_id": str(subject_id).zfill(5),
                "y_true": int(truth_values[0]),
                "y_pred": int(vote >= 0.5),
                "decision_score_pd": vote,
                "selected_candidate": f"{algorithm}_majority_vote_all8",
                "available_task_count": int(frame["task_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def summarize_algorithms(
    predictions: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeats = []
    summaries = []
    for index, algorithm in enumerate(ALGORITHM_ORDER):
        frame = predictions[predictions["algorithm"] == algorithm]
        repeat, summary = summarize_predictions(
            frame, bootstrap_resamples=resamples, bootstrap_seed=seed + index
        )
        repeat.insert(0, "algorithm", algorithm)
        summary.insert(0, "algorithm", algorithm)
        repeats.append(repeat)
        summaries.append(summary)
    return pd.concat(repeats, ignore_index=True), pd.concat(summaries, ignore_index=True)


def rankings(summary: pd.DataFrame) -> pd.DataFrame:
    result = (
        summary.groupby("algorithm", as_index=False)["macro_f1_mean"]
        .mean()
        .rename(columns={"macro_f1_mean": "task_mean_macro_f1"})
        .sort_values(["task_mean_macro_f1", "algorithm"], ascending=[False, True])
        .reset_index(drop=True)
    )
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def save_figures(
    task_summary: pd.DataFrame,
    task_ranking: pd.DataFrame,
    fusion_ranking: pd.DataFrame,
    selections: pd.DataFrame,
    output_root: Path,
) -> None:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10.0, 5.3))
    for algorithm in ALGORITHM_ORDER:
        frame = task_summary[task_summary["algorithm"] == algorithm].sort_values("task_id")
        axis.plot(frame["task_id"], frame["macro_f1_mean"], marker="o", label=ALGORITHM_LABELS[algorithm])
    axis.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
    axis.set(xlabel="PaHaW task", ylabel="Macro F1", xticks=range(1, 9), ylim=(0.3, 0.8), title="Classifiers on frozen SAZ images")
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_task_classifier_comparison.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for axis, ranking, column, title in (
        (axes[0], task_ranking, "task_mean_macro_f1", "Mean across separate tasks"),
        (axes[1], fusion_ranking, "macro_f1_mean", "Majority-vote participant fusion"),
    ):
        ordered = ranking.sort_values(column)
        axis.barh([ALGORITHM_LABELS[value] for value in ordered["algorithm"]], ordered[column])
        axis.axvline(0.5, color="#777777", linestyle=":", linewidth=1)
        axis.set(xlabel="Macro F1", xlim=(0.4, 0.75), title=title)
        for index, value in enumerate(ordered[column]):
            axis.text(value + 0.005, index, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(figure_dir / "task_and_fusion_rankings.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.2, 4.7))
    data = [
        selections[selections["task_id"] == task]["selected_epoch"].to_numpy()
        for task in range(1, 9)
    ]
    axis.boxplot(data, tick_labels=[str(task) for task in range(1, 9)])
    axis.set(xlabel="PaHaW task", ylabel="Selected epoch", title="Epoch selection inside outer training folds")
    fig.tight_layout()
    fig.savefig(figure_dir / "selected_epochs_by_task.png", dpi=220)
    plt.close(fig)


def format_frame(frame: pd.DataFrame, label_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    result = frame.copy()
    for column in label_columns:
        if column in result:
            result[column] = result[column].map(ALGORITHM_LABELS).fillna(result[column])
    for column in result.select_dtypes(include=[np.number]).columns:
        if column not in {"rank", "task_id", "n_subjects"}:
            result[column] = result[column].map(lambda value: f"{value:.4f}")
    return result


def write_report(
    config: dict,
    cnn_summary: pd.DataFrame,
    task_ranking: pd.DataFrame,
    task_pairwise: pd.DataFrame,
    cnn_vs_svm: pd.DataFrame,
    fusion_summary: pd.DataFrame,
    fusion_ranking: pd.DataFrame,
    fusion_pairwise: pd.DataFrame,
    selections: pd.DataFrame,
    device: torch.device,
    output_root: Path,
) -> None:
    cnn_global = float(cnn_summary["macro_f1_mean"].mean())
    svm_global = float(task_ranking[task_ranking["algorithm"] == "svm"]["task_mean_macro_f1"].iloc[0])
    delta_row = task_pairwise[
        (task_pairwise["candidate"] == "svm") & (task_pairwise["reference"] == "cnn")
    ]
    if len(delta_row):
        cnn_minus_svm = -float(delta_row.iloc[0]["delta_task_mean_macro_f1"])
        ci_low = -float(delta_row.iloc[0]["ci_high"])
        ci_high = -float(delta_row.iloc[0]["ci_low"])
        p_value = float(delta_row.iloc[0]["permutation_p"])
    else:
        delta_row = task_pairwise[
            (task_pairwise["candidate"] == "cnn") & (task_pairwise["reference"] == "svm")
        ].iloc[0]
        cnn_minus_svm = float(delta_row["delta_task_mean_macro_f1"])
        ci_low, ci_high = float(delta_row["ci_low"]), float(delta_row["ci_high"])
        p_value = float(delta_row["permutation_p"])
    best_task = task_ranking.iloc[0]
    best_fusion = fusion_ranking.iloc[0]
    compact_cnn = format_frame(
        cnn_summary[
            ["task_id", "n_subjects", "macro_f1_mean", "macro_f1_ci_low", "macro_f1_ci_high", "accuracy_mean", "sensitivity_pd_mean", "specificity_h_mean", "roc_auc_mean"]
        ]
    )
    compact_task_ranking = format_frame(task_ranking, ("algorithm",))
    compact_cnn_vs_svm = format_frame(
        cnn_vs_svm[["task_id", "delta_macro_f1", "ci_low", "ci_high", "permutation_p", "holm_p_global_8", "supported_holm_0_05_global_8"]]
    )
    compact_task_pairs = format_frame(task_pairwise, ("candidate", "reference"))
    compact_fusion = format_frame(
        fusion_summary[["algorithm", "macro_f1_mean", "macro_f1_ci_low", "macro_f1_ci_high", "accuracy_mean", "sensitivity_pd_mean", "specificity_h_mean", "roc_auc_mean"]],
        ("algorithm",),
    )
    compact_fusion_pairs = format_frame(fusion_pairwise, ("candidate", "reference"))
    report = f"""# CNN on frozen SAZ images

Experiment: `{config['experiment_id']}`. The CNN consumes the pixels of the selected SAZ representation with fixed stroke width; it does not use LPQ.

## Protocol

The model has {config['architecture']['trainable_parameters']:,} trainable parameters in four convolutional blocks. Images are foreground-cropped, square-padded, and resized to 128×128. Training uses only small rotation, translation, and scale changes; handwriting is never flipped.

For every task and external participant fold, the epoch count is chosen on the first frozen inner validation split. The model is then reinitialized and trained on all outer-training participants for exactly that many epochs. All 200 external models are saved for the later XAI stage. Execution device: `{device}`.

## Separate-task endpoint

CNN obtained an unweighted eight-task macro F1 of {cnn_global:.4f}, versus {svm_global:.4f} for SVM. The paired difference CNN minus SVM was {cnn_minus_svm:+.4f}, participant-bootstrap 95% CI [{ci_low:+.4f}, {ci_high:+.4f}], randomization p={p_value:.4f}.

The descriptive leader across separate tasks was {ALGORITHM_LABELS[best_task['algorithm']]} at {best_task['task_mean_macro_f1']:.4f}.

{markdown_table(compact_task_ranking)}

### CNN by task

{markdown_table(compact_cnn)}

### Paired CNN minus SVM by task

Holm correction covers the eight tasks.

{markdown_table(compact_cnn_vs_svm)}

### All global classifier pairs

Holm correction covers all six pairs among CNN, SVM, Random Forest, and Logistic Regression.

{markdown_table(compact_task_pairs)}

## Participant fusion endpoint

For a common comparison, each classifier uses majority vote over all available tasks. The descriptive leader was {ALGORITHM_LABELS[best_fusion['algorithm']]} at {best_fusion['macro_f1_mean']:.4f} macro F1.

{markdown_table(compact_fusion)}

{markdown_table(compact_fusion_pairs)}

## Training diagnostics

The selected epoch ranged from {selections['selected_epoch'].min()} to {selections['selected_epoch'].max()}, with median {selections['selected_epoch'].median():.1f}. This variation is decided entirely inside the external training set. XAI is intentionally deferred until these predictions and checkpoints are frozen; the next stage will explain only external-test images with their corresponding fold model.
"""
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cnn_report.md").write_text(report, encoding="utf-8")


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
    image_run = (project_root / config["image_run"]).resolve()
    classical_run = (project_root / config["classical_run"]).resolve()
    fusion_run = (project_root / config["fusion_run"]).resolve()
    static_run = (project_root / config["static_run"]).resolve()
    split_path = static_run / "metrics" / "cv_splits.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    encoding = config["selected_encoding"]
    metadata = pd.read_csv(
        image_run / "features" / f"{encoding}_metadata.csv", dtype={"subject_id": str}
    ).sort_values(["task_id", "subject_id"]).reset_index(drop=True)
    images = cache_images(metadata, int(config["input"]["image_size"]))

    if bool(config["training"]["deterministic_algorithms"]):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_probe = SmallHandwritingCNN(dropout=float(config["architecture"]["dropout"]))
    if parameter_count(model_probe) != int(config["architecture"]["trainable_parameters"]):
        raise ValueError("Declared CNN parameter count differs from implementation.")
    predictions, selections, history, fold_metrics = run_external_cv(
        metadata, images, split_payload, config, output_root, device
    )
    validation = config["validation"]
    cnn_repeat, cnn_summary = summarize_predictions(
        predictions,
        bootstrap_resamples=int(validation["bootstrap_resamples"]),
        bootstrap_seed=int(validation["bootstrap_seed"]),
    )
    for frame in (cnn_repeat, cnn_summary):
        frame.insert(0, "algorithm", "cnn")

    classical_predictions = pd.read_csv(
        classical_run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str}
    )
    classical_summary = pd.read_csv(classical_run / "metrics" / "task_summary.csv")
    all_predictions = pd.concat([classical_predictions, predictions], ignore_index=True)
    all_summary = pd.concat([classical_summary, cnn_summary], ignore_index=True)
    task_ranking = rankings(all_summary)
    method_frames = {
        algorithm: all_predictions[all_predictions["algorithm"] == algorithm].copy()
        for algorithm in ALGORITHM_ORDER
    }
    task_pairwise = compare_global_method_pairs(
        method_frames,
        list(ALGORITHM_ORDER),
        resamples=int(validation["paired_randomization_resamples"]),
        seed=int(validation["paired_randomization_seed"]),
    )
    svm_predictions = method_frames["svm"]
    cnn_vs_svm = compare_all_to_reference(
        predictions,
        svm_predictions,
        group_column="algorithm",
        resamples=int(validation["paired_randomization_resamples"]),
        seed=int(validation["paired_randomization_seed"]) + 1,
    )

    fused = majority_fusion(all_predictions)
    fused_repeat, fused_summary = summarize_algorithms(
        fused,
        resamples=int(validation["bootstrap_resamples"]),
        seed=int(validation["bootstrap_seed"]) + 100,
    )
    fusion_ranking = fused_summary[["algorithm", "macro_f1_mean"]].sort_values(
        ["macro_f1_mean", "algorithm"], ascending=[False, True]
    ).reset_index(drop=True)
    fusion_ranking.insert(0, "rank", np.arange(1, len(fusion_ranking) + 1))
    fused_frames = {
        algorithm: fused[fused["algorithm"] == algorithm].copy()
        for algorithm in ALGORITHM_ORDER
    }
    fusion_pairwise = compare_global_method_pairs(
        fused_frames,
        list(ALGORITHM_ORDER),
        resamples=int(validation["paired_randomization_resamples"]),
        seed=int(validation["paired_randomization_seed"]) + 2,
    )

    predictions.to_csv(metrics_dir / "outer_predictions.csv", index=False)
    selections.to_csv(metrics_dir / "selected_epochs.csv", index=False)
    history.to_csv(metrics_dir / "training_history.csv", index=False)
    fold_metrics.to_csv(metrics_dir / "outer_fold_metrics.csv", index=False)
    cnn_repeat.to_csv(metrics_dir / "repeat_metrics.csv", index=False)
    cnn_summary.to_csv(metrics_dir / "task_summary.csv", index=False)
    all_summary.to_csv(metrics_dir / "combined_task_summary.csv", index=False)
    task_ranking.to_csv(metrics_dir / "combined_task_ranking.csv", index=False)
    task_pairwise.to_csv(metrics_dir / "combined_task_pairwise.csv", index=False)
    cnn_vs_svm.to_csv(metrics_dir / "paired_contrasts_vs_svm.csv", index=False)
    fused.to_csv(metrics_dir / "fused_majority_predictions.csv", index=False)
    fused_repeat.to_csv(metrics_dir / "fused_majority_repeat_metrics.csv", index=False)
    fused_summary.to_csv(metrics_dir / "fused_majority_summary.csv", index=False)
    fusion_ranking.to_csv(metrics_dir / "fused_majority_ranking.csv", index=False)
    fusion_pairwise.to_csv(metrics_dir / "fused_majority_pairwise.csv", index=False)
    source_paths = [
        image_run / "features" / f"{encoding}_metadata.csv",
        classical_run / "metrics" / "outer_predictions.csv",
        classical_run / "metrics" / "task_summary.csv",
        fusion_run / "metrics" / "fused_predictions.csv",
        split_path,
    ]
    pd.DataFrame(
        [{"path": str(path), "sha256": sha256_file(path)} for path in source_paths]
    ).to_csv(metrics_dir / "source_artifacts.csv", index=False)
    save_figures(all_summary, task_ranking, fusion_ranking, selections, output_root)
    write_report(
        config,
        cnn_summary,
        task_ranking,
        task_pairwise,
        cnn_vs_svm,
        fused_summary,
        fusion_ranking,
        fusion_pairwise,
        selections,
        device,
        output_root,
    )

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
        "selected_encoding": encoding,
        "reused_cv_split_checksum": split_payload["checksum"],
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "prepared_images": list(images.shape),
        "trainable_parameters": parameter_count(model_probe),
        "saved_models": len(selections),
        "outer_predictions": len(predictions),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            package: version(package)
            for package in (
                "matplotlib",
                "numpy",
                "pandas",
                "pillow",
                "scikit-learn",
                "scipy",
                "torch",
                "torchvision",
            )
        },
    }
    save_json(manifest, output_root / "manifests" / "run_manifest.json")
    save_json(artifact_manifest(output_root), output_root / "manifests" / "artifact_manifest.json")
    print(task_ranking.to_string(index=False))
    print(fusion_ranking.to_string(index=False))
    print(f"complete in {manifest['duration_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
