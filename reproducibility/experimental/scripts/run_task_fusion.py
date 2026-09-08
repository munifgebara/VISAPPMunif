"""Fuse the eight frozen SAZ task classifiers into participant predictions."""

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
from sklearn.calibration import CalibratedClassifierCV

from pdhms_restart.comparison import compare_global_method_pairs
from pdhms_restart.data import sha256_file
from pdhms_restart.evaluation import Candidate, make_model, metric_values, summarize_predictions
from pdhms_restart.task_fusion import FUSION_METHODS, fuse_fold, reliability_weight


METHOD_LABELS = {
    "majority_vote_all8": "Majority vote",
    "calibrated_mean_all8": "Mean calibrated probability",
    "calibrated_reliability_weighted_all8": "Reliability-weighted calibrated probability",
}


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


def selected_rows(search: pd.DataFrame) -> pd.DataFrame:
    selected = search["selected"]
    if selected.dtype != bool:
        selected = selected.astype(str).str.lower().eq("true")
    result = search[selected].copy()
    counts = result.groupby(["repeat", "outer_fold", "task_id"]).size()
    if len(result) != 200 or not (counts == 1).all():
        raise ValueError("Expected one selected SVM for every outer fold and task.")
    return result


def candidate_from_row(row) -> Candidate:
    gamma: str | float = "scale"
    if row.kernel == "rbf":
        gamma = row.gamma if str(row.gamma) == "scale" else float(row.gamma)
    return Candidate(kernel=str(row.kernel), c=float(row.C), gamma=gamma)


def calibration_indices(
    training_subjects: np.ndarray,
    inner_splits: list[dict],
) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    for inner in inner_splits:
        inner_train = {str(value).zfill(5) for value in inner["train_subject_ids"]}
        validation = {str(value).zfill(5) for value in inner["validation_subject_ids"]}
        train_index = np.flatnonzero(np.isin(training_subjects, list(inner_train)))
        validation_index = np.flatnonzero(np.isin(training_subjects, list(validation)))
        if set(train_index) & set(validation_index):
            raise AssertionError("Inner calibration split overlaps.")
        if len(train_index) + len(validation_index) != len(training_subjects):
            raise ValueError("Inner calibration split does not cover outer training data.")
        result.append((train_index, validation_index))
    return result


def generate_calibrated_base_predictions(
    metadata: pd.DataFrame,
    features: np.ndarray,
    source_predictions: pd.DataFrame,
    selected_search: pd.DataFrame,
    split_payload: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = metadata.reset_index(drop=True).copy()
    metadata["subject_id"] = metadata["subject_id"].astype(str).str.zfill(5)
    source_predictions = source_predictions.copy()
    source_predictions["subject_id"] = source_predictions["subject_id"].astype(str).str.zfill(5)
    base_rows: list[dict[str, object]] = []
    fused_frames = []
    for outer in split_payload["outer_splits"]:
        repeat = int(outer["repeat"])
        outer_fold = int(outer["outer_fold"])
        train_subjects = {str(value).zfill(5) for value in outer["train_subject_ids"]}
        test_subjects = {str(value).zfill(5) for value in outer["test_subject_ids"]}
        fold_rows: list[dict[str, object]] = []
        for task_id in sorted(int(value) for value in metadata["task_id"].unique()):
            task_mask = metadata["task_id"].to_numpy() == task_id
            train_mask = task_mask & metadata["subject_id"].isin(train_subjects).to_numpy()
            test_mask = task_mask & metadata["subject_id"].isin(test_subjects).to_numpy()
            train_indices = np.flatnonzero(train_mask)
            test_indices = np.flatnonzero(test_mask)
            train_subject_array = metadata.loc[train_mask, "subject_id"].to_numpy()
            chosen = selected_search[
                (selected_search["repeat"] == repeat)
                & (selected_search["outer_fold"] == outer_fold)
                & (selected_search["task_id"] == task_id)
            ]
            if len(chosen) != 1:
                raise ValueError("Missing selected task model for calibration.")
            chosen_row = next(chosen.itertuples(index=False))
            candidate = candidate_from_row(chosen_row)
            cv = calibration_indices(train_subject_array, outer["tasks"][str(task_id)]["splits"])
            calibrated = CalibratedClassifierCV(
                estimator=make_model(candidate),
                method="sigmoid",
                cv=cv,
                n_jobs=-1,
                ensemble=True,
            )
            y_train = (metadata.loc[train_mask, "label"].to_numpy() == "PD").astype(int)
            calibrated.fit(features[train_indices], y_train)
            class_index = int(np.flatnonzero(calibrated.classes_ == 1)[0])
            probabilities = calibrated.predict_proba(features[test_indices])[:, class_index]
            source = source_predictions[
                (source_predictions["repeat"] == repeat)
                & (source_predictions["outer_fold"] == outer_fold)
                & (source_predictions["task_id"] == task_id)
            ].set_index("subject_id")
            reliability = reliability_weight(float(chosen_row.inner_macro_f1))
            for index, probability in zip(test_indices, probabilities, strict=True):
                sample = metadata.loc[index]
                subject_id = str(sample["subject_id"])
                if subject_id not in source.index:
                    raise ValueError("Calibrated test subject is absent from frozen predictions.")
                frozen = source.loc[subject_id]
                truth = int(sample["label"] == "PD")
                if truth != int(frozen["y_true"]):
                    raise ValueError("Calibrated and frozen task labels differ.")
                row = {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "task_id": task_id,
                    "subject_id": subject_id,
                    "label": sample["label"],
                    "y_true": truth,
                    "base_y_pred": int(frozen["y_pred"]),
                    "base_decision_score_pd": float(frozen["decision_score_pd"]),
                    "calibrated_probability_pd": float(probability),
                    "calibrated_y_pred": int(probability >= 0.5),
                    "selected_candidate": candidate.name,
                    "inner_macro_f1": float(chosen_row.inner_macro_f1),
                    "reliability_weight": reliability,
                }
                base_rows.append(row)
                fold_rows.append(row)
        fused_frames.append(fuse_fold(pd.DataFrame(fold_rows)))
        print(
            f"calibrated and fused repeat {repeat}/{split_payload['repeats']} "
            f"outer fold {outer_fold}/{split_payload['outer_folds']}",
            flush=True,
        )
    return pd.DataFrame(base_rows), pd.concat(fused_frames, ignore_index=True)


def task_calibration_metrics(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task_id, repeat), frame in base.groupby(["task_id", "repeat"], sort=True):
        values = metric_values(
            frame["y_true"].to_numpy(),
            frame["calibrated_y_pred"].to_numpy(),
            frame["calibrated_probability_pd"].to_numpy(),
        )
        rows.append({"task_id": int(task_id), "repeat": int(repeat), **values})
    return pd.DataFrame(rows)


def summarize_fusions(
    fused: pd.DataFrame,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeats = []
    summaries = []
    for method_index, (method, frame) in enumerate(fused.groupby("method", sort=False)):
        frame = frame.copy()
        frame["selected_candidate"] = method
        repeat, summary = summarize_predictions(
            frame,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + method_index,
        )
        repeat.insert(0, "method", method)
        summary.insert(0, "method", method)
        repeats.append(repeat)
        summaries.append(summary)
    return pd.concat(repeats, ignore_index=True), pd.concat(summaries, ignore_index=True)


def save_figures(
    ranking: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    base: pd.DataFrame,
    output_root: Path,
) -> None:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    ordered = ranking.sort_values("macro_f1_mean")
    fig, axis = plt.subplots(figsize=(8.8, 4.4))
    axis.barh(
        [METHOD_LABELS[value] for value in ordered["method"]],
        ordered["macro_f1_mean"],
        color="#2c6e9b",
    )
    axis.axvline(0.5751533486, color="#555555", linestyle="--", label="Mean of separate SAZ tasks")
    axis.set(xlabel="Mean external-repeat macro F1", xlim=(0.5, 0.72), title="Eight-task fusion")
    for index, value in enumerate(ordered["macro_f1_mean"]):
        axis.text(value + 0.003, index, f"{value:.3f}", va="center")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fusion_method_ranking.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.3, 4.8))
    positions = np.arange(len(FUSION_METHODS))
    values = [
        repeat_metrics[repeat_metrics["method"] == method]["macro_f1"].to_numpy()
        for method in FUSION_METHODS
    ]
    axis.boxplot(values, tick_labels=[METHOD_LABELS[method] for method in FUSION_METHODS])
    for index, current in enumerate(values):
        axis.scatter(np.full(len(current), positions[index] + 1), current, color="#2c6e9b", zorder=3)
    axis.set(ylabel="Macro F1", ylim=(0.45, 0.8), title="Variation across five external repetitions")
    axis.tick_params(axis="x", labelrotation=12)
    fig.tight_layout()
    fig.savefig(figure_dir / "fusion_repeat_variation.png", dpi=220)
    plt.close(fig)

    reliability = base.groupby("task_id")["reliability_weight"].agg(["mean", "std"])
    fig, axis = plt.subplots(figsize=(8.8, 4.4))
    axis.bar(reliability.index, reliability["mean"], yerr=reliability["std"], capsize=3)
    axis.set(
        xlabel="PaHaW task",
        ylabel="Reliability weight",
        xticks=range(1, 9),
        title="Weights derived only from inner-fold macro F1",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "inner_reliability_weights.png", dpi=220)
    plt.close(fig)


def write_report(
    config: dict,
    ranking: pd.DataFrame,
    summaries: pd.DataFrame,
    pairwise: pd.DataFrame,
    repeat_metrics: pd.DataFrame,
    base: pd.DataFrame,
    output_root: Path,
) -> None:
    compact_ranking = ranking.copy()
    compact_ranking["method"] = compact_ranking["method"].map(METHOD_LABELS)
    compact_summary = summaries[
        [
            "method",
            "n_subjects",
            "macro_f1_mean",
            "macro_f1_ci_low",
            "macro_f1_ci_high",
            "balanced_accuracy_mean",
            "accuracy_mean",
            "sensitivity_pd_mean",
            "specificity_h_mean",
            "roc_auc_mean",
        ]
    ].copy()
    compact_summary["method"] = compact_summary["method"].map(METHOD_LABELS)
    compact_pairwise = pairwise.copy()
    compact_pairwise["candidate"] = compact_pairwise["candidate"].map(METHOD_LABELS)
    compact_pairwise["reference"] = compact_pairwise["reference"].map(METHOD_LABELS)
    for frame in (compact_ranking, compact_summary, compact_pairwise):
        for column in frame.select_dtypes(include=[np.number]).columns:
            if column not in {"rank", "n_subjects", "task_id"}:
                frame[column] = frame[column].map(lambda value: f"{value:.4f}")

    primary_method = config["validation"]["primary_method"]
    primary = summaries[summaries["method"] == primary_method].iloc[0]
    best = ranking.iloc[0]
    missing = base.groupby(["repeat", "subject_id"])["task_id"].nunique()
    report = f"""# Fusão das oito tarefas por participante

Experimento: `{config['experiment_id']}`. A codificação de imagem permanece `speed_altitude_azimuth` (SAZ), com largura fixa, descritor LPQ-RGB e SVM específico para cada tarefa.

## Protocolo

Cada repetição externa produz uma única predição por participante após combinar todas as tarefas disponíveis. Os participantes do teste externo não participam da escolha do SVM, da calibração ou do cálculo dos pesos:

- os hiperparâmetros de cada SVM vêm dos três folds internos já congelados;
- a calibração sigmoide é ajustada somente no treino externo usando esses folds internos;
- o peso de uma tarefa é `max(macro-F1 interno - 0,5; 0,01)`;
- três regras foram declaradas: voto majoritário, média das probabilidades calibradas e média calibrada ponderada;
- os três participantes sem tarefa 1 são fundidos pelas sete tarefas disponíveis.

O endpoint principal pré-declarado é a média não ponderada das probabilidades calibradas das oito tarefas, avaliada por macro F1 médio nas cinco repetições externas.

## Resultado principal

A fusão calibrada não ponderada obteve macro F1 {primary['macro_f1_mean']:.4f}, IC95% [{primary['macro_f1_ci_low']:.4f}; {primary['macro_f1_ci_high']:.4f}], acurácia {primary['accuracy_mean']:.4f}, sensibilidade {primary['sensitivity_pd_mean']:.4f} e especificidade {primary['specificity_h_mean']:.4f}.

O melhor valor descritivo foi `{METHOD_LABELS[best['method']]}` com {best['macro_f1_mean']:.4f}. A média dos oito classificadores separados era 0,5752, mas ela representa outro endpoint; a fusão mede uma decisão clínica por participante.

## Ranking

{markdown_table(compact_ranking)}

## Métricas externas

{markdown_table(compact_summary)}

## Comparações pareadas entre regras

Os métodos compartilham pacientes e folds. Os intervalos usam bootstrap estratificado por diagnóstico e a randomização troca o método por participante em todas as repetições. Holm cobre os três pares.

{markdown_table(compact_pairwise)}

## Comparações históricas

O pipeline antigo informou 0,6925 para fusão de oito tarefas. Drotár et al. informaram acurácia 0,813 com atributos diretamente extraídos do sinal e tarefas 2–8. Os valores são referências descritivas: representação, métricas auxiliares e validação diferem. O resultado atual usa separação explícita por participante e mantém todo ajuste dentro do treino externo.

Foram geradas {len(base)} probabilidades externas calibradas e {len(repeat_metrics)} linhas de métricas por repetição. Cada participante teve entre {missing.min()} e {missing.max()} tarefas disponíveis.
"""
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "task_fusion_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = (project_root / config["output_root"]).resolve()
    if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
        raise FileExistsError(f"Refusing to resume or overwrite non-empty run: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    source_run = (project_root / config["source_run"]).resolve()
    static_run = (project_root / config["static_run"]).resolve()
    split_path = static_run / "metrics" / "cv_splits.json"
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    encoding = config["selected_encoding"]

    metadata = pd.read_csv(
        source_run / "features" / f"{encoding}_metadata.csv", dtype={"subject_id": str}
    )
    with np.load(source_run / "features" / f"{encoding}_lpq_rgb.npz") as payload:
        features = payload["features"].astype(np.float32)
    if len(metadata) != len(features):
        raise ValueError("Feature matrix and metadata lengths differ.")
    source_predictions = pd.read_csv(
        source_run / "metrics" / "outer_predictions.csv", dtype={"subject_id": str}
    )
    source_predictions = source_predictions[source_predictions["encoding"] == encoding].copy()
    search = pd.read_csv(source_run / "metrics" / "inner_search.csv")
    search = search[search["encoding"] == encoding].copy()
    selected = selected_rows(search)

    base, fused = generate_calibrated_base_predictions(
        metadata, features, source_predictions, selected, split_payload
    )
    validation = config["validation"]
    repeat_metrics, summaries = summarize_fusions(
        fused,
        bootstrap_resamples=int(validation["bootstrap_resamples"]),
        bootstrap_seed=int(validation["bootstrap_seed"]),
    )
    ranking = summaries[["method", "macro_f1_mean"]].sort_values(
        ["macro_f1_mean", "method"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    method_frames = {method: fused[fused["method"] == method].copy() for method in FUSION_METHODS}
    pairwise = compare_global_method_pairs(
        method_frames,
        list(FUSION_METHODS),
        resamples=int(validation["paired_randomization_resamples"]),
        seed=int(validation["paired_randomization_seed"]),
    )
    calibration_metrics = task_calibration_metrics(base)

    base.to_csv(metrics_dir / "calibrated_task_predictions.csv", index=False)
    calibration_metrics.to_csv(metrics_dir / "calibrated_task_repeat_metrics.csv", index=False)
    fused.to_csv(metrics_dir / "fused_predictions.csv", index=False)
    repeat_metrics.to_csv(metrics_dir / "repeat_metrics.csv", index=False)
    summaries.to_csv(metrics_dir / "method_summary.csv", index=False)
    ranking.to_csv(metrics_dir / "fusion_ranking.csv", index=False)
    pairwise.to_csv(metrics_dir / "global_pairwise_comparisons.csv", index=False)
    pd.DataFrame(
        [
            {"path": str(source_run / "metrics" / "outer_predictions.csv"), "sha256": sha256_file(source_run / "metrics" / "outer_predictions.csv")},
            {"path": str(source_run / "metrics" / "inner_search.csv"), "sha256": sha256_file(source_run / "metrics" / "inner_search.csv")},
            {"path": str(source_run / "features" / f"{encoding}_lpq_rgb.npz"), "sha256": sha256_file(source_run / "features" / f"{encoding}_lpq_rgb.npz")},
            {"path": str(split_path), "sha256": sha256_file(split_path)},
        ]
    ).to_csv(metrics_dir / "source_artifacts.csv", index=False)
    save_figures(ranking, repeat_metrics, base, output_root)
    write_report(config, ranking, summaries, pairwise, repeat_metrics, base, output_root)

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
        "source_run": str(source_run),
        "reused_cv_split_checksum": split_payload["checksum"],
        "calibrated_task_predictions": len(base),
        "fused_predictions": len(fused),
        "fusion_methods": list(FUSION_METHODS),
        "primary_method": validation["primary_method"],
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            package: version(package)
            for package in (
                "joblib",
                "matplotlib",
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
            )
        },
    }
    save_json(manifest, output_root / "manifests" / "run_manifest.json")
    save_json(artifact_manifest(output_root), output_root / "manifests" / "artifact_manifest.json")
    print(ranking.to_string(index=False))
    print(summaries[["method", "macro_f1_mean", "accuracy_mean"]].to_string(index=False))
    print(f"complete in {manifest['duration_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
