"""Summarize the prespecified new experiments from frozen outer predictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdhms_restart.comparison import _weighted_macro_f1, holm_adjust, paired_macro_f1_comparison, paired_task_mean_macro_f1_comparison
from pdhms_restart.data import sha256_file
from pdhms_restart.evaluation import metric_values


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"static": "Static", "auth9": "Selection among 9 encodings", "auth8": "Selection among 8 dynamic encodings",
          "circular_shift": "Within-stroke circular shift", "joint_permutation": "Within-stroke joint permutation"}


def seed_expanded(frame: pd.DataFrame, seed_order: list[int], repeats: int) -> pd.DataFrame:
    result = frame.copy()
    if "control_seed" in result and result.control_seed.nunique() > 1:
        positions = {seed: index for index, seed in enumerate(seed_order)}
        result["repeat"] = result["repeat"] + result.control_seed.map(positions) * repeats
        if result["repeat"].isna().any():
            raise ValueError("Unexpected control seed.")
    return result


def duplicate_reference(reference: pd.DataFrame, seed_order: list[int], repeats: int) -> pd.DataFrame:
    pieces = []
    for index, seed in enumerate(seed_order):
        current = reference.copy()
        current["repeat"] = current["repeat"] + index * repeats
        current["control_seed"] = seed
        pieces.append(current)
    return pd.concat(pieces, ignore_index=True)


def absolute_summary(frame: pd.DataFrame, *, resamples: int, seed: int) -> dict:
    subjects = frame[["subject_id", "y_true"]].drop_duplicates().sort_values("subject_id")
    if subjects.subject_id.duplicated().any():
        raise ValueError("Conflicting labels across repeated predictions.")
    ids = subjects.subject_id.tolist()
    truth = subjects.y_true.to_numpy(dtype=np.int8)
    rng = np.random.default_rng(seed)
    sampled = np.concatenate([rng.choice(np.flatnonzero(truth == label), size=(resamples, int((truth == label).sum())), replace=True) for label in (0, 1)], axis=1)
    weights = np.zeros((resamples, len(ids)), dtype=np.int16)
    np.add.at(weights, (np.repeat(np.arange(resamples), len(ids)), sampled.ravel()), 1)
    bootstrapped, observed = [], []
    for _, current in frame.groupby(["task_id", "repeat"], sort=True):
        if current.subject_id.duplicated().any():
            raise ValueError("Duplicate participant within task/repetition. Expand seeds first.")
        prediction = current.set_index("subject_id").y_pred.reindex(ids).fillna(-1).to_numpy(dtype=np.int8)
        available = prediction >= 0
        observed.append(metric_values(truth[available], prediction[available], prediction[available])["macro_f1"])
        bootstrapped.append(_weighted_macro_f1(prediction, truth, available, weights))
    values = np.mean(bootstrapped, axis=0)
    return {"macro_f1_mean": float(np.mean(observed)), "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)), "n_subjects": len(ids)}


def adjusted(rows: list[dict]) -> pd.DataFrame:
    result = holm_adjust(pd.DataFrame(rows)).rename(columns={"holm_p_global_32": "holm_p"})
    result["family_size"] = len(result)
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def make_figures(global_pairs, fusion_pairs, task_pairs, selections, output):
    target = output / "figures"
    target.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    contrast_labels = {("auth9", "static"): "9-encoding selection − static", ("auth8", "static"): "Dynamic selection − static",
                       ("auth8", "circular_shift"): "Aligned − circular shift", ("auth8", "joint_permutation"): "Aligned − joint permutation"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})
    for ax, frame, title in zip(axes, (global_pairs, fusion_pairs), ("Task-mean macro F1 (Holm family: 4)", "Participant fusion macro F1 (Holm family: 2)"), strict=True):
        values = frame.delta_task_mean_macro_f1.to_numpy()
        y = np.arange(len(frame))
        ax.errorbar(values, y, xerr=np.array([values - frame.ci_low, frame.ci_high - values]), fmt="o", color="#24516c", capsize=4)
        ax.set_yticks(y, [contrast_labels[(row.candidate, row.reference)] for row in frame.itertuples()])
        ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Paired difference with 95% pointwise interval")
        for index, row in enumerate(frame.itertuples()):
            ax.annotate(f"p(Holm)={row.holm_p:.3f}", (row.ci_high, index), xytext=(5, 8), textcoords="offset points", fontsize=8)
        ax.margins(x=0.4, y=0.3)
    fig.suptitle("Selection uses training participants only; intervals resample participants", fontsize=12)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(target / f"paired_effects.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, ((candidate, reference), title) in zip(axes.ravel(), contrast_labels.items(), strict=True):
        rows = task_pairs[(task_pairs.candidate == candidate) & (task_pairs.reference == reference)].sort_values("task_id")
        values = rows.delta_macro_f1.to_numpy()
        ax.errorbar(values, rows.task_id, xerr=np.array([values - rows.ci_low, rows.ci_high - values]), fmt="o", color="#24516c", capsize=3)
        ax.axvline(0, linestyle="--", color="#888888", linewidth=1)
        ax.set_title(title)
        ax.set_yticks(range(1, 9), [f"T{value}" for value in range(1, 9)])
        ax.set_xlabel("Difference in macro F1")
    axes[0, 0].invert_yaxis()
    fig.suptitle("Exploratory task contrasts: participant intervals; Holm family of 32", fontsize=12)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(target / f"task_effects.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    selection = selections[selections.phase == "outer_train"]
    frequency = pd.crosstab(selection.policy, selection.encoding, normalize="index").reindex(list(LABELS)).fillna(0)
    frequency.to_csv(output / "metrics/encoding_selection_frequency.csv")
    fig, ax = plt.subplots(figsize=(12, 5))
    left = np.zeros(len(frequency))
    colours = plt.get_cmap("tab20").colors
    for index, encoding in enumerate(frequency.columns):
        values = frequency[encoding].to_numpy() * 100
        ax.barh(np.arange(len(frequency)), values, left=left, label=encoding.replace("_", "+"), color=colours[index])
        left += values
    ax.set_yticks(np.arange(len(frequency)), [LABELS[name] for name in frequency.index])
    ax.set_xlabel("Percentage of outer-training encoding selections")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8)
    ax.set_title("One encoding per outer fold; controls summarize three fixed seeds")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(target / f"encoding_selection_frequency.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    metrics = output / "metrics"
    manifest_path = output / "manifests/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["completed_outer_units"] != 175:
        raise ValueError("All authentic and control outer units must finish before analysis.")
    task = pd.read_csv(metrics / "outer_task_predictions.csv", dtype={"subject_id": str})
    fused = pd.read_csv(metrics / "outer_fusion_predictions.csv", dtype={"subject_id": str})
    seeds = config["controls"]["seeds"]
    repeats = config["validation"]["repeats"]
    resamples = config["validation"]["bootstrap_resamples"]
    seed = config["validation"]["bootstrap_seed"]
    frames = {policy: seed_expanded(frame, seeds, repeats) for policy, frame in task.groupby("policy", sort=False)}
    repeats_rows, task_rows, global_rows = [], [], []
    for policy, frame in task.groupby("policy", sort=False):
        for (control_seed, task_id, repeat), current in frame.groupby(["control_seed", "task_id", "repeat"]):
            repeats_rows.append({"policy": policy, "control_seed": int(control_seed), "task_id": int(task_id), "repeat": int(repeat),
                                 **metric_values(current.y_true.to_numpy(), current.y_pred.to_numpy(), current.decision_score_pd.to_numpy())})
        n_seeds = int(frame.control_seed.nunique())
        for task_id, current in frames[policy].groupby("task_id"):
            task_rows.append({"policy": policy, "task_id": int(task_id), "seeds": n_seeds, **absolute_summary(current, resamples=resamples, seed=seed)})
        global_rows.append({"policy": policy, "endpoint": "task_mean", "seeds": n_seeds, **absolute_summary(frames[policy], resamples=resamples, seed=seed)})
    for policy, frame in fused.groupby("policy", sort=False):
        global_rows.append({"policy": policy, "endpoint": "participant_fusion", "seeds": 1, **absolute_summary(frame, resamples=resamples, seed=seed)})
    repeat_metrics = pd.DataFrame(repeats_rows)
    repeat_metrics.to_csv(metrics / "repeat_metrics.csv", index=False)
    pd.DataFrame(task_rows).to_csv(metrics / "task_summary.csv", index=False)
    global_summary = pd.DataFrame(global_rows)
    global_summary.to_csv(metrics / "global_summary.csv", index=False)
    repeat_metrics.groupby(["policy", "control_seed"])["macro_f1"].mean().reset_index(name="task_mean_macro_f1").to_csv(metrics / "seed_summary.csv", index=False)

    contrast_frames = [("auth9", "static", frames["auth9"], frames["static"]), ("auth8", "static", frames["auth8"], frames["static"])]
    for method in config["controls"]["methods"]:
        contrast_frames.append(("auth8", method, duplicate_reference(frames["auth8"], seeds, repeats), frames[method]))
    global_pairs, task_pairs = [], []
    for index, (candidate, reference, a, b) in enumerate(contrast_frames):
        result = paired_task_mean_macro_f1_comparison(a, b, resamples=resamples, seed=seed + index)
        global_pairs.append({"candidate": candidate, "reference": reference, **result})
        for task_id in config["tasks"]:
            result = paired_macro_f1_comparison(a[a.task_id == task_id], b[b.task_id == task_id], resamples=resamples, seed=seed + 10 + 8 * index + task_id)
            task_pairs.append({"candidate": candidate, "reference": reference, "task_id": task_id, **result})
    global_pairs = adjusted(global_pairs)
    task_pairs = adjusted(task_pairs)
    fusion_pairs = adjusted([{"candidate": policy, "reference": "static", **paired_task_mean_macro_f1_comparison(fused[fused.policy == policy], fused[fused.policy == "static"], resamples=resamples, seed=seed + 100 + index)} for index, policy in enumerate(("auth9", "auth8"))])
    global_pairs.to_csv(metrics / "task_global_comparisons.csv", index=False)
    fusion_pairs.to_csv(metrics / "fusion_comparisons.csv", index=False)
    task_pairs.to_csv(metrics / "task_specific_comparisons.csv", index=False)
    selections = pd.read_csv(metrics / "selections.csv")
    make_figures(global_pairs, fusion_pairs, task_pairs, selections, output)

    display = global_summary.copy()
    display["Method"] = display.policy.map(LABELS)
    display["Endpoint"] = display.endpoint
    display["Macro F1"] = display.macro_f1_mean.map(lambda x: f"{x:.4f}")
    display["95% CI"] = [f"[{row.ci_low:.4f}, {row.ci_high:.4f}]" for row in display.itertuples()]
    contrasts = global_pairs.copy()
    contrasts["Contrast"] = contrasts.candidate + " vs " + contrasts.reference
    contrasts["Delta"] = contrasts.delta_task_mean_macro_f1.map(lambda x: f"{x:+.4f}")
    contrasts["95% CI"] = [f"[{row.ci_low:.4f}, {row.ci_high:.4f}]" for row in contrasts.itertuples()]
    contrasts["Holm p"] = contrasts.holm_p.map(lambda x: f"{x:.4f}")
    supported = global_pairs[global_pairs.supported_holm_0_05]
    findings = "Nenhum dos quatro contrastes globais de tarefa sobrevive à correção de Holm." if supported.empty else "Contrastes globais de tarefa com p de Holm < 0,05: " + "; ".join(f"{row.candidate} vs {row.reference} (delta {row.delta_task_mean_macro_f1:+.4f})" for row in supported.itertuples()) + "."
    report = f"""# Seleção no treinamento e controles espaciais

Execução concluída a partir do protocolo registrado antes dos ajustes. São duas
análises novas, com LPQ+SVM fixos, usando 597 registros de 75 participantes.

## Resultados

{markdown_table(display[["Method", "Endpoint", "Macro F1", "95% CI"]])}

Estática é reavaliada no mesmo protocolo novo. auth9 pode escolher qualquer uma
das nove codificações, incluindo estática; auth8 escolhe apenas entre as oito
dinâmicas. Uma codificação global é escolhida em cada treinamento, com SVM
específico por tarefa. Os controles também escolhem entre oito codificações e
recebem o mesmo orçamento de ajuste interno. Seus valores são médias das três
sementes previamente fixadas, sem escolher uma semente ou combinar predições.

{markdown_table(contrasts[["Contrast", "Delta", "95% CI", "Holm p"]])}

{findings}

As diferenças de fusão estão em `metrics/fusion_comparisons.csv`, família
secundária de dois contrastes. Os 32 contrastes por tarefa são exploratórios e
estão em `metrics/task_specific_comparisons.csv`, com correção conjunta.

## O que foi executado

1. Reutilizadas as 25 divisões externas por participante, com novas divisões
   internas COMUNS a todas as tarefas. Toda seleção de codificação e parâmetros
   ocorre no treinamento externo. Alterar os dados de teste não altera a seleção,
   conforme teste de regressão do procedimento.
2. A fusão de cada política autêntica é escolhida por predições intermediárias
   fora da amostra. Cada treinamento intermediário repete a seleção de codificação
   e SVM em três subdivisões próprias. Calibração e pesos permanecem nesse
   treinamento. O conjunto externo não escolhe a regra de fusão.
3. Os sinais dos controles são deslocados circularmente ou permutados conjuntamente
   dentro de cada traço, em três sementes. Cada condição transforma treinamento
   e teste e repete sua própria seleção de codificação e parâmetros. Os controles
   avaliam apenas tarefas; não foi acrescentada fusão para essas condições.
4. Os cinco folds externos são concatenados antes do cálculo de cada métrica por
   repetição. Reamostragem e trocas pareadas usam participantes, preservando juntas
   tarefas, repetições e sementes. Os intervalos são pontuais de 95%.

## Interpretação e limites

Esta execução avalia um procedimento que escolhe a representação no treinamento;
não estima somente o desempenho da SAZ favorecida pela análise anterior. A
frequência de escolha de cada codificação está em `metrics/encoding_selection_frequency.csv`.
As diferenças em relação aos valores antigos também refletem a mudança dos folds
internos e do procedimento de seleção, portanto não constituem ganho pareado em
relação ao experimento antigo.

Uma vantagem sobre os controles apoiaria a utilidade da associação sinal–posição
para este descritor e classificador. Uma ausência de diferença não demonstra
equivalência nem inutilidade fisiológica dos sinais. A permutação também altera
a continuidade local, e conservar os vetores por segmento não conserva
necessariamente o histograma de pixels após cruzamentos, comprimentos diferentes
e antialiasing. O efeito não pode ser atribuído exclusivamente a um mecanismo
fisiológico ou separado de todas as mudanças de textura.

A separação agora abrange as decisões avaliadas, mas PaHaW já orientou o desenho
da pesquisa. Esta é uma reavaliação interna, não validação externa independente.
A comparação anterior entre SVM, LR, RF e CNN permanece exploratória e não foi
revalidada nesta etapa. As incertezas são condicionais às predições ajustadas;
não incluem toda a variabilidade de repetir o desenvolvimento e treinamento.

## Arquivos e reprodução

- `manifests/selection_plan.json`: participantes de todos os níveis de divisão.
- `units/`: buscas internas completas, seleção registrada antes do teste, OOF
  intermediárias e modelos ajustados. Os modelos não são versionados no Git.
- `metrics/`: predições externas, escolhas, métricas e todos os contrastes declarados.
- `figures/`: efeitos pareados, resultados por tarefa e frequências de seleção.
- `../spatial-controls-v1/`: descritores, diagnósticos e exemplos dos controles.
- `manifests/run_lock.json`: hashes de configuração, código e entradas, com
  bloqueio de retomada caso uma dessas fontes mude.

O código e os parâmetros usados nos experimentos anteriores foram preservados.
Não houve treino de CNN, nova família de classificadores ou alteração do artigo.
"""
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "nested_selection_controls_report.md").write_text(report, encoding="utf-8")
    manifest.update(status="analysis_complete", analysis_at_utc=datetime.now(timezone.utc).isoformat(),
                    analysis_source_sha256=sha256_file(Path(__file__)),
                    analysis_config_sha256=sha256_file(args.config),
                    global_task_contrasts=4, secondary_fusion_contrasts=2, exploratory_task_contrasts=32)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(global_summary.to_string(index=False))
    print(global_pairs.to_string(index=False))
    print(fusion_pairs.to_string(index=False))
    print(report.split("## O que foi executado")[0], flush=True)


if __name__ == "__main__":
    main()
