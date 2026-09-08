"""Analyze all declared competitive task-subset endpoints without selecting results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdhms_restart.comparison import _weighted_macro_f1, _permuted_macro_f1, holm_adjust
from pdhms_restart.data import sha256_file
from pdhms_restart.evaluation import metric_values
from pdhms_restart.nested_selection import macro_f1


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"lpq_static": "Static LPQ + SVM", "lpq_selected": "Selected encoding: LPQ + SVM",
          "kinematic_svm": "Kinematics + pressure: SVM", "resnet18_transfer": "ResNet18 transfer + linear head"}


def aligned_channels(frame):
    people = frame[["subject_id", "y_true"]].drop_duplicates().sort_values("subject_id")
    if people.subject_id.duplicated().any():
        raise ValueError("Conflicting participant labels.")
    ids, truth = people.subject_id.tolist(), people.y_true.to_numpy(dtype=np.int8)
    channels = {}
    for key, current in frame.groupby(["task_id", "repeat"], sort=True):
        if current.subject_id.duplicated().any():
            raise ValueError("Pool outer folds once per participant/task/repetition.")
        prediction = current.set_index("subject_id").y_pred.reindex(ids).fillna(-1).to_numpy(dtype=np.int8)
        channels[key] = prediction
    return ids, truth, channels


def bootstrap_weights(truth, count, seed):
    rng = np.random.default_rng(seed)
    sampled = np.concatenate([rng.choice(np.flatnonzero(truth == label),
                             size=(count, int((truth == label).sum())), replace=True) for label in (0, 1)], axis=1)
    weights = np.zeros((count, len(truth)), dtype=np.int16)
    np.add.at(weights, (np.repeat(np.arange(count), len(truth)), sampled.ravel()), 1)
    return weights


def absolute_summary(frame, settings):
    ids, truth, channels = aligned_channels(frame)
    weights = bootstrap_weights(truth, settings["bootstrap_resamples"], settings["bootstrap_seed"])
    bootstrapped, observed = [], []
    for predictions in channels.values():
        available = predictions >= 0
        observed.append(macro_f1(truth[available], predictions[available]))
        bootstrapped.append(_weighted_macro_f1(predictions, truth, available, weights))
    low, high = np.percentile(np.mean(bootstrapped, axis=0), [2.5, 97.5])
    return {"macro_f1_mean": float(np.mean(observed)), "ci_low": float(low),
            "ci_high": float(high), "n_subjects": len(ids)}


def paired_summary(candidate, reference, settings):
    keys = ["repeat", "outer_fold", "task_id", "subject_id"]
    merged = candidate.merge(reference, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(candidate) or len(merged) != len(reference):
        raise ValueError("Predictions are not fully paired.")
    if not np.array_equal(merged.y_true_a, merged.y_true_b):
        raise ValueError("Paired truth mismatch.")
    ids, truth, a = aligned_channels(candidate)
    other_ids, other_truth, b = aligned_channels(reference)
    if ids != other_ids or not np.array_equal(truth, other_truth) or a.keys() != b.keys():
        raise ValueError("Different participant or task/repetition sets.")
    weights = bootstrap_weights(truth, settings["bootstrap_resamples"], settings["bootstrap_seed"])
    swaps = np.random.default_rng(settings["paired_randomization_seed"]).integers(
        0, 2, (settings["paired_randomization_resamples"], len(ids)), dtype=np.int8).astype(bool)
    observed, bootstrapped, randomized = [], [], []
    for key in a:
        available = a[key] >= 0
        if not np.array_equal(available, b[key] >= 0):
            raise ValueError("Paired task availability mismatch.")
        observed.append(macro_f1(truth[available], a[key][available]) - macro_f1(truth[available], b[key][available]))
        bootstrapped.append(_weighted_macro_f1(a[key], truth, available, weights) -
                            _weighted_macro_f1(b[key], truth, available, weights))
        pa, pb = _permuted_macro_f1(a[key][None, :], b[key][None, :], truth[None, :], available[None, :], swaps)
        randomized.append(pa - pb)
    delta = float(np.mean(observed))
    low, high = np.percentile(np.mean(bootstrapped, axis=0), [2.5, 97.5])
    # Use the same arithmetic for the observed and swapped mean paired differences.
    randomized = np.mean(randomized, axis=0)
    p_value = (1 + np.count_nonzero(np.abs(randomized) >= abs(delta) - 1e-15)) / (len(randomized) + 1)
    return {"delta_macro_f1": delta, "ci_low": float(low), "ci_high": float(high), "permutation_p": float(p_value)}


def adjust(rows):
    frame = holm_adjust(pd.DataFrame(rows)).rename(columns={"holm_p_global_32": "holm_p"})
    frame["family_size"] = len(frame)
    return frame


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        destination = directory / f"{name}.{extension}"
        fig.savefig(destination, dpi=250, bbox_inches="tight")
        if extension == "svg":
            destination.write_bytes(re.sub(rb"[ \t]+(?=\r?$)", b"", destination.read_bytes(), flags=re.MULTILINE))
    plt.close(fig)


def figures(summary, reduction, task_summary, config, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    colours = {"literature_T234": "#24516c", "all8": "#bc6c25"}
    method_order = config["method_order"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), sharey=True)
    for ax, endpoint, title in zip(axes, ("participant_fusion", "task_mean"),
                                   ("Participant fusion (primary for T2/T3/T4)", "Mean of task-specific macro F1"), strict=True):
        for offset, (task_set, colour) in zip((-0.12, 0.12), colours.items(), strict=True):
            rows = summary[(summary.task_set == task_set) & (summary.endpoint == endpoint)].set_index("method").loc[method_order]
            values = rows.macro_f1_mean.to_numpy()
            ax.errorbar(values, np.arange(4) + offset, xerr=[values - rows.ci_low, rows.ci_high - values],
                        fmt="o", capsize=3, color=colour, label="T2/T3/T4" if task_set == "literature_T234" else "All eight tasks")
        ax.set_yticks(range(4), [LABELS[name] for name in method_order])
        ax.set_xlim(0.35, 1.0)
        ax.set_xlabel("Macro F1 with 95% pointwise participant interval")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_figure(fig, output / "figures", "competitive_methods")

    fig, ax = plt.subplots(figsize=(9, 4))
    rows = reduction.set_index("method").loc[method_order]
    values = rows.delta_macro_f1.to_numpy()
    ax.errorbar(values, range(4), xerr=[values - rows.ci_low, rows.ci_high - values], fmt="o", color="#24516c", capsize=4)
    ax.axvline(0, linestyle="--", color="#888888")
    ax.set_yticks(range(4), [LABELS[name] for name in method_order])
    ax.invert_yaxis()
    ax.set_xlabel("Three-task minus eight-task participant macro F1")
    ax.set_title("Does the literature-defined task subset improve participant classification?")
    for index, row in enumerate(rows.itertuples()):
        ax.annotate(f"p(Holm)={row.holm_p:.3f}", (row.ci_high, index), xytext=(7, 4), textcoords="offset points", fontsize=9)
    ax.margins(x=0.4, y=0.3)
    fig.tight_layout()
    save_figure(fig, output / "figures", "task_subset_effects")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for method in method_order:
        rows = task_summary[(task_summary.task_set == "literature_T234") & (task_summary.method == method)].sort_values("task_id")
        ax.plot(rows.task_id, rows.macro_f1, marker="o", label=LABELS[method])
    ax.set_xticks([2, 3, 4], ["T2: l", "T3: le", "T4: les"])
    ax.set_ylim(0.35, 1.0)
    ax.set_ylabel("Task macro F1, mean of five repetitions")
    ax.set_title("The three fixed tasks: all four methods")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    fig.tight_layout()
    save_figure(fig, output / "figures", "competitive_task_profiles")


def markdown(frame):
    rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    metrics = output / "metrics"
    manifest_path = output / "manifests/run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["completed_outer_units"] != 25:
        raise ValueError("All 25 units must complete before analysis.")
    frames = {endpoint: pd.read_csv(metrics / filename, dtype={"subject_id": str}, float_precision="round_trip")
              for endpoint, filename in (("task_mean", "outer_task_predictions.csv"), ("participant_fusion", "outer_fusion_predictions.csv"))}
    settings = config["validation"]
    summaries, repeats, pair_frames = [], [], []
    for endpoint, frame in frames.items():
        for (task_set, method), current in frame.groupby(["task_set", "method"], sort=False):
            scores = []
            for (task_id, repeat), group in current.groupby(["task_id", "repeat"], sort=True):
                result = metric_values(group.y_true.to_numpy(), group.y_pred.to_numpy(), group.score_pd.to_numpy())
                scores.append(result)
                repeats.append({"task_set": task_set, "method": method, "endpoint": endpoint,
                                "task_id": task_id, "repeat": repeat, **result})
            summaries.append({"task_set": task_set, "method": method, "endpoint": endpoint,
                              **absolute_summary(current, settings),
                              **{name: float(np.mean([row[name] for row in scores])) for name in scores[0] if name != "macro_f1"}})
        for task_set in config["task_sets"]:
            pairs = []
            for reference, candidate in combinations(config["method_order"], 2):
                a = frame[(frame.task_set == task_set) & (frame.method == candidate)]
                b = frame[(frame.task_set == task_set) & (frame.method == reference)]
                pairs.append({"task_set": task_set, "endpoint": endpoint, "candidate": candidate, "reference": reference,
                              **paired_summary(a, b, settings)})
            pair_frames.append(adjust(pairs))
            print(f"Analyzed {task_set}/{endpoint}", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(metrics / "global_summary.csv", index=False)
    repeat_metrics = pd.DataFrame(repeats)
    repeat_metrics.to_csv(metrics / "repeat_metrics.csv", index=False)
    task_summary = repeat_metrics[repeat_metrics.endpoint == "task_mean"].groupby(["task_set", "method", "task_id"])[list(metric_values(np.array([0, 1]), np.array([0, 1]), np.array([0., 1.])).keys())].mean().reset_index()
    task_summary.to_csv(metrics / "task_summary.csv", index=False)
    paired = pd.concat(pair_frames, ignore_index=True)
    paired.to_csv(metrics / "method_comparisons.csv", index=False)
    fused = frames["participant_fusion"]
    reduction = adjust([{"method": method, "candidate": "literature_T234", "reference": "all8",
                         **paired_summary(fused[(fused.task_set == "literature_T234") & (fused.method == method)],
                                          fused[(fused.task_set == "all8") & (fused.method == method)], settings)} for method in config["method_order"]])
    reduction.to_csv(metrics / "task_subset_comparisons.csv", index=False)
    selections = pd.read_csv(metrics / "selections.csv")
    selections.groupby(["task_set", "method", "encoding"]).size().reset_index(name="count").to_csv(metrics / "encoding_selection_frequency.csv", index=False)
    figures(summary, reduction, task_summary, config, output)
    display = summary.copy()
    display["Macro F1 [95% CI]"] = [f"{row.macro_f1_mean:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]" for row in display.itertuples()]
    primary = paired[(paired.task_set == config["primary_task_set"]) & (paired.endpoint == "participant_fusion")].copy()
    for name in ("delta_macro_f1", "ci_low", "ci_high", "permutation_p", "holm_p"):
        primary[name] = primary[name].map(lambda value: f"{value:.4f}")
    report = "# Competitive references on a fixed task subset\n\n"
    report += "All four declared pipelines were fitted with training-only selection. T2/T3/T4 were fixed from the Casademunt thesis before fitting; all eight tasks are a secondary comparator. This is development-cohort reanalysis, not external validation.\n\n"
    report += markdown(display[["task_set", "method", "endpoint", "Macro F1 [95% CI]", "accuracy"]])
    report += "\n\n## Primary paired comparisons: T2/T3/T4 participant fusion\n\n"
    report += markdown(primary[["candidate", "reference", "delta_macro_f1", "ci_low", "ci_high", "permutation_p", "holm_p"]])
    report += "\n\n## Within-method task-set comparisons\n\n" + markdown(reduction)
    report += "\n\nThe primary family has six contrasts. Three other endpoint families each have six; the task-set fusion family has four. Confidence intervals are pointwise and conditional on saved predictions. Folds are pooled before each repetition's score; participant resampling keeps all tasks and repetitions together. Bootstrap and permutation use separate declared seeds.\n\n"
    report += "LPQ-selected and ResNet18-transfer select their own encoding among nine candidates inside training. The transfer CNN has a frozen ImageNet backbone and a trained regularized linear head; it is not end-to-end fine-tuning. Kinematics retain native signal levels, pen-up movement and timing through a fixed 83-attribute descriptor, not an exact reproduction of Drotar. The comparison is between complete pipelines; feature dimension, input information, classifier and search budget differ.\n\n"
    report += "Reducing the task set changes the task-mean estimand. Only the paired participant-fusion contrasts directly compare decisions for the same people across task sets, and these may also involve a different internally selected encoding. No best subset, seed or architecture was chosen from these outer results.\n"
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "competitive_task_subset_report.md").write_text(report, encoding="utf-8")
    manifest.update(status="analysis_complete", analysis_source_sha256=sha256_file(Path(__file__)),
                    analysis_completed_at_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Analysis and figures complete; independent verification remains.", flush=True)


if __name__ == "__main__":
    main()
