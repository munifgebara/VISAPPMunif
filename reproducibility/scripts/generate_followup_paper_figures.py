"""Publish figures from the audited selection/control and competitive runs."""

from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdhms_restart.data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/2026-09-restart/runs"
OUTPUT = ROOT / "paper/figures"


def main():
    nested = RUNS / "nested-selection-controls-v1"
    competitive = RUNS / "competitive-task-subset-v1"
    paths = [nested / "metrics/global_summary.csv", nested / "metrics/task_global_comparisons.csv"]
    summaries, pairs = [pd.read_csv(path) for path in paths]
    order = ["static", "auth9", "auth8", "circular_shift", "joint_permutation"]
    labels = ["Static", "Select among nine", "Select dynamic only", "Circular-shift control", "Permutation control"]
    rows = summaries[summaries.endpoint == "task_mean"].set_index("policy").loc[order]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    values = rows.macro_f1_mean.to_numpy()
    axes[0].errorbar(values, range(5), xerr=[values - rows.ci_low, rows.ci_high - values], fmt="o", color="#24516c", capsize=3)
    axes[0].set_yticks(range(5), labels)
    axes[0].set_xlim(0.48, 0.63)
    axes[0].invert_yaxis()
    axes[0].set_title("All eight tasks: training-selected policies")
    axes[0].set_xlabel("Task-mean macro F1")
    values = pairs.delta_task_mean_macro_f1.to_numpy()
    axes[1].errorbar(values, range(4), xerr=[values - pairs.ci_low, pairs.ci_high - values], fmt="o", color="#24516c", capsize=3)
    axes[1].set_yticks(range(4), ["Nine minus static", "Dynamic minus static", "Dynamic minus shift", "Dynamic minus permutation"])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#888888", linestyle="--")
    axes[1].set_xlim(-0.026, 0.06)
    axes[1].set_title("Four paired contrasts")
    axes[1].set_xlabel("Difference in task-mean macro F1")
    for index, row in enumerate(pairs.itertuples()):
        axes[1].annotate(f"p(Holm)={row.holm_p:.3f}", (row.ci_high, index), xytext=(2, 9), textcoords="offset points", fontsize=8)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT / f"nested_selection_controls.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    names = ["nested_selection_controls"]
    for name in ("competitive_methods", "task_subset_effects", "competitive_task_profiles"):
        for extension in ("pdf", "png", "svg"):
            source = competitive / "figures" / f"{name}.{extension}"
            shutil.copyfile(source, OUTPUT / source.name)
            paths.append(source)
        names.append(name)
    source = RUNS / "spatial-controls-v1/figures/spatial_controls_examples.png"
    shutil.copyfile(source, OUTPUT / source.name)
    paths.append(source)
    names.append("spatial_controls_examples")
    manifest = {"generator": "scripts/generate_followup_paper_figures.py",
                "generator_sha256": sha256_file(Path(__file__)),
                "input_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths},
                "output_sha256": {path.name: sha256_file(path) for name in names for path in OUTPUT.glob(f"{name}.*")},
                "new_model_fits": 0, "figure_names": names}
    (OUTPUT / "followup_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"published_figures": names, "new_model_fits": 0}, indent=2))


if __name__ == "__main__":
    main()
