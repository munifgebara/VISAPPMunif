"""Rank all completed static and pseudodynamic image encodings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdhms_restart.comparison import compare_global_method_pairs
from pdhms_restart.data import sha256_file
from pdhms_restart.evaluation import metric_values


LABELS = {
    "static": "static",
    "speed": "speed",
    "pressure": "pressure",
    "altitude": "altitude",
    "azimuth": "azimuth",
    "speed_pressure_altitude": "speed + pressure + altitude",
    "speed_pressure_azimuth": "speed + pressure + azimuth",
    "speed_altitude_azimuth": "speed + altitude + azimuth",
    "pressure_altitude_azimuth": "pressure + altitude + azimuth",
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


def load_predictions(project_root: Path, specification: dict) -> tuple[pd.DataFrame, Path]:
    path = project_root / specification["run"] / "metrics" / "outer_predictions.csv"
    frame = pd.read_csv(path, dtype={"subject_id": str})
    if specification["encoding"] is not None:
        frame = frame[frame["encoding"] == specification["encoding"]].copy()
    return frame, path


def task_mean_macro_f1(predictions: pd.DataFrame) -> float:
    values = []
    for (_, _), frame in predictions.groupby(["task_id", "repeat"], sort=True):
        values.append(
            metric_values(
                frame["y_true"].to_numpy(),
                frame["y_pred"].to_numpy(),
                frame["decision_score_pd"].to_numpy(),
            )["macro_f1"]
        )
    return float(np.mean(values))


def artifact_manifest(output_root: Path) -> dict:
    target = output_root / "manifests" / "artifact_manifest.json"
    rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path != target:
            rows.append(
                {
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"artifact_count": len(rows), "artifacts": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = (project_root / config["output_root"]).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty comparison: {output_root}")
    metrics_dir = output_root / "metrics"
    figures_dir = output_root / "figures"
    reports_dir = output_root / "reports"
    for directory in (metrics_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    sources = []
    for name, specification in config["prediction_sources"].items():
        frame, path = load_predictions(project_root, specification)
        frames[name] = frame
        sources.append(
            {
                "encoding": name,
                "prediction_path": str(path.resolve()),
                "prediction_sha256": sha256_file(path),
                "rows_used": len(frame),
            }
        )
    pd.DataFrame(sources).to_csv(metrics_dir / "prediction_sources.csv", index=False)

    ranking = pd.DataFrame(
        [
            {"encoding": name, "task_mean_macro_f1": task_mean_macro_f1(frame)}
            for name, frame in frames.items()
        ]
    ).sort_values(["task_mean_macro_f1", "encoding"], ascending=[False, True]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(metrics_dir / "all_encoding_ranking.csv", index=False)

    comparisons = compare_global_method_pairs(
        frames,
        list(config["prediction_sources"]),
        resamples=int(config["validation"]["paired_randomization_resamples"]),
        seed=int(config["validation"]["paired_randomization_seed"]),
    )
    comparisons.to_csv(metrics_dir / "all_encoding_global_pairwise.csv", index=False)

    ordered = ranking.sort_values("task_mean_macro_f1")
    fig, axis = plt.subplots(figsize=(9.5, 6.0))
    colours = ["#777777" if value == "static" else "#2878B5" for value in ordered["encoding"]]
    axis.barh(
        [LABELS[value] for value in ordered["encoding"]],
        ordered["task_mean_macro_f1"],
        color=colours,
    )
    axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    axis.set(xlabel="Unweighted mean of task-specific macro F1", xlim=(0.48, 0.61))
    axis.set_title("Static and pseudodynamic encoding ranking")
    for index, value in enumerate(ordered["task_mean_macro_f1"]):
        axis.text(value + 0.001, index, f"{value:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "all_encoding_ranking.png", dpi=220)
    plt.close(fig)

    compact_ranking = ranking.copy()
    compact_ranking["task_mean_macro_f1"] = compact_ranking["task_mean_macro_f1"].map(
        lambda value: f"{value:.4f}"
    )
    compact_comparisons = comparisons.copy()
    numeric = [
        "delta_task_mean_macro_f1",
        "ci_low",
        "ci_high",
        "permutation_p",
        "holm_p_global_36",
    ]
    compact_comparisons[numeric] = compact_comparisons[numeric].map(lambda value: f"{value:.4f}")
    winner = ranking.iloc[0]
    report = f"""# Comparison of all image encodings

Experiment: `{config['experiment_id']}`. This analysis combines the frozen outer predictions from the static baseline, four isolated signals, and four three-signal RGB encodings. No model was retrained and all methods use the same participant-level splits.

## Ranking

The declared endpoint is the unweighted mean of task-specific macro F1 over eight tasks and five repetitions. The selected encoding is `{winner['encoding']}` with {winner['task_mean_macro_f1']:.4f}.

{markdown_table(compact_ranking)}

## Global paired comparisons

For each pair, delta is candidate minus reference. Bootstrap samples participants stratified by diagnosis and uses the same draw across tasks and repetitions. Randomization swaps method labels per participant across all tasks and repetitions. Holm correction covers all 36 comparisons.

{markdown_table(compact_comparisons)}

The ranking defines the representation used in the next classifier-comparison stage. Statistical uncertainty remains part of the interpretation: a numerical first place does not imply that the selected encoding is superior to every alternative.
"""
    (reports_dir / "all_encoding_comparison_report.md").write_text(report, encoding="utf-8")

    git_commit = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    save_json(
        {
            "experiment_id": config["experiment_id"],
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_git_commit": git_commit,
            "encodings_compared": len(frames),
            "global_pairwise_comparisons": len(comparisons),
            "selected_encoding": winner["encoding"],
            "selected_task_mean_macro_f1": float(winner["task_mean_macro_f1"]),
        },
        output_root / "manifests" / "run_manifest.json",
    )
    save_json(artifact_manifest(output_root), output_root / "manifests" / "artifact_manifest.json")
    print(ranking.to_string(index=False))
    print(f"selected: {winner['encoding']}")


if __name__ == "__main__":
    main()
