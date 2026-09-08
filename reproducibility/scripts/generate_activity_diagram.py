"""Create the approved illustrated UML activity diagram from saved experiments.

Run with the project Python environment and PYTHONPATH=src. No model is fitted,
and no manuscript text is edited. Images and maps are observed data.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from pdhms_restart.lpq import preprocess


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/2026-09-restart/runs"
OUT = ROOT / "paper/proposals/activity_diagram_v1"
PDF = ROOT / "output/pdf/activity_diagram_proposal.pdf"
H = 138
VERTICAL_SCALE = .86
INK = "#233648"
LINE = "#344655"
BLUE = "#28698b"
TEAL = "#087b76"
MUTED = "#566875"
PALE = "#f0f6f8"
SOURCES = {}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def record(path):
    SOURCES[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()


def read(run, file):
    path = RUNS / run / "metrics" / file
    record(path)
    return pd.read_csv(path, dtype={"subject_id": str})


def image_file(path):
    path = Path(path)
    record(path)
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    static = read("static-svm-v1", "image_manifest.csv")
    triplets = read("dynamic-triplet-svm-v1", "image_manifest.csv")
    examples = read("cnn-xai-v1", "figure_example_selection.csv")
    predictions = read("cnn-v1", "fused_majority_predictions.csv")
    controls = static[static.label == "H"]
    complete = controls.groupby("subject_id").task_id.nunique()
    subject = sorted(complete[complete == 8].index)[0]
    trace = image_file(controls[(controls.subject_id == subject) & (controls.task_id == 1)].image_path.iloc[0])
    example = examples[(examples.y_true == 1) & examples.correct].iloc[0]
    image_row = triplets[(triplets.subject_id == example.subject_id) &
                         (triplets.task_id == example.task_id) &
                         (triplets.encoding == "speed_altitude_azimuth")].iloc[0]
    cnn_input = preprocess(image_file(image_row.image_path), output_size=128)
    map_path = RUNS / "cnn-xai-v1/maps/spatial_maps.npz"
    record(map_path)
    with np.load(map_path) as maps:
        cam = maps["gradcam_maps"][int(example.prediction_index)]
    assert cam.shape == cnn_input.shape[:2] and cam.max() > 0
    matrices = []
    for _, frame in predictions[predictions.algorithm == "svm"].groupby("repeat"):
        assert len(frame) == 75 and frame.subject_id.is_unique
        counts = np.zeros((2, 2), dtype=float)
        np.add.at(counts, (frame.y_true.to_numpy(int), frame.y_pred.to_numpy(int)), 1)
        matrices.append(counts / counts.sum(axis=1, keepdims=True))
    matrix = np.mean(matrices, axis=0)

    fig = plt.figure(figsize=(8, H / 12.5 * VERTICAL_SCALE))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 100), ylim=(H, 0), aspect="auto")
    ax.axis("off")

    def text(x, y, s, size=11, weight="normal", color=INK, ha="center", **kw):
        return ax.text(x, y, s, ha=ha, va="center", fontsize=size,
                       weight=weight, color=color, zorder=6, **kw)

    def rounded(x, y, w, h, fill="white", edge=LINE, radius=1.15, lw=1):
        patch = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0.02,rounding_size={radius}",
                              facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2)
        ax.add_patch(patch)

    def action(x, y, w, h, title, lines=(), fill="white", title_size=11.3, sub_size=10.1):
        rounded(x-w/2, y-h/2, w, h, fill)
        if lines:
            total = 2.05 * len(lines)
            text(x, y-total/2, title, size=title_size, weight="bold")
            for i, line in enumerate(lines):
                text(x, y-total/2+2.05*(i+1), line, size=sub_size, color=MUTED)
        else:
            text(x, y, title, size=title_size, weight="bold")

    def arrow(points, color=LINE, lw=1.15):
        # Orthogonal routing; all arrowheads terminate at node boundaries.
        for p, q in zip(points[:-2], points[1:-1]):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=10, color=color, linewidth=lw,
                                    shrinkA=0, shrinkB=0, zorder=1))

    def diamond(x, y, w, h, label=""):
        ax.add_patch(Polygon([(x,y-h/2),(x+w/2,y),(x,y+h/2),(x-w/2,y)],
                             closed=True, facecolor="white", edgecolor=LINE, lw=1.1, zorder=3))
        if label:
            text(x, y, label, size=9.8, linespacing=1.0)

    def circle(x, y, radius, fill, lw=1):
        # Counteract the tighter vertical layout so UML endpoint nodes stay circular.
        ax.add_patch(Ellipse((x,y), 2*radius, 2*radius/VERTICAL_SCALE,
                            facecolor=fill, edgecolor=INK, lw=lw, zorder=5))

    def inset(x, y, w, h):
        a = fig.add_axes([x/100, 1-(y+h)/H, w/100, h/H])
        a.axis("off")
        return a

    # UML initial node, followed by an illustrated loading activity.
    circle(50, 1.65, .72, INK)
    arrow([(50,1.65+.72/VERTICAL_SCALE),(50,4.1)])
    rounded(9, 4.1, 82, 14.2, PALE, edge="#a8bac5")
    rounded(12.2, 6.0, 19.4, 10.5, "#dce7ec", edge=LINE, radius=.7)
    screen = inset(13.3, 6.8, 16.6, 8.8)
    screen.imshow(trace)
    circle(30.6,11.25,.22,LINE)
    # A schematic stylus, not a photograph of a participant.
    ax.add_patch(Polygon([(33.1,15.7),(36.8,6.4),(37.8,6.8),(34.1,16.1)],
                         facecolor=INK, edgecolor=INK, lw=.7, zorder=5))
    ax.add_patch(Polygon([(33.1,15.7),(34.1,16.1),(32.65,17.0)],
                         facecolor="#b88e68", edgecolor=INK, lw=.6, zorder=5))
    text(41, 7.8, "Load digitized handwriting", size=13.2, weight="bold", ha="left", color=BLUE)
    text(41, 11.2, "PaHaW: 75 participants / 597 recordings / 8 tasks", size=10.0, ha="left")
    text(41, 14.2, "Position, time, contact, pressure and pen angles", size=9.6, ha="left", color=MUTED)

    arrow([(50,18.3),(50,21.25)])
    action(50, 24.65, 70, 6.8, "Audit records and define participant folds",
           ["5 repeats × 5 outer folds; common across tasks"], fill=PALE)
    arrow([(50,28.05),(50,29.25)])
    diamond(50, 30.75, 3, 3)  # Merge of the initial flow and the encoding loop.
    arrow([(50,32.25),(50,33.55)])
    action(50, 37.3, 64, 7.5, "Render the next encoding",
           ["Static → 4 single signals → 4 RGB triplets"])
    arrow([(50,41.05),(50,43.0)])
    action(50, 47.1, 64, 8.2, "Extract LPQ; tune and evaluate SVM",
           ["Tune in 3 inner folds, refit, then predict outer test", "Repeat for every task and outer fold"], sub_size=9.8)
    arrow([(50,51.2),(50,53.2)])
    diamond(50, 56.3, 24, 6.2, "More encodings?")
    arrow([(38,56.3),(7,56.3),(7,30.75),(48.5,30.75)])
    text(23.3, 54.9, "[yes] next encoding", size=10.0, color=MUTED)
    arrow([(50,59.4),(50,63.0)])
    text(54.0, 61.1, "[no]", size=10.0, ha="left", color=MUTED)
    action(50, 67.0, 70, 8.0, "Compare all nine encodings; retain SAZ",
           ["Speed (R), altitude (G), azimuth (B)", "Exploratory selection using outer scores"], fill="#eef7f5")

    # Fork and join mean independent analysis branches, not a new patient split.
    arrow([(50,71.0),(50,73.0)])
    ax.add_patch(Rectangle((27,73.0),46,.65,facecolor=INK,edgecolor=INK,lw=0,zorder=3))
    arrow([(27,73.65),(27,76.0)])
    arrow([(73,73.65),(73,76.0)])
    action(27, 80.75, 39.0, 9.5, "Evaluate models on LPQ",
           ["SVM predictions already available", "Tune and evaluate LR and RF"], title_size=10.5, sub_size=9.8)
    action(73, 80.75, 39.0, 9.5, "Train a CNN on RGB pixels",
           ["Select epochs on inner holdout", "Refit and evaluate outer test"], title_size=10.5, sub_size=9.8)
    arrow([(27,85.5),(27,88.0)])
    arrow([(73,85.5),(73,88.0)])
    ax.add_patch(Rectangle((27,88.0),46,.65,facecolor=INK,edgecolor=INK,lw=0,zorder=3))
    arrow([(50,88.65),(50,90.6)])
    action(50, 94.35, 70, 7.5, "Compare methods and combine tasks",
           ["Task-level scores; participant-level majority vote"])
    arrow([(50,98.1),(50,100.1)])
    action(50, 103.85, 70, 7.5, "Explain held-out CNN predictions",
           ["Grad-CAM, occlusion and perturbation checks"])
    arrow([(50,107.6),(50,110.1)])

    # Illustrated reporting activity: a real aggregate and a real frozen XAI case.
    rounded(9, 110.1, 82, 23.3, PALE, edge="#a8bac5")
    text(50,112.2,"Inspect classification and explanation results",size=12.0,weight="bold",color=BLUE)
    text(28,114.4,"SVM: participant classification",size=10.1)
    cm = inset(22.4, 118.0, 14*VERTICAL_SCALE, 14)
    cm.imshow(matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    for i in range(2):
        for j in range(2):
            cm.text(j, i, f"{100*matrix[i,j]:.1f}%", ha="center", va="center", fontsize=9.8,
                    color="white" if matrix[i,j]>.65 else INK)
    for index, label in enumerate(["H", "PD"]):
        text(22.4+14*VERTICAL_SCALE*(.25+.5*index), 116.8, label, size=9.8)
        text(20.5, 121.5+7*index, label, size=9.8)
    text(37.5, 124, "Rows: observed\nColumns: predicted", size=9.8, ha="left", color=MUTED, linespacing=1.4)

    text(64.5, 115.1, "CNN input", size=10.1)
    text(81, 115.1, "Grad-CAM", size=10.1)
    inp = inset(58.5, 118.0, 14*VERTICAL_SCALE, 14)
    inp.imshow(cnn_input)
    heat = inset(75, 118.0, 14*VERTICAL_SCALE, 14)
    heat.imshow(cnn_input)
    heat.imshow(cam, cmap="magma", vmin=0, vmax=1, alpha=cam.astype(float)*.68)
    # Borders mark complete model inputs, including their white padded background.
    for x in [58.5, 75]:
        ax.add_patch(Rectangle((x,118.0),14*VERTICAL_SCALE,14,facecolor="none",edgecolor="#b9c6ce",lw=.55,zorder=7))

    arrow([(50,133.4),(50,135.5-.82/VERTICAL_SCALE)])
    circle(50,135.5,.82,"white",lw=1.25)
    circle(50,135.5,.48,INK)

    fig.savefig(OUT / "activity_diagram.png", dpi=300)
    fig.savefig(OUT / "activity_diagram.svg")
    fig.savefig(PDF)
    plt.close(fig)
    caption = (
        "Illustrated activity diagram of the incremental study. The tablet contains a reconstructed "
        "PaHaW spiral; its outline and stylus are schematic. The loop evaluates static geometry, four "
        "single-signal encodings and all four three-signal RGB combinations with LPQ and SVM. "
        "Participant-separated outer assignments are shared across tasks and methods. Selecting SAZ "
        "from the outer scores is exploratory. Independent classifier branches reuse those assignments; "
        "SVM predictions are retained from the encoding experiment. The CNN selects epochs on the first "
        "inner holdout. Task scores and participant voting are distinct endpoints. XAI uses only the "
        "CNN checkpoints responsible for the held-out predictions. The final illustrations show the "
        "SVM task-vote confusion matrix (row-normalized within each repetition, then averaged) and the "
        "previously selected median-confidence correctly classified PD case on task 4, with its original "
        "Grad-CAM. These two illustrations summarize different analyses and are not a matched case pair."
    )
    manifest = {
        "status": "approved_by_user_for_manuscript_insertion",
        "generator": "scripts/generate_activity_diagram.py", "date": "2026-09-08",
        "raster_size": [2400, int(round(H/12.5*VERTICAL_SCALE*300))], "dpi": 300,
        "height_mm_at_158mm_width": H/100*VERTICAL_SCALE*158,
        "minimum_text_pt_at_158mm_width": 9.6*158/(8*25.4),
        "caption": caption, "source_sha256": SOURCES,
        "illustrations": {
            "input_spiral_subject_id": subject,
            "xai_prediction_index": int(example.prediction_index),
            "xai_task_id": int(example.task_id),
            "xai_subject_id": str(example.subject_id),
            "xai_probability_pd": float(example.probability_pd),
            "svm_row_normalized_confusion_mean": matrix.tolist(),
        },
        "uml": {
            "initial_final_nodes": True,
            "loop": "Merge before rendering; decision after SVM evaluation, yes/no guards.",
            "fork_join": "Independent classical and CNN analysis branches; shared participant folds.",
        },
        "draft_page_policy": "Page limit may be exceeded for advisor review; cuts deferred to Yandre and Angel.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (OUT / "caption.txt").write_text(caption+"\n", encoding="utf-8")
    print(f"Created activity proposal: {OUT}")
    print(f"PDF: {PDF}")
    publish_approved()


def publish_approved():
    """Install the exact approved assets without rerendering or modifying them."""
    figure_dir = ROOT / "paper/figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "approved_by_user_for_manuscript_insertion"
    manifest["approved_date"] = "2026-09-08"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    shutil.copyfile(PDF, figure_dir / "activity_diagram.pdf")
    for extension in ("png", "svg"):
        shutil.copyfile(OUT / f"activity_diagram.{extension}", figure_dir / f"activity_diagram.{extension}")
    (figure_dir / "activity_diagram_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
