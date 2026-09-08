"""Rebuild manuscript figures exclusively from the frozen restart experiments.

Run from the repository root with PYTHONPATH=src. No fitted model is changed.
All plotted uncertainty is read from the original participant bootstrap outputs.
ROC curves and confusion matrices are summarized within repeat before averaging.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from pdhms_restart.cnn_model import prepared_rgb


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/2026-09-restart/runs"
OUT = ROOT / "paper/figures"
REVIEW = ROOT / "paper/review"
WIDTH = 158 / 25.4
INK = "#233648"
BLUE = "#236a91"
TEAL = "#138a83"
ORANGE = "#c5742a"
RED = "#ad4b66"
GRAY = "#72818a"
ALGS = ["svm", "logistic_regression", "random_forest", "cnn"]
ALG_LABEL = {"svm": "SVM", "logistic_regression": "Logistic regression", "random_forest": "Random forest", "cnn": "CNN"}
ALG_COLOR = dict(zip(ALGS, [BLUE, TEAL, ORANGE, RED]))
ENC = ["static", "speed", "pressure", "altitude", "azimuth", "speed_pressure_altitude", "speed_pressure_azimuth", "speed_altitude_azimuth", "pressure_altitude_azimuth"]
ENC_LABEL = dict(zip(ENC, ["Static", "Speed", "Pressure", "Altitude", "Azimuth", "SPA", "SPZ", "SAZ", "PAZ"]))
TASKS = ["Spiral", "l", "le", "les", "lektorka", "porovnat", "nepopadnout", "Sentence"]
SOURCES: dict[str, str] = {}
CATALOG: list[dict] = []

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#aab4bb", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#dfe5e9",
    "grid.linewidth": .5, "axes.axisbelow": True, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
})


def record(path: Path) -> None:
    SOURCES[str(path.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()


def read(run: str, filename: str) -> pd.DataFrame:
    path = RUNS / run / "metrics" / filename
    record(path)
    return pd.read_csv(path, dtype={"subject_id": str})


def save(fig: plt.Figure, name: str, caption: str, role: str = "alternative") -> None:
    if name == "xai_examples":
        # Retain the original physical PDF footprint after increasing annotation
        # sizes; the main manuscript uses this figure at 72% of text width.
        fig.set_size_inches(430.7348070417 / 72, 470.63952 / 72)
        fig.savefig(OUT / f"{name}.png", dpi=300)
        fig.savefig(OUT / f"{name}.pdf")
        with Image.open(OUT / f"{name}.png") as raster:
            # The PDF and Agg renderers differ by one pixel in their rounding.
            # Keep the previous PNG dimensions as well as the PDF MediaBox.
            raster.resize((1793, 1960), Image.Resampling.LANCZOS).save(
                OUT / f"{name}.png", dpi=(300, 300)
            )
    else:
        fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=.035)
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=.035)
    w, h = Image.open(OUT / f"{name}.png").size
    CATALOG.append({"name": name, "suggested_role": role, "caption": caption, "png_width": w, "png_height": h, "dpi": 300})
    if name == "xai_examples":
        CATALOG[-1].update({"annotation_font_pt": 10, "pdf_width_pt": 430.7348070417, "pdf_height_pt": 470.63952, "annotation_pt_at_113_76mm_width": 7.4864})
    elif name == "encoding_effects":
        CATALOG[-1]["value_label_decimals"] = 4
    plt.close(fig)
    print(f"Saved {name}: {w} x {h}", flush=True)


def panel(axis: plt.Axes, text: str) -> None:
    axis.text(-.01, 1.065, text, transform=axis.transAxes, weight="bold", fontsize=8.5, va="bottom")


def image_axis(axis: plt.Axes, arr: np.ndarray, title: str = "") -> None:
    axis.imshow(arr)
    axis.axis("off")
    if title:
        axis.set_title(title, pad=4)


def crop(arr: np.ndarray, pad: int = 12) -> np.ndarray:
    mask = np.any(arr < 250, axis=2)
    yy, xx = np.where(mask)
    if len(xx):
        return arr[max(0, yy.min()-pad):min(len(arr), yy.max()+pad+1), max(0, xx.min()-pad):min(arr.shape[1], xx.max()+pad+1)]
    return arr


static_manifest = read("static-svm-v1", "image_manifest.csv")
single_manifest = read("dynamic-single-signal-v1", "image_manifest.csv")
triplet_manifest = read("dynamic-triplet-svm-v1", "image_manifest.csv")


def rendered(subject: str, task: int, encoding: str, *, cropped: bool = True) -> np.ndarray:
    frame = static_manifest if encoding == "static" else single_manifest if encoding in ENC[1:5] else triplet_manifest
    filt = (frame.subject_id == subject) & (frame.task_id == task)
    if encoding != "static":
        filt &= frame.encoding == encoding
    path = Path(frame.loc[filt, "image_path"].iloc[0])
    record(path)
    arr = np.asarray(Image.open(path).convert("RGB"))
    return crop(arr) if cropped else arr


def sample_subject(label: str) -> str:
    frame = static_manifest[static_manifest.label == label]
    complete = frame.groupby("subject_id").task_id.nunique()
    return sorted(complete[complete == 8].index)[0]


SUBJECTS = [(sample_subject("H"), "Control example"), (sample_subject("PD"), "PD example")]


def input_figures() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 2.7), constrained_layout=True)
    for row, (subject, label) in enumerate(SUBJECTS):
        for col, enc in enumerate(["static", "speed", "speed_altitude_azimuth", "pressure_altitude_azimuth"]):
            image_axis(axes[row,col], rendered(subject, 1, enc, cropped=False), ENC_LABEL[enc] if row == 0 else "")
            if col == 0:
                axes[row,col].text(-.06, .5, label, rotation=90, va="center", ha="right", transform=axes[row,col].transAxes, fontsize=7)
    save(fig, "progressive_encoding", "Progressive rendering of task 1 (Archimedean spiral) for one control participant and one participant with Parkinson's disease. Static geometry is followed by speed in one channel and the SAZ and PAZ triplets. All images preserve the same contact-only geometry and foreground/background mapping. Examples are the lowest anonymous IDs with all eight recordings in each class, chosen without reference to prediction quality.", "main")

    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 2.7), constrained_layout=True)
    for row, (subject, label) in enumerate(SUBJECTS):
        for col, enc in enumerate(["speed", "pressure", "altitude", "azimuth"]):
            image_axis(axes[row,col], rendered(subject, 1, enc, cropped=False), ENC_LABEL[enc] if row == 0 else "")
            if col == 0:
                axes[row,col].text(-.06, .5, label, rotation=90, va="center", ha="right", transform=axes[row,col].transAxes, fontsize=7)
    save(fig, "single_signals", "Isolated signal encodings of the same task-1 examples as progressive_encoding. The tested signal varies in the red channel; green and blue are held at 30. Per-recording contact-segment 5th and 95th percentiles define the scale; speed is log-transformed, and azimuth is centred circularly.")

    subject = SUBJECTS[0][0]
    rgb = rendered(subject, 1, "speed_altitude_azimuth", cropped=False)
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.65), constrained_layout=True)
    for channel, signal in enumerate(["Speed", "Altitude", "Azimuth"]):
        gray = np.repeat(rgb[:, :, channel:channel+1], 3, axis=2)
        image_axis(axes[channel], gray, f"{['R','G','B'][channel]}: {signal}")
    image_axis(axes[3], rgb, "RGB: SAZ")
    save(fig, "rgb_semantics", "The semantic channels of an SAZ spiral, displayed individually in grayscale and jointly as RGB. Red stores normalized log-speed, green stores altitude, and blue stores circularly centred azimuth. Grayscale panels are exact channel values of the composite image, including antialiasing and the white background.", "main")

    for enc, name in [("static", "task_atlas"), ("speed_altitude_azimuth", "rgb_task_atlas")]:
        fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 5.75), constrained_layout=True)
        for task in range(1, 9):
            row, group = divmod(task-1, 2)
            for label_index, (subject, label) in enumerate(SUBJECTS):
                ax = axes[row, group*2+label_index]
                image_axis(ax, rendered(subject, task, enc, cropped=False), f"T{task}: {TASKS[task-1]}\n{'Control' if label_index == 0 else 'PD'}")
        save(fig, name, f"The eight PaHaW tasks rendered as {'static grayscale' if enc == 'static' else 'SAZ RGB'} images for the same two illustrative participants. Tasks progress from a spiral through repeated letters and words to a sentence. Each recording is scaled separately while preserving aspect ratio; absolute writing size is therefore not represented.")


def box(ax: plt.Axes, xy: tuple, width: float, height: float, text: str, color: str = "#edf3f7", fontsize: float = 7.5) -> None:
    ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.008,rounding_size=0.015", facecolor=color, edgecolor="#adc0cc", linewidth=.8))
    ax.text(xy[0]+width/2, xy[1]+height/2, text, va="center", ha="center", fontsize=fontsize, linespacing=1.45)


def arrow(ax: plt.Axes, a: tuple, b: tuple) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, linewidth=.9, color=GRAY))


def validation_diagram() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
    fig.subplots_adjust(left=.012,right=.988,bottom=.012,top=.988)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    box(ax, (.015,.81), .25,.155, "75 participants\n38 controls · 37 PD\n597 task recordings")
    box(ax, (.345,.81), .30,.155, "Participant-stratified folds\n5 outer folds × 5 repeats\nShared across tasks / methods")
    box(ax, (.725,.81), .26,.155, "Outer test participants\nHeld out during tuning\nand model training", "#f9f0e7")
    arrow(ax, (.275,.89), (.335,.89)); arrow(ax, (.655,.89), (.715,.89))
    box(ax, (.195,.58), .43,.145, "Outer training participants\nClassical tuning: 3 inner folds\nCNN epoch count: first inner fold")
    arrow(ax, (.495,.80), (.495,.735))
    box(ax, (.015,.335), .295,.16, "LPQ-RGB: 64 × 64 input\n768 histogram bins\nSVM · LR · random forest")
    box(ax, (.375,.335), .285,.16, "CNN: 128 × 128 input\n4 convolutional blocks\n105,873 parameters")
    arrow(ax, (.31,.57), (.16,.505)); arrow(ax, (.51,.57), (.51,.505))
    box(ax, (.725,.335), .26,.16, "External predictions\nEight task-level scores\nParticipant task fusion", "#eaf5f2")
    arrow(ax, (.67,.415), (.715,.415)); arrow(ax, (.85,.80), (.85,.505))
    ax.plot([.16,.16,.78],[.327,.267,.267],color=GRAY,lw=.9)
    arrow(ax,(.78,.267),(.78,.325))
    box(ax, (.02,.015), .435,.155, "Paired inference\nParticipant bootstrap across tasks / repeats\nPaired permutation tests · Holm correction", fontsize=7)
    box(ax, (.545,.015), .435,.155, "CNN explanations on frozen test models\nGrad-CAM · occlusion · channel removal\nDeletion · weight-randomization checks", fontsize=7)
    arrow(ax, (.87,.325), (.80,.18)); arrow(ax, (.725,.335), (.42,.18))
    save(fig, "grouped_validation", "Evaluation flow. Participant identities define every split. Outer assignments are shared across tasks and methods. Classical hyperparameters are selected in three inner folds; the CNN selects its epoch count using the first inner fold, then retrains on the entire outer-training set. LPQ-RGB uses 64×64 inputs, whereas the CNN uses 128×128 inputs. Confidence intervals resample participants jointly across tasks and repetitions. Every XAI map uses the checkpoint responsible for that external prediction.", "main")


def metric_figures() -> None:
    ranking = read("article-experiment-package-v1", "article_representation_table.csv")
    pair = read("encoding-comparison-v1", "all_encoding_global_pairwise.csv")
    contrasts = []
    for enc in ENC[1:]:
        r = pair[((pair.candidate == enc) & (pair.reference == "static")) | ((pair.candidate == "static") & (pair.reference == enc))].iloc[0]
        if r.candidate == "static":
            d, lo, hi = -r.delta_task_mean_macro_f1, -r.ci_high, -r.ci_low
        else:
            d, lo, hi = r.delta_task_mean_macro_f1, r.ci_low, r.ci_high
        contrasts.append({"encoding":enc, "delta":d, "low":lo, "high":hi})
    contrast = pd.DataFrame(contrasts).set_index("encoding")
    order = ranking.encoding.tolist()[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3), gridspec_kw={"width_ratios":[1, 1.17]}, constrained_layout=True)
    y = np.arange(len(order))
    vals = ranking.set_index("encoding").loc[order,"task_mean_macro_f1"]
    palette = [BLUE if enc == "static" else TEAL if enc == "speed_altitude_azimuth" else GRAY for enc in order]
    axes[0].scatter(vals, y, c=palette, s=29, zorder=3)
    axes[0].set(yticks=y, yticklabels=[ENC_LABEL[e] for e in order], xlabel="Mean task macro-F1", xlim=(.54,.603))
    axes[0].axvline(float(vals.loc["static"]), color=BLUE, ls=":", lw=.8)
    for yi, val in enumerate(vals): axes[0].text(val+.002, yi, f"{val:.4f}", fontsize=7, va="center", bbox={"facecolor":"white", "edgecolor":"none", "pad":.2})
    axes[0].grid(axis="x"); panel(axes[0], "(a) Encoding ranking")
    for yi, enc in enumerate(order):
        if enc == "static":
            axes[1].plot(0, yi, "|", color=BLUE, ms=9)
        else:
            r = contrast.loc[enc]
            axes[1].errorbar(r.delta, yi, xerr=[[r.delta-r.low],[r.high-r.delta]], fmt="o", color=palette[yi], capsize=2, markersize=4, lw=.8)
    axes[1].axvline(0, color=BLUE, ls=":", lw=.8)
    axes[1].set(yticks=y, yticklabels=[], xlabel="Difference from static macro-F1", xlim=(-.045,.055))
    axes[1].grid(axis="x"); panel(axes[1], "(b) Paired 95% intervals")
    save(fig, "encoding_effects", "SVM performance for all nine encodings, ordered by the mean of task-specific macro-F1 across eight tasks and five repeats. Right: paired differences from the static representation, with 95% stratified participant-bootstrap intervals. Participants are sampled jointly across tasks and repeats. The full 36-pair Holm family supports no encoding difference; the SAZ lead is descriptive.", "main")

    task_frames = []
    for run in ["static-svm-v1", "dynamic-single-signal-v1", "dynamic-triplet-svm-v1"]:
        df=read(run,"task_summary.csv")
        if "encoding" not in df: df["encoding"]="static"
        task_frames.append(df[["encoding","task_id","macro_f1_mean"]])
    task_encoding=pd.concat(task_frames).pivot(index="encoding",columns="task_id",values="macro_f1_mean").loc[ENC]
    fig,ax=plt.subplots(figsize=(WIDTH,3.5),constrained_layout=True)
    im=ax.imshow(task_encoding,vmin=.45,vmax=.70,cmap="YlGnBu",aspect="auto")
    ax.set(xticks=range(8),xticklabels=[f"T{i}" for i in range(1,9)],yticks=range(9),yticklabels=[ENC_LABEL[e] for e in ENC])
    for row in range(9):
        for col in range(8):
            v=task_encoding.iloc[row,col]
            ax.text(col,row,f"{v:.3f}",ha="center",va="center",fontsize=7,color="white" if v>.61 else INK)
    fig.colorbar(im,ax=ax,pad=.025,label="Macro-F1",fraction=.035)
    save(fig,"encoding_heatmap","Task-specific mean macro-F1 for the nine encodings under LPQ-RGB plus SVM. Every cell averages five out-of-fold repetition metrics. The shared colour scale makes the variation across tasks and encodings visible; cells are descriptive and do not imply separately significant contrasts.")

    fig,ax=plt.subplots(figsize=(WIDTH,2.55),constrained_layout=True)
    for enc,col,mark in [("static",BLUE,"o"),("altitude",ORANGE,"s"),("speed_altitude_azimuth",TEAL,"D"),("pressure_altitude_azimuth",RED,"^")]:
        ax.plot(range(1,9),task_encoding.loc[enc],marker=mark,ms=4,lw=1,color=col,label=ENC_LABEL[enc])
    ax.set(xticks=range(1,9),xticklabels=[f"T{i}" for i in range(1,9)],ylabel="Macro-F1",ylim=(.44,.72));ax.grid(axis="y")
    ax.legend(ncol=4,loc="upper center",frameon=False,bbox_to_anchor=(.5,1.19))
    save(fig,"encoding_tasks","Task profiles for static, altitude alone, SAZ, and PAZ under the same SVM protocol. Values are mean macro-F1 over the five repeats. Lines connect task identifiers only and do not indicate a continuous scale or a fitted trend.")

    classifiers=read("article-experiment-package-v1","article_classifier_table.csv").set_index("algorithm")
    task_models=read("cnn-v1","combined_task_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.35),gridspec_kw={"width_ratios":[2.3,1]},constrained_layout=True)
    offsets=np.linspace(-.25,.25,4)
    for offset,alg in zip(offsets,ALGS):
        df=task_models[task_models.algorithm==alg].sort_values("task_id")
        axes[0].plot(df.macro_f1_mean,np.arange(8)+offset,"o",ms=4,color=ALG_COLOR[alg],label=ALG_LABEL[alg])
    axes[0].set(yticks=range(8),yticklabels=[f"T{i}: {TASKS[i-1]}" for i in range(1,9)],xlabel="Task macro-F1",xlim=(.44,.71));axes[0].invert_yaxis();axes[0].grid(axis="x")
    panel(axes[0],"(a) Task-specific performance")
    for i,alg in enumerate(ALGS):
        v=classifiers.loc[alg,"task_mean_macro_f1"]
        axes[1].scatter(v,i,c=ALG_COLOR[alg],s=27)
        axes[1].text(v+.006,i,f"{v:.3f}",fontsize=7,va="center")
    axes[1].set(yticks=range(4),yticklabels=["SVM","LR","RF","CNN"],xlabel="Mean task macro-F1",xlim=(.47,.63));axes[1].invert_yaxis();axes[1].grid(axis="x")
    panel(axes[1],"(b) Overall")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,loc="lower center",bbox_to_anchor=(.58,-.11),frameon=False)
    save(fig,"classifiers_tasks","The fixed SAZ representation evaluated with SVM, logistic regression (LR), random forest (RF), and a compact CNN. Left: task-specific macro-F1 averaged over five repetitions. Right: unweighted mean over the eight tasks. Classical methods use LPQ-RGB descriptors; the CNN learns from the image. The CNN is below each classical method in the global six-pair Holm family, whereas differences among classical methods are unsupported.","main")

    model_pair=read("cnn-v1","combined_task_pairwise.csv")
    fig,ax=plt.subplots(figsize=(WIDTH,2.0),constrained_layout=True)
    for i,alg in enumerate(ALGS[1:]):
        r=model_pair[(model_pair.reference==alg)&(model_pair.candidate=="svm")].iloc[0]
        v,lo,hi=-r.delta_task_mean_macro_f1,-r.ci_high,-r.ci_low
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color=ALG_COLOR[alg],capsize=3,ms=5)
        ax.text(.051,i,f"Holm p = {r.holm_p_global_6:.3f}",ha="right",va="center",fontsize=7)
    ax.axvline(0,color=GRAY,ls=":");ax.set(yticks=range(3),yticklabels=[ALG_LABEL[a] for a in ALGS[1:]],xlabel="Difference from SVM: mean task macro-F1",xlim=(-.115,.055));ax.invert_yaxis();ax.grid(axis="x")
    save(fig,"classifier_contrasts","Paired differences from SVM on the task-mean endpoint. Error bars are 95% stratified participant-bootstrap intervals; annotations report Holm-adjusted permutation p-values from all six classifier pairs. Negative differences favour SVM.")

    fig,ax=plt.subplots(figsize=(WIDTH,2.2),constrained_layout=True)
    for i,alg in enumerate(ALGS):
        r=classifiers.loc[alg];v=r.fused_macro_f1
        ax.errorbar(v,i,xerr=[[v-r.fused_macro_f1_ci_low],[r.fused_macro_f1_ci_high-v]],fmt="o",color=ALG_COLOR[alg],capsize=3,ms=5)
        ax.text(.79,i,f"{v:.3f}",ha="right",va="center",fontsize=8)
    ax.set(yticks=range(4),yticklabels=[ALG_LABEL[a] for a in ALGS],xlabel="Participant-level macro-F1 after task voting",xlim=(.43,.80));ax.invert_yaxis();ax.grid(axis="x")
    save(fig,"participant_fusion","Participant classification by majority voting across available tasks for each classifier. Points show the mean macro-F1 of the five repeats, and error bars are 95% stratified participant-bootstrap intervals. No pairwise classifier difference survives Holm correction for this fused endpoint. These values evaluate one decision per participant and must not be read as task-level scores.","main")

    rules=read("task-fusion-v1","method_summary.csv")
    print("fusion rule columns",rules.columns.tolist(),flush=True)


def fusion_diagnostics() -> None:
    pred=read("cnn-v1","fused_majority_predictions.csv")
    fig,ax=plt.subplots(figsize=(WIDTH,3.4),constrained_layout=True)
    grid=np.linspace(0,1,501)
    for alg in ALGS:
        curves=[];areas=[]
        for repeat,frame in pred[pred.algorithm==alg].groupby("repeat"):
            assert frame.subject_id.is_unique and len(frame)==75
            fpr,tpr,_=roc_curve(frame.y_true,frame.decision_score_pd)
            curves.append(np.interp(grid,fpr,tpr));areas.append(roc_auc_score(frame.y_true,frame.decision_score_pd))
        mean=np.mean(curves,axis=0);mean[0]=0;mean[-1]=1
        ax.plot(grid,mean,lw=1.3,color=ALG_COLOR[alg],label=f"{ALG_LABEL[alg]}  (AUC {np.mean(areas):.3f})")
    ax.plot([0,1],[0,1],ls=":",lw=.8,color=GRAY)
    ax.set(xlabel="False-positive rate",ylabel="True-positive rate (PD)",xlim=(0,1),ylim=(0,1.02));ax.grid();ax.legend(loc="lower right",frameon=False)
    save(fig,"fusion_roc","Mean participant-level ROC curves for task-vote fractions. A ROC is computed separately in each of the five repetitions (75 distinct participants per repeat), interpolated to a common false-positive-rate grid, and then averaged. Legend AUCs are the mean of the five original repeat AUCs, not areas of pooled repeated predictions. Vote fractions are discrete scores, not calibrated disease probabilities.")

    fig,axes=plt.subplots(2,2,figsize=(WIDTH,5.0),constrained_layout=True)
    for ax,alg in zip(axes.flat,ALGS):
        counts=[];fractions=[]
        for _,frame in pred[pred.algorithm==alg].groupby("repeat"):
            cm=confusion_matrix(frame.y_true,frame.y_pred,labels=[0,1]);counts.append(cm);fractions.append(cm/cm.sum(axis=1,keepdims=True))
        count=np.mean(counts,axis=0);norm=np.mean(fractions,axis=0)
        ax.imshow(norm,vmin=0,vmax=1,cmap="Blues")
        for i in range(2):
            for j in range(2):ax.text(j,i,f"{norm[i,j]:.1%}\n({count[i,j]:.1f} / repeat)",ha="center",va="center",fontsize=8,color="white" if norm[i,j]>.65 else INK)
        ax.set(xticks=[0,1],xticklabels=["Control","PD"],yticks=[0,1],yticklabels=["Control","PD"],xlabel="Predicted class",ylabel="Observed class",title=ALG_LABEL[alg])
    save(fig,"fusion_confusion","Confusion matrices after task voting. Each matrix is computed from 75 distinct participants within a repeat, then averaged over five repetitions. Cell percentages are normalized by the observed class; parenthetical counts are mean participants per repeat (therefore fractional). Repeated predictions are not counted as independent observations.")


def xai_figures() -> None:
    samples=read("cnn-xai-v1","prediction_xai_metrics.csv")
    examples=read("cnn-xai-v1","figure_example_selection.csv")
    occlusion=read("cnn-xai-v1","occlusion_metrics.csv")
    sanity=read("cnn-xai-v1","weight_randomization_sanity.csv")
    bootstrap=read("cnn-xai-v1","global_participant_bootstrap.csv").set_index("metric")
    maps_path=RUNS/"cnn-xai-v1/maps/spatial_maps.npz";record(maps_path)
    with np.load(maps_path) as maps:
        cam=maps["gradcam_maps"]
        occ=maps["occlusion_maps"]
    fig,axes=plt.subplots(4,3,figsize=(WIDTH,6.55),constrained_layout=True)
    examples=examples.sort_values(["y_true","correct"],ascending=[True,False]).reset_index(drop=True)
    for i,r in examples.iterrows():
        raw=rendered(r.subject_id,int(r.task_id),"speed_altitude_azimuth",cropped=False)
        frame=triplet_manifest[(triplet_manifest.subject_id==r.subject_id)&(triplet_manifest.task_id==r.task_id)&(triplet_manifest.encoding=="speed_altitude_azimuth")]
        inp=prepared_rgb(Path(frame.image_path.iloc[0]),128)
        idx=int(r.prediction_index);oi=int(occlusion.loc[occlusion.prediction_index==idx,"occlusion_index"].iloc[0])
        for j in range(3):
            image_axis(axes[i,j],inp,["SAZ input","Grad-CAM","Occlusion"][j] if i==0 else "")
            if i == 0:
                axes[i,j].title.set_fontsize(10)
        for j,attention in [(1,cam[idx]),(2,occ[oi])]:
            # Attribution-controlled alpha leaves truly zero maps visibly empty.
            axes[i,j].imshow(attention,cmap="magma",vmin=0,vmax=1,alpha=np.asarray(attention,dtype=float)*.68)
        label="Control" if r.y_true==0 else "PD"
        outcome="correct" if r.correct else "error"
        axes[i,0].text(-.04,.5,f"{label}, {outcome}\nP(PD) = {r.probability_pd:.3f}",transform=axes[i,0].transAxes,rotation=90,ha="right",va="center",fontsize=10)
        if r.gradcam_degenerate:axes[i,1].text(.5,.05,"Zero Grad-CAM",transform=axes[i,1].transAxes,ha="center",fontsize=10,color=RED)
    save(fig,"xai_examples","Task-4 explanations for correct and incorrect predictions from both classes. The four frozen repeat-1 cases were selected at median predicted-class confidence within each class/outcome stratum, without selecting for visual plausibility. Grad-CAM and positive occlusion sensitivity are normalized per map; overlay opacity increases with attribution. The PD error has an all-zero Grad-CAM map and is retained. Maps explain the predicted class and use the exact external-test checkpoint. P(PD) is the CNN output, not a calibrated clinical probability.","main")

    fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.5),constrained_layout=True)
    specs=[(["gradcam_deletion_drop","random_deletion_drop","deletion_advantage"],["Top 10% CAM","Random 10%","Paired difference"],"(a) Deletion response"),(["speed_channel_drop","altitude_channel_drop","azimuth_channel_drop"],["Speed (R)","Altitude (G)","Azimuth (B)"],"(b) Channel sensitivity")]
    for ax,(keys,labels,title) in zip(axes,specs):
        for i,key in enumerate(keys):
            r=bootstrap.loc[key];v=r.participant_mean
            ax.errorbar(v,i,xerr=[[v-r.ci_low],[r.ci_high-v]],fmt="o",color=TEAL if i==2 else BLUE,ms=4,capsize=2,lw=.9)
        ax.set(yticks=range(3),yticklabels=labels,xlabel="Predicted-class confidence drop",xlim=(-.002,.063));ax.invert_yaxis();ax.axvline(0,color=GRAY,ls=":",lw=.8);ax.grid(axis="x");panel(ax,title)
    save(fig,"xai_validation","Perturbation checks for all 2,985 external CNN predictions. Left: removing the top 10% Grad-CAM pixels, equal-area random removal, and their paired difference. The random control matches pixel count but not foreground coverage or patch shape. Right: replacing each complete SAZ channel by white. Points are participant means and intervals are 95% stratified participant-bootstrap intervals after averaging repeats and tasks within participant. Channel removal changes image contrast and cannot isolate a physiological mechanism.","main")

    part=read("cnn-xai-v1","participant_task_xai_metrics.csv")
    channel=part.groupby("task_id")[["speed_channel_drop","altitude_channel_drop","azimuth_channel_drop"]].mean()
    fig,ax=plt.subplots(figsize=(WIDTH,3.25),constrained_layout=True)
    limit=max(abs(channel.to_numpy().min()),abs(channel.to_numpy().max()))
    im=ax.imshow(channel,cmap="RdBu_r",vmin=-limit,vmax=limit,aspect="auto")
    ax.set(yticks=range(8),yticklabels=[f"T{i}: {TASKS[i-1]}" for i in range(1,9)],xticks=range(3),xticklabels=["Speed (R)","Altitude (G)","Azimuth (B)"])
    for i in range(8):
        for j in range(3):ax.text(j,i,f"{channel.iloc[i,j]:+.3f}",ha="center",va="center",fontsize=8,color="white" if abs(channel.iloc[i,j])>.11 else INK)
    fig.colorbar(im,ax=ax,label="Confidence drop",pad=.025,fraction=.035)
    save(fig,"xai_channel_tasks","Mean channel-removal confidence changes by task. Repeated predictions are averaged within participant/task before the task mean. Positive values mean the original channel supported the predicted class; negative values mean its removal increased confidence. Colour uses a symmetric scale around zero. These are image perturbations, not isolated physiological effects.")

    fig,axes=plt.subplots(1,2,figsize=(WIDTH,2.6),constrained_layout=True)
    for ax,series,title,col in [(axes[0],occlusion.gradcam_occlusion_spearman,"Grad-CAM vs occlusion",BLUE),(axes[1],sanity.trained_random_spearman,"Trained vs random weights",RED)]:
        vals=series.dropna();ax.hist(vals,bins=np.linspace(-1,1,21),color=col,alpha=.8,edgecolor="white",linewidth=.4)
        med=vals.median();ax.axvline(med,color=INK,lw=1,ls="--");ax.axvline(0,color=GRAY,lw=.7,ls=":")
        ax.set(xlim=(-1,1),xlabel="Pixelwise Spearman correlation",ylabel="Number of maps",title=title)
        ax.text(.04,.93,f"Median {med:.3f}\n{len(vals)} nonconstant pairs",transform=ax.transAxes,ha="left",va="top",fontsize=7)
    save(fig,"xai_agreement","Descriptive distributions of spatial agreement. Left: Grad-CAM and positive occlusion maps for repeat 1 (one map per available participant/task recording). Right: trained and fully randomized networks, one external image for each of 200 checkpoints. Spearman correlations omit constant maps. These distributions are not independent-sample significance tests: participants recur across tasks, and checkpoints share training data.")


def fusion_rule_figure() -> None:
    df=read("task-fusion-v1","method_summary.csv")
    method_col="method" if "method" in df.columns else "fusion_method"
    labels={"calibrated_mean_all8":"Mean calibrated probability","calibrated_reliability_weighted_all8":"Reliability-weighted probability","majority_vote_all8":"Majority vote"}
    fig,ax=plt.subplots(figsize=(WIDTH,2.0),constrained_layout=True)
    for i,r in df.iterrows():
        v=r.macro_f1_mean
        ax.errorbar(v,i,xerr=[[v-r.macro_f1_ci_low],[r.macro_f1_ci_high-v]],fmt="o",color=BLUE,ms=5,capsize=3)
    ax.set(yticks=range(len(df)),yticklabels=[labels.get(x,x.replace("_"," ")) for x in df[method_col]],xlabel="Participant-level macro-F1 (SVM)",xlim=(.50,.79));ax.invert_yaxis();ax.grid(axis="x")
    save(fig,"fusion_rules","SVM task-fusion rules with 95% stratified participant-bootstrap intervals. Probability calibration and reliability weights use outer-training data only. The mean-probability rule was the primary fusion endpoint; the reliability-weighted mean and majority vote were declared sensitivity analyses. Task-fusion rules use all available tasks without selecting a subset from external results.")


def finish() -> None:
    activity_path = OUT / "activity_diagram_manifest.json"
    if activity_path.exists():
        activity = json.loads(activity_path.read_text(encoding="utf-8"))
        CATALOG.insert(0, {"name": "activity_diagram", "suggested_role": "main",
                           "caption": activity["caption"], "dpi": 300,
                           "png_width": activity["raster_size"][0],
                           "png_height": activity["raster_size"][1]})
        SOURCES.update(activity["source_sha256"])
    (OUT/"figure_manifest.json").write_text(json.dumps({"generator":"scripts/generate_paper_figures.py","dpi":300,"figures":CATALOG,"source_sha256":SOURCES},indent=2)+"\n",encoding="utf-8")
    lines=["# Publication figure catalogue", "", "All charts derive from frozen experiment outputs. The excluded experiment is absent from every figure and data input. Raster PNG files are 300 dpi; matching PDF files retain vector text and plots (handwriting and attribution are embedded raster data). Sizes below include the tight bounding box. Suggested manuscript width is 158 mm unless reduced deliberately after reviewing labels.", "", "## Figure selection", ""]
    for item in CATALOG:
        name=item["name"]
        lines.extend([f"### {name}","",f"Suggested placement: **{item['suggested_role']}**. PNG: {item['png_width']} × {item['png_height']} px. Files: `paper/figures/{name}.pdf` and `.png`.","",item["caption"],""])
        if name == "xai_examples":
            lines.extend(["Annotation and column-header type is 10 pt in the PDF, approximately 7.49 pt when included at 113.76 mm (72% of the 158 mm text width). The original 430.734807 × 470.639520 pt PDF footprint and 1793 × 1960 px PNG footprint are retained.", ""])
        elif name == "encoding_effects":
            lines.extend(["Numerical point labels use four decimals to distinguish SAZ (0.5752) and PAZ (0.5745).", ""])
    lines.extend(["## Statistical and visual safeguards","","- Encoding and task-level points are means of repeat-specific out-of-fold metrics, not metrics over pooled repetitions.","- Fusion ROC curves and confusion matrices are computed per repeat before averaging. Vote fractions are not probabilities.","- All shown confidence intervals come unchanged from the original stratified participant-bootstrap files. No sample-level bootstrap was introduced.","- Examples use anonymous class labels. The source manifest preserves exact file hashes; no participant identifier is printed in the figures.","- XAI selection retains the originally frozen median-confidence cases, including an all-zero Grad-CAM failure. Opacity is proportional to attribution to avoid tinting zero areas.","- The atlas deliberately retains per-image scaling and blank margins so the preprocessing can be assessed. Absolute handwriting size cannot be inferred.",""])
    (REVIEW/"figure_catalog.md").write_text("\n".join(lines),encoding="utf-8")
    thumbs=[]
    for entry in CATALOG:
        im=Image.open(OUT/f"{entry['name']}.png").convert("RGB");im.thumbnail((550,410))
        tile=Image.new("RGB",(580,450),"white");tile.paste(im,((580-im.width)//2,30+(410-im.height)//2));ImageDraw.Draw(tile).text((12,8),entry["name"],fill=INK);thumbs.append(tile)
    rows=(len(thumbs)+3)//4
    montage=Image.new("RGB",(4*580,rows*450),"#d9e1e7")
    for i,tile in enumerate(thumbs):montage.paste(tile,((i%4)*580,(i//4)*450))
    montage.save(OUT/"figure_contact_sheet.png")
    print(f"Finished {len(CATALOG)} paired figure assets",flush=True)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True);REVIEW.mkdir(parents=True,exist_ok=True)
    input_figures()
    validation_diagram()
    metric_figures()
    fusion_diagnostics()
    xai_figures()
    fusion_rule_figure()
    finish()


if __name__ == "__main__":
    main()
