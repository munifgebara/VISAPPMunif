"""Compose a scientific panel from the fixed spatial-control example selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from pdhms_restart.data import sha256_file


ROOT = Path(__file__).resolve().parents[1]
ENCODING = "speed_altitude_azimuth"
SEED = 20260909


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run = (ROOT / config["controls_output_root"]).resolve()
    feature_manifest_path = run / "manifests/feature_manifest.json"
    feature_manifest_hash = sha256_file(feature_manifest_path)
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest["status"] != "complete":
        raise ValueError("Freeze the complete feature generation before creating figures.")
    if SEED not in feature_manifest["seeds"]:
        raise ValueError("The illustrative seed is absent from the declared campaign.")
    participants = feature_manifest["panel_selection"]["subject_ids"][:2]
    frozen_keys = {tuple(key) for key in feature_manifest["panel_selection"]["recording_keys"]}
    metadata_paths = [
        ROOT / config["static_run"] / "features/metadata.csv",
        ROOT / config["triplet_run"] / "features" / f"{ENCODING}_metadata.csv",
        run / "features" / f"{ENCODING}__circular_shift__seed{SEED}_metadata.csv",
        run / "features" / f"{ENCODING}__joint_permutation__seed{SEED}_metadata.csv",
    ]
    frames = [pd.read_csv(path, dtype={"subject_id": str}) for path in metadata_paths]
    sources = {str(path): sha256_file(path) for path in [config_path, feature_manifest_path, *metadata_paths]}
    row_images = []
    selections = []
    ratios = []
    for display_id, subject in enumerate(participants, 1):
        for task in (1, 4):
            common_keys = sorted(key for key in frozen_keys if key[0] == subject and key[1] == task)
            if not common_keys:
                raise ValueError("A display example is not in the frozen selection.")
            key = common_keys[0]
            images = []
            paths = []
            for frame in frames:
                selected = frame[(frame["subject_id"] == subject) & (frame["task_id"] == task)
                                 & (frame["repetition_id"] == key[2])]
                if len(selected) != 1:
                    raise ValueError(f"Example key is not unique: {key}")
                path = Path(selected.iloc[0]["image_path"])
                with Image.open(path) as image:
                    images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
                paths.append(str(path))
                sources[str(path)] = sha256_file(path)
            if len({image.shape for image in images}) != 1:
                raise ValueError("The four views do not share one image geometry.")
            # A common display crop includes all foreground from the four actual
            # rasters. No signal value, stroke pixel or individual image is edited.
            mask = np.logical_or.reduce([np.any(image < 250, axis=2) for image in images])
            ys, xs = np.where(mask)
            margin = 10
            y0, y1 = max(0, int(ys.min()) - margin), min(images[0].shape[0], int(ys.max()) + margin + 1)
            x0, x1 = max(0, int(xs.min()) - margin), min(images[0].shape[1], int(xs.max()) + margin + 1)
            crops = [image[y0:y1, x0:x1] for image in images]
            row_images.append(crops)
            ratios.append((y1 - y0) / (x1 - x0))
            selections.append({"display_id": display_id, "recording_key": list(key), "image_paths": paths,
                               "common_display_crop_yxyx": [y0, x0, y1, x1]})
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 10})
    figure = plt.figure(figsize=(11.4, 2.25 * sum(ratios) + 1.7), facecolor="white")
    grid = figure.add_gridspec(len(row_images), 4, height_ratios=ratios,
                              left=0.105, right=0.99, top=0.84, bottom=0.18,
                              wspace=0.07, hspace=0.27)
    headings = ["Static", "Aligned SAZ", "Circular shift", "Joint permutation"]
    for row, images in enumerate(row_images):
        for column, image in enumerate(images):
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(image, interpolation="nearest", aspect="equal")
            axis.axis("off")
            if row == 0:
                axis.set_title(headings[column], fontsize=11, fontweight="bold", pad=13)
            if column == 0:
                task = selections[row]["recording_key"][1]
                task_name = "T1: spiral" if task == 1 else "T4: repeated les"
                axis.text(-0.06, 0.5, f"Example {selections[row]['display_id']}\n{task_name}",
                          transform=axis.transAxes, ha="right", va="center", fontsize=9)
    figure.suptitle("Spatial signal controls on the same handwriting", x=0.55, y=0.98,
                   fontsize=15, fontweight="bold")
    figure.text(0.55, 0.928, "SAZ channels: red = speed; green = altitude; blue = azimuth",
                ha="center", va="center", fontsize=10)
    caption = (
        "Illustrative SAZ examples, with the fixed seed 20260909. The first two participants of the "
        "previously fixed, diagnosis-independent example selection contribute T1 and T4. Geometry, "
        "pen-down segments and stroke width are unchanged. Circular shifts preserve cyclic vector "
        "order within each stroke; joint permutations do not. Both retain the joint-signal multiset "
        "of each stroke. The four views of a recording share one display crop; raster pixel histograms "
        "need not be identical. These examples are not selected by classifier performance."
    )
    figure.text(0.05, 0.115, textwrap.fill(caption, width=154), ha="left", va="top", fontsize=8.7,
                linespacing=1.4)
    figure_dir = run / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "spatial_controls_examples.png"
    svg = figure_dir / "spatial_controls_examples.svg"
    figure.savefig(png, dpi=240, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)
    if sha256_file(feature_manifest_path) != feature_manifest_hash:
        raise RuntimeError("Feature manifest changed during figure creation.")
    manifest = {
        "purpose": "Illustrative within-stroke spatial controls, not performance-selected examples",
        "encoding": ENCODING, "seed": SEED, "selection": selections,
        "source_sha256": sources, "generator_sha256": sha256_file(Path(__file__)),
        "feature_manifest_sha256": feature_manifest_hash,
        "outputs_sha256": {path.name: sha256_file(path) for path in (png, svg)},
        "caption": caption, "model_fitting_or_inference": False,
    }
    (figure_dir / "spatial_controls_examples_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"png": str(png), "svg": str(svg), "rows": len(row_images), "seed": SEED}))


if __name__ == "__main__":
    main()
