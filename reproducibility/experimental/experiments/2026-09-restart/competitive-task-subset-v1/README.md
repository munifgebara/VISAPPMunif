# Competitive methods on a fixed literature task set

The protocol in `DECISION.md` and the searches, seeds and comparison families in
`config.json` were fixed before the new classifiers were fitted on 8 September 2026.
The prior eight experiment stages and the nested-selection/spatial-control campaign
are preserved. This directory contains source configurations and documentation;
generated features and predictions belong in a separate local run directory.

## Scope

The primary endpoint is participant-level macro F1 after fixed majority voting over
T2/T3/T4, the repeated l/le/les tasks considered in the Casademunt Gonzalez thesis.
The thesis does not report this three-task ensemble. All eight tasks form a declared
secondary comparison; they are not a task subset selected from outer-test outcomes.

Four procedures share 25 outer participant partitions and three common inner folds:
static LPQ + SVM, internally selected LPQ encoding + SVM, an 83-attribute
kinematic/pressure SVM, and ResNet18 transfer + L2 logistic regression. Hyperparameters
are selected per task; eligible image methods select one of nine encodings for each
task set using only training folds. Standardization is fitted inside each training
partition. Fusion has no calibration, learned task weights or rule search. This is
reanalysis of the development cohort, not independent external validation.

## Environment and prerequisites

Use Python 3.13, `requirements-lock.txt`, and the `cnn` extra from `pyproject.toml`.
Run commands from the copied project root after installing it in editable mode:

```text
python -m pip install -r requirements-lock.txt
python -m pip install -e ".[cnn]"
```

First reproduce the static, single-signal and triplet runs, then the
`nested-selection-controls-v1` campaign as described in
`paper/review/experimental_reproduction.md` in the full repository, or the top-level
`README.md` in the packaged experimental sources. Those stages create the nine
image/LPQ inventories and the participant selection plan consumed here. A Git
checkout with a valid HEAD is required by the provenance recording. Only source
files and adjusted configuration copies need to be committed locally.

The supplied transfer extractor requires CUDA; it has no CPU device flag or silent
CPU fallback. Its local run used an NVIDIA GeForce RTX 3070, torch 2.11.0+cu128 and
torchvision 0.26.0+cpu. A compatible NVIDIA driver and a CUDA-enabled build of the
pinned torch version are needed for a complete fresh extraction. Check the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) for a build
compatible with the target machine. The feature matrices can subsequently be used
for the sklearn fitting and verification stages without a GPU. Reading the article,
figures or saved summaries also requires no GPU. Merely changing the JSON device
setting to CPU is rejected by this frozen extractor.

The backbone uses the official torchvision `ResNet18_Weights.IMAGENET1K_V1`
checkpoint, `resnet18-f37072fd.pth`. Put it at
`Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'`. This command can
retrieve the official weights into the default torch cache without fitting a model:

```text
python -c "from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)"
```

If `TORCH_HOME` redirects downloads, place a hash-verified copy at the explicit path
above; the extractor uses that path rather than the `TORCH_HOME` setting. Checkpoint
URL: [official PyTorch download](https://download.pytorch.org/models/resnet18-f37072fd.pth).
The required complete SHA-256 is listed below.

## Relocate paths without changing the protocol

`transfer_descriptor.json` is byte-identical to the configuration written before
the original extraction. Its historical absolute `output_root` and
`model.weights_path` remain intact for provenance. In a new working copy, create a
local derivative with Python so paths have the exact native spelling expected by
the extractor:

```python
import json
from pathlib import Path

root = Path.cwd().resolve()
directory = root / 'experiments/2026-09-restart/competitive-task-subset-v1'
config = json.loads((directory / 'transfer_descriptor.json').read_text(encoding='utf-8'))
config['output_root'] = str(root / 'experiments/2026-09-restart/runs/competitive-task-subset-v1/transfer_features')
config['model']['weights_path'] = str(Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
(directory / 'transfer_descriptor.local.json').write_text(
    json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
)
```

The following extraction command passes that derivative through `--config`, while
`--output-root` resolves to its matching absolute destination. The script compares
the complete configuration to the declared protocol and local checkpoint. Keep all
other fields unchanged. The relocated file has a different configuration hash,
which will be recorded in the new run; retain the archived file as the original.

Use regenerated stage-1--3 metadata, whose absolute image paths already point to
the new checkout. Do not patch inaccessible historical metadata inside a locked run.
Kinematic extraction resolves recordings from the explicit `--dataset-root` and
record keys, independently of the historical `source_path` column. Keep
`config.json` next to `DECISION.md`, and retain project-relative run references.
Its `kinematic_directory`, `transfer_directory`, `validation.outer_splits_from` and
`validation.selection_plan_from` must identify the newly generated artifacts.

## Run in this order

Start with empty extractor destinations in a new working copy. Replace
`PATH/TO/PaHaW` with a directory containing `PaHaW_public` and `PaHaW_files`.

```text
python scripts/prepare_kinematic_features.py --dataset-root "PATH/TO/PaHaW" --metadata experiments/2026-09-restart/runs/static-svm-v1/features/metadata.csv --output-root experiments/2026-09-restart/runs/competitive-task-subset-v1/kinematic_features --descriptor experiments/2026-09-restart/competitive-task-subset-v1/kinematic_descriptor.json
python scripts/prepare_transfer_features.py --config experiments/2026-09-restart/competitive-task-subset-v1/transfer_descriptor.local.json --output-root experiments/2026-09-restart/runs/competitive-task-subset-v1/transfer_features
python scripts/run_competitive_task_subset.py --config experiments/2026-09-restart/competitive-task-subset-v1/config.json
python scripts/verify_competitive_task_subset.py --config experiments/2026-09-restart/competitive-task-subset-v1/config.json --phase fits
python scripts/analyze_competitive_task_subset.py --config experiments/2026-09-restart/competitive-task-subset-v1/config.json
python scripts/verify_competitive_task_subset.py --config experiments/2026-09-restart/competitive-task-subset-v1/config.json --phase complete
```

An optional transfer check uses the same extraction command with `--benchmark-only`.
It checks the first batch's repeated inference and model-state integrity, writes no
feature artifacts and performs no classifier fit. The new tests can be run with:

```text
python -m pytest tests/test_kinematic_features.py tests/test_transfer_features.py tests/test_competitive_selection.py tests/test_competitive_analysis.py -q
```

The transfer extractor produces nine float32 matrices of shape 597 × 512 in the
static metadata row order. It reuses foreground crop and square white padding,
resizes to 224 × 224 with bilinear interpolation and applies fixed ImageNet
normalization. All ResNet18 parameters and BatchNorm buffers are frozen in eval
mode; no augmentation or end-to-end fine-tuning occurs. The only trainable CNN
component in the downstream comparison is the class-balanced L2 logistic head,
whose seven C candidates and encoding choice are evaluated inside training.

The fitting runner may resume completed outer units only when source, config,
protocol and input hashes match its lock. Do not modify fitting sources during a
run. Analysis requires all 25 units and produces the three campaign figures.
`--phase complete` then verifies the analyses as well as the fitting artifacts;
there is no separate competitive finalizer command. The complete verifier fits no
new model. Its report specifies which saved models and inferential contrasts it
independently reproduced.

## Original hashes and completed execution

| Source | Original SHA-256 |
|---|---|
| `config.json` | `a23214623041789e0b077b7a5f9d8f3de28fc1e809725c3fdbed9e74285d3671` |
| `DECISION.md` | `6bad95d8094cb6d5b9bb7fdbd5ba639cb3937a771da4328a6252e7c43edf2a34` |
| `kinematic_descriptor.json` | `ac56f9c774c1af4672bac15df67bcd9e80a8be98aceb838a36cc2e7c2686c248` |
| `transfer_descriptor.json` | `d95c6a81cdc9e2f3e51ef7a3a06deb232dff2fedf1a44052f1089936abc3e5ee` |
| Official ResNet18 checkpoint | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

The original transfer extraction verified 5,373 image inputs in nine encodings and
kept the same complete backbone-state digest before and after every encoding. The
classification campaign completed 25 outer units, 163,800 tuning fits and 873 final
fits. Its complete audit passed 237,893 checks and independently reproduced 73 saved
model predictions and the six primary bootstrap/randomization contrasts. These are
facts about the recorded execution, not guarantees of identical results on another
hardware/software stack.

## Local data and public source distribution

Obtain PaHaW separately under its access terms. The public source package needs no
participant recordings, participant lists, individual predictions, feature matrices,
trained heads, or attribution maps. Keep those generated artifacts in the local
research workspace. No reproduction step requires uploading identifiable or
participant-level data to GitHub. The package distributes source code, fixed
protocols, configuration descriptions and their hashes; the supplied article can
be compiled from its included artwork without access to the experiment data.
