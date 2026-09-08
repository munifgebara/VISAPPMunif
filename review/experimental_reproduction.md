# Experimental code accompanying the manuscript

`reproducibility/experimental/` contains exact copies of the existing Python package,
the run and verification scripts for ten stages, their JSON configurations and
protocol decisions, and the project environment files. Stages 1--8 preserve the
earlier experimental sequence. Stages 9--10 are new campaigns actually executed
on 8 September 2026: nested selection with spatial controls, followed by the
competitive task-subset comparison. `manifest.json` gives a SHA-256 digest for
each source. The packaging operation itself only copies files; model fitting,
feature extraction, analysis and verification are separate operations.

No PaHaW recordings, participant lists, individual predictions, model checkpoints,
attribution maps or saved experiment outputs are distributed in this directory.
The required PaHaW data must be obtained separately under the dataset's access terms.
Scripts and configurations for the width ablation and performance-gap audit are
excluded. The shared `dynamic_image.py` is preserved byte for byte and therefore
also contains unused width-rendering helpers; none of the supplied run
configurations invokes those helpers.

## Prepare an independent working copy

1. Copy the complete `experimental` directory to a new working directory. Preserve
   the archived source as the reference and make path changes only in the copy.
   Run the commands below from the copied directory, which contains `src`, `scripts`
   and `experiments`. The scripts resolve project-relative paths from their own
   location, not from the JSON configuration's parent directory.
2. Use Python 3.13 and install the supplied environment. The original lock file
   covers the classical environment; the `cnn` extra in `pyproject.toml` additionally
   pins PyTorch and torchvision. No dependency version has been changed for this
   release. Hardware and library builds can still affect floating-point results.

   ```text
   python -m pip install -r requirements-lock.txt
   python -m pip install -e ".[cnn]"
   ```

3. The runner manifests record a Git commit. Execute inside a Git checkout with a
   valid `HEAD`. For a standalone extraction, initialize a local repository and
   commit only the copied sources and adjusted configurations before starting.
   Do not add subsequently generated participant-level outputs to that commit.
4. Edit `dataset_root` in the copied configurations for `static-svm-v1`,
   `dynamic-single-signal-v1`, `dynamic-triplet-svm-v1` and
   `nested-selection-controls-v1`. Their original values are
   retained in the archive for provenance and point to the authors' local drive.
   The replacement must identify a PaHaW directory containing
   `PaHaW_files/corpus_PaHaW.xlsx` and `PaHaW_public/` with its `.svc` recordings.
   The original local inventory has 75 participants and 597 records; a different
   inventory is a different reproduction condition and must be reported as such.
5. In a fresh directory, keep the supplied project-relative `output_root` and run
   references. If outputs are relocated, update every downstream `static_run`,
   `single_signal_run`, `feature_run`, `source_run`, `image_run`, `classical_run`,
   `fusion_run`, `cnn_run` and `prediction_sources.*.run` reference consistently.
   The new campaigns additionally use `triplet_run`, `controls_output_root`,
   `render_config`, `kinematic_directory`, `transfer_directory`,
   `validation.outer_splits_from` and `validation.selection_plan_from`. Keep new
   run artifacts inside the copied project: several provenance checks require
   project-relative paths. Start with empty output directories. New model runners
   can resume completed outer units only if their frozen config/source/input
   signatures remain identical; extractors refuse incompatible existing bundles.
   Keep rendering settings, seeds, tasks, searches and comparison families unchanged.

The selected encoding is explicitly fixed to SAZ in the earlier downstream configurations,
matching the exploratory selection reported in the article. A rerun that produces a
different ranking does not silently redefine those configurations. Document any
subsequent representation choice as a new analysis. Stages 9--10 separately perform
encoding selection inside training; they do not change the earlier SAZ runs.

Fresh executions of stages 1--3 write image and source paths for the new location.
Do not reuse historical metadata containing inaccessible absolute image paths or
edit an already locked run in place. The kinematic extractor resolves recordings
from `--dataset-root` and the record keys. The transfer extractor reads image paths
from regenerated metadata and needs the local path adjustment described in the
stage-10 README. `DECISION.md` must stay beside each new campaign's `config.json`,
because its contents are included in the run lock.

## Stage order and commands

Each `run` command below executes the corresponding analysis and may fit models.
Stages 1--8 below retain their original commands and outputs as historical references;
they were not refitted for the two new campaigns. Stages 9--10 were executed and
audited locally. None of these commands is invoked by the archive builder. Run each
verification command at the indicated point in the sequence.

### 1. Static images, LPQ and SVM

```text
python scripts/run_static_svm.py --config experiments/2026-09-restart/static-svm-v1/config.json
python scripts/verify_static_svm_run.py --run experiments/2026-09-restart/runs/static-svm-v1
```

This stage generates the participant assignments reused downstream.

### 2. Four isolated colour signals

```text
python scripts/run_dynamic_single_signal.py --config experiments/2026-09-restart/dynamic-single-signal-v1/config.json
python scripts/verify_dynamic_single_signal_run.py --run experiments/2026-09-restart/runs/dynamic-single-signal-v1 --static-run experiments/2026-09-restart/runs/static-svm-v1
```

### 3. Four RGB triplets

```text
python scripts/run_dynamic_triplet_svm.py --config experiments/2026-09-restart/dynamic-triplet-svm-v1/config.json
python scripts/verify_dynamic_triplet_svm_run.py --run experiments/2026-09-restart/runs/dynamic-triplet-svm-v1 --static-run experiments/2026-09-restart/runs/static-svm-v1
```

### 4. Combined encoding comparison

```text
python scripts/run_encoding_comparison.py --config experiments/2026-09-restart/encoding-comparison-v1/config.json
python scripts/verify_encoding_comparison.py --run experiments/2026-09-restart/runs/encoding-comparison-v1
```

This stage compares saved predictions from stages 1--3 and fits no models.

### 5. Classical classifier comparison on SAZ

```text
python scripts/run_classical_classifier_comparison.py --config experiments/2026-09-restart/classical-classifier-comparison-v1/config.json
python scripts/verify_classical_classifier_comparison.py --run experiments/2026-09-restart/runs/classical-classifier-comparison-v1
```

This stage fits logistic regression and Random Forest and reuses the SAZ SVM predictions.

### 6. SVM task fusion and internal probability calibration

```text
python scripts/run_task_fusion.py --config experiments/2026-09-restart/task-fusion-v1/config.json
python scripts/verify_task_fusion.py --run experiments/2026-09-restart/runs/task-fusion-v1 --source-run experiments/2026-09-restart/runs/dynamic-triplet-svm-v1 --static-run experiments/2026-09-restart/runs/static-svm-v1
```

This stage evaluates the three SVM fusion rules, including internally calibrated
probabilities. It is required before the four-classifier participant comparison.

### 7. CNN and joint classifier comparisons

```text
python scripts/run_cnn.py --config experiments/2026-09-restart/cnn-v1/config.json
python scripts/verify_cnn_run.py --run experiments/2026-09-restart/runs/cnn-v1 --image-run experiments/2026-09-restart/runs/dynamic-triplet-svm-v1 --classical-run experiments/2026-09-restart/runs/classical-classifier-comparison-v1 --static-run experiments/2026-09-restart/runs/static-svm-v1
```

The runner trains and retains the outer CNN checkpoints. It also creates the joint
four-classifier task comparisons and majority-vote participant comparisons used in
the article. It depends on stages 1, 3, 5 and 6.

### 8. XAI of saved held-out CNN predictions

```text
python scripts/run_cnn_xai.py --config experiments/2026-09-restart/cnn-xai-v1/config.json
python scripts/verify_cnn_xai_run.py --run experiments/2026-09-restart/runs/cnn-xai-v1 --cnn-run experiments/2026-09-restart/runs/cnn-v1 --image-run experiments/2026-09-restart/runs/dynamic-triplet-svm-v1
```

The XAI stage reads stage-7 checkpoints and stage-3 images; it does not retrain the
CNN. The verification scripts check artifact integrity and protocol consistency,
and the CNN/XAI verifiers also reproduce selected checkpoint outputs. They are
checks of generated runs, not substitutes for the preceding analyses.

### 9. Nested encoding/fusion selection and spatial controls

This campaign depends on the generated images, LPQ matrices and participant splits
from stages 1--3. Its single configuration controls both the feature extractor's
`spatial-controls-v1` output and the `nested-selection-controls-v1` model output.
There is no separate `spatial-controls-v1/config.json`.

```text
python scripts/prepare_spatial_control_features.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json
python scripts/verify_spatial_control_features.py --run experiments/2026-09-restart/runs/spatial-controls-v1
python scripts/run_nested_selection_controls.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json --phase all
python scripts/analyze_nested_selection_controls.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json
python scripts/generate_spatial_control_figure.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json
python scripts/verify_nested_selection_controls.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json
python scripts/finalize_nested_selection_controls.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json
```

`--phase authentic` and `--phase controls` can divide the fitting work. Full analysis
requires all 175 outer units. The finalizer requires both verification reports to
pass and records completion; it fits no model. The experiment selects encodings,
task SVM parameters and authentic-data fusion rules within training. Circular shifts
and joint permutations each use all three declared seeds. Their task endpoint is
evaluated separately from participant fusion. The completed local campaign recorded
1,058,400 tuning SVM fits, excluding additional calibration and final refits.

### 10. Competitive methods on T2--T4 and all eight tasks

Read `experiments/2026-09-restart/competitive-task-subset-v1/README.md` before these
commands. It specifies the ImageNet checkpoint, GPU requirement for the supplied
extractor, exact path relocation, and hashes. Stage 10 requires stages 1--3 plus the
stage-9 participant selection plan. The primary T2/T3/T4 set is fixed from the
Casademunt thesis; the all-eight comparison is secondary. Neither subset is selected
from outer-test results.

The archived `transfer_descriptor.json` is a byte-identical copy of the completed
extractor's configuration. In a new checkout, first create
`transfer_descriptor.local.json` with only its two machine paths relocated as shown
in that README. The extractor checks the complete object passed to `--config`;
changing a scientific setting is not a supported relocation.

```text
python scripts/prepare_kinematic_features.py --dataset-root "PATH/TO/PaHaW" --metadata experiments/2026-09-restart/runs/static-svm-v1/features/metadata.csv --output-root experiments/2026-09-restart/runs/competitive-task-subset-v1/kinematic_features --descriptor experiments/2026-09-restart/competitive-task-subset-v1/kinematic_descriptor.json
python scripts/prepare_transfer_features.py --config experiments/2026-09-restart/competitive-task-subset-v1/transfer_descriptor.local.json --output-root experiments/2026-09-restart/runs/competitive-task-subset-v1/transfer_features
python scripts/run_competitive_task_subset.py --config experiments/2026-09-restart/competitive-task-subset-v1/config.json
python scripts/verify_competitive_task_subset.py --config experiments/2026-09-restart/competitive-task-subset-v1/config.json --phase fits
python scripts/analyze_competitive_task_subset.py --config experiments/2026-09-restart/competitive-task-subset-v1/config.json
python scripts/verify_competitive_task_subset.py --config experiments/2026-09-restart/competitive-task-subset-v1/config.json --phase complete
```

Replace `PATH/TO/PaHaW` with the dataset directory already used by stages 1--3.
Feature extraction is independent of participant labels. The ResNet18 backbone is
frozen; only a standardized, balanced L2 logistic head and the encoding choice are
learned in training partitions. This is fixed-feature transfer, not end-to-end
fine-tuning. The completed campaign recorded 25 outer units, 163,800 tuning fits
and 873 final fits. Its complete verifier passed 237,893 checks, reproduced outputs
of 73 saved models without new fitting, and independently reproduced the six primary
bootstrap/randomization contrasts. The verifier does not independently regenerate
every secondary interval or raw p-value.

The two new analyzers generate their campaign figures from saved predictions.
Running these analyzers repeats statistical calculations but does not fit models.
The transfer extractor also has `--benchmark-only`, which checks one batch without
writing feature artifacts; it is optional and is not a substitute for extraction.

## What this package establishes

The ten stages contain the implementation and configuration dependencies needed
to rerun the reported experimental sequence after the data and environment are
provided. Source packaging was checked by syntax parsing, import resolution within
the package, and byte-for-byte hashes. The two new campaigns were actually fitted,
analyzed and audited; the earlier eight stages and their source results were
preserved. Packaging and documentation checks do not constitute another full
experimental rerun, and do not establish bitwise reproducibility on other hardware
or an external cohort.

The separate generators in `reproducibility/scripts/` reproduce manuscript artwork
from saved experiment files in the full research repository. In particular,
`generate_paper_figures.py` also reads the historical `article-experiment-package-v1`
consolidation. That older consolidation includes an excluded width contrast, so its
runner and configuration are deliberately absent here. The present code package
supports the ten experimental stages, not a standalone one-command reconstruction
of every publication figure. The Overleaf archive already contains the reviewed
figures and LaTeX sources needed to compile the article without any experiment data.
