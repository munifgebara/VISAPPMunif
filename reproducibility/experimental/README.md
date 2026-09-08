# Experimental code accompanying the manuscript

`reproducibility/experimental/` contains exact copies of the existing Python package,
the run and verification scripts for eight stages, their original JSON configurations,
and the project environment files. `manifest.json` gives a SHA-256 digest for each
source. Packaging reads code and configurations only; it does not execute experiments.

No PaHaW recordings, participant lists, individual predictions, model checkpoints,
attribution maps or saved experiment outputs are distributed in this directory.
The required PaHaW data must be obtained separately under the dataset's access terms.
Scripts and configurations for the width ablation and performance-gap audit are
excluded. The shared `dynamic_image.py` is preserved byte for byte and therefore
also contains unused width-rendering helpers; none of the eight supplied run
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
   `dynamic-single-signal-v1` and `dynamic-triplet-svm-v1`. Their original values are
   retained in the archive for provenance and point to the authors' local drive.
   The replacement must identify a PaHaW directory containing
   `PaHaW_files/corpus_PaHaW.xlsx` and `PaHaW_public/` with its `.svc` recordings.
   The original local inventory has 75 participants and 597 records; a different
   inventory is a different reproduction condition and must be reported as such.
5. In a fresh directory, keep the supplied project-relative `output_root` and run
   references. If outputs are relocated, update every downstream `static_run`,
   `single_signal_run`, `feature_run`, `source_run`, `image_run`, `classical_run`,
   `fusion_run`, `cnn_run` and `prediction_sources.*.run` reference consistently.
   Output directories must be empty; the runners refuse to overwrite existing runs.
   Keep rendering settings, seeds, tasks, searches and comparison families unchanged.

The selected encoding is explicitly fixed to SAZ in the downstream configurations,
matching the exploratory selection reported in the article. A rerun that produces a
different ranking does not silently redefine those configurations. Document any
subsequent representation choice as a new analysis.

## Stage order and commands

Each `run` command below executes the corresponding analysis and may fit models.
These commands are reproduction instructions; they were not executed when preparing
this code package. Run the verification command after its stage finishes.

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

## What this package establishes

The eight stages contain all implementation and configuration dependencies needed
to rerun the reported experimental sequence after the data and environment are
provided. Source packaging was checked by syntax parsing, import resolution within
the package, and byte-for-byte hashes. No fit, prediction, explanation or statistical
comparison was rerun for this editorial release; agreement of a future rerun with
the published values has therefore not been newly established.

The separate generators in `reproducibility/scripts/` reproduce manuscript artwork
from saved experiment files in the full research repository. In particular,
`generate_paper_figures.py` also reads the historical `article-experiment-package-v1`
consolidation. That older consolidation includes an excluded width contrast, so its
runner and configuration are deliberately absent here. The present code package
supports the eight experimental stages, not a standalone one-command reconstruction
of every publication figure. The Overleaf archive already contains the reviewed
figures and LaTeX sources needed to compile the article without any experiment data.
