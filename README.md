# Incremental handwriting manuscript

This expanded advisor draft presents a new formulation of pseudodynamic handwriting
minimap signatures, followed by an incremental encoding study, training-only
selection, spatial controls, competitive baselines and an XAI case study.
The title and illustrated activity diagram approved by the authors are retained.
The width-ablation experiment is excluded. Yandre and Angel will advise on cuts;
the expanded draft is not the final conference submission.

## Read and compile

- `main.tex` is the Overleaf entry point; `sections/` contains the English text.
- `gallery.tex` contains thirteen alternative figures. Its page numbering follows
  the complete main manuscript through generated `gallery_start.tex`.
- `review/build_verification.json` records the current page, character, figure,
  table and reference counts, together with source and output hashes.
- `review/competitive_manuscript_review.md` and `review/competitive_revision.md`
  describe the new scientific review and applied changes. Earlier review reports
  retain their historical scope and should not be read as reviews of this version.

Compile main.tex with pdfLaTeX, BibTeX, and pdfLaTeX twice. Compile gallery.tex
separately. In the experiment workspace, `python scripts/build_paper.py` builds
both PDFs and the Overleaf ZIP under output/pdf. The optional --submission-check
also enforces the declared conference limits; the default permits this expanded
advisor draft. No experiments run during the document build.

## Evaluation stages

The initial eight-task encoding ranking and its SAZ classifier/XAI follow-ups
are exploratory and retained as such. The subsequent selection experiment chooses
encodings, SVM parameters and fusion inside training. Its spatial controls shift
or permute joint signal vectors within pen-down strokes using three fixed seeds.

The competitive extension fixes T2/T3/T4 from the tasks reported in the Casademunt
thesis before fitting models. It evaluates static LPQ+SVM, training-selected
LPQ+SVM, an 83-attribute kinematic/pressure SVM and ImageNet ResNet18 features
with a trained linear head. The backbone is frozen, not fine-tuned end to end.
All eight tasks are a declared secondary comparator; no best subset is selected
from outer-test results. Fixed majority voting is used for this extension.

Selected LPQ reaches participant macro F1 0.6449 on three tasks, compared with
0.6143 for static, 0.6279 for kinematics and 0.6021 for transfer features. The
paired differences remain uncertain after correction. Reducing tasks helps two
pipelines numerically and lowers two others. These results do not establish
superiority, equivalence, external validation or a clinical diagnostic system.

The cohort remains 75 participants and 597 records. Encodings, tasks, seeds and
repeated predictions do not increase the number of independent people. The
bootstrap and randomization analyses are conditional on saved predictions.

## Reproduction and editorial status

The Overleaf ZIP includes exact code/configuration copies for ten experimental
stages under reproducibility/experimental, with commands and SHA-256 manifests.
The original PaHaW recordings, individual predictions and checkpoints are not
redistributed. Read review/experimental_reproduction.md for dataset paths,
CUDA requirements of transfer extraction, stage order and reproduction limits.
Scientific figures are generated from archived results, not generative images.

The main file remains anonymous. The named author block is preserved for author
verification. References reuse the existing bibliography; cited additions to
that bibliography have verified DOIs, while the user-provided thesis has no
verified DOI. Bibliographic task-set evidence is in review/task_subset_literature.md.

The [VISAPP 2027 guidelines](https://visapp.scitevents.org/Guidelines.aspx?y=2027)
set submission character limits and publication page allowances. Current counts
are checked in the build manifest; this expanded draft may exceed them by the
user's instruction. The separate alternative gallery is for editorial selection.
The assistance disclosure remains recorded in review/ai_disclosure.md; final
submission declarations and placement remain for the authors to complete.
