# Competitive methods on a literature-defined task subset

Fixed on 8 September 2026 before fitting the new classifiers.

The user authorized option 3 (kinematic/pressure SVM and transfer CNN), a task
subset motivated by related work, manuscript revision, and publication to the
existing main branch. This extension follows the completed nested-selection and
spatial-control experiments; their sources and results remain immutable.

## Tasks and estimands

The primary task set is T2/T3/T4, the repeated l/le/les tasks reported in the
Casademunt Gonzalez thesis (T2 development and Appendix III for T3/T4). The
thesis does not report our proposed three-task fusion. Drotar uses T2--T8;
Diaz uses different five-task subsets for different representations. There is
no universal subset shared by these references. We will not search subsets
against outer-test outcomes. All eight tasks form a declared secondary comparator.

The primary endpoint is participant-level macro F1 of fixed majority voting
over T2/T3/T4. Secondary endpoints are the mean of individual task macro F1,
and the corresponding two endpoints for all eight tasks. Folds are pooled
within each repetition before scoring; five repetition scores are then averaged.
Missing spirals are omitted from all-eight votes. Tied votes are assigned to PD.
No task weights, fusion-rule search, threshold search, or new calibration is used.

## Four fixed methods

1. Static LPQ (768 attributes) with the original 21-candidate SVM grid.
2. LPQ with training-only selection among the nine original encodings, including
   static, and the same SVM grid.
3. A fixed 83-attribute kinematic/pressure descriptor with the same SVM grid.
4. ImageNet-pretrained ResNet18, with a frozen convolutional backbone and a
   trained L2 logistic linear head. Seven C values are declared in config.json.
   Its image encoding is selected internally among the same nine candidates.

Frozen-feature transfer is the fixed-feature-extractor form of CNN transfer
learning. It is not end-to-end fine-tuning and does not test every competitive
CNN training strategy. ImageNet weights, crop/padding, normalization and 224-pixel
resolution are fixed. Only the linear head and training-only normalization are
learned on PaHaW. Precomputing these unsupervised fixed features is permissible;
all participant labels remain outside feature extraction.

Kinematics retain native coordinate/pressure scales and seconds from timestamps
divided by 1,000. Temporal, geometric, speed, acceleration, jerk, pressure and
pressure-derivative summaries are fixed without supervised screening. Derivatives
do not cross pen lifts or invalid time steps. The pre-fit timestamp audit found
plausible gaps of 1--26.61 seconds and two enormous jumps. We therefore preserve
positive intervals up to 60 seconds and exclude nonpositive or larger intervals,
reporting counts rather than treating excluded intervals as observed duration.
This is a transparent reference baseline, not an exact reproduction of Drotar.

## Selection and inference

Reuse the original 25 outer participant partitions and the three participant-
common tuning folds in nested-selection-controls-v1. For each outer training set,
each task and each eligible encoding, tune hyperparameters by mean inner-fold
macro F1. Select one encoding by the mean of the selected scores across the
specified task set. Use declared candidate/encoding order for ties. Refit only
on outer training participants and persist all choices before outer evaluation.
Each method has its own search; neither another method's outer outcomes nor the
other task set's scores enter its choice. Training searches can be cached across
task sets because per-task fitting partitions and candidates are identical.

Report every method, both task sets, and all declared contrasts. The primary
Holm family comprises six method pairs for three-task fusion. Each other endpoint
has its own six-pair secondary family. Four within-method contrasts compare the
three-task and all-task fusion pipelines. These latter contrasts include any
change in training-selected encoding, not only removal of five votes.

Ten thousand diagnosis-stratified participant bootstrap draws and ten thousand
paired prediction swaps retain each person's tasks and repetitions together.
Intervals are pointwise, conditional on saved predictions; they do not capture
retraining or the full research-selection history. The cohort has already
informed development. A literature-motivated task set and internal selection
do not turn this reanalysis into independent validation.

No new task subset, architecture, descriptor or search grid will be selected
after seeing this run. Any genuine implementation correction must be documented
and must invalidate affected cached results rather than silently mixing versions.
