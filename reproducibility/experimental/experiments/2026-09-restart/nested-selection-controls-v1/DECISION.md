# Selection inside training and spatial controls

Protocol fixed on 2026-09-08 before fitting this campaign. The user authorized
improvements 1 and 2. Existing runs remain references and are not overwritten.

The 25 existing participant outer partitions are reused. All new inner partitions
are COMMON across tasks; the missing spiral is intersected with those partitions.
LPQ, rendering and the 21-candidate SVM search remain fixed. No classifier family
is selected in this campaign, and the previous CNN/LR/RF comparisons remain exploratory.

Three authentic policies are evaluated: static only; auth9, selecting among all
nine encodings; and auth8, selecting among the eight dynamic encodings. An encoding
is selected globally within a fitting partition, while SVM hyperparameters vary
by task. The score is the mean of the eight best task-wise inner-fold macro F1
scores, with deterministic declared-order tie breaks. Static has priority in auth9
ties. These are tuning scores, not unbiased performance estimates.

For the participant endpoint, outer-train is divided into three meta-folds. Within
each meta-train, three further tuning folds repeat the full encoding and SVM
selection. Sigmoid calibration and reliability weights are confined to meta-train.
Meta-validation predictions choose among the three existing fusion rules. The
selector is then fitted using all outer-train and tested on outer-test. Static
uses the same procedure. No outer-test labels choose encodings, models, calibration,
weights, fusion or thresholds.

The controls test the task endpoint. Each seed/condition independently selects
among eight transformed encodings and tunes SVM using exactly the same outer-train
tuning folds as auth8. The transformations apply to BOTH training and test records.
There is no fusion or extra classifier experiment for the controls.

Primary spatial control: circularly shift the joint channel vectors within each
contiguous pen-down stroke, using a nonzero shift of 25–75% of its segment count.
Secondary control: jointly permute the vectors within each stroke, also disrupting
local continuity. Single-segment strokes remain unchanged. Seeds are 20260909,
20260910 and 20260911; no seed is selected by its result. Geometry, segment count,
width, normalization and the multiset of signal vectors within each stroke are
preserved. Pixel histograms need not be preserved after drawing unequal segment
lengths, overlaps and antialiasing. Results cannot isolate all effects of texture.

For each repetition, concatenate outer-test folds before computing task macro F1.
Average those metrics across five repetitions, eight tasks, and three control
seeds where applicable. Seeds do not create extra participants and are not ensembled.
Use 10,000 diagnosis-stratified participant bootstrap draws and participant-wise
method swaps, keeping repetitions, tasks and seeds together. Comparisons retain
the conditional uncertainty of saved predictions, not the full variability of refitting.

Families fixed in config: four global task contrasts (Holm4), two secondary
participant-fusion contrasts (Holm2), and 32 exploratory task-specific contrasts
(Holm32). Report all outcomes, selection frequencies and seed variability. No new
experiment is chosen based on whether the desired advantage becomes significant.

These analyses repair selection inside cross-validation. They still reuse the
cohort that informed the method's development and are not independent external
confirmation or evidence of clinical readiness.
