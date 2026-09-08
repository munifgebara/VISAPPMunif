# Scientific and editorial review of the competitive extension

Reviewed on 8 September 2026: current `paper/main.tex` and every included section
after the nested-selection, spatial-control and competitive-task additions.
This review edits no manuscript source and requests no additional model fits.
PDF layout inspection remains a separate production check.

The new numerical claims agree with `global_summary.csv`,
`method_comparisons.csv`, `task_subset_comparisons.csv` and `task_summary.csv`
in the competitive run. In particular, the four primary fusion estimates,
confidence intervals, 0.0307 selected-LPQ minus static difference, class-specific
rates, vote AUCs, all-eight estimates and task-reduction contrasts are correctly
rounded. The individual T2/T3/T4 accuracies used beside the thesis comparison are
54.67/56.00/63.20%. There is no missing introductory paragraph between a section
and its first subsection. Novelty is appropriately scoped to the formulation,
and the prior use of dynamics in images is acknowledged.

## Corrections recommended before the final build

### R1. Mark the return to the initial fixed-SAZ experiment

**Priority: medium. Locations:** `05_classifiers_fusion.tex:5`, `:46`, `:50`;
`03_protocol.tex:39`.

The immediately preceding text now describes adaptive encoding selection and
its instability. The next section opens with “the selected representation” and
then fixes SAZ, which can appear to make SAZ the output of the protected
selection procedure. Its fusion section similarly leaves the initial SAZ
condition implicit. The validation caption says “The CNN” despite the new
ResNet18 pipeline having no epoch selection on PaHaW.

Suggested opening: “Returning to the initial exploratory comparison, we keep
SAZ and its participant partitions fixed to compare LPQ classifiers with a CNN
trained from scratch.” Identify the subsequent fusion results as those initial
fixed-SAZ predictions in the opening sentence and figure caption. Start the
validation caption with “Initial fixed-encoding evaluation” and identify the
“scratch-trained CNN”. These edits separate stages without rearranging them.

### R2. Identify what the 0.5991 versus 0.6171 contrast tests

**Priority: medium. Location:** `07_discussion.tex:11`.

“Selecting the fusion rule internally likewise provides no observed advantage
over static” puts the apparent treatment on fusion-rule selection. Both
pipelines in this contrast select fusion internally; their difference is the
encoding policy. It does not compare choosing a rule against a fixed rule.

Replace that clause with: “With the fusion rule selected internally for both
pipelines, nine-encoding fusion reaches 0.5991 against 0.6171 for static.”
The later 0.6330 static result is correctly described as fixed majority fusion
in `05b_competitive.tex:81`; it is not a contradictory estimate.

### R3. Distinguish recorded timing from measured protocol burden

**Priority: medium. Location:** `07_discussion.tex:19`.

“Acquisition time and clinical burden were not measured” is now too broad: the
83-feature baseline explicitly derives valid acquisition time from timestamps.
The unavailable claim is a measured benefit in total administration time or
clinical burden from adopting three tasks.

Suggested wording: “Using three tasks reduces the number of exercises; this
study did not evaluate the resulting change in protocol administration time
or clinical burden.” This retains the intended limitation without denying the
recorded temporal measurements.

### R4. Narrow the image-only SVM statement

**Priority: low. Location:** `04_representation_results.tex:15`.

“The SVM ... never receives the original signal measurements as tabular
features” applies to the LPQ experiment, whereas the manuscript now also has a
signal-based SVM. Replace “The SVM” with “The LPQ-based SVM in this comparison”.
No change to the underlying experiment or interpretation is needed.

### R5. Remove repeated numerical catalogues and defensive phrasing

**Priority: low. Locations:** `01_introduction.tex:24`;
`05b_competitive.tex:68`, `:81`.

The same four primary point estimates appear together in the abstract,
Introduction and competitive table, then the leading pair returns in the
Discussion and Conclusion. Retain the complete list in the abstract/table;
use the Introduction paragraph to state the substantive question and outcome:
three-task fusion ranks selected LPQ first numerically, with uncertain paired
differences and opposite task-reduction effects for different pipelines.

In the results, replace “The fixed voting threshold and coarse vote fractions
account for different aspects of the classification” with a direct metric
description: “Macro F1 evaluates the fixed voting decision; AUC ranks
participants by their vote fractions.” Also replace “their differences across
sections do not represent contradictory estimates of the same procedure” with
the positive definition: “Section [nested follow-up] selects among three
fusion rules; this comparison fixes majority voting.” These changes improve
precision and flow without changing any inference.

## Findings requiring no change

- The nine-encoding and dynamic-only nested estimates are correctly identified
  as adaptive procedures, including altered inner partitions. No decline is
  attributed solely to correcting selection bias.
- The primary three-task participant endpoint is separated from task means
  and from the secondary all-eight task population. Task reduction for image
  methods is correctly allowed to change the internally selected encoding.
- Kinematic extraction is described as an independent baseline, with its
  temporal conventions, native units, absent distributions and lack of
  smoothing disclosed. It is not presented as an exact Drotar reproduction.
- Frozen ResNet18 transfer is distinguished from end-to-end training and the
  earlier small CNN. XAI explicitly concerns the latter only.
- Spatial controls preserve joint vectors, with raster-histogram and local
  continuity qualifications. No physiological mechanism or corrected
  significance is inferred from the positive pointwise permutation interval.
- The manuscript reports uncertain gains without treating nonsignificance
  as equivalence. The literature table preserves differing protocols and
  does not assert a direct superiority test.

The main remaining source-level weakness is stage identification, rather than
numerical inconsistency. The five edits above are sufficient for this pass;
full layout and bibliography rendering should be checked after integration.
