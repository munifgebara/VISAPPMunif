# Abstract and introduction revision

Date: 2026-09-08. The authors requested a clearer account of the main findings,
with emphasis proportional to the evidence. The manuscript remains an expanded
draft for advisor review.

## Editorial changes

The abstract now opens with the representation and research questions, follows
the static-to-dynamic comparison, and gives the supported classifier contrast
explicitly. The introduction explains what the common descriptor, model search
and participant assignments make it possible to compare, then previews the
classifier, participant-voting and explanation findings. The abstract contains
190 words, within the template's 70–200-word range.

The revision changes emphasis and exposition. No new experiments, numerical
results, references or claims of clinical validity were added. The eight main
figures, twelve alternatives and all result sections are unchanged.

## Evidence and claim limits

An independent reading checked the saved article tables and original classifier
contrasts, in addition to Sections 3–7. Another review assessed the revised opening
for scientific accuracy and flow. Its suggestion to identify the exploratory,
SVM-selected representation beside the classifier finding was incorporated.

| Finding emphasized | Evidence | Scope retained in the opening |
|---|---|---|
| Incremental signal comparison | SAZ task-mean macro F1 0.5752; static 0.5643; paired difference 0.0109, 95% interval [−0.0146, 0.0370], Holm p=1.0000 | No demonstrated difference among the nine encodings; no assertion of equivalence or irrelevance of physical signals |
| Fixed texture features versus the tested CNN | SVM 0.5752; CNN 0.5105; paired difference 0.0646, 95% interval [0.0277, 0.1021], Holm p=0.0054; LR and RF also exceed the CNN | The small network trained from scratch, the SVM-selected encoding and exploratory comparison on this cohort |
| One decision per participant | Majority-vote macro F1 0.6498 for LR and 0.6488 for SVM | A distinct endpoint from the task mean; no claim of a statistically demonstrated fusion gain or LR superiority |
| Spatial explanation agreement | Grad-CAM foreground enrichment 1.803; median Grad-CAM–occlusion correlation −0.0121 | Concentration and perturbation sensitivity do not establish disease-specific regions or complete explanation faithfulness |

Inference remains conditional on the saved predictions, as detailed in Section 3.
Encoding selection is not independently validated. The opening does not generalize
the CNN result to pretrained or other architectures, and does not rank the method
above published studies with different protocols.

## Build and visual review

The manuscript has 14 pages; the selection edition has 26, with alternatives A1–A12
on physical pages 15–26. The approved activity diagram remains Figure 1 on page 2.
The contribution paragraph is kept together to avoid splitting a word between
pages 1 and 3 around the full-page diagram.

Final visual verification is recorded in `build_verification.json` after inspection
of the rendered pages. Historical insertion reviews refer to the earlier 13-page
manuscript. Build checks cover all eight figure captions, three tables, twelve cited
references, gallery labels, abstract length and unresolved LaTeX warnings.

All 26 pages were opened and inspected individually: the main reviewer checked
pages 1–3, a second reviewer checked 4–14, and a third checked 15–26. After the
paragraph-continuity correction, PNG checksums identified changes only on pages
1 and 3–7; all six were re-inspected. Pages 2 and 8–26 remained pixel-identical
to the previously inspected render. No clipping, overlap, missing content or
unresolved legibility issue remains. The gallery headings and printed page
numbers are continuous and correct. Result: passed.
