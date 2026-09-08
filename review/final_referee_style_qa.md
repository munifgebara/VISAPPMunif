# Final referee review: prose, XAI and closing pages

Scope: read-only review of the revised XAI, discussion and conclusion sources, with
individual visual inspection of rendered manuscript pages 11--14 in
`tmp/final-referee/render/`. No experiments, model inference or LaTeX edits were
performed during this check.

## Visual review

- Page 11: Figure 7 and its full caption are legible. All four diagnosis/outcome
  cases, the retained zero Grad-CAM, and the confidence labels remain visible.
  The continuation of the XAI analysis has no clipping or overlap.
- Page 12: Figure 8, uncertainty bars, axis labels and caveated caption are clear.
  The discussion retains its bridge paragraph before subsection 8.1.
- Page 13: Table 3 has readable cells and complete horizontal rules. Discussion,
  reproducibility and conclusion have no collisions, overflow or orphan headings.
- Page 14: the complete bibliography is readable; accented names, titles and
  publication details render without missing-glyph boxes.

## Scientific prose

The revised XAI account distinguishes foreground concentration, perturbation
sensitivity and attribution agreement. It explicitly retains zero maps in deletion
summaries, explains their tied pixel ranking, and reports the limited participant
coverage of the weight-randomization check. It does not infer disease-specific
regions from the heatmaps. The discussion retains observed encoding gains while
separating them from overall superiority, and the conclusion limits the classifier
result to the selected encoding and tested CNN. Repeated generic commentary has
been reduced without removing the six introductory section bridges. No new
numerical claim or scientific impropriety was identified in those passages.

## Required wording clarification

In subsection 8.4, `The experimental archive retains participant assignments...`
could refer to the distributed source archive described immediately before it.
That package deliberately excludes predictions and checkpoints. Specify
`Our internal experimental archive retains...` to distinguish retained research
outputs from the distributed code package. After this small edit, recheck page 13;
the remaining reviewed pages have no open visual defects.

## Closure after the final edit

The archive wording was corrected to `Our internal experimental archive retains...`.
The manuscript was rebuilt and all 26 pages were rendered again into
`tmp/final-referee/final-render/`. PNG SHA-256 comparison identified page 13 as
the only changed page. It was reinspected individually: the correction is visible,
Table 3 and the conclusion remain complete and legible, and no layout issue was
introduced. The other 25 pages are pixel-identical to the reviewed render.
The required clarification is resolved; final result: PASS, no open blocker.
