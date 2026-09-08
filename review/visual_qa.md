# Visual review of the 12-page manuscript

Reviewed the individual rendered pages `tmp/paper-qa/main-01.png` through `main-12.png`, not only the contact sheet. This review covers the rendered main manuscript before the author-choice gallery was appended. No manuscript or figure source was changed during this review.

## Findings

No blocking layout defect was found: text, equations, figure captions, tables and references stay within their columns and page margins. There are no clipped objects, overlapping labels, empty pages or unexplained large blank areas. The compact XAI labels below deserve a small readability improvement before release.

1. **Page 10, Figure 6 — small annotation text (minor but actionable).** The image is included at 72% of text width. Its native 7-point row annotations and zero-map label therefore print at approximately 5 points. The handwriting and attribution maps remain legible, but the vertical diagnosis/outcome/probability labels require close inspection. Increase these annotations in the figure asset, while retaining the present canvas and placement, to avoid changing pagination. The three column headers would also benefit from a small increase. Preserve all four examples and the zero-map failure.
2. **Page 6, Figure 3 — tied displayed rounding (optional).** SAZ and PAZ both display `0.575` although their underlying values differ. The order and text explain the difference, so this is not incorrect. Displaying four decimals would make the figure self-contained and match the manuscript's numerical precision.

## Page-by-page inspection

| Physical page | Content checked | Assessment |
| --- | --- | --- |
| 1 | Title, anonymous author block, keywords, abstract, introduction | Title fits; columns and paragraph continuation are correct. The template's top space is intentional. |
| 2 | Representation equations and signal definitions | Equations 1–5 and their numbers fit within the columns. Section wrapping is clean. No collisions or clipped mathematical symbols. |
| 3 | Figure 1; encoding and task-inventory tables | Spiral images correctly show static, speed, SAZ and PAZ for the two example classes. Colours are visible at manuscript scale. Both tables are readable; task T8 retains its sentence and diacritics. |
| 4 | Participant protocol, metrics, LPQ introduction | Text and equations 6–7 fit. The section transition is clean; no float-related gap. |
| 5 | Figure 2 validation diagram and caption; static/single-signal results | All box text fits and arrows avoid labels. “Shared across tasks / methods” is attached to the outer-fold box. The caption explicitly limits common assignments to the outer folds and describes the distinct inner selection procedures. |
| 6 | Figure 3 and paired-result discussion | Dot positions, zero-reference line and confidence intervals are readable and consistent. Optional four-decimal labels noted above. No interval clipping. |
| 7 | CNN methods, classifier comparisons, task-fusion equation | Dense but readable two-column text. Equation 8 fits with its number. No table or figure overlap. |
| 8 | Figure 4; XAI definition and Equation 9 | Task names, algorithm legend and both axes are readable. The figure distinguishes the task endpoint from its overall mean. Equation 9 fits. |
| 9 | Figure 5; XAI concentration, perturbation and channel results | Fusion intervals and point labels fit. Caption identifies the participant-level endpoint and conditional intervals. Subsection headings do not collide with the float. |
| 10 | Figure 6; beginning of discussion | Four diagnosis/outcome strata are present, with both successes and errors. The PD-error Grad-CAM is actually blank and labelled as zero. Grad-CAM and occlusion overlays have the intended differing spatial patterns. Annotation-size improvement noted above. White areas within the images are the actual padded model inputs, not missing image data. |
| 11 | Figure 7; literature comparison and limitations | Error bars and semantic channel labels are clear. Caption retains the unmatched-foreground limitation of the deletion control. No clipped axes or extraneous whitespace. |
| 12 | Comparison table, conclusion, references | Table 3 fits across the page. Wrapped study and method names are unambiguous, accuracy is identified in the header, and the multiple reported Casademunt task values align with the task list. Conclusion and references finish on this page without clipping or overflow. |

## Scope of the conclusion

This is a visual and figure-semantics review. It does not independently re-audit bibliographic transcription or reproduce numerical experiments. The gallery beginning on physical page 13 requires its own page review after assembly.

## Follow-up corrections applied to figure assets

Both findings were subsequently addressed in `scripts/generate_paper_figures.py` and the generated PNG/PDF assets:

- `xai_examples`: column headers, diagnosis/outcome/probability annotations, and the zero-Grad-CAM annotation are now 10 pt in the source PDF. At the manuscript inclusion width of 113.76 mm, they print at approximately 7.49 pt. The PDF MediaBox remains exactly 430.7348070417 × 470.63952 pt and the PNG remains 1793 × 1960 px, so the figure's physical footprint does not alter pagination. All four examples and their maps are preserved.
- `encoding_effects`: point labels now show four decimals, including SAZ 0.5752 and PAZ 0.5745. Small white label backgrounds prevent the static-reference line from crossing the numbers. The original PNG dimensions and PDF MediaBox are unchanged.
- The diagram's catalogue and manifest caption now explicitly states that participant identities define every split and that **outer** assignments are shared across tasks and methods, avoiding any suggestion of shared task-specific inner folds.

The regenerated figures were opened individually and visually inspected. No clipping or annotation overlap was introduced. This follow-up changed no LaTeX or build source; the final document should be rebuilt with the revised assets.
