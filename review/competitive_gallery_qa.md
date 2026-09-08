# Visual QA of the expanded competitive-baseline gallery

Reviewed on 2026-09-08. Each of the thirteen rendered pages was opened individually
with `view_image`, using `tmp/competitive-paper/render/page-20.png` through
`page-32.png` at 1650 pixels in height. This was not a contact-sheet-only review.
The selection PDF contains nineteen manuscript pages followed by this gallery.

Independent PDF text extraction confirms 32 physical pages, continuous page labels
1–32, headings A1–A13 in order, and printed gallery footers 20–32. Each gallery page
contains one composite figure, a scientific caption and a separate editorial note.

| Physical page | Heading | Individual visual check |
|---|---|---|
| 20 | A1. Isolated dynamic signals | Both participant rows and all four signal columns are intact. Spiral detail, channel titles and caption are legible; footer 20 is correct. |
| 21 | A2. Reading the RGB representation | The three grayscale channels and RGB composite retain their aspect ratios. Channel labels and caption fit; footer 21 is correct. |
| 22 | A3. The eight tasks as static images | All sixteen atlas panels and their task/class labels remain visible. Handwriting and caption are not clipped; footer 22 is correct. |
| 23 | A4. The eight tasks in pseudodynamic colour | All sixteen colour panels, task names and caption remain legible. Aspect ratios agree with the static atlas; footer 23 is correct. |
| 24 | A5. Encoding performance across tasks | All 72 annotated cells, row labels, task identifiers and colour-bar ticks are readable. Caption retains its descriptive interpretation; footer 24 is correct. |
| 25 | A6. Selected encoding profiles | Four curves, marker shapes, legend and axes are readable. Caption clearly treats task identifiers as categorical; footer 25 is correct. |
| 26 | A7. Classifier differences relative to SVM | Three complete intervals, zero reference and Holm p-values fit. The scientific caption and editorial note do not collide; footer 26 is correct. |
| 27 | A8. Three ways to combine SVM task predictions | The long fusion-rule names and all interval endpoints are visible. Caption is legible and separated from the note; footer 27 is correct. |
| 28 | A9. Discrimination from task-vote fractions | Four ROC curves, diagonal and the full AUC legend are readable. Caption explains repetition-wise averaging and vote scores; footer 28 is correct. |
| 29 | A10. Errors after task voting | Four confusion matrices are intact. Percentages, mean counts, axes and the caption remain readable; footer 29 is correct. |
| 30 | A11. Channel-removal sensitivity by task | All 24 signed values, channel labels, task names and colour scale are clear. Caption includes the image-contrast limitation; footer 30 is correct. |
| 31 | A12. Agreement between spatial explanations | Both histograms, median annotations and valid-pair counts are legible. The left median annotation overlaps a coloured bar but remains readable, consistent with earlier gallery reviews; footer 31 is correct. |
| 32 | A13. Competitive methods on the three fixed tasks | All four method profiles, T2/T3/T4 labels, the two-column legend and caption fit. The caption explicitly makes the task profiles descriptive and states selection within training. The nearly coincident T2 image-method points reflect their similar values. Footer 32 is correct. |

**Disposition: passed.** No truncated captions, misplaced footers, missing glyphs,
clipped figures, stretched graphics or layout collisions were found. Large vertical
spaces around shorter figures follow the intended one-figure-per-page selection
layout. No LaTeX or figure changes are required by this review, and none were made.

The numerical audit of the new competitive figure is recorded separately in
`experiments/2026-09-restart/runs/competitive-task-subset-v1/reports/figure_qa.md`.
As noted there, distinct marker shapes would be an optional monochrome-accessibility
improvement if A13 is later chosen for the main article; the current colour version
is legible.
