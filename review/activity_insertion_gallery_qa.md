# Gallery visual QA after activity-diagram insertion

Reviewed on 2026-09-08. The updated selection PDF contains 25 pages: the 13-page expanded manuscript followed by twelve alternative-figure pages. This check concerns the gallery only.

- PDF: `output/pdf/manuscript_with_figure_alternatives.pdf`
- Build identification: expanded manuscript with thirteen main pages and twelve gallery pages. The main manuscript was subsequently recompiled to adjust the introduction; no PDF checksum is fixed here because that changes the container while the gallery content and pagination remain unchanged.
- Rendered pages inspected individually: `tmp/activity-insertion-qa/page-14.png` through `page-25.png`.
- PDF metadata and text extraction independently confirm 25 pages, continuous page labels 1–25, gallery headings A1–A12, and printed gallery footers 14–25.

Each full-page image was opened and visually inspected; this was not a contact-sheet-only check. The twelve gallery pages retain one composite figure per page, a readable scientific caption, and a separate editorial selection note. No cropping, stretched images, overlapping caption blocks, missing glyphs, or displaced footers were found. The large white areas surrounding shorter plots follow the intended one-figure-per-page selection layout.

| Physical page | Figure | Visual inspection |
|---|---|---|
| 14 | A1. Isolated dynamic signals | Both subject rows and the four signal columns remain clear; stroke detail and caption are legible. Footer 14 is correct. |
| 15 | A2. Reading the RGB representation | Individual channels and the combined colour image retain their proportions. Channel semantics and caption are legible. Footer 15 is correct. |
| 16 | A3. The eight tasks as static images | All sixteen panels remain visible with intact task labels. No clipping of handwriting or text. Footer 16 is correct. |
| 17 | A4. The eight tasks in pseudodynamic colour | All sixteen colour panels and their labels remain clear; the atlas retains the same proportions as the static version. Footer 17 is correct. |
| 18 | A5. Encoding performance across tasks | All 72 values, nine row labels, eight task labels, and colour-bar ticks are readable. Caption retains its descriptive scope. Footer 18 is correct. |
| 19 | A6. Selected encoding profiles | Four curves, distinct markers, legend, axes, and caption remain legible. Footer 19 is correct. |
| 20 | A7. Classifier differences relative to SVM | Confidence intervals, zero reference, classifier labels, and Holm-adjusted p-values are clear. Caption fits without collision. Footer 20 is correct. |
| 21 | A8. Three ways to combine SVM task predictions | Long fusion-rule labels fit; three estimates and intervals are readable. Caption and editorial note are separated. Footer 21 is correct. |
| 22 | A9. Discrimination from task-vote fractions | Four ROC curves, diagonal, full axes, and AUC legend are visible. Caption clearly distinguishes repetition-level averaging from pooled predictions. Footer 22 is correct. |
| 23 | A10. Errors after task voting | Four confusion matrices remain readable, including percentages, mean counts, and observed/predicted class labels. Caption and note fit below the figure. Footer 23 is correct. |
| 24 | A11. Channel-removal sensitivity by task | All 24 signed values, task labels, channel names, and symmetric colour scale are clear. Caption retains the contrast-change caveat. Footer 24 is correct. |
| 25 | A12. Agreement between spatial explanations | Both histograms, correlation axes, sample counts, and median annotations are readable. The left annotation lies over a coloured bar but remains decipherable; it does not require a layout correction. Footer 25 is correct. |

**Disposition:** passed. Insertion of the activity diagram has correctly moved the gallery from physical pages 13–24 to 14–25. No LaTeX changes are required by this review, and none were made.
