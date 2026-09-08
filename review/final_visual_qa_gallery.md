# Final visual QA: figure-selection gallery

**Result: passed. No TeX or figure change was required.**

Inspected each final render individually, `tmp/paper-final-qa/page-13.png` through `page-24.png`, rather than relying on contact sheets. The renders were checked against `output/pdf/manuscript_with_figure_alternatives.pdf` (24 pages; SHA-256 `ccaa2da86d04d07e0e41d439afb1c69d715ce5486018e3e553073806659f4ffa`).

The combined PDF has consecutive logical page labels 1–24. Every gallery page has its matching visible footer 13–24, the matching A1–A12 title, one composite figure, one scientific caption, and a discrete editorial substitution note. All twelve gallery pages measure 595.276 × 841.89 points (A4).

| Final page | Figure | Individual visual check |
|---:|---|---|
| 13 | A1, isolated signals | Four signal columns and two class rows are clear; spiral shapes retain their aspect ratio; labels and caption fit. |
| 14 | A2, RGB semantics | Grayscale channels and RGB composite align; low-contrast channel values remain visible; no clipping or geometric stretching. |
| 15 | A3, static task atlas | All sixteen task/class panels are present; task labels are separated; letter and sentence shapes remain undistorted. |
| 16 | A4, RGB task atlas | Same complete inventory and geometry as A3; coloured segments, labels and caption remain legible. |
| 17 | A5, encoding heatmap | All nine encoding rows, eight task columns, cell values and colour scale are readable; no legend/caption collision. |
| 18 | A6, encoding profiles | Four-series legend, markers, task IDs and axis labels fit; complete line profiles are visible. |
| 19 | A7, classifier contrasts | Three estimates, interval endpoints, zero reference and Holm annotations are readable; negative signs render correctly. |
| 20 | A8, fusion rules | Long rule names and all interval endpoints fit; the metric axis and caption are unobstructed. |
| 21 | A9, fusion ROC | Both axes span 0–1; all four curves, reference diagonal, legend names and AUC values are visible. |
| 22 | A10, confusion matrices | Four matrices are complete; observed/predicted axes, percentages and fractional mean counts are readable and correctly placed. |
| 23 | A11, channel sensitivity | Eight task rows and three channel columns are complete; signed cell values and the symmetric colour scale are legible. |
| 24 | A12, spatial agreement | Both histograms, median annotations, valid-pair counts, axes and explanatory caption are visible; no clipped elements. |

Typography and rendering checks: the English captions show no missing glyphs, replacement boxes or malformed ligatures; mathematical minus signs, percentages, apostrophes and dashes render correctly. These gallery pages do not contain accent-bearing author names, so no claim about bibliography diacritics is implied by this check. Figure aspect ratios are preserved throughout. The generous blank space on shorter figures follows the requested one-figure-per-page selection layout.

The captions preserve the necessary scope distinctions: descriptive task cells; conditional participant-bootstrap intervals; per-repetition ROC and confusion calculations; uncalibrated vote scores; image-level channel perturbations; and nonconstant-map denominators in the agreement analysis. No alternative figure repeats a main-body graphic in this gallery.
