# VISAPP 2027 draft — author checklist

Complete draft of `main.tex` with **real experimental results**. Compiles clean
(`pdflatex → bibtex → pdflatex ×2`), **12 pages**, no undefined references or citations, no
overfull table.

> See also `../PROJECT_LOG.md` — the cross-session handoff record, including how to give an
> agent GitHub write access.

## 0. Blocking items

### 0.1 Nothing was committed — the draft must be pushed manually

The GitHub connector available while drafting is **read-only**: it can read
`munifgebara/VISAPPMunif` but `PUT /contents` and `POST /git/trees` both return
`403 Resource not accessible by integration`. No commit was made.

All deliverables are in the project folder under `visapp-draft/`:

```
visapp-draft/
├── main.tex
├── refs.bib
├── NOTES.md
├── main.pdf                          (compiled proof, 12 pages)
└── figures/
    ├── ablation-2x2.png
    ├── viewer-animated.png
    └── viewer-side-by-side.png
```

Copy `main.tex`, `refs.bib` and the whole `figures/` directory into the repository root and
commit. `main.pdf` is a build artifact. The two screenshots have already been processed: the
header nav row containing the "Project page" link was whited out for double-blind.

### 0.2 One `[NEEDCITE]` remains

Section 5.6, the clipping-based CNN baseline (Alberto Casademunt González, supervised by Ángel
Sánchez Calle, ETS Ingeniería Informática, 2022/2023, plus the public repository). A citable
form must be decided — `@mastersthesis` with a stable URL, or an in-text statement that it is
unpublished work reproduced with permission plus a software citation.

**Double-blind conflict:** the thesis supervisor is a co-author of this paper. Naming the thesis
in the anonymous submission reveals authorship. The draft body does not name it; only this file
does. Restore the full citation for the camera-ready.

## 1. Length limits (verified 27 Jul 2026)

VISAPP constrains **submissions by character count**, not page count:

| Type | Submission | Publication page limit |
|---|---|---|
| Regular Paper | 10,000–50,000 characters excluding whitespace, incl. references, tables, appendices | 12 pages if classified Full, 8 if Short (+4 extra for a fee) |

Current draft: **39,818 characters excluding whitespace, 12 pages**. Inside the character
window, but **exactly at the 12-page Full limit with zero slack** — and `algorithm2e` renders
differently on Overleaf than in the local verification stub, so the count may shift by a line.
Check the page count on Overleaf before submitting. If it must shrink, the natural cuts are
Section 5.1 (the eight-task screening table), Section 5.5 (channel attribution) and part of
Section 2. A Short Paper classification would require losing four pages.

Source: <https://visapp.scitevents.org/Guidelines.aspx>

## 2. Where every number in the paper comes from

All figures were read from `C:\Users\munif\PycharmProjects\PseudodynamicHandwritingMinimapSignatures\`.
Nothing was estimated or invented.

| Paper location | Source file |
|---|---|
| 597 minimaps, 512×512, v2.0, 0 failed | `outputs/manifests/minimap_v2/image_manifests.json` |
| 72 images for task 1, 75 for tasks 2–8 | directory counts in `dataset/minimaps_v2/task_*/` |
| Validation passed, widths 1–8, 0 broken | `outputs/metrics/minimap_v2/validation_summary.json`, `outputs/reports/minimap_v2/validation_report.md` |
| CPU/CUDA: 80 files, RGB diff 0, speed diff 2.22e-16 | `outputs/metrics/minimap_v2/cpu_cuda_equivalence.csv` |
| Table 2 (eight-task screening) + ensemble 0.6925 | `outputs/reports/minimap_v2_baseline/ieee_minimap_v2_baseline_report.md` |
| Table 3 (fixed descriptor, tasks 1/3/7) | `outputs/reports/minimap_v2_best_ensemble/ieee_minimap_v2_best_ensemble_report.md` |
| Ensemble vs. task paired tests | `outputs/metrics/minimap_v2_best_ensemble/statistical_comparison.json` |
| Tables 4 and 5 (factorial ablation) | `outputs/reports/minimap_v2_color_width_ablation/ieee_color_width_ablation_report.md` |
| ResNet-18 results + augmentation audit | `outputs/reports/minimap_v2_resnet18_safe_aug/ieee_resnet18_safe_augmentation_report.md` |
| CNN vs. handcrafted paired test | `outputs/reports/minimap_v2_resnet18_safe_aug/comparison_with_handcrafted_baseline.md` |
| Channel attribution (R .352 / G .406 / B .243) | `outputs/reports/minimap_v2_resnet18_xai/semantic_channel_analysis.md` |
| Protocol, seed 42, split checksum | `outputs/manifests/minimap_v2_best_ensemble/run_manifest.json` |

## 3. Headline results, for quick reference

- **Factorial ablation is the paper's main claim.** `MONO_FIXED` 0.5597 → `RGB_SPEED_V2` 0.7321
  macro F1; paired Δ = 0.1725, 95% CI [0.0589, 0.2868], Holm-adjusted *p* = 0.031. **Supported.**
- No individual main effect survives Holm correction. Colour is worth +0.040 at fixed width and
  +0.092 at speed width; width is worth +0.080 in monochrome and +0.133 with colour. The paper
  reads this as a positive interaction but explicitly declines to claim it.
- Best pipeline: RGB LPQ + visual geometry/width (812 features) + logistic regression,
  reliability-weighted over tasks 1/3/7 → 0.7321 macro F1, balanced accuracy 0.7326, MCC 0.4685,
  ROC AUC 0.7575.
- The ensemble does **not** beat task 1 alone (Δ = 0.0012, Holm *p* = 1.0). The paper says so.
- **Negative result:** ResNet-18 is significantly worse than the texture pipeline
  (Δ = −0.1720, Holm *p* = 0.040). Reported as a finding about sample size and about the
  augmentation constraint that semantic channels impose, not as a flaw in the representation.

## 3b. The Drotar comparison — what was verified in the source

The full text of Drotar et al. (2016) was read (arXiv 2411.03044, mirror of Artif. Intell. Med.
67:39-46). Four facts anchor the Discussion paragraph; all are quoted or paraphrased from that
paper, none inferred:

1. **81.3% accuracy, 87.4% sensitivity, 80.9% specificity**, SVM, merged tasks 2-8.
2. **No demographic or clinical covariate is used as a model input.** Sex, age, disease duration,
   UPDRS-V and levodopa equivalent dose appear only in Table 1 and Appendix A as cohort
   description, plus a balance check ("No significant differences related to gender or age were
   found between the PD and healthy control groups"). The twenty most relevant features in their
   Table 3 are all handwriting-derived. The comparison is fair on this axis.
3. **Feature selection was done on the full labelled corpus.** Section 3.3: "From all computed
   features we kept only those that passed the Mann-Whitney U test, i.e. those that showed a
   statistically significant (p < 0.05) difference between the PD and control groups." It is
   described before the cross-validation results, with no indication of being recomputed per
   fold. Validation was "stratified 10-fold cross-validation", repeated 10 times.
4. **Task 1 was excluded from their fused model.** "We did not find any statistically significant
   kinematic features for the tasks 1 and 4"; "Task 1 contained data from only 69 subjects and
   did not show any significant discrimination potential, therefore we did not include this
   task." Their own explanation: "we did not utilise any spiral specific features."

Point 4 is the strongest argument in the paper's favour and is now stated in the Discussion: the
spiral, which their descriptor set could not exploit, is our best single task (0.7336 macro F1,
0.7855 ROC AUC).

Point 3 is phrased carefully in the paper — "not, as described, recomputed inside each
cross-validation fold" and "We raise this to calibrate the gap, not to dispute the finding".
**Do not sharpen this into an accusation.** It was ordinary practice in 2016, and the wording as
it stands is defensible against a reviewer who knows the paper.

## 4. Caveats the draft states explicitly — verify you are comfortable with each

1. **Preselection bias.** Tasks 1, 3 and 7 were chosen in exploratory analysis on this same
   corpus. Section 6 says so and points to the unbiased eight-task screening as the conservative
   estimate. Do not remove this.
2. **`screening_reproduction: outside_descriptive_tolerance`.** The fixed configuration did not
   exactly reproduce the exploratory reference metrics. The draft reports this as an unresolved
   discrepancy. **If you can explain or fix it before submission, do — a reviewer who reads the
   artifacts will ask.**
3. Task 7 is at chance (MCC −0.0398) and is reported as such.
4. Per-task scaling profiles are computed over all files of a task, not training folds only.
   Label-agnostic, so no supervision leakage, but weakly transductive. Flagged as future work.

## 5. Author block `TODO:` markers

In the `\else` branch of the anonymization switch:

- affiliation, city, country for authors 1 and 3 (Munif Gebara Junior, Yandre) and author 2
  (Ángel Sánchez Calle)
- e-mail addresses; ORCIDs if `\orcidAuthor{}` is used as in the template
- ~~the third author's full name~~ — **resolved**: Yandre Maldonado e Gomes da Costa. Note that
  `refs.bib` deliberately keeps the abbreviated published byline *Yandre M. G. Costa* for the two
  self-citations, because that is the form printed on those papers; expanding it would also break
  BibTeX name parsing, since it would read the compound surname as `Costa` alone.
- Acknowledgements bullets: funding, contributions, repository URL, AI-tools declaration.

## 6. Double-blind checklist

- `\anonymoustrue` at the top of `main.tex`. Set `\anonymousfalse` for the camera-ready.
- No author name, affiliation, e-mail or ORCID in the compiled anonymous PDF — verified by text
  extraction.
- Acknowledgements suppressed in the anonymous branch.
- Viewer screenshots: nav row whited out. **Re-verify visually after adding them.**
- The two minimap self-citations are third person ("Gebara Jr. et al. ..."), never "our previous
  work". Standard practice, but citing two of the authors' own works as the lineage makes
  authorship guessable. Unavoidable — the lineage is the novelty argument.
- **VISAPP requires AI-generated text to be disclosed in the acknowledgements, and also requires
  the acknowledgements removed for double-blind review.** These conflict. Consider asking the
  secretariat.

## 7. Self-citation budget

12 references, 2 self-citations = **16.7%**, under the 20% SCITEPRESS ceiling. Recheck if you
add or remove references.

## 8. References — verification status

Verified against publisher or primary sources during drafting: `Drotar2016`, `Diaz2019`,
`Moetesum2019`, `Pereira2016`, `Wang2015`, `Eckmann1987`, `Hatami2018`, `Shi2022`, `Cortes1995`,
plus the two supplied self-citations.

**Not independently verified — confirm before submission:**

- `LeCun1998` (Proc. IEEE 86(11):2278–2324) and `He2016` (CVPR 2016, 770–778). Canonical, but
  written from memory rather than checked against a source.
- `Impedovo2019`: sources disagree on the year. IEEE Rev. Biomed. Eng. vol. 12 is a 2019 volume,
  but the paper appeared online in 2018 and some databases list 2018. `refs.bib` uses 2019.
- `Diaz2019` DOI (`10.1016/j.patrec.2019.08.018`) was inferred from journal and volume, not read
  off the publisher page.
- `Gebara2026`: the SBES proceedings series name and edition number were written generically.
  Replace with the exact official title once available.

## 9. Local compilation caveat

`algorithm2e` was not installed in the environment used to verify this draft, so a minimal stub
was substituted **for the local test only**. It is not in the repository, and `main.tex` uses
the real `algorithm2e` exactly as the SCITEPRESS template does. Overleaf provides it.
**Confirm Algorithm 1 renders correctly on Overleaf** — the local render used a simplified
layout, and the page count may shift by a line or two.

## 10. Figure legibility

Both screenshots are UI captures roughly 1490 px wide, placed as `figure*` spanning both
columns — effective resolution around 240 dpi, below the 300 dpi the template asks for. The
smallest interface labels are near the limit of legibility in print. Before the camera-ready,
recapture at a higher device pixel ratio or crop to the informative region.

## 11. The ablation figure — what was changed and why

`figures/ablation-2x2.png` is the supplied infographic with the **"Main findings" box removed**.
That box asserted that stroke width "showed the largest isolated contribution" and that semantic
colour "also added useful information". Neither main effect survives Holm correction, and
Table 5 in the paper says so explicitly. A figure contradicting the paper's own statistics is
exactly what a careful reviewer flags. Everything else was kept: the four variant panels, the
bar chart, the metric table, the 2×2 layout and the "+0.173" conclusion banner all match the
reported numbers to three decimals.

If you prefer to restore those bullets, they must be reworded to descriptive form — e.g.
"stroke width showed the largest descriptive contribution; no single factor reached
significance after correction".

## 12. What the paper still lacks

- **A figure of the minimaps themselves.** The paper describes the encoding and shows the viewer
  and the ablation, but never shows a bare H minimap beside a bare PD minimap with a colour
  legend. That is the single most valuable remaining addition, and
  `outputs/figures/minimap_v2/` already contains generated examples and legends. Add it in
  Section 3.
- Draft markers: `\draftmarkerstrue` renders the remaining `\needcite` in red. Set
  `\draftmarkersfalse` to read without it. The macro must be gone entirely before submission.
