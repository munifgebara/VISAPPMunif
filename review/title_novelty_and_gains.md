# Title, proposed formulation and observed encoding gains

Date: 2026-09-08. The author selected the title **Encoding Pen Dynamics in
Handwriting Images: An Incremental Study of Parkinson's Disease Classification**,
requested emphasis on the proposed technique, and asked whether the colour
encoding had actually improved the results.

## Observed gains and statistical interpretation

The source CSVs confirm numerical improvements. SAZ improves four tasks (T1, T6,
T7 and T8) and decreases four (T2–T5). The revised opening reports both directions
and distinguishes these observed results from a general superiority claim.

| Endpoint | Static macro F1 | SAZ macro F1 | Difference |
|---|---:|---:|---:|
| Mean of eight tasks | 0.564286 | 0.575153 | +0.010867 |
| T1, spiral | 0.591456 | 0.671641 | +0.080185 |
| T6, porovnat | 0.547122 | 0.599537 | +0.052415 |
| T7, nepopadnout | 0.510519 | 0.556059 | +0.045540 |
| T2, repeated l | 0.567505 | 0.506308 | −0.061197 |

The global difference has a 95% participant-bootstrap interval of
[−0.014605, +0.036992], unadjusted p=0.409559 and Holm p=1.000000 across 36
encoding pairs. The T1 contrast has interval [+0.010517, +0.153413], unadjusted
p=0.030597 and Holm p=0.948505 in the 32-comparison triplet-versus-static family.
Its unadjusted evidence must not be reported as surviving that declared correction.
The largest observed spiral gain across all triplets belongs to SPA, not SAZ;
the abstract explicitly identifies the SAZ signal combination rather than claiming
it is the best spiral encoding.

Sources, under `experiments/2026-09-restart/runs/`:

- `static-svm-v1/metrics/task_summary.csv`
- `dynamic-triplet-svm-v1/metrics/task_summary.csv`
- `dynamic-triplet-svm-v1/metrics/paired_contrasts_vs_static.csv`
- `encoding-comparison-v1/metrics/all_encoding_ranking.csv`
- `encoding-comparison-v1/metrics/all_encoding_global_pairwise.csv`

No model was refitted, comparison family changed, task dropped or new favourable
endpoint selected for this revision. The statement is descriptive and retains
uncertainty about overall superiority. Lack of statistical support is not proof
that the observed gains are absent or that the encodings are equivalent.

## Scope of the methodological contribution

The abstract now introduces a **new formulation** of pseudodynamic handwriting
minimap signatures. The introduction specifies its construction: contact-only
trajectory rendering, within-recording percentile scaling, circular azimuth centring,
and an explicit family of nine encodings tested under common participant partitions
across all eight tasks. Novelty is attributed to this formulation and investigation,
not to the broad idea of converting dynamics to images or to exclusive priority
for each component operation.

The relevant RGB precedent is Cilia et al., *From Online Handwriting to Synthetic
Images for Alzheimer's Disease Detection Using a Deep Transfer Learning Approach*,
IEEE Journal of Biomedical and Health Informatics 25(12), 4243–4254, 2021.
Its DOI **10.1109/JBHI.2021.3101982** and metadata were checked against the
[publisher-deposited Crossref record](https://api.crossref.org/works/10.1109/JBHI.2021.3101982).
The [authors' institutional abstract](https://iris.unicas.it/handle/11580/88343)
describes RGB synthetic handwriting images and their comparison with binary images
using deep learning. The full IEEE article was not retrieved in this check; the
manuscript's short description stays within what the abstract establishes.
This reference has been added beside Diaz et al. in the introduction.

A further overlapping [public implementation](https://github.com/firekern/parkinson-handwriting-dl)
was located, using individual signals and RGB triplets on the PaHaW spiral. No DOI
or verified publication chronology was established for it, and no manuscript
citation or performance comparison was added. The search was targeted, not an
exhaustive novelty review; it cannot support a claim to be the first RGB handwriting
method or the first use of each individual preprocessing step.

## Document verification

The abstract contains 184 words. The main manuscript remains 14 pages with eight
figures, three tables and thirteen cited references; the selection edition has
26 pages, with alternatives A1–A12 on pages 15–26. The activity diagram remains
Figure 1 on page 2. Build checks found no unresolved references, missing characters,
oversized floats or overfull boxes. The final visual review is recorded in the
build verification manifest after inspecting the rendered pages.

Visual result: passed. Pages 1 and 3 were inspected by the main reviewer, 4–8 by
the novelty reviewer, and 9–14 by the results reviewer, with every page opened
individually. There are no clipping, overlap or unresolved legibility issues.
PNG SHA-256 comparisons confirm that page 2 and all twelve gallery pages are
pixel-identical to the previously inspected version documented in
`opening_revision.md`. The new title fits on two lines, page 1 ends with a complete
paragraph, and page 3 resumes with the complete contribution paragraph.
