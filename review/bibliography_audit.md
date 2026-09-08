# Bibliography and venue evidence audit

Audit date: 2026-09-07. Sources: the previous `D:/doutorado/VISAPPMunif/main.tex` and `refs.bib`, the supplied Casademunt thesis, primary publisher pages, author-deposited papers, and the thesis author's source repository. Percentages below are published values unless explicitly identified as current results. This memo is editorial evidence; it is not an additional model experiment.

## 1. The reference supervised by Ángel Sánchez Calle

Alberto Casademunt González, *Análisis automático de alteraciones en la escritura manuscrita debidas al párkinson*, undergraduate thesis, academic year 2022/2023. The cover names Casademunt as author and Ángel Sánchez Calle as supervisor. It should be cited as Casademunt (2023), not as a paper authored by Ángel. No DOI was found. It is retained because the user explicitly identified this work as the required comparator; no identifier has been invented.

Primary supplied file: `D:/doutorado/PseudodynamicHandwritingMinimapSignatures/TFG_AlbertoCasademunt (1).pdf`, 80 PDF pages. The following page numbers distinguish PDF position from printed numbering:

| Evidence | PDF page | Printed page | Finding |
|---|---:|---:|---|
| Cover | 1 | unnumbered | Author, supervisor, academic year |
| Table 5.1 | 43 | 29 | 62 training participants and 13 test participants |
| Metric definitions | 47–48 | 33–34 | Positive class is PD; F1 is positive-class F1 |
| Section 6.1 | 48 | 34 | Crop sizes 71, 91, 111, 131, 151 are evaluated using the group called test |
| Appendix II | 73–74 | 59–60 | Public code URL; seed 17, 30 epochs; crop size 111 recommended |
| Table III.6 | 79 | 65 | Participant/task outcomes below, visually checked |

| PaHaW task | Published accuracy | Published PD-class F1 |
|---|---:|---:|
| 2, repeated letter l | 69.23% | 66.67% |
| 3, repeated bigram le | 61.54% | 61.54% |
| 4, repeated word les | 69.23% | 66.67% |

The code is [vibrantsalt/TFG](https://github.com/vibrantsalt/TFG), audited at commit `c721578e7e5acd8a9e0eda11aeca62a79902a5f8`. In [PD_Prediction.ipynb](https://github.com/vibrantsalt/TFG/blob/c721578e7e5acd8a9e0eda11aeca62a79902a5f8/PD_Prediction.ipynb), zero-based code cell 11 shuffles healthy and PD participant ID lists separately with seed 17, reserves seven healthy and six PD participants for testing, and assigns the remaining IDs to training before generating crops. Cell 21 calls `f1_score(..., pos_label='PD')`. Thus the problem is not an observed crop-level train/test mixing: it is a single small holdout whose outcomes were consulted during crop-size exploration. Exact split IDs were not reconstructed, and no model was rerun. The [CNN source](https://github.com/vibrantsalt/TFG/blob/c721578e7e5acd8a9e0eda11aeca62a79902a5f8/nn.py) uses grayscale crops and a two-convolution network followed by fully connected layers; this differs substantially from the full-task RGB CNN in the current study.

Safe comparison: report the three task accuracies with their evaluation design. If F1 appears, label it `F1 (PD)` explicitly. Do not compare these F1 values to macro-F1, declare superiority, or imply the same held-out subjects were used.

## 2. Drotár et al. (2016)

DOI verified: [10.1016/j.artmed.2016.01.004](https://doi.org/10.1016/j.artmed.2016.01.004); [PubMed primary index](https://pubmed.ncbi.nlm.nih.gov/26874552/). Full author-deposited text: [arXiv PDF](https://arxiv.org/pdf/2411.03044), deposited later than publication.

Section 3.3 (PDF p.12) describes stratified 10-fold CV repeated ten times, with averages across runs. Features pass a Mann–Whitney screen at p<0.05; the text does not document whether this screening is repeated inside each training split. Avoid claiming either confirmed nesting or proven leakage. The SVM uses an RBF kernel with grid search. Tables 4–5 (PDF p.14) report 81.3% accuracy for combined kinematic and pressure features from tasks 2–8, and 82.5% for pressure alone. The spiral is excluded from the combined endpoint; the paper reports 69 spiral records. The combined-feature task accuracies for tasks 2/3/4 are 72.3/71.0/66.4%. These are accuracy, not balanced accuracy or macro-F1. Our 75-person, eight-task fusion is a different endpoint. Retain `Drotar2016` from the original bibliography. Do not reuse the previous manuscript's spiral superiority statement.

## 3. Diaz et al. (2019)

DOI verified at [publisher](https://www.sciencedirect.com/science/article/abs/pii/S0167865518307013): [10.1016/j.patrec.2019.08.018](https://doi.org/10.1016/j.patrec.2019.08.018). Full author-deposited text: [arXiv PDF](https://arxiv.org/pdf/2405.13438).

The method uses sampled points and in-air movements, pretrained CNN features, additional image representations, and classifier ensembles. Sections 3.3–3.5 (PDF pp.5–6) describe stratified 10-fold CV, explicitly training-only feature selection, and an ensemble of the five tasks with highest measured accuracy. They do not explicitly establish training-only task selection. Table 4 (PDF p.9, visually checked) reports accuracy 86.67%, AUC 83.33%, sensitivity 89.17%, specificity 80.83% for the enhanced ensemble; static accuracy is 72.50%, dynamic 81.67%, and temporal enhancement alone 81.25%. These figures contextualize a different representation and task-selection procedure. Retain `Diaz2019`. Do not equate its transfer-learning method with our small CNN trained from scratch or characterize its 86.67% as a single-task result. Table 6 distinguishes Drotár 2016's 81.30% from another study's 88.13%; do not attribute the latter to `Drotar2016`.

## 4. Same-metric current values for contextual tables

Values read directly from `experiments/2026-09-restart/runs/cnn-v1/metrics/combined_task_summary.csv` and `fused_majority_summary.csv`. These are averages across the five repetitions; they must not be represented as an independent test cohort or a reproduction of the historical protocols.

| Current SAZ endpoint | Accuracy | PD-class F1 | Macro-F1 |
|---|---:|---:|---:|
| SVM, task 2 | 51.20% | 46.42% | 50.63% |
| SVM, task 3 | 58.13% | 58.50% | 58.02% |
| SVM, task 4 | 64.00% | 62.09% | 63.84% |
| SVM, eight-task majority fusion | 65.07% | 66.97% | 64.88% |
| Logistic regression, eight-task majority fusion | 65.07% | 66.67% | 64.98% |

Use an explicit protocol column alongside any literature accuracy column. Present the gap candidly. Different splits, modalities, feature selection, and task aggregation prevent a controlled cross-paper ranking. Within our campaign, SAZ and the downstream models were chosen after development on PaHaW: inner-fold model selection does not turn representation selection into an untouched external validation.

## 5. New references with confirmed DOI

Only three methodological references are proposed, plus the explicitly requested thesis. BibTeX is in `new_verified_refs.bib`.

| Proposed key | Reason | Verified DOI and primary record |
|---|---|---|
| Ojansivu2008 | Credit the LPQ descriptor | [10.1007/978-3-540-69905-7_27](https://link.springer.com/chapter/10.1007/978-3-540-69905-7_27) |
| Breiman2001 | Credit Random Forest | [10.1023/A:1010933404324](https://link.springer.com/article/10.1023/A:1010933404324) |
| Selvaraju2017 | Credit Grad-CAM | [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74), resolves to IEEE document 8237336; metadata also verified with [Crossref](https://api.crossref.org/works/10.1109/ICCV.2017.74) |

For LPQ, describe the exact implementation actually run. The original paper includes decorrelation; a citation alone must not imply that every implementation step or invariance guarantee was reproduced. Random-weight comparison can be specified as the diagnostic actually performed, without adding an unverified DOI. Adebayo et al., *Sanity Checks for Saliency Maps*, was located in the [official NeurIPS proceedings](https://proceedings.nips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf); no proceedings DOI was verified, so no new BibTeX entry is proposed under the user's DOI rule.

Retain previous references only where the new argument needs them. `Drotar2016`, `Diaz2019`, `Impedovo2019`, `Cortes1995`, `LeCun1998`, and the relevant visual-signature precedent are directly useful. ResNet-18 should not appear as the current CNN. Do not retain a lengthy generic history of signal-to-image methods simply to preserve the former citation count. The `Gebara2026` entry is marked “To appear” in the old bibliography; its publication status was not independently resolved in this audit, so omitting it is preferable if the argument does not require it.

## 6. Published VISAPP papers used as quality comparators

These articles were read for concrete editorial and experimental practices. They are not PD benchmarks and need not be cited in the manuscript. Their presence in proceedings is evidence of venue fit, not a guarantee that every statement or reporting choice is sound.

### Figueroa et al., VISAPP 2024, pp.757–764

*A Learning Paradigm for Interpretable Gradients*, [publisher record](https://www.scitepress.org/PublishedPapers/2024/124668/), DOI [10.5220/0012466800003660](https://doi.org/10.5220/0012466800003660), [full paper](https://www.scitepress.org/Papers/2024/124668/124668.pdf).

Sections 4–5 specify the method mathematically and with an algorithm; the experiment separates predictive performance from attribution quality. Section 5.2 defines confidence-based metrics; Section 5.3 defines insertion/deletion; Figures 2–3 compare methods on the same inputs. Adopt: a compact encoding definition, matched visual comparisons, and quantitative XAI tests next to examples. Our review criterion: every heatmap must state its target and explain what the independent perturbation check supports. Do not describe visual plausibility alone as fidelity.

### Karatsiolis and Kamilaris, VISAPP 2023, pp.39–46

*A Model-agnostic Approach for Generating Saliency Maps to Explain Inferred Decisions of Deep Learning Models*, [publisher record](https://www.scitepress.org/PublishedPapers/2023/116124/), DOI [10.5220/0011612400003417](https://doi.org/10.5220/0011612400003417), [full paper](https://www.scitepress.org/Papers/2023/116124/116124.pdf).

Section 4 and Figures 3–6 connect side-by-side maps with insertion/deletion evaluations; Table 1 reports model-dependent outcomes and the discussion acknowledges a model where DE-CAM loses. Adopt: report the tested CNN's weaker result openly, show correct and incorrect cases, and retain the disagreement between Grad-CAM and occlusion. The paper also discusses computational cost. Our criterion: acknowledge the exact model and sample scope of every explanatory claim rather than generalizing from one architecture.

### Aires, de Morais and Lin, VISAPP 2025, pp.855–862

*Handwriting Trajectory Recovery of Latin Characters with Deep Learning: A Novel Exploring the Amount of Points per Character and New Evaluation Method*, [full publisher paper](https://www.scitepress.org/Papers/2025/133831/133831.pdf), DOI [10.5220/0013383100003912](https://doi.org/10.5220/0013383100003912), verified in its first-page imprint.

Sections 3.4–3.5 and Figures 3–7 illustrate cases where a geometric reconstruction and the metric disagree, before introducing the revised evaluation. Tables 1–4 separate model configurations and evaluation definitions. Adopt: explain why task-mean macro-F1 and participant fusion answer different questions, using explicit labels. Our criterion: a change of metric or aggregation must never be described as a model improvement. Unlike an unrestricted gallery, each body figure should answer the question raised by its surrounding paragraph; alternative designs belong in the requested selection gallery.

## 7. Reviewer checklist for the first rewrite

1. The title and abstract should name the representation and incremental evaluation without claiming a verified dynamic advantage or state of the art.
2. Introduce geometry, one dynamic quantity, RGB combinations, texture classification, model comparison, fusion, then XAI. Define `task` once for the eight acquisition exercises.
3. State which comparisons are exploratory and that patients, not raster images, determine splits and uncertainty.
4. Every table caption should state the metric, aggregation, and evaluation population. Every figure should have a specific evidential purpose and readable legends at the intended column width.
5. Separate tested low-performing CNN from claims about CNNs generally; explain the resolution/architecture distinction from cited pretrained methods.
6. Replace unqualified claims such as “preserves dynamic information,” “clinically meaningful areas,” or “velocity is best” with the measurable property and its uncertainty.
7. Keep the main manuscript within the venue limit. Pages 13 onward are an editorial selection gallery and must be excluded from the submission PDF.
8. Keep an editorial note about the venue's AI-disclosure placement versus anonymous acknowledgements rule. Resolve the final placement consistently with the submission guidance; do not silently omit a required disclosure.
