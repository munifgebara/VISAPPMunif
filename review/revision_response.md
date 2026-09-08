# Response to the scientific review of manuscript version 1

This follow-up reviews the revised sources in `paper/main.tex`, `paper/sections/*.tex`, and `paper/refs.bib` against comments R1–R11 in `reviewer_report_v1.md`. It is a closure review of the edits, not a new experiment or a second numerical campaign. No TeX, data, or model code was changed during this review. The final compilation and visual inspection after the latest text edits remain the responsibility of the main editing pass.

**Scientific assessment:** the substantive review comments are addressed in the current sources. The principal estimates and their interpretation remain consistent with the audited experiment. No new scientific blocker or material ambiguity was introduced by the condensation. The former layout issues have corresponding source changes and an existing 12-page build, but final pagination and figure readability must be verified after the last rebuild.

## R1–R11 closure

| Comment | Status | Revision verified in the current source |
|---|---|---|
| R1 — 15 pages and stranded XAI figures | Source changes addressed; final rendering confirmation pending | `06_xai.tex:7` reduces the example panel to 0.72 text width. Both XAI figures now have explicit calls at lines 23 and 37. All seven selected figures remain. The existing build log reports 12 pages, and `visual_qa.md` records the two XAI figures on pages 10 and 11, before the final references. Those checks concern an earlier compiled state; the latest source still needs its final build and page review. |
| R2 — Duplicated ranking tables | Resolved | The nine-row encoding ranking and the four-model endpoint table were removed. `04_representation_results.tex:28` points directly to the encoding figure. `05_classifiers_fusion.tex:62` gives the principal fused estimates and points to the fusion figure. The three remaining tables have separate functions: channel assignments, task inventory, and protocol-aware literature comparison. |
| R3 — Casademunt's compound surname | Resolved in bibliography source; final citation rendering to confirm | `refs.bib`, entry `Casademunt2023`, now uses `author = {Casademunt Gonz{\'a}lez, Alberto}`. The literature table names “Casademunt González (2023).” This preserves the two-word family name and separates authorship from supervision. |
| R4 — Exploratory SVM-selected encoding in the abstract | Resolved | The abstract explicitly says “In an exploratory comparison on this SVM-selected encoding” and identifies task-mean macro F1 as the endpoint for the classical-model/CNN comparison. The Introduction and `03_protocol.tex:46` retain the lack of an independent test of representation selection. |
| R5 — Diaz 81.25% condition label | Resolved by omission | `07_discussion.tex:37` no longer includes or renames the auxiliary 81.25% condition. It retains the audited 86.67% enhanced-ensemble and 72.50% static-counterpart accuracies, together with pretrained features, selected tasks, and protocol differences. |
| R6 — Binary Grad-CAM target | Resolved | `06_xai.tex:19` defines a single model output logit `g`, with `g_c=g` for PD and `g_c=-g` for H. This matches the actual binary Grad-CAM implementation and removes the apparent assumption of two independent output logits. |
| R7 — Participant coverage of random-weight checks | Resolved | `06_xai.tex:43` states that the 200 checks cover 11 distinct participants because the deterministic rule repeatedly chooses the first identifier in each test partition. It retains the 160 valid nonconstant pairs and the conditional interpretation of low agreement. |
| R8 — Exact LPQ implementation | Resolved | `04_representation_results.tex:8` specifies four complex 5×5 kernels with frequency offsets (1,0), (0,1), (1,1), and (1,−1), scaled by 1/5; reflected boundaries; eight-bit codes; normalized square-root histograms; 768 concatenated attributes; and no decorrelation or spatial pyramid. The description does not imply that the complete canonical LPQ preprocessing was reproduced merely because the original method is cited. |
| R9 — Tablet pressure versus calibrated force | Resolved | `02_encoding.tex:30` now calls pressure “the tablet's contact-pressure signal.” The native-unit speed computation and per-record normalization are preserved, without claiming physical force calibration. |
| R10 — BCE, calibration, and randomization details | Resolved | `05_classifiers_fusion.tex:16` specifies positive BCE weight as the H/PD training-count ratio. Line 49 describes sigmoid calibration through the three inner participant folds with ensemble averaging. `03_protocol.tex:61` calls the randomization draws 10,000 participant-wise swap patterns. These statements match the implemented procedures. |
| R11 — Literature scope and table overflow | Scientific/source issue resolved; final table rendering to confirm | `07_discussion.tex:26` includes the thesis year; lines 29–30 label current rows as T1–T8, 5×5 participant CV, majority fusion. All accuracy values remain correctly identified. The revised fixed-width column specification at line 22 addresses the former overflow. `visual_qa.md` reports the compiled table fitting on page 12, but the final build should confirm the last source state. |

## Specific checks requested for this pass

**LPQ:** the revised method remains the executed fixed descriptor. It crops and pads the foreground, resizes to 64×64, forms Fourier-sign codes, and builds square-root histograms by channel. Static RGB duplication is still disclosed. Adding the frequency directions and explicitly removing decorrelation introduces no claim about learned filters or the original LPQ paper's full invariance guarantees.

**CNN loss:** “H/PD training-count ratio” correctly identifies negative/positive counts. The surrounding procedure distinguishes epoch selection on the first inner holdout from the reinitialized final fit to all outer-training participants. This gives the reader the appropriate training set for each weighting calculation; it does not imply use of outer-test labels.

**Probability calibration:** the three-fold sigmoid/ensemble wording is faithful to the SVM fusion code. The manuscript still reports the originally declared calibrated-mean endpoint before introducing majority voting as the common descriptive endpoint. It does not retrospectively present the best observed majority rule as the initial primary rule.

**Fusion antecedent:** the sentence at `05_classifiers_fusion.tex:62` reads “logistic regression at 0.6498 and SVM at 0.6488 after fusion, both with mean accuracy 0.6507.” “Both” now unambiguously refers to logistic regression and SVM. Random Forest and CNN scores are given in the next sentence. Accuracy 0.6507 is therefore not attributed to all four classifiers. The LR–SVM difference, interval, Holm p-value, and absence of an equivalence test are preserved.

**Abstract:** the algorithm claim is now limited to task-mean macro F1, the tested small CNN trained from scratch, and the exploratory SVM-selected encoding. It cannot reasonably be read as claiming significant superiority for the fused participant endpoint. The final sentence reports the uncertain colour gain directly instead of evaluating the authors' rhetorical restraint.

**Grad-CAM:** the explicit ±logit target, corresponding outer checkpoint, 2,985/597 prediction scopes, zero-map handling, 540 nonconstant occlusion pairs, and 160 nonconstant trained/random pairs remain clear. The whole-channel perturbation is still distinguished from isolated physiological importance. No clinical localization claim was added.

## Editorial and scientific coherence after condensation

The revised heading “From Texture to Learned Features” correctly describes the progression without implying that logistic regression is more complex than an RBF SVM. The Introduction is shorter and keeps the representation problem, terminology, closest precedent, and exploratory contribution. The erroneous attribution of a single nominal sampling frequency has been removed; speed explicitly uses recorded timestamps. The discussion now states that XAI does not explain the cause of CNN underperformance, and the unbounded transfer-learning citation has been replaced by the directly relevant Diaz precedent.

The paper still advances from static geometry to one dynamic signal, RGB combinations, fixed texture classification, pixel learning, participant fusion, and explanations. Removing the duplicated tables improves the visual argument without hiding an unfavourable estimate. The source still contains some optional repetitions of limits, but none obstructs interpretation enough to justify reopening the settled revision. If the final page limit holds, further condensation is a matter of author preference rather than a scientific requirement.

## Published-VISAPP quality criteria

The following criteria were taken from the three papers documented in Section 6 of `bibliography_audit.md`; they are editorial benchmarks, not acceptance predictions.

1. **Figueroa et al. (2024), _A Learning Paradigm for Interpretable Gradients_: separate predictive performance from attribution evidence, define the explanation target, and place quantitative perturbation checks alongside matched examples.** The revised manuscript meets this in its explicit ±logit Grad-CAM definition, checkpoint-linked maps, deletion estimates, and occlusion agreement. It preserves the unmatched-foreground limitation, so the larger deletion effect is not promoted to full attribution faithfulness. Final figure placement/readability must be confirmed in the build.

2. **Karatsiolis and Kamilaris (2023), _A Model-agnostic Approach for Generating Saliency Maps to Explain Inferred Decisions of Deep Learning Models_: report model-dependent outcomes candidly and connect explanatory examples to measurable perturbations.** The revised manuscript keeps the tested CNN's lower score, correct/error cases from both classes, the all-zero Grad-CAM failure, and channel/occlusion results. The added 11-participant coverage makes the random-weight check more accurately bounded.

3. **Aires et al. (2025), _Handwriting Trajectory Recovery of Latin Characters with Deep Learning_: make the evaluation definition explicit when different forms of reconstruction or aggregation change the score.** The revised manuscript retains repeat-wise task macro-F1, participant majority voting, the tie rule, and the distinction between literature accuracy and positive-class F1. The current rows of the literature table now identify their participant-CV/fusion scope. Figures and prose have complementary roles after removal of the duplicate tables.

## Remaining closure items

Only final production checks remain from this review: rebuild the latest sources and bibliography; verify that the main PDF still contains at most 12 pages with all seven body figures; confirm the corrected compound surname in printed citations; check the literature table for overflow and the compact XAI annotation sizes; and ensure that the alternative gallery begins at physical page 13 in the selection edition. The main editor is already handling that final build. This follow-up has not independently certified an unbuilt PDF.

No new model experiment was needed to address these review comments. The findings above record source-level resolution and outstanding production verification; they do not imply that the authors have approved the text, figures, or submission.

## Production closure after the final build

The final build and individual page inspection subsequently resolved the production
items above: 12 main pages, all seven body figures, three tables and twelve cited
references, with no unresolved-reference or overflow warnings. The selection copy
has 24 pages and its alternatives start on physical page 13. Corrected XAI labels,
the Casademunt reference, and the final literature table are legible. See
`final_visual_qa_main.md`, `final_visual_qa_gallery.md`, and `build_verification.json`
for the completed checks. No scientific revision remains open from this review.
