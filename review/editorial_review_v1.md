# Editorial review of manuscript version 1

Reviewed: `paper/main.tex`, all seven section files, bibliography, and the 15-page `paper/build/main.pdf` available at review time. The root author is concurrently condensing the manuscript; locations below use section names and textual anchors rather than assuming stable line numbers. This review recommends changes without editing TeX or generating experiments.

## Recommendation

Revise before submission. The narrative now follows the requested progression: static image, isolated signals, RGB combinations, fixed descriptor, classifier alternatives, task fusion, and CNN explanation. The numbers are presented with substantially more restraint than the earlier manuscript. The main remaining issues are a sampling-rate attribution error, the treatment of a compound surname in BibTeX, repeated qualifications that slow the prose, and delayed XAI floats that make the manuscript exceed its intended length.

## Corrections required

1. **Sampling rate, Participants and Tasks.** The opening sentence attributes 150 Hz to `Drotar2016`. The author-deposited Drotár PDF p.5 says 100 samples per second; the local `info.txt` says 150 Hz. The current speed computation uses actual timestamps, so this nominal discrepancy need not become a new experiment. Prefer: “The local PaHaW documentation describes acquisition with a Wacom Intuos 4M tablet. Speed is computed from recorded timestamp differences.” If retaining 150 Hz, explicitly attribute it to the local documentation and do not make Drotár appear to support that value.

2. **Casademunt's surname.** The compiled bibliography reads “González, A. C. (2023)”. Set the BibTeX author as `Casademunt Gonz{\'a}lez, Alberto` so the compound surname remains intact. The thesis authorship is Alberto Casademunt González, with Ángel Sánchez Calle as supervisor; the current prose correctly describes that relationship. The title, academic year, task outcomes, crop size, and 62/13 division are supported by the supplied thesis.

3. **XAI placement.** In the inspected PDF, body text and references extend through p.13, while Figures 6 and 7 appear alone on pp.14 and 15, after the references. These are intended main-body figures, not the user's alternative gallery. Resolve the float queue before appending the gallery. The four-case XAI panel has height/width 1.093 and occupies much of a page at 0.83 text width. Reduce its physical size only while labels remain readable, or use a compact arrangement that preserves the full image background. Do not crop away the background merely to make a heatmap look more focused. Ensure the two XAI figures appear in or adjacent to their analysis before the Discussion/References boundary.

4. **Complexity claim.** “Increasing Model Complexity” introduces logistic regression after a linear/RBF SVM search. Logistic regression is a simpler reference, not a monotonic increase in model complexity. “From Texture Descriptors to Learned Features” would describe the actual progression. Keep the paragraph explaining why the linear comparator is included.

5. **He2016 attribution.** The last classifier paragraph attaches `He2016` to “Transfer learning”. The ResNet paper is primarily the architecture reference; `Diaz2019` directly supports the relevant pretrained-feature route for this application. Replace with that already-used reference, or write “pretrained residual networks” if the architectural reference is retained. The current CNN is correctly described as trained from scratch and must not be called a fine-tuned ResNet.

## Literature and DOI check

The contextual table's values match the audited sources: Casademunt accuracy 69.23/61.54/69.23% for T2/T3/T4; Drotár 81.30% for tasks T2–T8; Diaz 86.67% for the selected-task ensemble. The current SVM and logistic-regression majority ensembles both have accuracy 65.07%. The table correctly avoids substituting macro-F1 for these accuracies. Its caveat about incomparable protocols should remain. A compact single row for the two current LPQ classifiers could replace two equal-accuracy rows, with the distinct macro-F1 values retained in the classifier/fusion results.

The source DOI strings for the three added methodological references are correct: LPQ `10.1007/978-3-540-69905-7_27`, Random Forest `10.1023/A:1010933404324`, and Grad-CAM `10.1109/ICCV.2017.74`. The thesis has no verified DOI and is the explicitly requested comparator. No further reference is needed for the condensation. The reference to the supervisor in its note is factual, but may be omitted from the printed entry if space is tight; the editorial audit already records it.

The current account of Drotár's screening is appropriately cautious: lack of documented nesting is not reported as proof of leakage. Diaz's explicit training-only feature selection deserves one short clause if the literature paragraph is retained; selecting five tasks still differs from the current all-task fusion. Do not add its static/dynamic intermediate percentages merely to fill a comparison paragraph if they are no longer discussed.

## Concrete route from 15 to 12 pages while retaining seven figures

The two late floats account for two complete pages in the first build, although fitting them into the body still consumes real area. Combine float management with approximately 900–1,200 words of cuts and remove redundant tabular displays. Keep the font, margins, and template geometry unchanged.

| Location | Specific cut or consolidation | Preserve |
|---|---|---|
| Introduction | Remove the final section-by-section roadmap. Compress the generic recurrence/time-series paragraph to the directly relevant Diaz precedent. Merge the two paragraphs about earlier classifiers. | Opening representation problem, meaning of “signature”, incremental contribution, exploratory selection disclosure |
| Encoding | Combine the normalization and channel-mapping equations into one display, with the degenerate-range rule in the adjoining sentence. Move branch-cut and overlap limitations into one compact end paragraph. | Contact-only geometry, per-record normalization, fixed colour range, exact channel assignments, losses of absolute magnitude/size |
| Protocol | State participant-level independence once. Compress the repeated 200-model/2,985-prediction explanation and combine the conditional-inference limitations with the exploratory-selection paragraph. | Five repeats/five outer folds, three inner folds, shared participant IDs, training-only fitting, CNN's different inner selection, metric aggregation |
| Representation results | Remove the nine-row ranking table if Figure 3 retains all encodings and labelled estimates. Delete the second static-baseline paragraph and most task-by-task numerical anecdotes. | Static and SAZ values, all nine encodings visible, paired SAZ–static and SAZ–PAZ uncertainty |
| Classifier results | Delete the selected-epoch-range interpretation paragraph; epoch-selection mechanics remain in Methods. Reduce task anecdotes to one sentence. | Four model definitions, LPQ/CNN preprocessing difference, SVM–CNN effect and interval, lack of classical separation |
| Fusion | Remove the four-row classifier table if Figures 4–5 and the text already supply the same endpoints. Reduce sensitivity/specificity/AUC commentary to one compact operating-point sentence or move those plots to the gallery. | Available-task denominator, PD tie handling, per-repeat aggregation, exploratory common majority endpoint, near tie between LR and SVM |
| XAI | Merge repeated statements that maps are not biomarkers. Do not repeat the four-case selection procedure in both full caption and full paragraph. Keep the whole-channel result to one paragraph. | Exact checkpoint/target, 2,985/597 coverage, constant-map denominators, matched-area control limitation, near-zero occlusion agreement |
| Discussion | Reduce “What the Incremental Comparison Establishes” to two paragraphs and merge the last two limitations subsections. Remove closing exhortations about what a defensible paper should do. | Scientific explanation of information discarded, literature gap, sample size/age/development limits, concise reproducibility statement |

The ranking and classifier tables are the clearest redundant space consumers. Keeping the required figures while deleting duplicate tables respects the user's preference for a visual article. If exact values are removed from a display, retain the key estimates in prose and all values in the reproducibility files.

## Sentence-level edits for more natural prose

- Abstract, “The results favour modest claims”: replace this editorial judgement with the observation: “Colour encoding produced inspectable inputs, but its gain over static reconstruction remained uncertain.”
- Abstract, “SVM, logistic regression and Random Forest outperform a small CNN”: add “in task-mean macro F1” so the statement cannot be read as the nonsignificant fused comparison.
- Introduction, “We report that selection and the associated uncertainty rather than treating the best score as a new independent test”: shorten to “Downstream results are conditional on that exploratory selection.” The fuller explanation belongs in the protocol.
- Static result, “Writing content matters even before dynamics are introduced”: “Static performance varied across tasks.” The experiment does not isolate writing content from other task characteristics.
- Representation results, “Neither the presence of a dynamic signal nor an intuitively relevant physiological measurement guarantees a better image descriptor”: delete; the immediately preceding four values and following uncertainty already establish the point.
- Discussion, “A defensible comparison acknowledges that gap and describes its conditions”: replace with a direct result, “Our participant-level accuracy is lower under a different, all-task evaluation protocol.”
- Conclusion, “needs to earn its place”: remove the rhetorical closing. End on the bounded result: “Combining task predictions yielded participant-level macro F1 near 0.65; the added colour signals did not show a supported advantage over the static baseline in the task-level comparison.” Keep clear that task fusion and task-average metrics are different endpoints, not a paired fusion gain.

The manuscript repeatedly uses “not X” clauses to guard against plausible misunderstandings. Most are scientifically justified individually, but repeating them in captions, results, discussion and conclusion creates a defensive rhythm. State each limitation fully once, then use short qualifying terms such as “exploratory”, “task-mean”, or “conditional” where needed.

## Explicit quality criteria from published VISAPP papers

1. **Figueroa et al. (2024), [A Learning Paradigm for Interpretable Gradients](https://www.scitepress.org/PublishedPapers/2024/124668/).** Their method definition, matched examples and quantitative attribution tests give separate evidence for prediction and interpretation. Our draft meets this criterion through Grad-CAM, deletion and occlusion; it should retain the distinction between targeted sensitivity and complete faithfulness. Fixing the late floats will place that evidence beside the argument.

2. **Karatsiolis and Kamilaris (2023), [A Model-agnostic Approach for Generating Saliency Maps to Explain Inferred Decisions of Deep Learning Models](https://www.scitepress.org/PublishedPapers/2023/116124/).** Their evaluation acknowledges model-dependent losses and connects saliency panels to perturbation measurements. The four correct/error cases and retained zero map satisfy that spirit. Keep those cases during compression; remove repetitive warnings instead of suppressing the weak result.

3. **Aires et al. (2025), [Handwriting Trajectory Recovery of Latin Characters with Deep Learning](https://www.scitepress.org/Papers/2025/133831/133831.pdf).** Their geometric examples clarify why evaluation definitions can change reported outcomes. Our paper's analogous strength is the explicit separation of task means, participant fusion, accuracy and PD-class F1. Preserve the single protocol-aware literature table, but avoid repeating every metric distinction in every subsection.

These are concrete editorial comparators, not a claim that the manuscript will be accepted. The proposed revision should improve economy while preserving the evidence behind its narrower contribution.
