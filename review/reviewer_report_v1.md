# Independent scientific review of manuscript version 1

Reviewed the complete `paper/main.tex`, all seven section files, bibliography, seven body figures, and the compiled 15-page `paper/build/main.pdf`. Local results were checked against the source CSVs for the encoding, classifier, fusion and XAI runs. Review criteria follow Section 6 of `bibliography_audit.md`: explicit method definition, matched figures, attribution evidence beyond visual plausibility, precise evaluation units, candid model-dependent limitations, and figures that answer the adjacent scientific question. No additional experiment is requested.

**Recommendation: revise before circulation.** The central numerical account is accurate and the manuscript is substantially more transparent than a conventional best-score comparison. Its contribution is an empirical evaluation of an explicit representation, not a new classifier or an established clinical advance. That is a coherent paper, but the current length and repeated defensive commentary obscure the result. The main corrections are editorial, with one bibliography-name error and several short reproducibility clarifications.

## 1. Highest-priority changes

### R1 — The main manuscript is 15 pages and two essential XAI figures are stranded after the references

**Location:** PDF pages 13–15; `06_xai.tex:5` and `:26`; `main.tex` before bibliography.

Text and references extend through page 13. The XAI example figure occupies page 14 alone, and the perturbation figure occupies page 15 alone. These are essential body figures, not the author's requested alternative gallery. They are also absent from explicit `Figure~\ref{...}` calls in the XAI body. A reader encounters the XAI conclusions several pages before the evidence. The standalone full-page placement of the much shorter perturbation figure leaves most of its page empty.

**Required change:** make both references explicit; bring both floats into the XAI discussion; reduce/rearrange the tall example panel or its rendering size sufficiently to satisfy the normal two-column float placement constraints; prevent body floats from passing into the bibliography. Preserve the main template's font and margins. The primary PDF must end at or before page 12, and the alternative selection document must begin its extras at page 13. All seven scientific body figures can remain if the duplicated tables and prose below are removed.

**Evidence:** page-text extraction and visual checks of pages 1, 8, 9, 12, 14, and 15. The first PDF is 15 pages; the last two pages contain approximately 88 and 97 extracted words, respectively, almost entirely figure labels and captions.

### R2 — Remove duplicated results before compressing methods or figures

**Location:** `04_representation_results.tex:30`–`:49`; `05_classifiers_fusion.tex:60`–`:74`; PDF pages 6 and 9.

Table 3 reproduces the nine encoding scores already shown and discussed in Figure 3. Table 4 repeats task means from Figure 4 and fused estimates/intervals from Figure 5. These consume valuable space without contributing a new comparison. The manuscript repeatedly recites the same four-decimal point estimates in figure labels, tables and prose.

**Required change:** retain all seven figures, delete the encoding-ranking table, and either delete the four-model table or replace it with a compact table of genuinely complementary operating-point quantities. The latter are already discussed in prose, so deletion is the simpler option. Keep exact principal scores and paired intervals in the adjacent paragraphs; the supplementary data tables retain all digits. Update figure/table references accordingly. This also removes the table-4 overfull box warning.

### R3 — Fix the compound surname of the principal thesis comparator

**Location:** `refs.bib`, `Casademunt2023`; PDF page 12 reference list; `07_discussion.tex:14` and `:26`.

The body calls the author “Casademunt González,” but BibTeX currently parses the record as “González, A. C.” and generates “(González, 2023).” This inconsistency makes the requested comparison look like two references.

**Required change:** use `author = {Casademunt Gonz{\'a}lez, Alberto}` and recompile BibTeX and LaTeX. In the table use “Casademunt González (2023)” or a proper citation. The thesis is correctly credited to the student, with Ángel Sánchez Calle as supervisor, not coauthor.

### R4 — Qualify the high-level algorithm comparison as conditional on SVM-selected SAZ

**Location:** abstract in `main.tex`; `01_introduction.tex:16`; `05_classifiers_fusion.tex:30`–`:36`.

The main text appropriately discloses outer-score representation selection, but the abstract still reads as a general comparison in which three algorithms outperform a CNN. SAZ was selected using SVM scores from this cohort, then fixed for the other methods. The algorithm comparison is conditional on that SVM-guided representation choice. The findings remain reportable, but their scope should be recognizable in the abstract and the comparison should not be presented as an equal search over representations for every model.

**Required change:** add a short phrase such as “In the exploratory comparison on this SVM-selected encoding...” before the classifier claim. Keep one explicit statement of the selection limitation in Data and Evaluation and one short reminder in Discussion; eliminate repeated qualifications elsewhere. No fresh experiment is necessary for this draft's stated developmental contribution.

## 2. Scientific clarifications and small factual corrections

### R5 — Use the source's term for the auxiliary Diaz result, or omit that extra number

**Location:** `07_discussion.tex:37`.

The manuscript labels 81.25% “velocity-only enhancement,” while the verified bibliography memo records “temporal enhancement alone.” The main comparable result is 86.67%, and the extra static/temporal scores are not necessary to make the comparison.

**Change:** verify the exact source label with the bibliography reviewer, or remove the 72.50%/81.25% sentence during condensation. Retain 86.67% with its selected-task ensemble and protocol differences. Do not silently rename a source's experimental condition.

### R6 — Explicitly define the binary target used for Grad-CAM

**Location:** `06_xai.tex:14`–`:19`.

The CNN has one output logit, but the Grad-CAM equation refers to a predicted-class logit `g_c` as if there were two outputs. The code correctly uses the output logit for PD and its negative for H. A reader attempting reproduction needs that sign convention.

**Change:** append “For the single output logit, the H target is its negative and the PD target is the logit itself.” This is more useful than the repeated generic warning about the semantic meaning of heatmap colours, which can stay once in the caption.

### R7 — Report the participant coverage of the 200 random-weight checks

**Location:** `06_xai.tex:43`.

The 200 checkpoints are not 200 independent images or participants. The deterministic first-test-ID rule causes these checks to cover only **11 distinct participants**, repeated across tasks and folds. The current text does not claim independence, but “one ... image from each of the 200 checkpoints” can still suggest broader coverage than obtained.

**Change:** write “Across 200 checkpoints, covering 11 distinct participants, ...” and retain the 160 nonconstant-pair denominator. The saved `weight_randomization_sanity.csv` gives 11 unique subject IDs. This is a descriptive sanity check rather than a broad robustness result.

### R8 — Tighten the LPQ description to match the exact implementation

**Location:** `04_representation_results.tex:8`.

The description is generally faithful. “No ... learned decorrelation” leaves open whether a fixed covariance decorrelation was applied; none was. The four Fourier kernels are mentioned without their frequency positions.

**Change:** replace the ending with “The four frequency offsets are (1,0), (0,1), (1,1), and (1,−1), scaled by 1/5; no decorrelation or spatial pyramid is applied.” Reorder offsets freely because the histogram code-bit order, if needed, is in source. The important point is not to imply the complete canonical LPQ preprocessing solely through its citation.

### R9 — Describe device pressure as a sensor quantity, not calibrated force

**Location:** `02_encoding.tex:30`.

“Pressure reflects contact force” can be read as calibrated physical force. The experiment uses raw tablet pressure with within-record normalization, not force measurements.

**Change:** “Pressure is the tablet's contact-pressure signal.” The native-unit qualification already used for speed is appropriate. Keep the distinction between stylus orientation and trajectory direction.

### R10 — Add two inexpensive reproducibility details while deleting redundant prose

**Location:** `05_classifiers_fusion.tex:16` and `:49`; `03_protocol.tex:61`.

The network loss is called weighted BCE without specifying the ratio, and the calibrated fusion rule only says sigmoid calibration occurs inside training. The implementation is more specific and can be described without a large parameter table.

**Change:** state “positive weight = H/PD training count” for BCE, and “sigmoid calibration in the three inner participant folds, with ensemble averaging” for SVM. Describe the 10,000 randomization draws as participant-wise swap patterns rather than simply “10,000 exchanges.” No need to add software implementation trivia to the main narrative; library versions and hashes belong in the reproducibility package.

### R11 — Make literature rows consistent and complete enough to interpret

**Location:** `07_discussion.tex:18`–`:33`.

The comparison table correctly uses accuracy and different scopes, avoiding the former macro-F1/F1 mismatch. However, the current rows state the present task fusion but omit its repeated participant-separated evaluation, whereas the historical rows name CV or holdout. Casademunt's row also lacks a year. The current table overflows its text box by roughly 6 points.

**Change:** label current scope “T1–T8; 5×5 participant CV; majority fusion” with a short caption explaining nested classifier tuning and available-task policy. Add the thesis year and reduce column widths or reflow cells to remove overflow. SVM and logistic regression have identical accuracy here; one combined row can say “LPQ + SVM / LR” with “65.07 / 65.07,” or retain two rows if macro-F1 complement is used in the text. Do not use the same table to rank incomparable protocols.

## 3. Novelty, narrative and condensation plan

The paper's novelty is the controlled progression from geometry through isolated measured signals to all triplets, with explicit loss of absolute quantities and common participant partitions. The negative or uncertain gain is scientifically useful when it is the result being explained. The Introduction should state that contribution directly and spend fewer sentences defending every result against a hypothetical overclaim. The current prose has a recurring pattern—result, caution, another caution, future alternative—which makes a careful study feel hesitant and adds length.

The following cuts preserve the seven figures and all essential scientific qualifications. Approximate savings are editorial targets, not exact page predictions.

| Location | Concrete condensation | Approximate saving |
|---|---|---:|
| `01_introduction.tex:12`–`:18` | Merge the general learned-feature and signal-to-image background into one paragraph; delete the roadmap paragraph. Preserve Diaz as the nearest precedent, the minimap definition, and the contribution paragraph. | 140–190 words |
| `02_encoding.tex:10`, `:20`, `:39`, `:53`, `:77`, `:79` | Use a distinct recording symbol rather than explaining `p_i` twice; retain one sentence each on lost size, relative signal level, azimuth branch cut, channel permutation, and crossing overwrite. | 100–150 words |
| `03_protocol.tex:28`, `:30`, `:44`, `:46`, `:63`–`:65` | Keep audit counts and study unit, but reduce the family enumeration to the global 36/6/6 families, with taskwise families in reproducibility notes. Combine selection and fixed-prediction uncertainty into one compact limitations paragraph. | 130–180 words |
| `04_representation_results.tex:18`, `:28`, `:51`, `:53`–`:57` | Delete the repeated scanned-image/selection paragraph; keep static baseline, isolated-signal range, SAZ/PAZ contrast and one positive/negative task example. Delete Table 3. | 200–250 words plus one table |
| `05_classifiers_fusion.tex:20`, `:30`, `:34`, `:36` | Keep selected epoch range in one sentence. Report classical contrasts jointly, preserving SVM–CNN effect/CI/p. Replace general architecture disclaimers with one sentence identifying the tested CNN and preprocessing/selection difference. | 100–150 words |
| `05_classifiers_fusion.tex:51`, `:76`, `:78`, `:80` | Keep original calibrated endpoint and common majority definition. Report LR/SVM near-identical estimates and complementary sensitivity/specificity together; move ROC AUCs to their alternative figure. Delete Table 4 and the final endpoint-warning paragraph. | 100–150 words plus one table |
| `06_xai.tex:12`, `:23`, `:35`, `:37`, `:43`, `:47`–`:51` | Put the exact example-selection rule only in the caption; retain measured enrichment, zero maps, deletion comparison and its control limitation, 540-pair spatial agreement, 160-pair random-weight check. Let Figure 7 carry channel intervals; keep only point values and one caution in prose. | 180–250 words |
| `07_discussion.tex:6`–`:10`, `:39`, `:43`–`:53`, `:58`–`:60` | Merge the summary with reasons for lost information; remove repeated negative framing; keep literature gap, selection/age/cohort scope and reproducibility. Make the conclusion one focused paragraph. | 250–350 words |

Together these changes can remove roughly 1,200–1,600 words plus two duplicated tables. Do not spend that recovered space on more caveats or a generic history of CNNs. Use it to place both XAI figures beside their analysis while maintaining readable labels.

The six essential qualifications to retain once in their proper locations are: per-record normalization loses absolute quantities; SAZ selection is exploratory; resampling is conditional on fixed predictions; fusion differs from task averaging; the CNN is one architecture trained from scratch; and XAI perturbation/agreement evidence does not establish a disease-specific region. Their repetition is not additional rigor.

## 4. Specific wording recommendations

- Abstract: replace “The results favour modest claims” with a direct conclusion such as “The colour representation is inspectable, but its small observed advantage over static remains uncertain.” The current phrase evaluates the authors' rhetoric rather than their scientific finding.
- Introduction, line 6: “simplest possible image” is unnecessarily absolute; “a monochrome reconstruction” is sufficient.
- Representation, line 53: make the replacement reference explicit: “Starting from PAZ, replacing pressure, azimuth or altitude with speed gives SAZ, SPA or SPZ.” This avoids forcing the reader to infer which triplet is being modified.
- Discussion, line 39: remove the sentence about selecting favourable tasks and obscuring the question. The protocol already states the actual choices; speculation about worse alternatives sounds adversarial and adds no evidence.
- Discussion, lines 43–45: retain the data-size and XAI-agreement interpretations, but avoid repeatedly calling the model weak or inferior once its quantitative result has been established.
- Conclusion, line 60: replace “needs to earn its place” with a scientific statement about the result. Example: “Under the tested normalization and participant partitions, neither additional colour signals nor the trained-from-scratch CNN produced a clear advantage over the corresponding simpler alternatives.”
- The title currently names a representation but not its application. If space permits, append “for Parkinson's Disease Classification” or use a subtitle; otherwise the abstract and keywords already identify the application. This is optional and lower priority than a concise main narrative.

## 5. Checks completed and positive findings to preserve

- Dataset totals, task-1 class counts, static values, isolated-signal examples, SAZ task deltas, all four classifier means, main CIs/p-values, fusion operating points and XAI channel drops match the saved files at the reported precision.
- The specific static T4 estimate 0.6502 and CI [0.5582,0.7351] are correct. RF T4 0.6791 is correct. The reported CNN accuracies for T2/T3/T4 are 55.20/52.00/53.60%, matching the saved summaries.
- Macro-F1, PD-class F1 and accuracy are distinguished correctly. The literature comparison does not manufacture a paired superiority claim.
- The shared outer participant partition, single-inner-holdout CNN procedure, per-record normalization, and age imbalance are disclosed.
- The per-map zero Grad-CAM example is retained instead of being hidden; confidence perturbations are evaluated quantitatively; the low agreement with occlusion is reported with its valid denominator. These meet the substantive XAI criteria drawn from the published VISAPP examples.
- The author-excluded stroke-width experiment is absent from all article sections reviewed. Fixed thickness appears only as a renderer setting.
- The body reads from encoding to classical learning, pixels, fusion and XAI. That progression should survive the condensation; it is the manuscript's organizing strength.

## 6. Completion criteria for revision 2

1. Main body including references and all seven selected figures is at most 12 pages with no template squeezing; alternatives begin at page 13 only in the selection edition.
2. Every body figure is explicitly called out near the paragraph that interprets it; no main figure follows the bibliography unintentionally.
3. Duplicate ranking tables are removed or given a nonredundant role; table overflow warnings are resolved.
4. Casademunt's surname and the Diaz condition label are corrected; new DOI records remain those already verified.
5. Abstract states exploratory comparison scope; binary Grad-CAM target and the 11-participant randomization coverage are clear.
6. Core numeric results remain unchanged, and the conditional/single-cohort limits remain once each in the appropriate section.
7. The final visual check assesses the full PDF at reading scale, including table alignment, figure label sizes, XAI caption placement, and the clean transition into the optional gallery.

