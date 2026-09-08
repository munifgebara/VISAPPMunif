# Publication figure catalogue

All charts derive from frozen experiment outputs. The excluded experiment is absent from every figure and data input. Raster PNG files are 300 dpi; matching PDF files retain vector text and plots (handwriting and attribution are embedded raster data). Sizes below include the tight bounding box. Suggested manuscript width is 158 mm unless reduced deliberately after reviewing labels.

## Figure selection

### activity_diagram

Suggested placement: **main**. PNG: 2400 × 2848 px. Files: `paper/figures/activity_diagram.pdf` and `.png`.

Illustrated activity diagram of the incremental study. The tablet contains a reconstructed PaHaW spiral; its outline and stylus are schematic. The loop evaluates static geometry, four single-signal encodings and all four three-signal RGB combinations with LPQ and SVM. Participant-separated outer assignments are shared across tasks and methods. Selecting SAZ from the outer scores is exploratory. Independent classifier branches reuse those assignments; SVM predictions are retained from the encoding experiment. The CNN selects epochs on the first inner holdout. Task scores and participant voting are distinct endpoints. XAI uses only the CNN checkpoints responsible for the held-out predictions. The final illustrations show the SVM task-vote confusion matrix (row-normalized within each repetition, then averaged) and the previously selected median-confidence correctly classified PD case on task 4, with its original Grad-CAM. These two illustrations summarize different analyses and are not a matched case pair.

### progressive_encoding

Suggested placement: **main**. PNG: 1819 × 805 px. Files: `paper/figures/progressive_encoding.pdf` and `.png`.

Progressive rendering of task 1 (Archimedean spiral) for one control participant and one participant with Parkinson's disease. Static geometry is followed by speed in one channel and the SAZ and PAZ triplets. All images preserve the same contact-only geometry and foreground/background mapping. Examples are the lowest anonymous IDs with all eight recordings in each class, chosen without reference to prediction quality.

### single_signals

Suggested placement: **alternative**. PNG: 1819 × 805 px. Files: `paper/figures/single_signals.pdf` and `.png`.

Isolated signal encodings of the same task-1 examples as progressive_encoding. The tested signal varies in the red channel; green and blue are held at 30. Per-recording contact-segment 5th and 95th percentiles define the scale; speed is log-transformed, and azimuth is centred circularly.

### rgb_semantics

Suggested placement: **main**. PNG: 1856 × 500 px. Files: `paper/figures/rgb_semantics.pdf` and `.png`.

The semantic channels of an SAZ spiral, displayed individually in grayscale and jointly as RGB. Red stores normalized log-speed, green stores altitude, and blue stores circularly centred azimuth. Grayscale panels are exact channel values of the composite image, including antialiasing and the white background.

### task_atlas

Suggested placement: **alternative**. PNG: 1737 × 1720 px. Files: `paper/figures/task_atlas.pdf` and `.png`.

The eight PaHaW tasks rendered as static grayscale images for the same two illustrative participants. Tasks progress from a spiral through repeated letters and words to a sentence. Each recording is scaled separately while preserving aspect ratio; absolute writing size is therefore not represented.

### rgb_task_atlas

Suggested placement: **alternative**. PNG: 1737 × 1720 px. Files: `paper/figures/rgb_task_atlas.pdf` and `.png`.

The eight PaHaW tasks rendered as SAZ RGB images for the same two illustrative participants. Tasks progress from a spiral through repeated letters and words to a sentence. Each recording is scaled separately while preserving aspect ratio; absolute writing size is therefore not represented.

### grouped_validation

Suggested placement: **main**. PNG: 1842 × 1075 px. Files: `paper/figures/grouped_validation.pdf` and `.png`.

Evaluation flow. Participant identities define every split. Outer assignments are shared across tasks and methods. Classical hyperparameters are selected in three inner folds; the CNN selects its epoch count using the first inner fold, then retrains on the entire outer-training set. LPQ-RGB uses 64×64 inputs, whereas the CNN uses 128×128 inputs. Confidence intervals resample participants jointly across tasks and repetitions. Every XAI map uses the checkpoint responsible for that external prediction.

### encoding_effects

Suggested placement: **main**. PNG: 1862 × 985 px. Files: `paper/figures/encoding_effects.pdf` and `.png`.

SVM performance for all nine encodings, ordered by the mean of task-specific macro-F1 across eight tasks and five repeats. Right: paired differences from the static representation, with 95% stratified participant-bootstrap intervals. Participants are sampled jointly across tasks and repeats. The full 36-pair Holm family supports no encoding difference; the SAZ lead is descriptive.

Numerical point labels use four decimals to distinguish SAZ (0.5752) and PAZ (0.5745).

### encoding_heatmap

Suggested placement: **alternative**. PNG: 1861 × 1045 px. Files: `paper/figures/encoding_heatmap.pdf` and `.png`.

Task-specific mean macro-F1 for the nine encodings under LPQ-RGB plus SVM. Every cell averages five out-of-fold repetition metrics. The shared colour scale makes the variation across tasks and encodings visible; cells are descriptive and do not imply separately significant contrasts.

### encoding_tasks

Suggested placement: **alternative**. PNG: 1862 × 761 px. Files: `paper/figures/encoding_tasks.pdf` and `.png`.

Task profiles for static, altitude alone, SAZ, and PAZ under the same SVM protocol. Values are mean macro-F1 over the five repeats. Lines connect task identifiers only and do not indicate a continuous scale or a fitted trend.

### classifiers_tasks

Suggested placement: **main**. PNG: 1862 × 1109 px. Files: `paper/figures/classifiers_tasks.pdf` and `.png`.

The fixed SAZ representation evaluated with SVM, logistic regression (LR), random forest (RF), and a compact CNN. Left: task-specific macro-F1 averaged over five repetitions. Right: unweighted mean over the eight tasks. Classical methods use LPQ-RGB descriptors; the CNN learns from the image. The CNN is below each classical method in the global six-pair Holm family, whereas differences among classical methods are unsupported.

### classifier_contrasts

Suggested placement: **alternative**. PNG: 1862 × 595 px. Files: `paper/figures/classifier_contrasts.pdf` and `.png`.

Paired differences from SVM on the task-mean endpoint. Error bars are 95% stratified participant-bootstrap intervals; annotations report Holm-adjusted permutation p-values from all six classifier pairs. Negative differences favour SVM.

### participant_fusion

Suggested placement: **main**. PNG: 1862 × 655 px. Files: `paper/figures/participant_fusion.pdf` and `.png`.

Participant classification by majority voting across available tasks for each classifier. Points show the mean macro-F1 of the five repeats, and error bars are 95% stratified participant-bootstrap intervals. No pairwise classifier difference survives Holm correction for this fused endpoint. These values evaluate one decision per participant and must not be read as task-level scores.

### fusion_roc

Suggested placement: **alternative**. PNG: 1862 × 1015 px. Files: `paper/figures/fusion_roc.pdf` and `.png`.

Mean participant-level ROC curves for task-vote fractions. A ROC is computed separately in each of the five repetitions (75 distinct participants per repeat), interpolated to a common false-positive-rate grid, and then averaged. Legend AUCs are the mean of the five original repeat AUCs, not areas of pooled repeated predictions. Vote fractions are discrete scores, not calibrated disease probabilities.

### fusion_confusion

Suggested placement: **alternative**. PNG: 1702 × 1495 px. Files: `paper/figures/fusion_confusion.pdf` and `.png`.

Confusion matrices after task voting. Each matrix is computed from 75 distinct participants within a repeat, then averaged over five repetitions. Cell percentages are normalized by the observed class; parenthetical counts are mean participants per repeat (therefore fractional). Repeated predictions are not counted as independent observations.

### xai_examples

Suggested placement: **main**. PNG: 1793 × 1960 px. Files: `paper/figures/xai_examples.pdf` and `.png`.

Task-4 explanations for correct and incorrect predictions from both classes. The four frozen repeat-1 cases were selected at median predicted-class confidence within each class/outcome stratum, without selecting for visual plausibility. Grad-CAM and positive occlusion sensitivity are normalized per map; overlay opacity increases with attribution. The PD error has an all-zero Grad-CAM map and is retained. Maps explain the predicted class and use the exact external-test checkpoint. P(PD) is the CNN output, not a calibrated clinical probability.

Annotation and column-header type is 10 pt in the PDF, approximately 7.49 pt when included at 113.76 mm (72% of the 158 mm text width). The original 430.734807 × 470.639520 pt PDF footprint and 1793 × 1960 px PNG footprint are retained.

### xai_validation

Suggested placement: **main**. PNG: 1862 × 746 px. Files: `paper/figures/xai_validation.pdf` and `.png`.

Perturbation checks for all 2,985 external CNN predictions. Left: removing the top 10% Grad-CAM pixels, equal-area random removal, and their paired difference. The random control matches pixel count but not foreground coverage or patch shape. Right: replacing each complete SAZ channel by white. Points are participant means and intervals are 95% stratified participant-bootstrap intervals after averaging repeats and tasks within participant. Channel removal changes image contrast and cannot isolate a physiological mechanism.

### xai_channel_tasks

Suggested placement: **alternative**. PNG: 1861 × 970 px. Files: `paper/figures/xai_channel_tasks.pdf` and `.png`.

Mean channel-removal confidence changes by task. Repeated predictions are averaged within participant/task before the task mean. Positive values mean the original channel supported the predicted class; negative values mean its removal increased confidence. Colour uses a symmetric scale around zero. These are image perturbations, not isolated physiological effects.

### xai_agreement

Suggested placement: **alternative**. PNG: 1862 × 775 px. Files: `paper/figures/xai_agreement.pdf` and `.png`.

Descriptive distributions of spatial agreement. Left: Grad-CAM and positive occlusion maps for repeat 1 (one map per available participant/task recording). Right: trained and fully randomized networks, one external image for each of 200 checkpoints. Spearman correlations omit constant maps. These distributions are not independent-sample significance tests: participants recur across tasks, and checkpoints share training data.

### fusion_rules

Suggested placement: **alternative**. PNG: 1862 × 595 px. Files: `paper/figures/fusion_rules.pdf` and `.png`.

SVM task-fusion rules with 95% stratified participant-bootstrap intervals. Probability calibration and reliability weights use outer-training data only. The mean-probability rule was the primary fusion endpoint; the reliability-weighted mean and majority vote were declared sensitivity analyses. Task-fusion rules use all available tasks without selecting a subset from external results.

## Statistical and visual safeguards

- Encoding and task-level points are means of repeat-specific out-of-fold metrics, not metrics over pooled repetitions.
- Fusion ROC curves and confusion matrices are computed per repeat before averaging. Vote fractions are not probabilities.
- All shown confidence intervals come unchanged from the original stratified participant-bootstrap files. No sample-level bootstrap was introduced.
- Examples use anonymous class labels. The source manifest preserves exact file hashes; no participant identifier is printed in the figures.
- XAI selection retains the originally frozen median-confidence cases, including an all-zero Grad-CAM failure. Opacity is proportional to attribution to avoid tinting zero areas.
- The atlas deliberately retains per-image scaling and blank margins so the preprocessing can be assessed. Absolute handwriting size cannot be inferred.
