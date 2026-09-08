# Final methodological and visual review

Date: 2026-09-08. Result: PASS; no blocking issue identified.

Reviewed the final LaTeX changes in sections 2–8 and individually inspected rendered manuscript pages 4, 5, 6, 7, 8, 9 and 10 in `tmp/final-referee/render/`. The reviewed main PDF has 14 pages (SHA-256 `5f5bff91cfd12ab52689263bc0b8387ccc7d23e28aded577bffac4045a557c85`); the selection PDF has 26 pages (SHA-256 `421ba37d5e587b03ea3a9f210e0de020af4680bf70d3bb97e00977e8df650c51`).

- Pages 4–5: normalization equations, RGB assignments, task inventory and participant protocol are legible. Table placement and continued text preserve the reading order.
- Pages 6–7: the validation diagram, LPQ/SVM settings, encoding results and classifier definitions are legible and agree with the saved procedure. Participant separation and exploratory encoding selection remain explicit.
- Pages 8–9: encoding and classifier figures retain readable axes, values and captions. Conditional classifier inference and the distinction between task means and participant fusion are preserved; no new result or superiority claim was introduced.
- Page 10: fusion intervals, XAI target definition and enrichment/deletion summaries are legible. The new statement that deletion retains all-zero maps is present and consistent with the implementation; its continuation follows on the next page.
- Scientific corrections were checked against local sources: 95% percentile intervals; three Holm-adjusted fusion-rule pairs; inclusion of all 2,985 predictions in deletion; and the interpretation of Grad-CAM/occlusion disagreement. The discussion and conclusion retain the observed gains on four tasks and qualify overall superiority and cross-model inference.

No clipping, overlapping text, truncated table, missing mathematical symbol or incompatible numerical claim was found in the assigned pages. Ordinary column/page continuations around floats do not omit content. No experiments were rerun and no LaTeX was changed during this review.

Final-build closure: the subsequent archive-wording correction affected only page
13. PNG hash comparison confirmed that pages 4–10 are unchanged in the final PDFs;
their current hashes are recorded in `build_verification.json`. The scope and
PASS result of this review therefore remain valid.
