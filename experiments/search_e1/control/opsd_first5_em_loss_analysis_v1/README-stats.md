# First-five retrospective correctness/loss statistics (CPU)

`stats.py` reads only explicitly SHA-bound outputs of the native first-five
extractor. No model, tokenizer, strict scorer, network, GPU, RNG restoration,
checkpoint writes or training is used. The root publisher must publish and
verify source before running it. Raw input/output question IDs remain private.

CLI: `python3 -B stats.py --manifest /ABS/input.private.json
--manifest-sha256 EXACT --output /ABS/NEW/output`.

Manifest schema: `opsd-first5-em-loss-stats-inputs-v1`, explicit references
`pair_features`, `cohort_features`, `extraction_summary` and `root_execution`,
each `{path, sha256, bytes?}`; `seed:42`,
`cv_folds:5`, `bootstrap_replicates:1000` (500–2000 accepted). Three groups are
C0/C1/M and exactly five512-question outer cohorts each. No changing cohort
denominator, invented all-wrong loss, silently skipped partial JSONL or duplicate
pair is accepted. Features/outcomes must match the common group/question cohort.
The extractor summary must be accepted with explicit counts4529 pairs,
940295 valid tokens,7680 cohort rows,15 outers and151 updates. Statistical row
and token totals reproduce those accepted counts. Root execution proof requires
`returncode:0`, explicit64-hex `extractor_source_sha256`, `request_sha256`,
`stdout_sha256`, `stderr_sha256`, with request identity equal to the accepted
summary. It is SHA-bound evidence of the actual extractor execution, not a new native
token/logit audit by the statistics program.

The extractor schemas are `opsd-first5-pair-feature-v1` and
`opsd-first5-cohort-feature-v1`. Region means use `teacher_kl`, `outer_old_kl`,
`target_post_loss`, `student_entropy`, `student_observed_logp` and
`student_teacher_observed_logp_gap`. Whole-policy values and final-answer-region
values remain distinct. Region absence contributes exactly zero to whole-
trajectory loss; an absent answer region has no observed-token log probability
and is NA, never logp0. The extractor owns reverse direction, offsets, native
loss reconstruction and strict labels; this program never re-scores a response.

## Questions and outcomes

First verify the actual native rule: selected D is correct if and only if all
five siblings are correct. Report deviations/counts instead of assuming this
rule describes an unbiased policy sample. All-five-wrong questions remain in
the full cohort success histogram and empirical success fraction, with no pair
loss; available-loss correlations explicitly condition on eligibility.

`s/5` is the empirical success fraction of five outer-start sampled siblings.
`pass_at5` means at least one strict success, not an extrapolated independent
event probability. Selected-D strict EM and these outcomes precede that outer's
training. Token forward features use the already complete D and a current
pre-update state, potentially after earlier Adam updates in the same outer.
They are not the processed scores emitted at sampling time, nor an ex ante
forecast or a calibrated probability of a future answer. Teacher quantities
also use a privileged successful B reference.

## Association and uncertainty

Six predefined main features: whole-policy teacher raw KL, target post-clip
loss, ordinary outer-old raw KL, final-answer student observed-token logp,
final-answer student-minus-teacher logp gap, whole-policy student entropy.
For each group and pooled eligible questions: Pearson/point-biserial, Spearman,
oriented AUC for binary D EM, correct-minus-incorrect mean difference and Cohen
d; correlate continuous s/5 separately. AUC uses higher recorded feature as
the positive direction, without choosing orientation after observing labels.

Bootstrap common question IDs jointly across groups, within outer/data-source
strata; every selected pair of a drawn question receives the same multiplicity.
Replicates recompute weighted exact tied ranks, AUC and moments. Intervals are
95% marginal descriptive percentile intervals. They are not independent-run
variance or simultaneous/multiple-comparison-adjusted claims. The program
reports no p-values. Full-sample decile bins are descriptive eligible-sample
plots, not prediction cut points; ties can reduce the number of bins.

Region associations are exploratory. Six predeclared pair contrasts use
equal-trajectory regional outer-old KL contributions for C0→C1 and C1→M on
think/search/final-answer. Common eligible question identity is paired, but D,
reference and current model history need not match after outer0. These remain
post-selection associations, not identical-token or causal comparisons.
Think/search/final-answer regional token means of teacher KL, student observed
logp and observed student-teacher gap also receive per-group/pooled binary-D
and empirical-s/5 associations with common-question bootstrap intervals.
Fifteen group/outer cells report main-feature Pearson/Spearman/AUC and counts
without another bootstrap, so sign variation across different cohorts is
visible instead of being hidden in pooled estimates. No cross-outer lag
correlation or learning curve is inferred.

## Out-of-fold retrospective discrimination

Five fixed-seed question-block folds hold all copies of a question across
groups together. Standardization, median imputation and category levels are
learned on training folds only. Missing-answer indicators and group-specific
feature slopes are explicit. A fixed ridge logistic objective uses summed
binary logloss plus `0.5 * lambda * sum(nonintercept beta^2)`, lambda1, fitted
with SciPy L-BFGS; no model selection, nested CV or sklearn installation.

Ordinary controls: group, outer, data source, log valid-token count, student
searches and normalized within-outer pre-update progress. A separate stage
adds reference searches, log reference character length and reference recorded
token count. Do not add sibling
success count, CC/CW pair label or selected-D correctness to predictors: they
leak the target under failed-priority mining. Ordinary-source drift is an
outer-reset target; it is not cumulative source100 drift.

Four comparable core families: ordinary controls; those plus ordinary-student
distribution features (old KL/answer student logp/entropy); then plus successful
reference metadata; then plus privileged-teacher distributions (teacher KL/
target loss/answer logp gap). The first two exclude privileged B metadata,
although they still concern an already completed D response rather than a
real-time pre-generation deployment. No coefficients imply causal impact.

Three additional binary models add only one region's teacher KL, observed
student-minus-teacher logp gap and target post loss to the student-plus-reference
baseline: think, search query or final answer. These report conditional
region-specific OOF information increments, not regional training causality.
Empty-region means are imputed within training folds with an explicit absence
indicator. Four core models also fit bounded empirical s/5 with fractional
cross-entropy and report OOF RMSE, Brier_fraction and correlation. This is
retrospective regression of a five-sibling fraction, not a binomial likelihood
assuming five independent rollouts. Seven binary and four fractional models
over fivefolds yield55 small CPU fits; regularization remains fixed, not tuned.

Report actual OOF AUC/Brier/logloss and question-bootstrap right-minus-left
increments. Higher AUC or lower Brier/logloss is better retrospective
discrimination. Bootstrap conditions on fixed OOF predictions; it does not
refit models per draw or estimate independent training-run variation.
Scores are not used to claim sampling-probability calibration.

Outer0 additionally compares six features on exact common student response,
successful reference and token/mask/offset SHA across the three groups. For
C0→C1/C1→M, report paired deltas and bootstrap intervals overall and by recorded
D correctness. This layer is cleaner history matching, but the current
pre-update model state can already differ along each optimization path. It
does not imply identical checkpoints or gained/lost EM.

Clip count/fraction/student-teacher-target probability mass and raw-minus-post
loss are exploratory associations by group/region. They do not receive another
large bootstrap family and carry no significance/causal claim.

No TRAIN question can be joined to disjoint native512 heldout questions.
Three group-endpoint EM scalars are too few for a useful loss/EM correlation;
the existing native512 paired-score helper handles those endpoints separately.

Output: `summary.json`, `univariate.csv`, `descriptive-bins.csv`, private OOF
predictions and input proofs, plus a content-SHA artifact manifest. New output
directories only. The initial implementation has not run on actual data until
the root publishes and executes it.
