# First-five static scientific figures

CPU-only `plots.py` reads exact SHA-bound accepted statistics summary and
descriptive-bin CSV. It performs no inference, scoring, token/trajectory read,
training, HTTP or GPU work. English labels use bundled DejaVu Sans. No question
IDs or private paths are included in figure-data or the plots.

CLI: `python3 -B plots.py --manifest /ABS/NEW/input.json
--manifest-sha256 EXACT --output /ABS/NEW/output`.

Manifest: `schema:opsd-first5-em-loss-plot-inputs-v1`, explicit `{path,sha256}`
refs `summary` and `bins`. They must be the accepted stats output4529 pairs,
7680 cohorts. Root publishes/readbacks source before executing it, then views
the actual rendered PNGs and checks labels/values and PNG/SVG/JSON SHA.

Four aggregate figures (PNG and SVG each):

1. Six primary features vs empirical five-sibling s/5: group-specific Pearson r
   and marginal95% common-question bootstrap intervals, plus pooled comparison.
2. Think/search/final-answer teacher KL, observed student logp and logp gap vs
   selected-D EM and s/5, preserving absent regions and actual denominators.
3. Four core question-block OOF information families: binary strict-D AUC and
   empirical-fraction RMSE. These are actual OOF point metrics; marginal
   increment confidence intervals are retained in JSON, not invented as
   absolute-metric intervals. Ordinary/source controls, added student
   distributions, added reference metadata and added teacher distributions
   remain distinct.
4. Per-group descriptive teacher-KL and answer-geometric-probability bins vs
   selected-D correctness and s/5. Answer x is exp(midpoint of bin logp bounds),
   not an arithmetic mean probability or the probability of a correct whole
   answer. Bins are full-sample descriptive quantiles, not OOF cut points.

`figure-data.json` records all plotted aggregate numbers, source SHA values,
denominators and retained absence rows. `inputs.private.json` records fixed
path/metadata proofs separately; it stays private. `derived-manifest.json`
binds the exact8 rendered files plus JSON/source SHA. Every output uses a new
directory. The plot helper does not change accepted statistics or production
code, and plotting failure preserves its output directory for root diagnosis.

Correlation/OOF figures concern recorded eligible D and completed teacher-
forced prefixes. All-five-wrong questions do not gain fabricated loss. Native
mining can constrain D correctness to all-five-correct fallback. None of these
figures is a causal regional-gradient study, future probability calibration,
five-outer heldout512 result or the full seven-group final report.
