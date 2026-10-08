# OPSD CPU token-region analysis

Independent, read-only postprocessing of completed `opsd_debug` artifacts. This
helper does not import Torch, models, tokenizers or HTTP clients. Root reviews,
backs up and publishes this source before deploying/running it. Actual traces and
outputs are private data and must not enter Git. No live worker/source is changed.

Run with an installed Python 3, substituting existing selection/run directories:

```sh
python3 summarize_regions.py --cases-dir /path/to/frozen-selection \
  --run-dir /path/to/diagnostic-run --output /path/to/NEW-private-analysis
# Optional previously produced frozen strict-protocol cases.jsonl:
# --protocol-cases /path/to/protocol-analysis/cases.jsonl
python3 -m unittest -v test_summarize_regions.py
```

Inputs follow the actual `opsd_debug/worker.py` and `token_probe.py` schema:
`worker-*/case-results.jsonl`, each case's `token-summary-*.json`,
`tokens-*/tokens.jsonl`, and `student/teacher-generation-*.json` (teacher uses
`teacher-generation.json`). The case selection's original pair/siblings supply
the exact response and authored spans. Stable JSON/JSONL SHA256/stat proofs,
selection semantic/manifest hashes, completed-result identities, producer source
manifest from run config and this helper's hash are recorded. Token paths from
summary are not blindly trusted: the expected case-local file is read and must
match its recorded bytes/rows/hash. Summary fields must match the corresponding
completed result. No source file is imported to execute its code.

The case table stores offsets/hashes only (at most 4096 cases); only one case and
one bounded JSONL row (maximum 16MiB) are decoded at a time. Teacher comparisons
stream two rows concurrently rather than retaining full-vocabulary arrays. Small
question-level sufficient statistics are kept for aggregation. Growing result
logs are read to their initial size: only complete-line prefixes count, with
prefix SHA/stat and explicit partial-line skips, never a claimed final log hash.
Completed files must remain byte/stat stable; changed/duplicate/foreign artifacts
fail. Output is exclusive/new; failure retains `failed.json` and partial data
marked unaccepted, with no overwriting or retrying generation.

## Categories and mathematical reduction

Every token retains the recorded frozen `policy_mask`; the same offset/span and
information-range rule is checked for agreement, never repaired. The literal
`<information>…</information>` range (including unclosed trailing information),
zero-width tokens and non-authored tokens cannot receive supervision.

Character categories use exact recognized think/search/answer/information tags
as `structural_tags` first. Remaining content is `search_query` or `think_text`
under the innermost open relevant tag. The final complete answer payload wholly
in the last authored span is `final_answer`; remaining text is `other`. Tokens
touching multiple categories or zero-width offsets are `boundary`. Unknown or
malformed tags stay text. These labels describe text, not protocol acceptance:
`final_answer` is not a claim of strict validity or factual correctness.

For every trajectory/region the output gives valid-token count, metric sum,
sum/count token mean and sum/M whole-trajectory contribution. Metrics are clipped
teacher→student KL, unclipped KL, ordinary-base→student KL, clipped vocabulary
entry count and clipped teacher probability mass. Region contributions sum back
to the corresponding valid-policy mean. Clipped loss must additionally reconstruct
the producer's native trajectory_mean/frozen_pair_loss/contribution total within
the native 2e-5 absolute, 2e-4 relative tolerance; clip counts also reconcile.
Every token metric must be finite; the clipped value may legitimately be negative.
These are loss contributions, **not gradient contributions**.

Common-original comparisons require all four versions to have the same response,
authored spans, token IDs/offsets and masks. For mined pairs, observed privileged
teacher target logprob/probability, entropy and ordered topk IDs/logprob/probability
are compared at every position (1e-6 abs/relative tolerance, exactness and maxima
recorded). Missing fields are explicitly counted and cannot prove invariance;
topk equality cannot prove full-vocabulary equality. This applies to fixed
original history only, not independently generated teacher/student histories.

Hard cases have no privileged teacher and exactly five original failed siblings.
First average **all five trajectory contributions** within question, retaining
zero contribution when a region is absent, then average questions. Region token
means average only nonempty trajectories/questions, with denominator counts.
Clip-count/probability-mass aggregates are their per-policy-token means; raw
per-trajectory sums remain in `cases.jsonl`. Aggregate token counts are question
means, potentially fractional for hard cases, rather than pooled token weighting.

`summary.json` separates original four-version common-history cells/overall from
new-response unpaired cells/overall. New responses never yield a paired checkpoint
loss-change curve. Original loss/drift changes are associated with recorded strict
0→1/1→0/0→0/1→1 flips (six cells and overall); association is not training causality.
Optional protocol outputs are identity/trace/EM-bound and copied without rescoring;
an invalid tag's literal gold match remains only that named diagnostic cue, never
strict EM or semantic correctness. Whole-transcript structure risk belongs to the
separate protocol/source audit, not a factual-error label here. These balanced
training strata do not estimate native512 or benchmark EM.

All probe skips and full-vocabulary statuses (including `cap_skip`) are counted
by response origin/version. A full-vocabulary cap does not exclude accepted token
metrics. Missing probes exclude that entire question from the four-version common
comparison but retain its individual accepted records and unpaired new responses.
No actual data has been analyzed by the fixture tests.
