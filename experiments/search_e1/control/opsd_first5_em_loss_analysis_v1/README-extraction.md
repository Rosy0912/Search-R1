# First five online OPSD outers: native EM/loss extraction

`extract.py` is a standalone, stdlib-only, read-only CPU extractor. It does not
modify the trainer, import Torch, instantiate a model/tokenizer, run inference,
read optimizer tensors, issue HTTP requests, or write a Node file. Root must
publish the exact source and verify the remote Git bytes before execution.
Outputs include private question IDs and source paths: keep the NDJSON out of Git.

## Invocation

Feed the published script as stdin to the original HKUST1 environment Python:

```text
python -B - --request-json '<exact compact JSON>' --request-sha256 <SHA256-of-UTF8-argument>
```

Root sets `CUDA_VISIBLE_DEVICES` empty, numeric CPU thread counts to one, and
`nice 19`. No deployment or GPU queue lock is required; analysis never delays
the running seven-group queue. `--request-json` must stay below Linux's single
argument limit; the current explicit request is about 26 KiB.

Request schema: `opsd-first5-em-loss-extraction-request-v1`. Required keys:

- `facts_reference`: root's already accepted original facts `{path, sha256,
  bytes}`. The Mac path is provenance; the Node does **not** reread that file.
- `groups`: exactly ordered `C0`, `C1`, `M`, each with `group`, SHA-bound native
  `config`, and five `outers` (0–4).
- Each outer: `variant`, `outer`, `status=COMPLETE_FACTS`, `run_id`, `contract`,
  `counts` (`P`, `K`, `skipped`, `optimizer_updates`,
  `optimizer_updates_before`, `optimizer_updates_delta`, `valid_tokens`),
  `sampling.strict_reward_hits`, and `sources` SHA-bound references for
  `completion`, `checkpoint`, `outer_manifest`, `encoding`, `order`.

The script rereads those immutable native documents, then follows only the
outer-end checkpoint's `outer_update_receipts` and their committed token-file
references. It verifies complete fresh SHA256 and stable metadata for every
read JSON/JSONL/source file. Failed/uncommitted retry token files are excluded.
The 12-file source100 manifest is reread and its identity checked; large weight
payloads are not rehashed. Frozen core and authored-final QA parser source bytes
are SHA-checked and compiled as pure scoring functions. No model import occurs.

## Output and acceptance

Stdout is private NDJSON with `record_kind`:

- `pair_feature`: one `opsd-first5-pair-feature-v1` per actual usable pair.
- `cohort_feature`: one `opsd-first5-cohort-feature-v1` per sampled question;
  exactly `3 × 5 × 512 = 7,680`, including all-five-wrong questions.
- `outer_proof`: 15 immutable native count/update/token reconstruction proofs.
- `summary`: one final accepted result with exact row/update/token counts,
  per-kind output hashes, complete input read proofs, and scope flags.

Partial stdout is not accepted. Root must preserve stderr and actual process
return code and require rc0 plus the terminal accepted summary; root then
splits records without changing their serialized bytes.

Each pair binds question ID, data source, group/outer, exact original B/D rows,
encoded identity, current pre-update adapter, outer-old adapter, config,
source/runtime manifest, update receipt and token SHA. `optimizer_update` is
the true cumulative Adam count. `normalized_update_progress` is
`(local_update_number − 1) / max(outer_update_count − 1, 1)` and labels the
pre-update point within the outer, not an interpolated checkpoint curve.

`selected_student_strict_em` is the original selected D's strict score.
`siblings_success_count=s`, `sampling_success_fraction=s/5`, and
`sampling_pass_at5=int(s>0)` come from all five original rollouts. The fraction
is a five-sample estimate, not the true success probability. The extractor
calls the frozen strict scorer on original authored spans and checks recorded
diagnostics. C0/C1/M's selected D is correct iff all five siblings are correct;
this rule is explicitly verified and is a selection confound, not a new
finding that independent model confidence perfectly predicts correctness.

All-wrong questions have `loss_available=false`,
`loss_missing_reason=all_five_wrong`, and no invented zero loss. Eligible pairs
excluded by native context/mask encoding have `native_encoding_skip`. Sampling
EM labels use outer-start autonomous weights; training loss/logp use the
particular batch's current pre-update weights on fixed D history. They are not
generation-time processed scores. The heldout512 cohort is separate; this
extractor does not join TRAIN loss to heldout per-question EM.

## Token metrics and regions

Regions: `all_policy`, `think_text`, `search_query`, `structural_tags`,
`final_answer`, `other`, `boundary`. `character_regions`/`category` reuse the
pure textual logic of `opsd_token_region_analysis/summarize_regions.py`.
Classification does not redefine protocol validity. The original frozen
`response_encoding` mask is reconstructed from header character offsets,
original authored spans, and information spans. Prompt/reference/observations
are conditioning text and have no training loss rows. `<information>` tags,
payloads and trailing unclosed information are excluded. Tokens crossing
regions go to `boundary`. `token_text` is single-token decoding and is never
required to equal the original character slice or concatenated to reconstruct D.

Each region stores valid count, `sums`, `means`, per-trajectory contributions
(sum divided by **all** trajectory valid tokens), and actual-batch
contributions (weight times sum divided by trajectory tokens and actual
global pair count). Empty regions have sum/contribution zero but mean,
geometric probability and hit fraction null. This preserves absence rather
than inventing probability one.

Exact metric keys include `teacher_kl`, `outer_old_kl`, `mixed_kl`,
`target_raw_kl`, `target_post_loss`, `clipped_vocabulary_count`,
`clipped_vocabulary_fraction`, `clipped_student_probability_mass`,
`clipped_teacher_probability_mass`, `clipped_target_probability_mass`,
`student_entropy`, `teacher_entropy`, `outer_old_entropy`, `target_entropy`,
four `*_observed_logp`, `student_teacher_observed_logp_gap`, and
`clip_loss_removed`. The four distributions are student/teacher/outer_old/target.

Observed D token log probabilities are summed and averaged within each region;
`probability_geometric_means` is exp(mean logp), including the final-answer
payload. It is length-normalized fixed-history confidence, not probability of
a correct autonomous final answer. Final-answer absence stays null.

M target is the **arithmetic probability mixture**
`0.1 teacher + 0.9 outer-old`, followed by log; it is not a weighted sum of KL
losses. `target_raw_kl`/`mixed_kl` are that target divergence. `teacher_kl` is the
separately comparable teacher divergence. Clip masses use the actual target's
trigger set. The logged upper clip10 is per signed vocabulary contribution;
negative terms are not lower-clipped. Native token→trajectory→actual batch
means and region reconstruction use rel/abs tolerance2e-6, not a claim of
recomputed logit equality.

Full vocabulary snapshots/top32 are not required for this all-token scalar
analysis. Auxiliary complete/cap-skip rows are counted; no NPY payload or
logit is reread/recomputed. No per-token parameter gradients are available;
region loss concentration and outcome relationships remain associations.

Cohort `question_semantic_row_sha256` hashes exactly the canonical four original fields `question_id`, `question`, `answers`, and `data_source`, excluding other sampling metadata. It supports an actual source-contract and row-identity cross-check against heldout records. A question ID prefix alone is not an overlap check.
