# Search-G1 baseline review against the reviewed Search-R1 fork

Date: 2026-10-02. Read-only review of the user-provided extracted repository `/Users/shaoshitong/Documents/Search-G1/repo`; no training, process, code or Git mutation was performed. The provided baseline is **Search-G1**, attributed by its README to Rosy0912/Search-G1, and the user subsequently confirmed that identity. It must not be published as Search-E1.

The [Search-G1 paper, arXiv 2608.07531v1](https://arxiv.org/html/2608.07531v1) describes correctness-first representation shaping, masked GRPO and periodic recalibration. This is a different objective from Search-R1's outcome-only reward. The audit below evaluates the actual supplied implementation; paper/README aggregate scores are not independently reproduced results.

## Verdict

The baseline has sensible method-specific engineering and a recognizable masked Search-R1/GRPO foundation, but the archived code contains concrete evaluation and checkpoint defects. Preserve the original ZIP/source attribution, publish corrections separately, and require regression plus integration evidence before using it as a strict comparator. Its larger shaped reward values are not evidence that our EM-only Search-R1 reward is wrong.

## Reward comparability

`recipes/train_search_g1.sh:27-33` selects the probe branch, necessity gating, grounding coefficient 0.6, direct-answer bonus 0.5, repeated-marker cost 0.02, and wrong-grounding scale **0**. `main_ppo_eco.py:271-315` implements:

- Correct searched: `1 + 0.6 * (1-boundary) * flip - 0.02 * bounded_cost`, ordinarily 0.98 to 1.6.
- Correct direct: `1 + 0.5 * boundary`, ordinarily 1 to 1.5.
- Valid wrong answer: 0.05 under the provided recipe; invalid/no answer: -0.1.
- Validation/evaluation uses plain EM, separately from the shaped training reward (`main_ppo_eco.py:438-444`).

These bounds assume ordinary finite probe probabilities in [0,1] and default optional base/cost settings. They are not enforced for arbitrary environment overrides. Executing the actual `_score_probe` method with controlled CPU inputs produced 1.6 and 0.98 for two correct searched cases, and 0.05 for both wrong cases despite different probe predictions. Therefore the default recipe restores credit variation in tied **all-correct** groups, but not in uniformly valid **all-wrong** groups. Some comments describing all-wrong variance refer to the nonzero class default that the recipe explicitly overrides.

For comparisons, log G1 shaped reward and both methods' standalone full-set EM separately. Use a shared model, data split, retrieval assets, action budget, checkpoint-selection rule and evaluation parser. Retain probe fitting/refresh/scoring overhead in resource accounting. Do not copy G1 shaping into the paper-R1 profile: that would create a different method.

## Confirmed defects and consequential boundaries

### P1: validation omits the tail

`verl/trainer/ppo/ray_trainer.py:409-413` uses `drop_last=True` for validation. The default recipe selects 200 validation examples and batch size 64: the actual DataLoader source expression, executed on Torch CPU, evaluates **192 of 200**. Final evaluation with batch 32 likewise drops examples whenever dataset size is not divisible by 32; datasets smaller than one batch fail the loader-size assertion.

Recommendation: keep training drop-last unchanged, retain all validation examples, log evaluated counts per dataset, and test a non-divisible tail. Our reviewed R1 fork already makes this correction.

### P1: greedy metadata fix is incomplete

`generation.py:246-251` and `:291-296` copy metadata into active batches, which is a sensible improvement over original Search-R1. However, `_generate_with_gpu_padding` at `:198` constructs the padded batch without metadata. Executing that actual method on CPU with three active samples and nominal two GPUs yielded an empty metadata dictionary at the worker, despite the input containing `do_sample=False`, `validate=True`, and `recompute_log_prob=False`.

Thus multi-GPU training-time validation can switch back to sampling when the number of still-active trajectories is not divisible by the GPU count. Standalone `evaluate.sh` defaults to one GPU and temperature zero, so that default path does not exhibit this particular padding failure. Recommendation: copy metadata at the padding constructor too, then test both active filtering and tail padding. Our R1 fork covers all three constructors.

### P1: outcome parsing can reward an answer authored by the retriever

`main_ppo_eco.py:369-379` and `main_ppo.py:87-99` parse the concatenated prompt plus interleaved response. `qa_em.py:73-82` selects the last answer-tag match and relies on the prompt's example to create the first match. It does not consult `info_mask`.

Actual scorer reproduction: the normal prompt example followed by model search, an observation containing `<answer>Paris</answer>`, and another model search receives **EM 1**, although the model never emitted a final answer. This is a demonstrated parser weakness, not evidence that the real corpus contains such markup or that the trained agent exploited it.

Recommendation: score only complete answer tags within continuous model-authored spans; exclude prompt examples and observations without joining spans across observation boundaries. Apply identical provenance handling to EM, G1 outcome classification and counterfactual requests. Our R1 fork has the span-based correction; preserving G1's shaped formula alone will not fix this issue.

### P1: default final evaluation has no corresponding final checkpoint

`train_search_g1.sh:13-15` sets threshold 250 and save/test interval 25. The unchanged trainer starts at update 1 and stops after increment reaches 250 (`ray_trainer.py:852-862`), so it performs **249 updates**. Saves occur at 25 through **225**, and the final policy after update 249 is evaluated without being saved. README's example `actor/global_step_250` is therefore not produced by the unmodified default recipe.

Recommendation: explicitly choose whether the intended update count is 249 or 250, align the threshold, and save the exact final updated model before final evaluation. Our R1 fork saves the terminal policy and uses threshold 501 to obtain 500 updates.

### P2: Search/Q and search cost count observation markup too

`main_ppo_eco.py:378-381` applies `_SEARCH_RE` to the complete response string, which includes observations. It excludes the initial prompt's markers but not retrieved content. A controlled response with two model searches and one observation-side `<search>` was counted as **three** markers. This is also not a physical retrieval-call counter.

Recommendation: count model-authored parsed actions or use trusted rollout `valid_search_stats`; report executed retrieval calls and token/latency costs separately. This instrumentation is useful for R1 but must not add a search-cost reward to its paper objective.

### P2: probe response validity is not checked before training continues

`main_ppo_eco.py:243-258` retries transport failures, correctly avoiding an intentional outage fallback. However, it accepts `results` without checking cardinality/finiteness/range; `:424-428` maps whichever rows arrive, and `_score_probe` treats an absent row as `flip=boundary=0`. An HTTP-success response containing an empty or short result list can silently remove representation shaping for affected trajectories. The actual `_score_probe(..., None)` method returns 1.0 for a correct single-search row rather than aborting.

Recommendation: reject missing, misaligned or nonfinite readouts before constructing any rewards. Keep the latest complete manifest or pause/fail explicitly; record its version per batch. The provided server normally returns one row per request, so this is a robustness gap rather than evidence that its normal response is truncated.

### Conditional P1: cross-file question grouping can leak calibration examples

`ts_probe_train.py:28-48` adds a new `gid_off` for every dump file. This correctly separates different datasets/files, but if multiple dumps revisit the same questions at different checkpoints or overlapping offsets, those repeated questions get different group IDs. `GroupShuffleSplit` can then place the same underlying question in both folds. Dumps store local `qi` rather than stable original dataset/question IDs (`ts_probe_dump.py:302-318`).

Single-file, single-round calibration is grouped correctly. The issue appears when combining repeated-question files, not automatically in the documented one-round recipe. Preserve stable dataset-qualified question identities across checkpoints/offsets, or prohibit such combinations. Also separate estimator-selection CV from an untouched final fidelity assessment; choosing the best layer/model on the same CV scores makes that selected score an optimization estimate, not an independent final estimate.

### Conditional P1: merged policy datasets need global group IDs

`scripts/prepare_qa.py:94` assigns dataset-local integer indices, while `ray_trainer.py:754` uses index alone as GRPO UID. The README example trains on one NQ parquet, which is safe. Concatenating separately prepared NQ and HotpotQA files without reindexing merges unrelated questions with equal IDs. Our paper-R1 preprocessor assigns global unique IDs; apply the same safeguard if G1's training scope expands.

## Sound choices worth retaining

- `scripts/prepare_qa.py:105-158` rejects overlapping calibration/validation ranges and duplicate destinations, writes them separately, and excludes their row indices from policy training. README explicitly chooses `--split train` for those holds and `--split test` for final evaluation. The `auto` split default prefers test and should not be relied on for training. Row-index separation does not prove normalized question-text deduplication across independently supplied datasets.
- `ts_probe_train.py:64-85` uses GroupShuffleSplit within a correctly identified question group; multiple sampled trajectories from one question do not randomly leak across these folds.
- Observation shadow-mask construction and actor `dp_actor.py:240-269` mask PG, explicit KL and entropy while retaining retrieved context in attention. This matches the appropriate Search-R1 foundation.
- Probe servers and refresh manifests separate measurement from optimization, and the documented interface can keep the last completed readout during refresh. Calibration deletes evidence while retaining reasoning; its signal is an answer-stage sensitivity proxy, not a proof of total causal retrieval dependence.
- Validation plainly reports EM rather than training shaping. Conditional trust metrics and response-marker costs are separate, method-specific diagnostics.

## Fair-comparison decisions

The supplied G1 recipe uses a smaller batch (128), default four GPUs, a 250 threshold, sample top-p inherited as 0.95, instruct models in README, and `max_turns=4` **plus** an extra final generation. The calibration rollout also uses five potential generations and samples with top-p 1 (`ts_probe_dump.py:106-138`). These are not our R1 paper-GRPO settings (base model, batch 512, eight GPUs, 500 updates, top-p 1, four total action rounds).

Choose one of two honest comparisons: reproduce each method's own published recipe with fully disclosed differences, or run a separate common-budget ablation with identical model/data/interface/training budgets. Do not silently redefine either paper reproduction to make plots look similar. Our current R1 scope should remain outcome-only; hold G1-specific shaping, calibration data exclusion and readout fitting in its own branch/experiment.

Publish the source archive identity and attribution separately from the reviewed corrections. Before training corrected G1, require regression tests for tail coverage, metadata propagation, reward provenance, malformed probe outputs, unique group IDs and exact saved-policy evaluation. A high shaped reward or an aggregate README table cannot substitute for those checks.
