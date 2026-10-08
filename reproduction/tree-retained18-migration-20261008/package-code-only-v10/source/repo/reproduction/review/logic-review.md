# Search-R1 independent logic review

Reviewed 2026-10-02. Baseline: upstream commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3`. Scope: rollout, GRPO grouping, rewards, observation masking, validation and local launch/preprocessing adaptation. This review establishes specific properties and defects; it does not certify that all bugs are absent or that paper scores have been reproduced.

## Release assessment

The original adaptation was a runnable **NQ quickstart** rather than a faithful paper experiment. The integrated fork now contains explicit corrections and a separate paper-GRPO recipe. Final independent static/CPU review found no remaining blocking defect in the reviewed data/grouping/budget/masking/reward/validation paths. Remote mixed-data short training and eventual complete benchmark scores remain separate acceptance requirements; a passed code review does not imply paper performance has already been reproduced.

## Findings

### P1: validation sampling flags are discarded (upstream; minimally patched)

- `verl/trainer/ppo/ray_trainer.py:503` constructs validation metadata with `do_sample=False`, `validate=True`, `recompute_log_prob=False`.
- `search_r1/llm_agent/generation.py:243` and `:286` construct active batches from tensors only. The GPU padding constructor at original `:198` has the same omission.
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py:162` falls back to `do_sample=True` when metadata is absent. Therefore validation uses training sampling, rather than its explicitly requested greedy path. `fsdp_workers.py:435` also falls back to unnecessary reference-policy log-probability recomputation.
- The minimal patch copies `meta_info` at those three batch constructors. It preserves training sampling and avoids sharing the metadata dictionary with workers.
- Regression: actual CPU `run_llm_loop` methods with a recording worker exercise normal and final generations, 1/2 nominal GPUs, padded and divisible batches, and training/validation flags. The fixed source passes two test methods (six subcases); the pinned source produces five failing subcases. Files: upstream `tests/reproduction/test_generation_metadata.py`, local `metadata-regression-before.txt`.
- Workstation coverage uses real Torch and TensorHelper, but a DataProto stand-in because tensordict is absent. The test automatically uses real DataProto when training dependencies are present. A remote CPU rerun is still required for integration acceptance.

### P1: validation silently drops the final partial batch (upstream; patched)

- `verl/trainer/ppo/ray_trainer.py:409-413`: validation DataLoader sets `drop_last=True`.
- Recorded NQ test size 3,610 with validation batch 256 evaluates only 3,584 examples, omitting 26. The denominator changes with batch size. This is not full-test EM.
- The patch changes only validation to `drop_last=False`, preserves training's `drop_last=True`, and adds `val/evaluated_count/{dataset}`. The rollout manager already pads generation inputs to nominal GPU divisibility and trims outputs. The regression executes the actual source DataLoader expression with the real Torch CPU loader: all 3,610 unique indices are evaluated and the last batch contains 26. Another test executes the actual source metric aggregation statements and verifies per-dataset counts/EM. A real remote validation-tail batch remains an integration acceptance requirement.

### P1: the quickstart recipe is not the paper recipe (upstream defaults and adaptation)

The [paper v5, sections 4.3 and B.2](https://arxiv.org/html/2503.09516v5) specifies mixed NQ/HotpotQA training, seven-dataset EM, 500 GRPO updates, group size 5, top-p 1, action budget 4, and top-3 Wikipedia/E5 retrieval. Its main table defaults to PPO; GRPO is a separate variant.

The reviewed launcher's `prepare_data.py` only prepares NQ; `train_full.sh:63` sets step threshold 1,005; `:66` sets `max_turns=2`; top-p is inherited as 0.95 from `ppo_trainer.yaml:71`. These faithfully follow much of the public quickstart, not the above experiment.

Keep this profile explicitly named quickstart. Add a distinct paper-GRPO profile with mixed training, seven-dataset evaluation, top-p 1 and the intended search budget. `max_turns` counts rounds that can call search; a final generation happens afterwards with `do_search=False`, so document this implementation detail rather than treating the value as the total number of generation calls.

The fork now adds `GenerationConfig.force_final_answer=True` to preserve the quickstart's extra final generation, propagating the root configuration into both training and validation. A paper profile uses `max_turns=4` and `force_final_answer=False`, so a trajectory still active after four actions receives no fifth generation. This minimal compatibility switch avoids changing archived quickstart behavior. CPU regressions execute the actual loop: four persistent search actions produce exactly four worker calls in the paper mode and five in the legacy mode. An early valid answer continues to terminate the trajectory through the existing action parser.

The existing loop starts updates at global step 1, increments, then checks `>= total_training_steps`. Threshold 1,005 permits 1,004 updates. The paper profile's threshold of 501 produces 500 updates under the unchanged loop. The actual optimizer uses a constant schedule after `int(0.285 * horizon)` warm-up steps; horizons 500 and 501 both give 142. A regression executes the real terminal loop and scheduler: there are exactly 500 updates, the final checkpoint is step 500, and all 500 learning rates are identical between those two horizons. No loop/scheduler rewrite is necessary for this specific profile.

### P1 boundary: merged datasets can accidentally share GRPO group IDs

- `rl_dataset.py:152` derives the row index from `extra_info.index`; `ray_trainer.py:744` sets group UID to this index.
- Current NQ preparation has one unique integer index per row and is safe. Separately preprocessing NQ and HotpotQA from index zero and concatenating them creates collisions. Rebalancing does not fix collisions; `core_algos.py:136` groups solely by UID.
- Assign globally unique integer IDs after concatenation, or dataset-qualified IDs. Assert uniqueness before repeating prompts; after `n_agent=5`, assert each logical prompt contributes exactly five group members regardless of ordering. This is a required safeguard for the forthcoming mixed-data profile, not a claim that the current NQ run already mixes groups.
- Final review of `prepare_paper_data.py` confirms that `index=len(output)` assigns global IDs across NQ and HotpotQA; dataset-local `source_index` is separate provenance. Its uniqueness assertions cover full and balanced mixed smoke training. A real-source core-algorithm regression confirms independent five-member groups, unchanged advantages after permutation, and finite zero advantages for the equal-reward group.

### P2: outcome reward can select answer markup from an observation (upstream; patched)

- `main_ppo.py:75` decodes prompt plus the entire interleaved response. `qa_em.py:73-82` chooses the last `<answer>...</answer>` match and requires at least two matches because the standard prompt contains an example.
- Real scorer reproduction: normal prompt plus model `<answer>Paris</answer>` yields 1; the same prompt plus a search observation containing `<answer>Paris</answer>`, followed only by another model search, also yields 1. There is no model-generated final answer in the second case. `logic-evidence.json` records both results.
- The standard prompt's example explains why a standalone response with one answer tag scores zero; that standalone API behavior is not evidence that normal current training rewards are all zero.
- The new `compute_score_generated_em` helper chooses the last complete answer wholly contained within one contiguous model-authored span. RewardManager derives spans from response tokens and `info_mask`; it excludes prompt/observations and never joins spans across observations. With no `info_mask`, the complete response is one model span. Empty responses receive zero without a negative reward index. The legacy `compute_score_em` standalone API and normalization/default scoring remain unchanged.
- Tests invoke the actual RewardManager methods with real Torch CPU: prompt-only markup and observation answers score zero, real generated answers and alias normalization score one, cross-observation stitched tags/content score zero, incomplete tags score zero, last-answer behavior is preserved, and empty responses are safe. This is an explicit fork correction to the reward input provenance, not an undisclosed upstream-equivalent reward change. The original evidence proves a parser weakness, not that the downloaded Wikipedia corpus contains this markup or that reward hacking has occurred in the run.

## Properties checked without a confirmed defect

- Sequential interaction: `generation.py:229-310` generates, parses an action, retrieves, appends observation to rolling context, then generates again. Search is an environment result rather than an independently optimized output.
- Grouping: `ray_trainer.py:702` repeats each question five times; group IDs survive reordering because they are stored in the same DataProto. `core_algos.py:111-156` sums outcome scores and normalizes within UID groups. Equal-reward groups produce zero advantages; this is expected.
- Loss mask: observations, including invalid-action feedback, receive padding-valued shadow tokens in `_info_masked_concatenate_with_padding`; final `info_mask` retains model tokens. `_create_loss_mask` slices the response section. `dp_actor.py:240-269` uses the same mask for PG, entropy and explicit KL. The full attention mask still retains observations for contextual conditioning. Nonzero observation advantages prior to actor masking therefore do not by themselves imply observation optimization.
- Reference: `fsdp_workers.py:480-499` uses the reference module on the same interleaved input, temperature-aligned log probabilities. The reviewed GRPO recipe uses explicit KL loss, not double KL shaping in rewards.
- Reward placement: `main_ppo.py:82` places the scalar EM at the last valid response token. GRPO sums token scores, so scalar placement does not change the group outcome under this recipe. No additional process reward or format bonus is enabled by the reviewed defaults.
- Data adaptation: `prepare_data.py` imports the actual pinned upstream `make_prefix`, preserves `golden_answers`, uses base-model prompts, and keeps train/test separate. Adding a final question mark matches the original preprocessor. No ground truth is inserted into the prompt.

## Server adaptation assessment

- Isolated environment, local pinned assets, offline HF caches and console logging change deployment rather than policy objective. NCCL NVLS disablement is a communication compatibility setting; it does not remove observation masking or alter GRPO groups.
- Retriever wrapper retains the upstream encoder/retrieval implementation and changes the HTTP bind address to localhost. CPU FAISS FP32 is a documented departure from the sample GPU/FP16 search path; rankings can differ around ties or small score gaps. Do not claim bitwise retrieval equivalence without a sampled top-k comparison. The same corpus/index/model plus top-3 remains the correct high-level retrieval setup.
- This review does not independently certify the remote cache contents, driver workaround, numerical checkpoint differences or final EM. The deployment reviewer and root agent own remote integration evidence.

## Publication acceptance checklist

1. Preserve upstream ancestry and attribution, with explicit patches and reproducible dependency/asset revisions.
2. Distinguish the archived quickstart from the paper-GRPO run; compare its profile to the paper and name GRPO results accordingly.
3. Pass metadata, validation coverage, unique grouping and ordinary/observation reward regressions using real remote dependencies.
4. Run a short real search rollout/update, verify masked loss behavior and checkpoint parameter changes, then initialize the full run from the original base model.
5. Report completed update count, checkpoint-selection rule and full seven-dataset EM; until these finish, state reproduction in progress.

Code source: [pinned Search-R1 repository](https://github.com/PeterGriffinJin/Search-R1/tree/598e61bd1d36895726d28a8d06b3a15bed19f5d3). Review evidence describes behavior at that commit; line positions shift after fixes.

## Post-fix CPU check

`python3 -m unittest discover -s tests/reproduction -p 'test_*.py' -v` passes 15 test methods on the workstation, including four final-checkpoint tests authored by the root agent. Workstation distributed dependencies remain absent, so RewardManager/DataProto tests explicitly use the documented minimal DataProto fallback. Use `SEARCH_R1_REQUIRE_REAL_DATAPROTO=1` in the remote environment to reject that fallback and require real integration imports. No GPU training was started by this reviewer.

## Final integration review

Re-reviewed `/Users/shaoshitong/Documents/Search-R1/repo`, `prepare_paper_data.py`, `train_paper_grpo.sh` and the reviewed mixed-data `train_smoke.sh` after integration. No active launch script was changed during this final review.

- Paper profile uses mixed NQ/HotpotQA training, all seven evaluation sources, group size five, top-p one, top-three retrieval, four actions without a fifth forced generation, and 500 updates. It is explicitly a **GRPO variant**; compare against the relevant GRPO results, not the default PPO main-table scores.
- Smoke uses 16 NQ plus 16 HotpotQA examples with original globally unique IDs, two updates, and nine validation examples. A newly added actual-loop CPU case proves that 8+1 input batches are sent to nominal eight-worker generation as 8+8 padded batches, return 8+1 outputs, and retain `do_sample=False`, `validate=True`, `recompute_log_prob=False` in both calls.
- Reward provenance and actor loss provenance share the same response-section `info_mask`: the former splits continuous model spans; the latter excludes observations from PG, explicit KL and entropy. A real-core-function CPU test verifies zero gradients at observation positions and nonzero gradients at model positions for all three finite objectives. Attention remains full-context for conditioning.
- The earlier 15 CPU tests have passed with real remote DataProto, as reported by the root agent. Four additional final-review tests (three core/schedule checks and the explicit 8+1 greedy-tail case) pass locally and await the root agent's unified remote CPU rerun.
- At the final local snapshot, all 19 logic/checkpoint tests pass; the combined suite, including the runtime reviewer's additional tests, runs 29 tests with one expected macOS skip for Linux procfs. `git diff --check` passes.

**Verdict:** the reviewed integrated recipe and fork corrections are reasonable for proceeding with the isolated mixed-data eight-GPU smoke and publishing the reproducible fork. Acceptance of full reproduction still requires the smoke's actual update/checkpoint/coverage evidence, then the complete 500-update GRPO run and seven-dataset EM. The review makes no absolute guarantee of bug-free code or numerical equivalence with the authors' run; CPU FP32 retrieval and disclosed upstream corrections remain recorded variations.
