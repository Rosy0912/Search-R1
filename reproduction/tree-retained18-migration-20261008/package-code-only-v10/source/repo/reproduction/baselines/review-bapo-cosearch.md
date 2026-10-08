# BAPO / CoSearch implementation review

Reviewed 2026-10-02 in `/Users/shaoshitong/Documents/Search-R1-RL-Baselines`, branch `codex/search-r1-rl-baselines`, against reviewed Search-R1 base `b251ba6b503217a59cd3395c1326d7b66b602989`. This report covers the working-tree implementation, not a released commit. The reviewer read code and ran CPU tests; no implementation edits or GPU jobs were performed.

## Outcome

The five concrete findings from the first two review rounds are fixed in the current working tree. No remaining blocking defect was identified in the reviewed BAPO/CoSearch CPU paths. This is acceptance of an independent Search-R1 adaptation and its CPU evidence, **not** verification of original-paper result reproduction or cluster execution.

## Findings and fixes

| Round / severity | Finding | Current fix and evidence |
|---|---|---|
| 1 / P1 | BAPO plateau gate enabled the IDK bonus for high-diversity groups, reversing the paper. | `search_r1/baselines/bapo.py::boundary_scores` now gates on diversity `< G/2`, including normalized IDK among answer values. The low-diversity and high-diversity counterexample test passes; a group with any positive base correctness gets no bonus. |
| 1 / P1 | BAPO and CoSearch considered the presence of an answer tag sufficient for format validity. Malformed/recovered trajectories could receive positive reward, and CoSearch could update the ranker from them. | `legal_action_records` validates every recorded model action, enforces one complete action, checks its prefix/suffix, nonempty contents, and terminal answer placement. Reward/resampling and CoSearch `main_f1_reward`/`after_actor` use `decode_answers(..., require_legal=True)`. Tests reject recovered invalid actions and multiple answer tags. Invalid main trajectories are filtered from ranker updates. |
| 1 / P2 | Independent BAPO stage holdout could be bypassed by expressing the final-test path as a list or alias. | `controller.py::canonical_paths` compares resolved path sets. BAPO controller additionally compares serialized prompt keys against both training and final-test datasets. Required `stage_validation_files` is unset by default and fails fast; no stage holdout has been selected and no BAPO training was authorized by that missing choice. |
| 1 / P2 | CoSearch invalid ranking used `-1`, inconsistent with the selected paper format penalty. | `cosearch.py::ranker_reward` now uses `format_penalty=-0.2`; the parser rejects duplicates, noninteger IDs, IDs outside candidate bounds, wrong length, and malformed ranking structure. The invalid-ranking penalty test passes. |
| 2 / P1 | Ranker generation inherited checkpoint repetition processors, while old/new PPO log probabilities were computed from raw temperature-scaled logits. Official Qwen2.5 Instruct checkpoints specify repetition penalty 1.05. | `GenerativeRanker.rank` constructs a fresh `transformers.GenerationConfig` with temperature sampling, top-p 1, top-k 0, one beam, and required token IDs/EOS. It no longer inherits repetition/no-repeat/forced-token processors. The tiny HF ranker test deliberately sets checkpoint repetition penalty 1.05 and asserts the actual supplied config has penalty 1 and no-repeat size 0. Rollout scoring and optimization both use eval-mode logits divided by the same configured temperature. |

The Qwen checkpoint configuration supporting the second-round finding is [the official Qwen2.5-7B-Instruct generation_config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/generation_config.json); the 3B Instruct configuration was also checked and has the same repetition penalty.

## Source fidelity and deliberate adaptation choices

BAPO primary references:

- [Paper v2, method section](https://arxiv.org/html/2601.11037v2#S4).
- Official source pinned to `8ddbb4fa23f105d52d3621d232f839212e895c3f`: [group reward scorer](https://github.com/Liushiyu-0709/BAPO-Reliable-Search/blob/8ddbb4fa23f105d52d3621d232f839212e895c3f/RL_train/src/verl/verl/utils/reward_score/re_search_faith.py), [stage manager](https://github.com/Liushiyu-0709/BAPO-Reliable-Search/blob/8ddbb4fa23f105d52d3621d232f839212e895c3f/RL_train/src/verl/verl/workers/reward_manager/re_search_faithful.py), [training dispatch](https://github.com/Liushiyu-0709/BAPO-Reliable-Search/blob/8ddbb4fa23f105d52d3621d232f839212e895c3f/RL_train/src/verl/verl/trainer/ppo/ray_trainer.py#L873-L875).

This implementation intentionally chooses paper semantics: early batch IDK ratio gate, low-diversity plateau gate scaled with actual group size, consecutive patience with ties counted as nonimprovement. It does not copy the source's hard-coded group of eight, exclusion of refusals from diversity, or non-resetting strict-below-best counter. Search-R1 keeps its own tool protocol and model-authored action spans instead of the original `<result>`/boxed-answer parser. Empty-or-one-think prefixes are accepted to preserve Search-R1's action format; this parser is an adaptation rather than the author's exact format validator. Group reward and complete-trajectory resampling are integrated on the driver, before old logprob computation and group-breaking batch balancing. Independent stage validation keeps base QA reward; stage selection does not consume the seven final benchmarks. BAPO changes reward/sampling and is not a loss-only replacement.

CoSearch primary references:

- [Paper v2, method section](https://arxiv.org/html/2604.17555v2#S3).
- Official source pinned to `763bf8c4d7caa363ad845d39ddfd53b81ae377bd`: [dual-agent trainer](https://github.com/snap-research/CoSearch/blob/763bf8c4d7caa363ad845d39ddfd53b81ae377bd/verl/verl/trainer/ppo/dual_agent_ray_trainer.py), [ranker score assignment](https://github.com/snap-research/CoSearch/blob/763bf8c4d7caa363ad845d39ddfd53b81ae377bd/verl/verl/experimental/agent_loop/score_assign_functions.py), [grouping](https://github.com/snap-research/CoSearch/blob/763bf8c4d7caa363ad845d39ddfd53b81ae377bd/verl/verl/experimental/agent_loop/uid_group_functions.py), [official training recipe](https://github.com/snap-research/CoSearch/blob/763bf8c4d7caa363ad845d39ddfd53b81ae377bd/scripts/train_co_search_grpo.sh).

This implementation selects paper-style continuous main F1, composite reward threshold 0.5, token-F1 semantic grouping without the official code's additional answer-availability split, and minimum group size three. It does not adopt the official recipe's main-reward binarization or conditional threshold 0.8. Main-format-invalid trajectories are excluded from ranker optimization. Gold answers are consulted only after rollout for reward; the reviewed ranker prompt contains initial context, issued query, candidate documents, and ranking instructions, not gold labels. Main observation tokens remain excluded from policy loss, and ranker inputs are excluded from its response-token objective.

## Backend and experiment limitations

- CoSearch's portable backend is an actual **unsharded Hugging Face ranker with a separate AdamW optimizer**, default CPU float32, not the authors' distributed dual-Ray/FSDP/reference-worker system. It has no ranker reference-policy KL term. The backend, precision, regularization, batching, ranking prompt, and resources must be listed in results; a paper-equivalent distributed backend has not been validated.
- The primary Search-R1 actor/model/data/retriever assets remain fixed. CoSearch nevertheless adds an auxiliary model, increases dense candidate retrieval to 50, and changes observations by ranking. It is a joint-system extension, not a policy-only control. Returned documents stay at actor top-k=3, whereas the original paper uses 5; consequently Hit@5 sees only the selected three documents. This is a controlled-task adaptation and must not be presented as the author's exact retrieval setup.
- Failed ranker generation falls back to deterministic dense top-k so the search rollout can continue; the invalid ranker action still receives its format penalty. This fallback is an explicit environment choice.
- Stage/ranker artifacts are saved, and ranker optimizer state is saved, but `resume_supported=False` remains declared: a complete joint-state reload and bitwise continuation path has not been implemented or verified.
- Ranker generated/prefill tokens, score-forward tokens, training-forward tokens, optimizer updates, and separate validation generated/prefill tokens are tracked. These are operation/token counters, not FLOP measurements. Reused main trajectories do not make auxiliary ranker computation free; report GPU/CPU wall time and memory in the real run.
- The shared Search-R1 rolling-context/reconstructed-trajectory limitation is now disclosed in README and diagnosed without changing default behavior. A context mismatch can invalidate strict main-policy on-policy assumptions; neither this CPU review nor the clean auxiliary ranker sampling configuration certifies those assumptions for long trajectories. The optional error mode is a fail-fast guard, not a new algorithm.
- No GPU forward/backward, real Ray collective/dispatch, multi-GPU ranker, long-context 50-document workload, live retrieval server, or full end-to-end learning run was tested. CPU default is a portability path; no claim is made that a 7B float32 ranker plus optimizer is practical on this Mac.

## Verification performed

Command: `.venv-cpu/bin/python -m pytest tests/baselines -q`.

Final reviewer rerun after the shared-context diagnostic addition: **24 passed**. Existing urllib3/LibreSSL and Ray deprecation warnings were observed; no test failures.

Evidence scope:

- Analytical reward/gate/stage/format/grouping/mask counterexamples and real old-verl DataProto source selection.
- A real tiny HF causal model executes ranker forward scoring, backward gradients, and optimizer parameter updates on a nonzero GRPO group; invalid main trajectories do not add an optimizer step. Valid ranking text in this focused test is supplied by a scripted generation stub, so it is not evidence of a production ranker learning to output valid rankings. The stub also verifies the clean generation config actually passed to the model.
- Driver `fit()` transport tests use real DataProto with lightweight stand-ins for distributed workers/retrieval. These establish callback integration and shape/metadata transport, not real cluster worker execution or benchmark improvement.

No edits were made to the original Search-R1 agent's checkout, remote training process, corpus/index, checkpoints, or credentials by this review.

## Final README wording check

The final `reproduction/baselines/README.md` wording for BAPO reward/stage/holdout and CoSearch composite reward/grouping/backend/fallback was checked against the implementation and the scope above. No additional correction was required. The README clearly calls CoSearch a joint-system adaptation, retains the changed K=3 setting, requires an independent BAPO holdout, and limits PASS to CPU tests/source review. Its shared-context caveat appropriately avoids claiming strict on-policy equivalence for long trajectories.
