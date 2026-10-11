# Recursive A acceleration handoff

Use branch `codex/recursive-bran-a-hkust1-20261011` in `shaoshitong/Search-R1`. A's local source is `/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/recursive-bran-experiment-a-20261011`; published HKUST1 source directories are under `/data/shaoshitong/recursive-bran-a-8gpu-20261011/`. Adapt these changes to B's own authenticated checkpoint and resource contract; A's pinned PID/run stop scripts are not B launchers.

## Changes and locations

| Change | Code | Enabled scope and preserved behavior |
| --- | --- | --- |
| Skip unused per-action HF scoring | `verl/workers/fsdp_workers.py: generate_sequences` | Native Tree emission with sampling-consistency diagnostic explicitly disabled. Preserve actual sampled token IDs/log-probs and raw probability provenance. |
| Skip complete old-actor HF beta | `verl/trainer/ppo/ray_trainer.py: prepare_behavior_mu_old_policy`, `RayPPOTrainer.fit`; `search_r1/baselines/policy_safety_runtime.py`; `verl/workers/actor/dp_actor.py: update_policy` | Explicit `RECURSIVE_A_OMIT_FULL_BETA=1` and guarded behavior-mu profile, with both reference KL coefficients zero. PPO denominator remains emitted FP32 mu. No fabricated beta tensor; full-beta comparisons are null. Current gradient forward/backward and fixed-subset safety forwards remain. |
| Explicit training temperature after scoring removal | `prepare_behavior_mu_old_policy` | Native mu validation proves plain temperature-one emissions. Carry temperature 1 explicitly; do not rely on the deleted HF pass to populate metadata. Reject conflicting temperature. |
| Reuse one vLLM weight version across the sampling phase | `verl/workers/sharding_manager/fsdp_vllm.py: prepare_policy_cache`, `__enter__`, `__exit__`, `release_model_weights`; `verl/workers/fsdp_workers.py: generate_sequences`, `release_rollout_policy_cache` | Version is actual committed optimizer count plus scheduler step. First call loads/synchronizes, subsequent unchanged-policy calls reuse. Release before HF/ref/training/checkpoint or on errors. Preserve per-call training/sampling CUDA RNG handoff. Validation without sampling follows the ordinary release path. KV caching still uses the original lifecycle. |
| Sparse evaluation, state save and detailed recording | `search_r1/baselines/recording_schedule.py`; `rollout.py`; `tree_rank_audit.py`; `verl/trainer/ppo/ray_trainer.py`; `runtime_audit.py` | Human schedule is 5/50/100/150/200 only. No every-outer full state or detailed tree dumps outside those points. Preserve internal executor state, selected prefixes, raw probability ledgers, essential scalar/error/optimizer logs. |
| Separate generation RPC and synchronization timing | `SearchRollout._advance`; `FSDPVLLMShardingManager.release_model_weights` | Generation RPC count/wall time and cache phase calls/sync count/wall time are actual counters. RPC includes synchronization/backend work; whole recursive rollout also includes retrieval, parsing, tree construction and recording. Neither is called pure decode time. |

The runtime opt-in environment variables must be propagated by `search_r1/baselines/runtime_audit.py: runtime_env_vars` into Ray workers. B should adapt the A-only activation checks to its own isolated profile, preserving default behavior for unrelated jobs.

## Evidence and limits

The cache has four targeted CPU lifecycle checks. `release/recursive_bran_a/test_mu_only_old_policy.py` has three real CPU contract checks: identical mu yields identical PPO scalar/gradient/one Adam state; safety still runs with full-beta comparisons absent; unsupported/missing/fabricated probability inputs refuse. The temperature check uses a synthetic batch with no inherited HF temperature metadata. These do not establish GPU speed or complete numerical equivalence.

A's complete fourth checkpoint has all eight ranks at Adam80. Cold continuation restores the actual model, Adam, scheduler, training Python/NumPy/Torch CPU/CUDA state and driver cursor. The independent vLLM manager `gen_random_states` was not present in that parent checkpoint; no bitwise sampling-restart claim is made. Never attach a mid-round sampling RNG to an earlier completed checkpoint. Future complete boundaries should preserve that independent sampling stream explicitly, as B has identified.

Published source and deployment are distinct from operational acceptance. A is switching from the completed fourth checkpoint with the metadata fix; no new complete outer or speedup is claimed until actually observed. B should resume its own complete boundary, preserve incomplete outputs, and report actual generation/reference/update timings after a healthy outer. There is no established 14-minute guarantee.
