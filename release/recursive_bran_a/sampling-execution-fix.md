# Sampling execution corrections

Implemented in an isolated source, not hot-patched into the running job.

* Cross-question action pipeline: retain exactly one pending branch per question, but refill completed questions without waiting for other trajectories to finish their full search loop. The shared RNG consumption order changes; the uniform selection law and per-question dependencies remain. No bitwise trajectory equivalence claim.
* Keep vLLM KV allocation for the same versioned sampling phase. Free it before HF/ref/training, policy change, or exception. Existing weight cache is retained. Ordinary non-opt-in lifecycle remains unchanged.
* `TREE_LINEAR_PREFIX_SCAN=1`: derive UTF8 boundaries from the already-bound ByteLevel decoder, then perform real native full/cut comparisons. Explicit proof records 1–2 native calls rather than claiming every prefix was checked. Historical proof/default path remains supported.
* `actor_rollout_ref.rollout.tree_action_tag_stop=true`: native string stopping on `</search>` and `</answer>`, include closing tag, detokenize required. Capture and validate string-stop identity and original emitted IDs/mu; no fabricated logprob. Raw/behavior-mu validators admit only these two bound strings. Native GPU end-to-end acceptance is still required before claiming this path works in production.
* Scheduled detailed rounds emit `TREE_ACTION_TIMING` after each action RPC including total wall/driver CPU, generation RPC time, tokens, and prefix scan delta. No question text or IDs are printed.

Evidence boundary: CPU KV lifecycle, scripted actual recursive executor, and native stop metadata tests passed. Lightweight ByteLevel tests passed; real HF fixture tests require the existing Node environment and have not run on Mac (transformers/tokenizers unavailable). No local environment/model download was performed. None of these tests establishes GPU throughput.

Corrected comparisons: old OPSD submits 320 requests per GPU but does not require all 320 full sequences to reside in memory. Current 64 requests does not itself prove a 5x slowdown. The referenced fast Tree run105 has recorded generation367.235s, 696656 emitted tokens, 362991 discarded tokens and prefix scan33.819s; thus discarded tail alone is not a sufficient explanation for the current slower run. Hardware and algorithm/call-work must be compared using actual segment measurements.

Deployment is pending a fresh reachable HKUST process/checkpoint boundary. SSH observation timeout must never trigger blind restart. Preserve current source, partials and all complete states; resume actual full model/Adam/scheduler/training RNG/cursor. Historical complete parent4 lacks independent vLLM sampling RNG; do not claim bitwise sampler continuation.

A second fast-Tree difference: its recorded launch uses GPU E5 / sharded FP16 FAISS; A keeps the approved18004 CPU FP32 E5 / full FP32 oracle. This is not a measured causal attribution. New counters separately measure actual HTTP+JSON time, interaction processing, generation RPC, and prefix scan, without new calls or retriever changes. The fast Tree also has per-action KV allocation and quadratic prefix checks, so those repairs must not be represented as the sole proven reason it is faster.
