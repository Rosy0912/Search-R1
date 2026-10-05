# Tree-GRPO PPO safeguards

The user approved the attached PPO三项防护实施与验收说明 on 2026-10-06. That specification is the design authorization; this work does not require a second permission to implement its three safeguards.

## Preserved experiment

Parent source 502e9fafa8e3aba722dbfcb493abc2b9d3ece326 and failed job3234 remain immutable. The candidate has its own worktree and later frozen source/run directory. Restore the best evaluated and complete independently accepted checkpoint110, not degraded128. Keep LR1e-6, train640, mini384, globalmicro16, DP8, ten optimizer calls per successful outer, one PPO epoch, original tree advantages/equal-leaf weights, temperatures, reward, lengths, splits, masks, loss denominators and reference KL unchanged. All assets and execution stay on h100-9.

## Three changes

1. Opt-in token Dual-Clip c3 with original epsilon. Mask old/current/advantages/weights before subtraction. Positive branches cap delta at log(1+epsilon); negative branches bound at log(1-epsilon)..log(c); zero advantage never evaluates unbounded exp. Original row denominator and leaf-token weights remain.
2. Fixed deterministic length-stratified monitor rows across the entire local outer, held unchanged before and immediately after every optimizer write. Use actual current actor forward and fixed old policy probabilities. Reduce k3 numerator/weight denominator and negative-tail numerator/negative denominator globally. Compute exact global negative p95/p99/max from gathered values, not average rank quantiles. Also log counts, weights, saturation and sample identities/coverage limits. Calibrate explicit KL/tail candidate thresholds on healthy checkpoint110; 0.01 is a candidate, not a guaranteed safe value. First-update probability consistency must be independently accepted. Reference KL remains a separate regularizer.
3. Check valid old/new probabilities, advantages, weights, and KL log bound before exponentiation. Every rank participates at the same ordered synchronization points; one-rank invalid data becomes a global refusal before backward or optimizer writes. Keep full gradient clipping and verify measured postclip norms and all local parameter/Adam floating tensors after writes.

## Refusal semantics

Current driver requires ten calls per full outer. Before a separately reviewed partial-update/fresh-rollout controller exists, reject with a synchronized controlled exit. Save diagnostic statistics, monitor indices, retained complete rollout paths, refusal reason and exact committed optimizer call count; do not increment outer or scheduler for a partial outer. A post-update brake cannot undo the committed step. No rollback or exact original-executor replay is claimed. Recovery uses a healthy accepted complete checkpoint in a new run.

## Validation and delivery

CPU: exact values and gradients versus direct Dual-Clip including boundary convention; extreme positive/negative delta and zero advantage; masked NaNs versus valid rejection; unchanged split objective; k3/tail/empty-sample rejection. Real Gloo and NCCL8 acceptance/refusal collectives, including one-rank NaN and post-write brake accounting. Healthy3B checkpoint110 probability consistency and a short GPU segment, actual ten writes/outer, saved complete trajectories, original fixed512 EM/F1 and action validity. Independently verify all-rank progress, full HF/tokenizer reload and complete state before long continuation. Record code diff/config/source SHA, threshold calibration and all historical costs separately. CPU-only success is never deployment or final300 success.

## Alternatives considered

Dual-Clip alone prevents loss overflow but cannot stop rollout reuse after drift; rejected. Global raw ratio clamping changes the PPO objective; rejected. The authorized combination of bounded equivalent Dual-Clip, sampled-old-policy checks, and synchronized numerical refusal is selected. An extra change, if evidence requires it, must have a separate diff, reason and validation.

Implementation clarification: monitor length selection, k3 and initial probability checks use response attention intersected with the existing loss mask, without mutating the original mask. Runtime initialization catches local validation/filesystem failures and collectively refuses before any monitor forward. NCCL checks and fixed monitor inputs use the current CUDA device even when rollout input tensors arrive on CPU. Functional attention dropout must be zero.
