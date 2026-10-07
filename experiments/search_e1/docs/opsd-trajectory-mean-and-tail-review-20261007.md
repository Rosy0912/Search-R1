# Source100 OPSD trajectory mean and final batch review

This read-only review found no normalization or final-batch collective mismatch in the frozen preparation-3 OPSD implementation. It does not constitute the real GPU/NCCL five-update acceptance, which remains pending. No remote source, configuration, process or GPU workload was changed.

## Actual configuration and loss

The root is `/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007`. The preparation-3 pipeline explicitly launches eight ranks with accumulation4, so the global optimizer batch is32. The trainer default accumulation8 is not the effective argument. Frozen `search_e1/opsd.py` SHA256 is `61df9cc630bca7466f3edfb06da18ad92ddff1c713bf7c6c4a104ba88672c037`.

`encoded_pair` filters empty responses, zero valid policy masks and over-limit student/teacher contexts; an empty usable dataset fails explicitly. `pair_loss` sums KL over valid policy-token positions and divides by that trajectory's number of valid positions. Each trajectory then contributes its loss divided by the actual global batch length. Gradients are manually summed across ranks. This implements the user's requested mean within each trajectory followed by a mean across trajectories; it does not pool tokens from different trajectories.

The global batch slice preserves the final remainder without padding, duplication or dropping. Its denominator is the actual remainder, not32. Every batch clears gradients. Thus one update uses `(1/m) sum_j gradient(L_j)`, where m is32 or the final remainder. All ranks perform ceil(usable_pairs/32) optimizer steps. The final usable pair count is not yet known because full generation/mining/encoding have not completed.

## Empty ranks and initialization

The model has no DDP/FSDP wrapper in this OPSD implementation. Per-trajectory forward/backward has no distributed collective. The rank slice may be empty when the final remainder is less than8; empty ranks create zero gradients and join the same parameter-order SUM, MAX checks, optimizer step and audit/save collectives outside the local loop. Unequal backward counts therefore do not imply a collective mismatch here.

All ranks load the same archive100 path explicitly as BF16 and seed Torch and Python with42 before model/LoRA construction. The actual Torch2.4.0+cu121 seed call seeds CUDA as well; PEFT0.13.2 defaults to random Kaiming LoRA_A and zero LoRA_B. Base attention dropout and LoRA dropout are0. There is no rank-offset training seed or rank-specific initialization branch. The rollout worker seeds42–49 are a separate generation setting.

Code and installed dependency inspection support identical initialization followed by identical summed gradients and Adam updates. This is an inference from those code paths, not a measured all-eight-rank initial parameter hash comparison. The external five-update check's exact Adam-state equality and rank0 adapter evidence must not be represented as such an initial parameter comparison. No additional runtime change was made for this review.

## Archive comparison and CPU verification

`encoded_pair` and `pair_loss` are byte-identical to the retained compressed archive implementation. Its original update loop uses the same actual-batch divisor, remainder slicing, zero gradients on empty ranks, manual SUM, clipping and Adam step. Added finite-value checks and update evidence do not alter that normal finite path.

A CPU reference check extracts the actual frozen global-batch update loop and runs eight CPU model copies with real float64 gradients and AdamW, using a blocking SUM/MAX collective simulator. N=32 and N=32+r for all r=1..31 are compared with serial actual-batch trajectory-mean AdamW updates. All32 cases give maximum weight difference0.0 and equal rank step counts, including empty ranks. Audit/save collectives were separately inspected statically; this test does not execute real token KL, GPU kernels or NCCL.

[CPU evidence](receipts/opsd-tail-cpu-check-20261007.json), SHA256 `5bfbd194f5c1a93759c59eb9e9294d1e80ad7e7db6b658605680a81bb4903b1a`.

The full independent report and retained source/check files remain locally at `/Users/shaoshitong/Documents/Codex/2026-10-03/xia/analysis-output/opsd-tail-review-20261007/`; report SHA256 `ae621ccd1a2687e0f020b314fbee22fdaf22553ea720475471dd2cecddda6215`. CPU proof is not a substitute for actual source100 plus five finite, changed-weight OPSD updates.
