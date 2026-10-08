# BAPO / Search-R1 baseline alignment candidate

This is an independent source profile based on 60f566b. It does not modify job3315, its source, checkpoints or counting. No new training job or retriever service has been submitted or restarted by this change.

## Implementation

1. Train, stage, debug512 and final evaluation share four interaction rounds followed by one final generation with retrieval disabled. Archive closing-tag truncation, retokenization, observation head crop, response head crop, rolling-context tail crop, pre-crop position IDs and original prompt padding are preserved.
2. Fixed reference is regularized in actor loss: `use_kl_loss=true`, coefficient0.001, `low_var_kl`. Reward-level KL stays zero.
3. `old_log_probs` remains the frozen pre-update actor HF recomputation. Native selected-token probabilities and original request/completion journals remain separate. No native probabilities are attached to retokenized training tokens.
4. Every complete outer retains2560 training trajectories; mini128, epoch1,20 actual Adam commits per rank throughout. Actor global micro16/rank2 and old/reference32/rank4 remain. No outer50 batch transition. A complete200 run requires4000 writes per rank; any numerical fault preserves partial writes and does not count a completed outer.
5. Ordinary clipped PPO epsilon0.2; Dual-Clip disabled. Finite-value/domain checks precede exponentiation without clipping finite log-ratios. Overflow/nonfinite loss, gradient or Adam candidate is a fault, never a successful write or a finite-policy SKIP.
6. HF drift and raw-sampling diagnostics do not control update admission. Full frozen-HF scans occur only at the outer boundaries. Bounded raw diagnostics use at most4 retained rows per rank, original raw contexts and actual native probabilities; cross-backend discrepancies include backend drift and are explicitly distinct from pure policy KL. Unsupported provenance is reported unavailable. No sampler/HF equality is assumed.
7. Prepared data is imported unchanged; ordered question identities, prompts, targets and file hashes are pinned. Actual exported baseline batch order is replayed with a recoverable cursor and checked before every rollout. Seed42 alone is insufficient evidence. Stage64 is pinned separately: train exclusion must already match the common prepared data, or an explicitly training-derived stage is reported as such. No hidden resplitting or train/dev subsampling.
8. CPU parity fixtures cover identical actions, search requests, prompt/response tokens, masks and position IDs through the full final-generation chain. Greedy debug/final evaluation shares this flow.
9. Shared `qa_protocol.py` is byte-identical to Search-E1. BAPO scores the terminal model-authored answer; no extra whole-chain invalidation. BAPO F1, IDK bonus, early/late stage control, prescribed invalid-answer penalty, resampling and group advantages remain.
10. Aligned preflight requires a pinned retriever manifest, live local `/identity`, inspected CUDA encoder and GPU FAISS, matching E5/corpus/index assets and golden ordered top3 results/format/truncated observation tokens. The identity endpoint is opt-in and read-only. No A800 or cross-node retrieval.

The archive objective also revealed a non-BAPO reduction mismatch: aligned actor uses the archive microbatch masked-token mean divided by accumulation count, rather than sequence-mean weighting. An unequal-answer-length gradient oracle checks this behavior.

## Entry and inputs

`reproduction/baseline-aligned/run.sh` is a candidate entry for an independently allocated and supervised node. It requires explicit official model, common prepared data/order/retrieval manifests and their independently obtained hashes, loopback GPU retrieval, and the task's real UUID lease/resource supervisor. It refuses a dirty source and prior run artifacts and claims a new output directory exclusively. It does not submit Slurm or perform resume. Existing resource-watch deployment is an external prerequisite; old source-bound receipts must not authorize this new source.

`reproduction/baseline-aligned/prepare_inputs.py` imports existing shared parquet files and the baseline's actual exported epoch indices. It never regenerates train/dev splits. Retriever identity manifest/probe format is documented in `reproduction/baselines/baseline-aligned-retrieval.md`.

## Scope of validation

CPU tests and independent source reviews establish code behavior only. They do not establish target Linux/CUDA, real eight-rank throughput, current retriever parity, completed iterations or experimental effectiveness. Those require a separately authorized new run with its own fixed-source acceptance. Existing experiment3315 retains its original configuration and supervision.

Final frozen-source combined regression: **189 passed**, seven library/config warnings, 25.13s. Independent cross-component reviewers accepted the source/CPU scope after closing all must-fix findings. Full output is in `validation.log`; this is not target GPU or training acceptance.
