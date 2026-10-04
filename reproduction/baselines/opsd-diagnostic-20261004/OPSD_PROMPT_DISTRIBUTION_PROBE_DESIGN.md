# OPSD initial-state distribution diagnostic — local review candidate

This is a diagnostic, not training, search evaluation, a reward-corrected EM/F1 result, or a step80/500 checkpoint acceptance. No remote modification or GPU submission has been performed by this agent.

## Exact experiment

Use the first64 ordered rows of the pinned formal3007 `train.parquet` (nq/hotpotqa, split=train). Save every original prompt, alias, extra_info, ordinal, ChatML string and token IDs. Student prompts contain the original question only. The teacher receives the accepted training answers through the externally pinned `teacher_inputs` candidate, separately in `legacy` and `user_context` modes. These modes also differ in the added instruction text; results cannot isolate placement alone.

For each row save four initial-state full-vocabulary distributions: original base student, actual accepted HFstep5 student (BF16 parameters/autocast, FlashAttention2), original fixed base teacher FP32/eager with either prompt mode. Sequential residency avoids keeping a student and the teacher together. No model gradients, generation, search, retrieval, FAISS, optimizer or parameter updates occur.

Save the full token sequence and all full-vocabulary conditional logit rows for `<think>`, `<search>`, `<answer>`, and the candidate `<answer>` + first accepted target + `</answer>`. These are forced candidate sequence probabilities, not successful generation or accuracy. An answer exceeding the actual model context is explicitly skipped with its untruncated tokens/reason. All aliases remain in the raw identity.

CPU replay computes actual source full-vocabulary KL, component-clipped .05 loss, clipping counts/probability mass, logit-gradient cosine, and change in unclipped KL after a hypothetical normalized L2=.01 logit-space step. This step is not a model or optimizer update. Replay rechecks the complete model/data/source pins and regenerates the first64 original train row identities rather than trusting output-written questions. Per-row raw `.npz` files use no pickle; conditional probabilities must reproduce exactly from the saved float32 logits. Four pairings per row give256 comparisons; aggregates count clipping/opposed gradients/KL increases.

## Gates and invocation

The parent must externally prepare and pin two immutable JSON files before any collection:

* `binding`: exact version1, scope `train-initial-state-distribution-only`, training_commit8e60976d364ab7d7ae51714415813c1359dafe9a, questions64, selection `first64-ordered-train-rows`; `train`, `source`, and `base_pins` are exact absolute path/SHA records; `probe_sha256` pins this executing script. `models.base` and `models.hf5` each contain the exact directory and complete file-name/SHA dictionary. The source is the reviewed teacher candidate snapshot, not a hot-modified production repository.
* `allocation`: exact version1, actual fresh job_id, actual UID, node `h100-10`, externally selected GPU UUID. Future IDs cannot be inferred or filled in beforehand.

Base pin record must be the historical accepted nine-file record SHA105c9616c25c8b574608fc0a05d467ce05efd7110ffdbddf945a816f42e52d5c. Fixed original base has434 state keys (tied head omitted), HF5 has435 (explicit lm_head); exact36-layer Qwen key sets and the added head relation are checked. Loaded unique parameter count remains3085938688 with the input/output embedding Parameter actually tied. HF5 three full shard SHA and index SHA3a899e7f6ef595e15c34b5ffc7e4a1df6131d7c371418b1a6e8f823ab8a7302d are hard-pinned. Actual metadata is separately pinned, exact3085938688 parameters and every loaded parameter finite are checked, architecture is compared against the original base, and model loading must report no missing, unexpected, mismatched or error keys. This is not an HF500 gate.

The wrapper must use its own new singleton-H100 Slurm allocation, inherited original global fd9 plus per-GPU kernel lease, and `SEARCH_R1_GPU_PREFLIGHT=1`. The script independently checks real scheduler UID/node/GRES/RUNNING, visible device count, actual CUDA0 PCI-to-NVML UUID mapping (Torch2.4 lacks properties.uuid), initial <=15% utilization/memory eligibility, descriptor inode/lock exclusivity, and selected GPU process four-UID/cgroup/PID-start ownership. Never use another job or GPU. Existing cancelled3007 is expressly rejected. Initial and periodic/final raw resources are retained.

Collection invocation (placeholders deliberately unfilled):

```
python opsd_prompt_distribution_probe.py collect --binding ABS_BINDING --binding-sha256 EXTERNAL_SHA --allocation ABS_ALLOCATION --allocation-sha256 EXTERNAL_SHA --output NEW_OWN_IMMUTABLE_DIRECTORY
```

After collection and independently checking actual Slurm terminal/cleanup, use CPU replay with the observed externally pinned manifest SHA:

```
CUDA_VISIBLE_DEVICES='' python opsd_prompt_distribution_probe.py replay --directory ACTUAL_DIRECTORY --manifest-sha256 OBSERVED_SHA --output NEW_REPLAY_JSON
```

A failure preserves partial raw files and writes a failure marker; no complete manifest is manufactured. A manifest does not substitute for scheduler terminal/cleanup or an independent actual GPU review.

## Cost and limits

One H100, sequential models: FP32 teacher parameters roughly12GiB; temporary full heads/workspace vary with input lengths. Approximately20GiB expected, not measured. The actual85% allocator headroom gate is enforced. Raw evidence size depends on full answer token lengths and vocabulary; expect roughly1–several GiB, not a guaranteed upper bound. Time and peak allocation must be reported from the actual job.

Local49 CPU tests passed, including real torch CPU toy-model conditional forwards, source mathematics, private teacher/student separation, raw full-vocabulary save/replay, explicit untruncated long-answer skip, malformed logits/JSON/import rejection, path pins and immutable files. CUDA mapping in toy forwards is explicitly stubbed. This proves neither real model loading nor CUDA compatibility. Real FSDP training forward equivalence, late-history teacher behavior and actual search effects remain outside this diagnostic.
