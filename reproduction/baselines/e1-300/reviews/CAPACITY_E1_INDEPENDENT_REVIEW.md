# Independent E1 hardware capacity candidate review

2026-10-04. Initial reviewed candidate fingerprint: `capacity-e1-reviewed-candidate.json`; actual CPU counterexample/compile stdout: `capacity-e1-independent-cpu.txt`. No GPU, service, Slurm or training was started by this review.

## Blocking input error

The helper repeats ordinary text tokens to length6144, slices its first2048 as `prompts`, and sets `teacher_prompt_mode='user_context'` with nonempty privileged targets. That plain text has no ChatML control suffix. `opsd_loss` calls `teacher_inputs`, which requires the actual prompt to end in the verified user-end/assistant-start suffix. This fails before teacher/student forward with `Student prompt does not end with the verified chat suffix`.

The counterexample used the same helper text, repetition, widths, micro2, targets and real teacher_inputs, with the existing real CPU ChatML-tokenizer fixture (not the official tokenizer assets and not a model). It reproduces the exact boundary failure. This is sufficient to demonstrate the algorithmic invalidity of the synthetic construction; actual official asset/GPU validation remains pending.

Minimal repair: build a legitimate user message with `apply_chat_template(...,add_generation_prompt=True)`, fill only the user content to the desired prompt width while preserving its complete ChatML suffix, then append the4096 response/history IDs. Preserve all widths, mixed observation masks, targets and null clipping. Validate the exact constructed tensors with teacher_inputs on CPU before submitting eight GPUs. Do not weaken the production suffix check to accommodate a bad probe.

## Other alignment requirements

The helper hardcodes AdamW weight_decay0, while the production worker default is0.01 when the profile supplies no weight_decay override. State residency is similar but this is not the exact production optimizer definition. The helper should read/lock the actual fixed configuration, or explicitly use its exact reviewed weight_decay and betas. An unchanged LR1e-6 and two steps do not remove this difference.

Native/default allocator values are recorded but not asserted. The E1 requirement must reject experimental allocator configurations and non-native backend, either in the probe itself or an explicitly reviewed outer wrapper checking the complete actual records before acceptance. Recording a setting alone is not a gate.

The script verifies eight distributed ranks and the current job cgroup/node but does not itself verify unique allocated GPU UUIDs, the same-job resident GPU retriever PID/UUID identity, its loopback endpoint, leases or pre-probe15% free-memory guard. Its header says those are prerequisite conditions. They must be enforced by the forthcoming owned wrapper and independently bound to this current job; no old proof can substitute.

## Valid design already present and limits

Two actual forward/backward/clip/Adam calls keep Adam state resident during the next graph. Training mode and non-reentrant HF decoder recomputation are audited per-layer, each reaching4 calls over two steps. FP32 master weights/BF16 Flash FSDP and FP32 eager staged teacher/TF32-disabled audit match the intended numerical policy. CPU compilation passed, but none of these GPU behaviors has run in this review.

Clearing only unreferenced initialization cache before resetting the update peak is explicitly checked not to change allocated live bytes. Reported capacity sums pre-model external device usage + PyTorch peak reserved +1GiB NCCL slack, and requires85% total. This is a conservative estimate with a fixed sampled external baseline, not a continuous NVML peak or proof that retriever usage stayed constant. After-context CPU teacher return is separately checked. The wrapper must keep the own retriever alive and preserve strict headroom.

These are synthetic worst-width capacity steps, not RL outer steps,20 production minibatches, real histories/teacher efficacy, vLLM residency, scheduler correctness or full first-five acceptance. Even repaired all-eight capacity success cannot approve production5 or complete300. Initial result: **BLOCKED pending valid ChatML synthetic prompt and exact runtime/config gates**, not OOM and not a failed GPU run.

## Corrected candidate incremental review

Corrected helper exact SHA256: `bbe157e70f959fb654f0d10836d91f32a1c6af0919eacdec8b2caf5c446e9b2b`. The original rejection remains in the initial CPU evidence and is not described as a GPU failure.

The new `synthetic_tensors` renders a legitimate user ChatML message, preserves the full verified user-end/assistant-start suffix, and fills exactly2048 prompt tokens plus4096 response tokens. The truncated portion is user content, not the suffix; the rendered prompt already contains its opening role controls. It runs the real user_context adapter before GPU loss and records actual widths/hint expansion. AdamW now uses weight_decay0.01, matching the production default. Both allocator env names are rejected when nonempty, and the actual allocator backend must be `native`.

Independent CPU execution using the existing real ChatML tokenizer fixture passed exact widths, unchanged concatenated student inputs, all8192 response next-token alignments via teacher positions, and8160 supervised tokens. This fixture intentionally has different suffix/hint token counts from the official tokenizer. Evidence: `capacity-e1-delta-independent-cpu.txt`.

The actual h100-10 official-tokenizer CPU stdout in `capacity-chatml-native-cpu.json` was inspected as raw stdout containing a warning followed by one JSON record. Its helper SHA matches the corrected source; `gpu_verified=false`; the official suffix has5 tokens, teacher context6198, hint expansion54, and8160 loss tokens. I parsed and checked that record independently. This is an actual tokenizer/input-construction check, not official model forward, NCCL, FSDP, memory or GPU proof.

The three identified source-level issues are closed for this exact candidate. **Static/CPU capacity-helper prelaunch review passes, conditional on the separately reviewed owned wrapper enforcing leases, exact8 allocated unique UUIDs, same-job resident GPU retriever/loopback identity, source/import/asset pins and pre/post strict15% headroom.** No eight-rank capacity or production outer result is claimed. Two synthetic Adam updates still do not replace first5, dev/stage evidence or final300 acceptance.
