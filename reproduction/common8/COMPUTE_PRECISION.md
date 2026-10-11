# A800 BF16 performance correction (2026-10-08)

The latest user instruction aligns this run with Search-E1 BF16 execution.
New local A800 manifests explicitly select `bf16_policy_fp32_master`: actor and
reference CUDA forward use BF16 autocast, actor FSDP compute parameters and
vLLM generation use BF16, while actor master weights/Adam and Qwen RoPE buffers
remain FP32. Selected-response log-probability reductions use FP32; global
L2 clipping still accumulates FP64. Torch TF32 flags are no longer overridden.
No optimizer clipping, batch, reward, reference weight, or safety limit changes.

`sampling_journal_no_extra_hf_forward` uses the actual generated selected token
log-probabilities as frozen old. Full actual token/context/mask/sampler/finite
provenance remains checked, but the additional full-outer independent HF forward
and maximum discrepancy gate are omitted under the newer explicit instruction.
Receipts record unmeasured HF discrepancy/parity as null, never as passed.
Runtime global32 policy KL0.2/tail0.01 and finite gradient/parameter/Adam checks
remain. The initial original-model probe artifacts and historical dev512 scalar
baseline are reused with their recorded provenance. A new run starts from the
official weights and fresh optimizer/scheduler; no in-memory partial old run is
promoted to a completed outer or inherited. This is a configuration correction,
not a measured speedup or successful five-step/final acceptance claim.

The FP32 observations below describe historical diagnostic execution only.

# Explicit common8 compute precision

Native3260 tested an isolated FP32 candidate with the original probability
acceptance bound unchanged at 0.05. All eight ranks completed the same64 ordered
single-action questions, one actual DTensor weight sync, the internal and separate
actor recomputation paths, and zero optimizer writes. Actor/reference error was
0; maximum actor/vLLM valid-token log-probability error was0.00055694580078125.
The preceding BF16 experiments retained a maximum error of1.550987. This supports
a precision adaptation, not a proof of the complete numerical root cause.
The candidate's original wrapper and scalar/hash proof are retained separately.

The explicit implementation replaces that diagnostic's global autocast override:

* Actor/reference parameters use FP32. Actor FSDP parameter/reduction/buffer
  dtypes are FP32; the fixed reference retains its existing no-mixed-cast mode.
* Both policies use the same `ComputePrecision` helper. FP32 locally disables
  CUDA autocast; missing/default configuration retains the original BF16 path.
* Both workers explicitly disable the Torch matmul and cuDNN TF32 flags. These
  settings affect that worker process; conflicting runtime flags are rejected.
* vLLM uses FP32 parameters and cache tensors. Actual runtime evidence is required.

The model revision, canonical shared data, LR1e-6, outer512/G5, mini128,
actor globalmicro16, old/reference32, epoch1, rewards, masking, GRPO advantage,
reference regularization, sampling settings, loss denominator, gradient clip1,
300-update budget and original probability bound remain unchanged. Runtime
precision is a separately disclosed numerical adaptation.

`model_smoke` observes actual forwards after initialization/profiling. Outer FSDP
hooks measure storage/pre-FSDP tensor dtypes; inner Linear hooks measure actual
compute parameter dtypes after unshard/cast. Embedding activations, logits,
autocast state, vLLM hidden activations and sampler-before logits, actual KV
cache tensors, and backend class are recorded. Observation hooks return the
original outputs and are removed afterward. Missing or mismatched observations
reject the model smoke collectively. Synthetic CPU hook tests are not native
runtime evidence.

An FP32 tensor label and disabled Torch TF32 flags do not prove all internal
attention arithmetic is IEEE FP32: the selected backend can use FastF32
operators. The original log-probability gate remains the measured acceptance
criterion. Initial64-question admission also does not prove a full2560-trajectory
memory peak, resident-retriever/probe coexistence, successful optimizer updates,
healthy fixed512 EM/F1/action rates, or final51713 evaluation.

Native3260 is a candidate result, not production admission. The explicit source
must pass its own native runtime evidence and healthy calibration before the
fresh300 training run starts. No candidate or calibration weights may replace
its official fresh initialization.

49 actual-target Torch2.4 CPU/private-group Gloo regressions passed in6.154s.
This includes rank1 module/TF32/logits faults causing both policy ranks to reject
without simulated writes; the legacy default adds no collectives. Runtime
precision invariant checks use the corresponding FSDP process group before
forward and after matching logits. Evidence is in
`g1-common8-explicit-precision-calibration-cpu.json`; it is not CUDA admission.

The authorized native H100 recovery profile `native8_bf16_varlen_fp32_master`
restores the original full CP5 model and optimizer state, then uses BF16 actor,
reference and rollout computation. Actor master weights, Adam moments, rewards,
probe calculations, response log probability reductions and objective remain
FP32. This is an explicitly recorded execution transition, not bitwise
continuation of the previous FP32 suffix. Global micro16/local2/accumulation8,
mini128, twenty opportunities, selection limits and scheduler remain unchanged.

Padding removal selects Qwen2 FlashAttention2 (HF 4.46.3, FlashAttention 2.7.4)
with packed position resets and varlen sequence boundaries; packed SDPA is not
used. The behavior denominator uses the actual selected sampling journal with
existing token/context/mask provenance checks, as in the A800 BF16 execution.
It does not claim independently recomputed HF probability parity. The original
0 and CP5 FP32 validation scores are reused with pinned source SHA, explicitly
identified as historical measurements. New CP10 is saved and read back before
continuing11; it was unavailable in canceled job3368. Speed is measured from
actual candidate filling plus training, not predicted from configuration.

The later direct user authorization permits Slurm shared scheduling while
preserving the video job and the actual eight-GPU training budget. Shared native
execution reserves96CPUs and512GiB host RAM. It uses an explicit per-run physical
GPU authorization (indices0–7, independently observed UUIDs and native video job
birth), rather than falsely claiming a second Slurm GPU reservation while the
video owns all eight GRES. The pipeline retains own Slurm process/cgroup tracking,
UUID locks, actual CUDA/rank validation and bound foreign process checks. Global
Slurm/cgroup/device configuration and the video's reservation are not modified.
The admission receipt distinguishes zero newly reserved Slurm GPUs from eight
actual shared physical GPUs. CPU fixtures are not actual GPU acceptance.

Human instruction: reduce extraneous training checks and retain training correctness. Startup no longer performs packed/single diagnostic forwards or numerical logit gates. Actual loaded Flash2 layer/dtype configuration remains checked; CP5 HF/Adam/scheduler/RNG/cursor/probe readback and real finite optimizer-update accounting remain. Necessary isolated failure diagnostic3380 reproduced max0.8359375 for different GEMM shapes, while equal-shaped masked CP5 forwards matched exactly and sequence isolation was exact. This is no quality or bitwise-continuation claim.
