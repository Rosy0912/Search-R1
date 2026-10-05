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
