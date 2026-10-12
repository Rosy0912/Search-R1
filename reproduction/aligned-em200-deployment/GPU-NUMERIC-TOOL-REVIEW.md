# Independent synthetic GPU numerical tool review

Reviewed locally on 2026-10-09 by the actor review agent. Scope is the new `numerics_gpu_fixture.py` compared with `work/numericfix/remote/numerics-gpu-fixture.py`, including the author's output-admission repair after review feedback. No remote or GPU execution, training, optimizer write, or production source modification was performed.

**Conclusion: no remaining must-fix finding in the reviewed bytes.** The tool preserves the original eight-rank synthetic FSDP checks and adds an independent ordinary PPO value/gradient oracle. Its output admission now confines observations to the canonical new hardware run. This conclusion is a tool review and local mathematical check; actual eight-rank GPU results must be reviewed separately after execution.

Current tool SHA-256: `adb451471ccf7d75b74ab78a30a01d074e460c69dd1cbcd4ebad7eced86c4028`.

## Admission and output isolation

The tool rejects optimized Python before relying on assertions. It requires the new environment prefix, exact new repository path, clean source HEAD `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`, and actual imported preflight/gradient/core module paths from that repository. Before CUDA initialization it requires an existing canonical `ROOT/runs/hardware-aligned-em200-…-<SLURM_JOB_ID>` directory, exact canonical `RUN/numeric-fixture` output, and absence of the current rank's prior JSON. It invokes the actual source080 live resource-guard validator before CUDA setup. Final output creation rechecks canonical run/output paths, creates only the direct numeric-fixture directory, and exclusively creates each rank file. It does not overwrite prior rank evidence or create arbitrary ancestor directories.

The original output-order defect was repaired: the first version created an arbitrary output directory before validating its relationship to the run and did not bind that run to the new root. Current reviewed bytes validate the new run and output before any directory creation or GPU work.

Seven local checks against the exact admission block passed: valid run accepted; foreign run, symlink run, foreign output, nonexistent run, mismatched job suffix, and existing rank receipt rejected. Temporary canonical local roots were substituted only in the review process; shipped source bytes were not changed.

Source080's live guard requires the actual watcher ancestor command to name `ROOT/tools/resource_watch.py`, with exact same-run marker, PID/start epoch, and stable ancestry. A watcher executed only as `ROOT/tools/deployment/resource_watch.py` cannot satisfy this check. The eventual supervisor must meet this explicit integration requirement; this review has not inspected or accepted an eventual GPU launch receipt.

## Numerical checks and scope

Inherited tests build small synthetic tied-weight/sharded/replicated modules, exercise FSDP original-parameter and CPU-offload combinations, compare FP64 full-vector gradient norm and clipping to native behavior, check FP32 square-overflow-safe norms, empty shards/replicas, and collective rejection of one-rank NaN/Inf. They require NCCL world size eight and Torch 2.4.0. The saturated low-variance KL value/backward oracle and nonfinite KL refusal remain unchanged.

The new ordinary PPO block calls actual `core_algos.compute_policy_loss` with clip range 0.2 and `log_ratio_clip=None`. Its independently written FP64 oracle takes the maximum of unclipped and ordinary clipped surrogate losses, masks tokens, and divides by supervised-token count; it compares both loss and derivatives. A negative-advantage log-ratio of 30 gives ratio `exp(30)`, proving the branch remains uncapped by a log-ratio-20 clamp or DualClip. Masked tokens and both signs of advantages are included. NaN, positive Inf, and float32 exponent-overflow log-ratio 89 must raise before any success receipt is written.

The exact new PPO block passed locally on CPU against source080, including FP64 value/gradient comparisons and the three rejection cases. An additional independent counterexample showed ordinary loss divided by optional log-clamp-20 loss equals approximately 22026.46, demonstrating this oracle distinguishes that legacy clamp. This local calculation does not establish CUDA execution or distributed FSDP correctness.

There is no optimizer construction or step and no real actor/model/checkpoint loading. The script changes only synthetic module gradients, process-local RNG/device state, and its bounded run-local output. Each success JSON labels `ACTUAL_8RANK_SYNTHETIC_GPU_ORACLES_NOT_TRAINING`. A partial failure cannot produce the missing ranks' PASS records. All eight receipts and process success must be checked independently before calling an actual GPU fixture passed.

Signed: independent actor review agent, 2026-10-09.
