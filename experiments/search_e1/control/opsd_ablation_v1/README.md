# Seven reverse-KL online ablations: execution contract

This independent namespace implements the human execution goal created at 2026-10-08 23:18:19 CST. C0/C1/M/R/W/P/J each run **ten online sampling/training outers**, then immediately evaluate their raw final adapter on the same native512. Ten outers is not ten Adam steps. No historical GRPO/OPSD optimizer is restored.

| Group | Constant LR | D selection | Reverse target | Training/sampling precision |
| --- | --- | --- | --- | --- |
| C0 | 5e-6 | native farthest, failed priority | frozen privileged teacher | BF16 base/compute, FP32 LoRA |
| C1 | 1e-6 | same | same | same |
| M | 1e-6 | same | .1 teacher + .9 frozen outer-old student | same |
| R | 1e-6 | unbiased uniform among all five | teacher | same |
| W | 1e-6 | same, positive correctness exposure weight | teacher | same |
| P | 1e-6 | native | teacher | FP32 master, FP16 main compute/activation and sampling |
| J | 1e-6 | random, weighted | mixture | same as P |

All groups use the exact source100 twelve-file inventory and the actual common504-FP32-tensor initial LoRA template, with252 zero B tensors. One seed42 TRAIN shuffle freezes5120 unique questions: ten cohorts of512, with no validation-ID or exact-question-text overlap. Every group generates its own five fresh siblings per question from its latest raw student. The frozen scorer, successful B rule, generation budgets and complete D prefix remain unchanged. Random D may be B. All-correct questions remain eligible; all-wrong questions have no invented successful B.

## Code and evidence boundaries

| Requirement | Implementation | Acceptance evidence |
| --- | --- | --- |
| Full-vocabulary reverse KL with both student coefficient/logprob gradients | objective.py | Non-symmetric reference/autograd/finite-difference, clip and alpha fixtures; actual token preupdate records |
| Trajectory mean, post-clip positive weight, actual batch/tail denominator, eight-rank SUM | objective.py, trainer.py | Actual batch receipts/token contributions and all-rank update states |
| Common cohorts, native/random D and positive weight | cohort_v1.py, miner_v1.py, weights_v1.py | Bound source/cohort/pair files, deterministic rejection proof and native selection shadow check |
| Whole-outer old target, current graph stable through backward | targets.py, trainer.py | Original outer-start adapter SHA; same-outer resume binding |
| Actual FP32 masters and FP16 major compute, synchronized overflow handling | precision.py, retry_state.py, trainer.py | Actual dtype/recompute hooks, all-rank scaler states; overflow capsule and no advanced Adam/cursor |
| Eight complete optimizer/RNG/cursor states | state_storage.py, checkpoint.py, trainer.py | Eight actual equal Adam digests plus one complete shared tensor payload, eight distinct rank files |
| Source/ownership/oracle/storage/memory admission and isolated CPU merges | prepare.py, gates.py, controller.py, sampling.py | Real file SHA/PID/UID/starttick/UUID/locks, child return code and direct memory.free |
| Autonomous privileged teacher, separate from training on D history | teacher_diag.py, controller.py | At most8 per outer, independent BF16 greedy process, raw IDs/HTTP/strict EM and explicit diagnostic labels |
| All valid-token scalars, four observed distributions and bounded full snapshots | objective.py, vocabulary_dump.py | Every valid scalar; top32 positions capped; actual NPY SHA and study-wide24GiB reservations |
| Immediate raw10 native512 and independent precision deployment checks | sibling opsd_ablation_eval_v1 | Actual ten-outer terminal, resource release, complete unique coverage and same-question comparisons |

CPU fixtures are necessary correctness checks, not GPU or performance acceptance. They do not prove that P/J long contexts fit on24GiB cards. The first actual model forwards, actual Adam updates and peak/direct-free samples must establish that. Initialization and a successful CPU test command are never reported as experimental completion.

## Physical checkpoint storage

`shared_exact_eight_rank_optimizer_v1` changes disk representation only. Before saving, each rank inspects its actual Adam moments/steps/groups and computes its full optimizer digest. Any inequality rejects the checkpoint. Rank0 then writes one complete `optimizer-shared.pt`; each of eight distinct `rank-*-state.pt` files retains its own RNG/cursor/scaler/identity and the explicit shared file proof. The complete manifest binds every file and all eight actual optimizer digests. Restore expands the referenced full optimizer, verifies its SHA/run/config/update/digest, then performs the original rank-state, optimizer-load and actual digest checks.

Nothing is discarded from the logical state. No historical file is rewritten or removed. All new checkpoints and debug remain under keep-all retention. This avoids saving eight byte-equivalent Adam tensor sets: approximate mature checkpoint size is360MiB rather than2GiB. Ten outers have at most160 updates per group; even the maximum1120 updates have a substantially smaller footprint. Merge caches, debug, actual free space and Mac backup space remain independently monitored.

Full-vocabulary NPY capacity is reserved durably **before** files are written. An interrupted dump may waste its conservative reservation, but cannot silently reset the study cap. `complete`, `cap_skip`, and unrequested positions are distinct; top32 does not reconstruct full KL. The persisted vectors are original-forward FP32 log probabilities, with exponentiation available for probabilities. Scalar token loss is never truncated by this cap.

## Preparation and execution

`prepare_bundle.py` creates seven private CPU candidate configs, not launch authorization. It copies only the immutable old model/runtime/data identities; no old controller, Adam, observer, storage exception or permissions are reused. Missing newly scoped storage/publication/resource/oracle inputs remain explicit HOLD. This is a GPU-start dependency, not a pause of CPU implementation, publication or existing backups.

All new runnable helpers must be committed/pushed to `search-e1` and read back from the fetched remote commit before node execution. Deployment uses an independent source directory and the unchanged305 runtime. No hot edits are made to an active run. The same group/run/config continues after first1/5 updates and on accepted full-state recovery; no short test followed by a reset to zero.

`README-controller.md` documents exact config/observer/lease admission and the fixed controller CLI. The sibling evaluator owns the sequential driver. Actual trainingrc0, `TRAIN_COMPLETE` ten-outer evidence, complete child release and fresh oracle/free-memory checks precede each group's512. Seven raw adapters plus commonBF16 zero and FP16 zero/P/J produce5632 official predictions. P/J commonBF16 and FP16 deployment results are separate; no best-model fallback is used.

Private questions, full trajectories, token distributions, model weights, environments and credentials stay outside Git. Final delivery includes actual config/source commits, node/run/PID identities, true K/Adam/exposure counts, checkpoint/log/report absolute paths, strict EM/paired failures and verified persistent-backup coverage. Unsupported causal attribution and unmeasured intermediate curves are marked unknown.
