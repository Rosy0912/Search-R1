# Evaluation-only continuation, 2026-10-13

User authorizes preserving already scored questions, stopping slow final evaluation and evaluating remaining questions faster. This does not authorize restarting training. Old3360 has 200 credited outer updates, eight ranks each4000 Adam writes and zero skips.

The old evaluator has no durable per-question score journal. A read-only external CPython3.10 snapshot preserves its ordered qa_rows twice without signals or injection. Bind and fsync the recovered metrics to exact original source/config, ordered final parquet and actualHF200 file hashes before stopping. Current accepted prefix30880; remaining20833 of51713. Prefix positions follow frozen unshuffled/no-filter evaluator, compared every source row and annotated with question hashes.

Replacement: same frozen custom vLLM0.6.3/XFORMERS BF16 greedy native-token parser/search rules, eight persistent TP1 engines, localbatch64/global scheduling512. No actor optimizer/reference/training construction. Resident weights and cache remove per-call FSDP synchronization/offload; question queues independent. Four rounds and2048/4096/500/500 limits unchanged. Greedy output can differ numerically with changed batch/kernel shapes; do not claim bit-identical executor resume.

All model/data/index/score tensors remain h100-10/10.33.0.18; full8GPU FP16 FAISS and GPU E5 use loopback. ExclusiveSlurm8GPU, inheritedUUID leases, exact original foreignMEM/SM15% guard and own-cgroup pidfd cleanup remain. Durable atomic batch scores and SHA receipts allow future job continuation through the same score directory. Final merge rejects gaps/duplicates/source/question/hash mismatch and requires seven datasets/all51713.

Dispatch journal creates SUBMISSION_UNKNOWN before one sbatch; inspect uncertain outcomes, never repeat. Replacement queued afterany3360, then explicit old3360-only cancellation after source review, prefix durability and actual replacement allocation checks. Preserve original native/HF200, old cost/resource evidence and user-authorized handoff record. Old canceled evaluator is not a failed training or a completed final evaluation. Actual8GPU batch progression and full final acceptance remain runtime requirements.

## Latest human sharing override

2026-10-13 user explicitly permits shared GPUs with free memory and selecting any of the four sameH100 nodes; empty/exclusive GPU reservation is not required. Actual four-node inventory found h100-10 all8physicalGPU0memory/0util despite Slurm3391 holding reservation, and assetsalready local. Shared mode is an independent own systemd service (CPUQuota32 MemoryMax256G), with no Slurm device-policy/config changes and no exclusive reservation claim: physical8, Slurm_reserved_gpus0. GPUUUID leases and own-cgroup pidfd cleanup remain. Startup needs40GiB free each, realtime12GiB headroom/own36GiBcap; foreignSM15% gate replaced by explicit authorized memory-sharing mode. Cache utilization20% (old40%) is an execution-only memory bound, recorded separately. No foreign process/job is stopped. Only known pending own3392 is retired after independent review and durable prefix binding; one recorded own-unit launch, unknown outcomes must be inspected. Static review passed; GPU fit and real batch throughput still need actual verification.

## Truthful completed-training / evaluation-handoff CPU acceptance

The isolated `verify_handoff_hf.py` accepts the composite result only after the
replacement shared unit is terminal with worker exit 0 and empty own cleanup
residual. It never requires or creates the original `run-result.json` or
`train200-terminal.json`: both are actually absent after the authorized
slow-evaluation cancellation. Original Slurm state remains CANCELLED / 0:15,
from the SHA-bound historical scontrol observation; current original receipt
absence and queue absence are checked separately. The original training body
exit code stays null and no original normal-exit claim is made.

Training completion is independently recomputed from the frozen JSON-only
contracts: 200 credited outers, 8 ranks × 4000 actual Adam writes, direct-rank
records, Tree/policy/safety evidence, finite saved native200 manifests and
unchanged native shard hashes. Native tensors are only hashed, never loaded,
restored, or imported into an optimizer. The original and evaluation source
commits, config, base model, split bytes, HF200 inventory and reviewed release
are bound separately. Preserved30880 and replacement20833 must jointly cover
exact51713 with all question hashes and seven source counts. Immutable batch
receipts, rank provenance and published final metrics are reread and the
metrics recomputed. Old partial validation cost files remain separately
hashed: unpreserved additional slow-evaluator compute is never counted as
preserved final scores.

Only then the wrapper invokes the unchanged full frozen CPU HF verifier and
performs the original wrapper's complete tokenizer backend, AddedToken,
special-role, semantic-config, vocabulary, PAD/BOS/EOS/UNK, encode/decode and
chat-template checks. All GPUs must be explicitly hidden; at least64GiB free
CPU memory is required. It never reruns training or evaluation. All artifacts
and caches remain on the asset-local remote filesystem.

Arguments are `--original-job-terminal`, `--handoff-journal`, `--manifest`,
`--release` and `--review`, each an existing absolute remote path. The honest
original snapshot is under `output/eval-handoff-acceptance-inputs`, separate
from the newly created `output/eval-handoff-hf-acceptance` output namespace.
The original evaluation handoff journal is the original `dispatch-once.json`
with HANDOFF_REQUESTED / cancel_returncode0; it is not a shared repair journal.
The immutable accepted shared release and asset manifest retain their pinned
deployed hashes and source0e00998. An existing acceptance output namespace
causes refusal; a failed CPU check preserves its log and is not automatically
retried or overwritten.

The composite output is `identity-bound-handoff-hf.json`, schema
`tree_train200_eval_handoff_composite_v1`, status
`JOB3360_TRAIN200_AUTHORIZED_EVAL_HANDOFF_FULL_HF_ACCEPTED`. Its before/after
input hashes are descriptor maps with absolute path and SHA256, covering
frozen/reviewed code, historical cancellation, current replacement terminal,
native shards, all direct training evidence, datasets and complete score
ledger. `independent_final_acceptance` stays false until a separate independent
review binds this receipt, the wrapper and the actual reports. Changed greedy
batching/kernel shapes retain the numerical-equivalence caveat; original
sampler/actor gate and prospective production thresholds are not newly
claimed.

The wrapper also consumes every actual200×8 rank's published successful-write
gradient norms and fixed local sample evidence. Norms must be finite and
positive with exactly recomputed counters; sample coordinates/layout, full
sample finiteness, nonzero bounds, extrema and before/after SHA fields are
validated. Outer1 has zero learning rate and an unchanged sample; every
nonzero-LR outer must show actual sampled movement. The numerical report
explicitly limits its claim to the fixed up-to8192 local coordinates, while
the separate full-HF verifier proves complete final parameter finiteness.
Profile0 is not credited as a real training outer.

The supplied original handoff journal must equal the journal embedded in the
SHA-pinned historical scheduler observation, identify own3360 RUNNING before
cancellation, and preserve request/stop/observation epoch ordering. The entire
attempt-results directory must be exactly the consumed attempt sequence
ending at outer200, with no later/pending/fault/unconsumed attempt artifacts;
its full inventory is hashed and rechecked after CPU verification.
