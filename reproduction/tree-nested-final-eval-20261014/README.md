# Fixed778 trained200 final-only evaluator

This is a separately committed operational wrapper. It does not modify frozen
778 training code, restore an optimizer, stop a training/evaluation process, or
claim independent review. It is not deployed or executed by its implementation.
The original training instance must reach effective200 and publish its complete
HF/native200 before any handoff. Current training continues unchanged.

The copied `eval_worker.py`, `eval_merge.py` and `shared_guard.py` remain byte
identical to the actually completed old evaluator source0e00998. The new
wrapper binds only the nested778 run and its config SHA355edf52...799b3.
No old3360 model, scores, prefix, rank statistics or job cancellation is reused.

## Future remote preparation

Export committed code only with `build_code_package.py --repo REPO --output OUT`.
Its small code manifest names remote `ROOT/instruct300/eval-nested-SOURCE` and
pins the complete standalone wrapper and unchanged external runtime/retriever.
Publication and remote byte verification must precede deployment or launch.

After real effective200, invoke `prepare.py --mode archive` on retained H10018.
It uses frozen778 JSON accounting: original0bef outer1, ba9 outer2–5 and778 outer6–200.
It pins actual778 native5 load receipt SHA cceb1cfe...c348d6 and requires all
credited source/config segments. Lost original outer2, failed ba9 outer6 and
never-loaded eb6 startup are excluded from effective credit; their bound terminal,
rollout and startup cost evidence remains preserved and unaccepted. It verifies real optimizer totals,
complete eight-rank native200/scheduler200, HF shard/tokenizer inventory, hashes
and fsyncs every saved native/HF file on existing persistent remote storage.
It does not deserialize native states or claim CPU/GPU cold-load acceptance.
The archive proof, asset manifest, immutable empty score prefix and pending
handoff template appear in `TRAIN_RUN/output/final-eval-fast`. All tensors and
model bytes stay remote. Native model identity refers to the official base;
trained HF identity comes from its exported per-file inventory. The native and
evaluator aggregate HF SHA encodings differ; their per-file inventories agree.

Frozen778 automatically starts its old slow live FSDP/vLLM batch32 final
validation after saving200. If a later explicit human instruction authorizes
stopping only that final evaluation, the caller handles the exact own unit
outside this wrapper, after archival completes. No stop action is supplied here.
If that authority is uncertain, leave the template pending and keep the live
experiment unchanged. Original normal exit must never be synthesized.

After the original own training unit is truly terminal and its cgroup is empty,
`prepare.py --mode capture-terminal` records actual systemd status plus the
failed supervisor result proving trained200 and actual empty own cgroup. A missing
cleanup-success field does not substitute for the independent live own-cgroup check.
It retains the original cleanup error and partial-final/terminal cost inputs.
If the original full final finishes normally, replacement is unnecessary and
this interrupted-final handoff path refuses it.

With the actual original human instruction and time, the caller may run
`prepare.py --mode bind-handoff --explicit-final-only-authority
--human-instruction TEXT --instruction-time TIME`. This creates a SHA-bound
`handoff-authorized.json`; it does not launch or stop anything. A scoring resume
can use a distinct `--handoff-output OUTPUT/handoff-authorized-ID.json` and
additionally pin `--prior-evaluator-terminal PATH` for a previous known
failed own evaluator with empty cleanup. Unknown startup outcomes require
manual reconciliation, never a blind restart.

## Future exact owned launch

Use the exported `dispatch_once.py --tools TOOLS --code-manifest
TOOLS/code-manifest.json --code-manifest-sha SHA --handoff
TRAIN_RUN/output/final-eval-fast/handoff-authorized.json --handoff-sha SHA
--unit sst-tree-nested-final-eval-ID.service --journal
TRAIN_RUN/output/final-eval-fast/dispatch-once-ID.json`.
The dispatcher requires original explicit final-only handoff, actual original
terminal and no live descendants. It owns a durable once record and rejects
unknown or active launches. A failed scoring continuation needs its exact prior
terminal pin; completed results forbid another launch.

The separate service has CPUQuota32/MemoryMax384GiB,448GiB host availability,
all eight pinned same-node UUID leases and own cgroup. Existing shared evaluator
startup40GiB/free, realtime12GiB/free and own36GiB/GPU guards remain. GPU E5 plus
the full eight-GPU sharded FP16 index use loopback18082/top3. The supervisor
starts eight persistent TP1 engines with local batch64, cache utilization0.20,
BF16/XFORMERS/vLLM0.6.3, greedy native tokens, four turns and original
2048/4096/500/500 limits. No actor optimizer or reference worker is constructed.
Changed batching/kernel shapes can change greedy answers; no exact executor
resume or numerical equivalence is claimed.

Full fresh scoring has51713 questions across seven exact source counts:
NQ3610, TriviaQA11313, PopQA14267, HotpotQA7405,2Wiki12576,MuSiQue2417,
Bamboogle125. It produces809 immutable contiguous blocks, last block1.
Scores live in `TRAIN_RUN/output/final-eval-fast/scores`. Existing worker code
validates batch JSON/.done SHA and each question/source, completes valid fsynced
orphan receipts, and skips complete64 blocks on a future explicit continuation.
It rejects partial blocks, duplicates and identities that changed.

Complete merge requires all51713 indices, question hashes and seven counts,
then publishes final metrics/.done. The supervisor additionally requires
worker exit0 and actual own cleanup. The result preserves the original training
unit's interrupted terminal and claims no independent final acceptance.
Actual HF loading/tokenizer proof, scoring progress, final output, resource
cost and cleanup must be observed before the goal can be accepted. None of
those future outcomes is promised by this code artifact.

Current readiness is pending real200. The actual778 run has loaded5 and is
continuing training; this separate code package neither proves200 nor authorizes
termination. No evaluation deployment, launch, GPU test or new review accompanies
this adaptation.
