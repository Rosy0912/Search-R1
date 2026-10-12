# Aligned SOURCE080 final evidence: phase1 candidate

This candidate implements terminal/raw accounting only. It has not been deployed or executed against a formal job. It always records `acceptance:false`, `goal_complete:false` and `release_receipt:false`; no status or process exit can approve training or complete the goal. Full final verification remains pending.

Its supported path is one original SOURCE080 formal job with200 complete no-skip outers and no infrastructure fault or resume. In that path, SOURCE080's real eight-rank progress replay yields4000 Adam writes per rank/32000 overall. This is the no-fault reference total, not a rule that erases partial writes. Fault/resume chains are explicitly rejected by this phase and listed as unsupported pending work. The full goal still includes truthful cross-job recovery provenance and all retained partial writes; it must not be silently narrowed to a fault-free experiment.

## Future invocation, not performed

The root, source and isolated environment are fixed. Actual new job/token/RUN identity must come from `ROOT/scheduler-head/formal-submit-receipt.json`; historical3315/3345 roots and sources are refused. The output must be a new unused direct child of `ROOT/evidence`.

```sh
CUDA_VISIBLE_DEVICES='' /mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/env/bin/python /mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/tools/final-evidence-v1/aligned_final_evidence.py --output /mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/evidence/final-phase1-NEW-UNUSED-NAME
```

The candidate uses only a read-only scheduler query, reads/copies original files, and creates exclusive own evidence. It never submits, resumes, stops or contacts another process. It does not issue a release receipt, re-run GPU roundtrips, deserialize native/driver tensors, scan official weights or load HF models. The only hashes now are small source/tool/receipt metadata and raw terminal logs/ledgers/EM audit files. No model, corpus, index or large native archive is hashed at this stage.

## Evidence collected

The original scheduler query is saved before interpreting it, including unavailable/error observations. An actual strongly bound terminal scontrol record is required before any RUN read: exact actual job, formal job name, source/token comment, fixed work directory, command and h100-5 node, real End/Submit timestamp and exit code; completed jobs additionally require valid Start/End order. Pending or running jobs cannot become terminal from a token or old log. Failed terminal jobs retain the raw resources/cleanup/exit observations and cannot produce complete phase1 status.

For a completed0:0 formal job, phase1 archives original source, supervisor/train logs, resource watcher, cleanup/exit records, all200 driver attempt ledgers, prepared/stage proof files, eight rank prepared/final events per outer, and all200 training EM audits. Each copied file has an exclusive binding sidecar, remains available on later failure, and is rehashed before collected status. The source080 checkout, deployed formal tools, actual submit receipt and helper/provenance bytes are also bound and rechecked.

The actual SOURCE080 `Fresh200Progress` and aligned rank contract replay every rank's200 outcomes; stage64 proof/source/input binding precedes scheduler acceptance, all20 writes remain prepared until stage success, and exact before/after scheduler LR uses the source warmup57 function. Current main config is validated with the source production validator, so mini128/token_mean/referenceKL.001/frozenHF-old/noDualClip/no-finite-SKIP are the production profile. Per-rank producer monitors retain the frozenHF-old diagnostic role. EM verification calls the actual shared authored-span/terminal-answer verifier against each audit, with original stable question IDs, ordinal/resample/snapshot provenance,512 questions×5 siblings, source, RUN, driver PID and common/order/retrieval bindings; recomputed aggregate values must match that outer's actual metrics. This does not reconstruct original training tensors from logs.

The original fifth manifest/independent receipt and same-PID pause/release markers are byte-bound, matched to source080 aligned fixed20 identity, and ordered before attempt6. This archives the pre-existing release record; it does not sign or fully revalidate that record's native state. All200 attempts must use that same driver PID. The four dev512 evaluations must be at50/100/150/200 with256 nq+256 hotpotqa each. The unique original `FRESH200_COMPLETED_RUN_JSON` must equal independently replayed progress and appear after outer200 training metrics and before the final evaluation summary. The one final51713 summary must follow outer200, cover all seven exact benchmark counts, and have finite count-weighted EM/F1/format/cost aggregates consistent with its source means. No per-question final answer file exists in the source evaluation path; phase1 therefore checks original aggregate consistency only and does not claim per-question final recomputation.

Small checkpoint sidecars at5/50/100/150/200 are archived and checked for exact complete counters/progress, same driver PID, eight native ranks, scheduler epoch, version3, final rank event and cleared pending state. SOURCE080 creates the self-contained gate manifest only at5; the later native/HF inventories and cryptographic full-file manifests must be built in the next phase and identified as derived evidence, rather than pretending source080 wrote them. Later sidecars are producer metadata, not a subsequent actual GPU reload. Native archives and real driver state200 remain unopened in phase1.

Resource observations must belong to this RUN/job, have an exact boolean pause and finite ordered times. Earlier resource pauses are retained with their actual intervals, reasons and recovery; the final observation must be healthy. These metadata checks do not finish UUID/GPU-exit/lease/cleanup validation, which remains explicit pending work; raw observations and GPU-exit output remain archived for later independent numeric inspection. Cleanup must show this own RUN marker with an empty remaining list and zero cleanup exit. The collector never invokes cleanup itself. Wall time, CPU time and peak RSS describe its actual extraction/replay costs; no price or GPU allocation cost is invented.

## Pending full target

The saved report's pending list includes all official12-file hashes; complete HF/native inventories and full hashes at5/50/100/150/200; actual CPU inspection of driver/native state200 with truthful DTensor unsupported status; real offline HF/tokenizer/tied/all-parameter-finite checks for official and every export; full dtype-aligned weight-change scans beyond a first witness or chunk; fault/recovery cross-job identities and retained partial-write accounting; complete own cleanup/GPU-exit/UUID/lease/resource validation; and independent final review. Those phases must continue from immutable original identity/bindings and write new exclusive output, rechecking previous originals. A resume/continuation CLI for those phases is not implemented yet. No partial phase1 result is final acceptance.

## Local tests and limits

```sh
PYTHONPATH=../bapo-baseline-alignment-20261009 ../fresh200-local-test-env/bin/python -m pytest -q reproduction/aligned-em200-deployment/test_aligned_final_evidence.py
```

35 code/CPU fixture tests pass. They cover exact terminal identity and timestamp negatives, pending-before-RUN-I/O refusal, full200 real SOURCE080 progress replay with eight ranks and explicit post50 switch/skip/stage/scheduler/EM/PID/rank faults, complete51713 aggregate consistency, four dev512 counts, and same-PID fifth original receipt binding/sequence. One512×5 initial+retained authored audit is produced and recomputed by the actual journal/verifier within the200-loop plumbing test; the other199 EM calls there are explicit provenance spies, not199 actual audits. Files/Slurm/native/state/checkpoint paths elsewhere are synthetic fixtures. No actual target scheduler, formal job,200-step run, official3B asset, state200, cleanup or final benchmark was accepted by these tests.

The candidate imports the already-reviewed generic math/first5 file-safety functions and read-only scheduler/raw-copy functions, retaining their provenance/source bindings. It imports no historical final3315 source, ROOT, variable mini20/10 policy, production21 helper or raw-native denominator requirement.

## Deployment dependency layout (candidate only)

Future installation must copy this file together with exact reviewed `aligned_first5_cpu_evidence.py`, `aligned_first5_math.py`, `aligned-first5-math-provenance.json` and `collect_hardware_evidence.py` into one new canonical own `ROOT/tools/final-evidence-v1` directory. Do not depend on the first5 helper's separate nested installation or a collector elsewhere on PYTHONPATH. Runtime checks the actually imported module file paths equal this helper directory and binds/rechecks all those bytes. Root/actor review must pin that inventory before any installation. No such installation was performed by this implementation task.
