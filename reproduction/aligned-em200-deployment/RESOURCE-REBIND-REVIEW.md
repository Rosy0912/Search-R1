# Independent resource supervisor rebind review

Reviewed locally and read-only on 2026-10-09 by the actor review agent. Scope is the new deployment-tools `resource_watch.py` against the prior controller-initfix version, plus exact byte comparison of `owned_processes.py`. This review grants no new GPU allocation, launch, or training permission. Nothing was executed remotely.

**Conclusion: no must-fix finding in this rebind.** The watcher diff changes only the new task root, frozen source, launch protocol, and disables every legacy root. The ownership and recovery implementations are unchanged. This is code/boundary review, not an actual target Linux pidfd or GPU runtime acceptance.

| File | Current SHA-256 |
| --- | --- |
| `resource_watch.py` | `5ea7c46d2a8b4cda5e666cacb21b781a1d2a8908cd605950f3fae6c0d1774025` |
| `owned_processes.py` | `2c8cdb3363461b402c3155091cab89739524256b1f89ab58c0f5e62334a877bd` |

## Binding checks

The exact root is `/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009`; source is `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`; protocol is `BAPO_ALIGNED_EM200_LAUNCH_V1`. `LEGACY_E1_ROOTS` is empty, so `load_runtime_anchor` cannot take its retained legacy approval branch for any foreign root.

Approval validation requires the exact protocol and equal source in both the independently read source file and receipt, exact root and canonical resolution, and a typed/hash-bound original runtime executable record. It explicitly rejects legacy E1 acceptance status. Runtime loading additionally checks regular canonical files, runtime-record bytes/SHA, own environment prefix, host h100-5, interpreter identity/SHA, source-bound tool manifest and this watcher's own SHA. Optional current-process verification compares actual interpreter and `/proc/self/exe`. The new-root trainer command requires a real Python `-m search_r1.baselines.main` prefix; a legacy `verl.trainer.main_ppo` command or arbitrary argument substring cannot satisfy it.

These are machine identity and source bindings. Their presence is not independent human authorization to launch. The old comment immediately above the source constant still refers to an unbound 0.2/skip source; the actual constant and validation bind source080. This comment is stale but does not change behavior or this review's permission scope.

## Ownership and cleanup boundaries

`owned_processes.py` is byte-identical to the prior controller tool. It selects processes by exact inherited `BAPO_RUN_ID`, exact numeric current `SLURM_JOB_ID`, and own UID, excluding its own PID and parent. Before signalling, it opens a pidfd and rechecks the selected process start epoch against a fresh ownership snapshot. SIGTERM and subsequent SIGKILL use the held pidfd, rather than a potentially reused numeric PID. It neither selects foreign markers nor uses a global kill fallback.

The unchanged watcher freeze path requires run/job/UID/allocated UUID and process/parent epoch certificates. It rejects uncertified or changed identities; rechecks after pidfd open; stores held descriptors for thaw or bounded failure cleanup; and refuses unavailable safe pidfd signalling. Recovery pauses and resumes only certified owned epochs. External occupancy and unknown observations fail closed. No rebind broadens this selection or introduces an old-root fallback.

## Local validation

Twelve explicit boundary checks passed: valid new binding; old protocol refused; old60/3315 receipt-source values refused; old root refused; legacy acceptance status refused; old60/3315 independently supplied source values refused; empty legacy-root allowlist; exact new trainer entrypoint admitted; legacy entrypoint refused; and argument-substring imitation refused.

The two existing recovery/scalability test modules were loaded locally with their watcher global replaced by the actual new module, leaving shipped files untouched. Of 51 discovered tests, 48 passed and 3 existing Linux-only actual-pidfd tests were skipped on the local platform. Passing cases cover foreign-run/job/UID ownership, allocation mismatches, changed child/parent epochs, post-pidfd rejection, no-pidfd refusal, bounded recovery, retained descriptor cleanup, and frozen-original versus PID replacement behavior. These simulated tests do not establish actual Linux signal behavior or GPU occupancy detection.

Exact file comparison confirmed no `owned_processes.py` difference. The watcher textual diff was limited to its four binding assignments. No publication, installation, live process signalling, remote receipt acceptance, GPU allocation, or training was performed during this review.

Signed: independent actor review agent, 2026-10-09.
