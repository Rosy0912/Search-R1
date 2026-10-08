# Independent hardware-only launch code review

Local read-only review on 2026-10-09 by the actor review agent. Scope: `hardware.sbatch`, `hardware_worker.sh`, `hardware_inner.sh`, `submit_hardware.py`, and only the entrypoint interaction with `verify_live_retrieval.py`. The live retrieval validator is independently reviewed by its assigned reviewer. No Slurm command, remote operation, GPU allocation, training, or production-source modification was performed by this review.

**Conclusion: no remaining must-fix findings in the final reviewed bytes.** The author repaired all three independently identified issues: unknown response-file admission, truncating unchecked worker lock, and symlink-sensitive run/log directories. The live-retrieval reviewer has separately finalized the dependent validator/binder; their detailed proof checks are outside this review.

## Submission and unknown outcomes

The submitter requires the exact new environment/host, clean source080 repository, canonical root and scheduler directory, SHA-bound hardware files, exact imported new watcher, actual interpreter-anchor verification, and hardware-only approval status. It holds a nonblocking submission-global flock, refuses any existing intent, receipt, or response including symlinks, and checks for already active new-source BAPO comments before submission. Queue inspection does not cancel, change, or manage any old/global allocation.

The exclusive intent write flushes/fsyncs the file and containing directory before the sole `sbatch` invocation. Timeout, nonzero exit, malformed output, or response-write failure retain the intent and block further submissions. There is no retry or requeue branch. A valid numeric job result produces a separate receipt with source/token/job/run binding and explicit submission-only scope. The response alone is never hardware acceptance. Independent review caught the initial missing response-file precheck; the author added it before any scheduler submission.

Nine final local scenarios executed the actual submission control flow with all scheduler subprocesses replaced by local mocks: success (exactly one submission); timeout, nonzero result, malformed result (each retained intent and subsequent invocation made no further submission); preexisting intent, receipt, response (each made zero submissions); and symlink runs/logs directories (each rejected with zero submissions). Real local exclusive writes and fsyncs ran in temporary canonical directories. Interpreter/host/source/tool prerequisites were fixture-bound, so these tests establish submission state semantics rather than actual target admission.

## Resource and cleanup flow

The sbatch specification requests one h100-5 node, eight typed H100 GPUs, exclusive allocation, no requeue, and an isolated initial environment. The worker requires a hexadecimal token, actual host, numeric Slurm job, no restart, visible GPUs, clean source080, and a global execution lock before creating its new run. Cache/offline and library environment changes are process-local.

The worker checks canonical root/head/lock and an existing ordinary, single-link, own-UID lock before opening it in append mode and taking nonblocking flock. Independent review caught the original truncating unchecked lock open; this was repaired. Seven final exact worker admission-block tests passed: absent and ordinary existing locks accepted without modifying contents; symlink, hardlink, and directory locks rejected, with foreign contents preserved; symlink runs/logs directories also rejected.

The worker and submitter now require canonical existing runs/logs/tools/deployment/repository/environment directories before any output or submission. After mkdir, the worker checks its new RUN is a canonical directory before writing source/log files or starting GPU work. The worker creates a fresh token/job-named run, exports its exact run marker, and wraps all hardware work in EXIT cleanup. Cleanup calls the unchanged owned-process tool with this run and current job; that tool selects own UID, exact marker/job, verifies process birth after pidfd open, and signals held descriptors. It does not cancel Slurm jobs or select foreign experiments. Cleanup logs original and cleanup exit status separately and promotes cleanup failure to failure when hardware work itself succeeded.

The actual source080 preflight acquires idle-device UUID leases and execs the watcher. The watcher command is `ROOT/env/bin/python ROOT/tools/resource_watch.py`, deliberately without a Python `-B` flag before its script, meeting the live guard's exact argv position. The supervisor then exports a live PID/start-epoch/same-run guard to the inner child and retains inherited lease descriptors. These contracts were traced in actual source; this review did not execute GPU inventory, leases, pidfd cleanup, or guard ancestry on the target.

## Hardware-only sequence

The inner script runs eight-rank synthetic numerical checks first; on failure `set -e` prevents the following steps. It then captures the archive-reference retrieval golden in a separate process. That process exits before the live local CUDA/E5/GPU-FAISS retriever starts, so the two full indices do not coexist. Capture completion is durably bound before service start, and the live child's exact birth and parent PID are bound before readiness. A port-bind probe precedes service start; readiness checks require the new child to remain alive. The child's PID is explicitly passed to the live validator, which receives the same run/job markers. The validator's exact provenance and parity checks remain the other reviewer's scope.

There is no production trainer invocation, real actor optimizer update, accepted-outer accounting, resume, prior RNG/checkpoint import, or first-five gate release in these files. Submission, synthetic oracle output and retrieval parity remain hardware evidence with explicit scopes; they do not create training permission or claim an official outer. Old60/3315 receipt or launch identities cannot satisfy the new root/source/protocol/tool anchor requirements. Old allocation state was neither changed nor operated upon.

Shell syntax checks passed for all three shell files. All nine final submission hardware-file pins match their local files. The final shell syntax and the 9 submission plus 7 worker-boundary tests passed. These are local fixture checks, not an actual target hardware run.

## Final reviewed file hashes

| File | SHA-256 |
| --- | --- |
| `hardware.sbatch` | `afc53d15545dea79d3c32fd5ec58216ff104474b634173c8d5346af7029f0101` |
| `hardware_worker.sh` | `15e170b11c7e36a16600d501388d326bad1f93c01a112a8828583e8a6a1e0886` |
| `hardware_inner.sh` | `d89e90ba172fb3ade4ad670de39abe656fb55addb2274c77fc454fc25cfb6ac1` |
| `submit_hardware.py` | `4fdfa4c9b2116ec1fbbfba955380ab97fbce7c4a232d56d8373e85daa59593bb` |

Signed: independent actor review agent, 2026-10-09. This signature applies to these exact bytes and does not authorize publication, allocation, GPU work, or training on its own.
