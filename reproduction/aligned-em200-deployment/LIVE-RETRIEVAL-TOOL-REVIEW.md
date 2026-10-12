# Independent source review of live retrieval verification

Reviewer: `alignment_generation_review`. Date:2026-10-09 (Asia/Shanghai). Scope: unexecuted deployment-tools `verify_live_retrieval.py`, `hardware_inner.sh`, relevant worker/preflight and source080 retriever/identity consumer. No remote operations, GPU execution, HTTP service launch, training or fabricated actual GPU PASS occurred.

## Result: fixes resolved; source review passes

**PASS for the latest local source-review scope only.** No remaining blocking defect was identified. The tools have not been deployed or executed on GPUs by this reviewer; this document is not live-service, training or hardware acceptance.

| Final reviewed file | SHA256 |
| --- | --- |
| `verify_live_retrieval.py` | `66dbe3d7cf77fc60a86207ec6d17210848a5f0e4c7e769109b1a895a80e06e2c` |
| `bind_retrieval_proof.py` | `3ae4f0095b775716b0f2d3180b513906a7be51837d1f2786b3ec72447733704d` |
| `hardware_inner.sh` | `d89e90ba172fb3ade4ad670de39abe656fb55addb2274c77fc454fc25cfb6ac1` |

The originally reported must-fix issues were independently reread after repair:

1. **Golden hash is independently frozen before service start.** The new binding helper runs after foreground archive capture exits and before the port check/service launch. It reads the collector's actual final completion JSON, verifies output path/hash/probe count/scope and claim equality, and exclusively writes `capture-completed-before-live.json` binding manifest/claim/log/collector hashes and source/run/job. Live verification checks these bytes against that prior record, binds exact collector identity, archive hard hash, independent scopes and eight current/captured lease UUIDs. The consumer's expected manifest SHA now comes directly from the completion record; returned consumer SHA must equal it, and the golden file is rehashed after validation. This closes the final transient-substitution window left by using a fresh self-hash as the trusted expected value.
2. **Actual launched service birth is bound.** Immediately after shell `$!`, before readiness polling, the helper records a stable live `/proc` start epoch, PPID, UID, exact own executable path/device/inode, exact CUDA/FAISS/loopback/port18083 command arguments and RUN/job markers. Expected PPID is the actual phase shell `$$`. The helper rechecks birth during observation and refuses zombies; its bounded retry only accommodates initial exec transition. Live verifier checks the same recorded identity before and after long asset/probe validation. The exact helper module file is checked too. Numeric PID plus markers alone is no longer the admission basis.
3. **Verifier checks clean source080.** It requires HEAD080 and empty porcelain status before importing actual new-checkout preflight/identity modules. Existing actual module-file checks remain. Mandatory guards use ordinary exceptions and are not disabled by optimization.

Independent review also recomputed every `HARDWARE_FILES` pin in `submit_hardware.py`: all nine pinned files equal their actual current bytes, including helper, verifier, inner script and archive collector/reference. No submit action was performed. This checks publication binding, not execution of those files.

## Checks that are correct in the current source

Canonical ROOT/runs hardware name and same job, own environment/host, source080 HEAD, exact imported preflight/validator module file paths, live inherited GPU-UUID flock and same-run resource-watch ancestry are checked. Guard uses ordinary `require` exceptions, so Python optimization cannot remove the verifier checks. Requests uses a Session with `trust_env=False`; endpoint is fixed loopback18083. Golden claim equality, query batch size1 and twelve probes are required. The actual source080 `validate_retriever_identity` independently hashes complete pinned corpus/index/encoder/production service/identity sources, checks loaded/current live identities and actual CUDA/FAISS runtime, requests singleton probes and compares exact ordered contents, passage formatting and head500 tokenizer IDs. It returns the live PID, verified count and GPU identity while explicitly leaving full GPU parity/training claims false. Scope text correctly limits the tool result to archive golden versus own local service.

The production `retriever_local.py` runs the actual source080 retrieval server and forces loopback. Its parser supports the inner script's CUDA/FAISS/host/port/thread flags and matches archive fp16 mean/query-prefix/query256/topk3/batch512 settings. Identity endpoint is enabled before starting the service.

## Startup and exit order

`hardware_inner.sh` runs the eight-rank numerical fixture foreground, then the archive golden collector foreground. Shell `set -euo pipefail` prevents advancing after a failed collector. The collector therefore exits before the port18083 availability bind and live service launch; simultaneous full archive/live index allocation is not intentionally requested. This is source ordering, not evidence that a real GPU process exited or memory was released. The bind-and-close probe has a small unavoidable race, but own-PID endpoint verification must reject a foreign service that wins it. Readiness uses loopback curl with proxies disabled and confirms the launched PID remains alive.

The outer hardware worker creates a new canonical RUN exclusively, sets source/environment/run markers, takes its own execution lock, enters preflight GPU leases and live resource watcher, and arranges own-process cleanup on exit. This review did not execute or approve that eventual hardware job. No training appears in the inner script.

## Locally executed rejection evidence and limits

AST parsing of the verifier succeeded. Two actual subprocess rejection branches were run locally: an unowned `/tmp` RUN rejected with `same new hardware run required`; a canonical-looking target RUN under the wrong local environment rejected with `new local environment required`. Both exited nonzero before network/GPU/service use. After repair, those branches were rerun and the new binding helper also actually rejected an unowned RUN with `canonical same-job hardware run required`. Additionally the exact verifier frozen-completion refusal expression was extracted via AST and executed on a changed temporary golden file; it rejected with `archive capture bytes changed after completion`. This only tests a rejection expression, without preflight, network or GPU execution. Final Python AST parsing and shell syntax checking passed. No mocks produced acceptance, and no actual GPU PASS was manufactured.

Separate CPU/tool reviews and target CPU input receipts do not prove this live GPU service validation. Eventual capture, completion binding, actual stable service PID/runtime/assets, twelve exact singleton probes and owned cleanup must be recorded on the real target. A future live-service result still is not training acceptance, first-five release, or200-outer completion.

Signed: `alignment_generation_review`, independent source reviewer; identified must-fix items resolved for the exact final hashes above. No runtime approval or actual GPU acceptance is claimed.
