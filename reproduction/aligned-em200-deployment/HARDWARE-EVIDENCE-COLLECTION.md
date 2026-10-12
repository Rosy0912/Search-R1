# Job3345 terminal raw evidence collector

`collect_hardware_evidence.py` is fixed to source080, the new NFS ROOT, job3345 and token `13e07285c97044fd97974d94f53fea1b`. It is a read-only target CPU utility: its only subprocess commands are scheduler observations and source HEAD inspection. It never submits, stops or signals jobs/processes, invokes GPU computation, or rehashes corpus/index/encoder assets. It writes only a newly exclusive own `ROOT/evidence` archive. Original golden asset hash records are copied as evidence.

Invocation by the operator after publication/deployment, using the existing new target environment:

```text
ROOT/env/bin/python -B ROOT/tools/deployment/collect_hardware_evidence.py --output ROOT/evidence/hardware3345-terminal-raw-UNIQUE
```

Replace literal ROOT with the fixed full NFS root in the collector. The parent `ROOT/evidence` must already exist canonically. Pending/running/unverified state prints raw scheduler observations and `acceptance=false`, creates no terminal archive and does not claim GPU use. It never treats disabled sacct as a job failure. Actual strong `scontrol` records bind job/name/comment/source/token/command/workdir/required node/state/exit/timestamps; an available primary sacct record can corroborate or, after controller retention expiry, provide bound accounting evidence. Conflicting scheduler states/identity remain unverified. If both scheduler records have vanished or accounting is disabled and no strong controller record survives, terminal state is honestly unavailable.

At a strongly observed terminal state, it exclusively copies the fixed RUN's raw files, original submission records, Slurm stdout/stderr and tiny launcher/validator/watcher source. It records pre/post scheduler observations, missing required eight-rank numeric and retrieval/resource/UUID/cleanup evidence, per-original SHA/size/inode/time and an archive inventory SHA for every saved file. File mutation, symlink escape and missing files are retained as archival issues. Only the documented numeric/log/resource/retrieval/cleanup RUN artifact allowlist is copied; unexpected regular files, including any corpus/index/model/checkpoint bytes, are reported and excluded; the golden's original asset SHA pins remain in its copied JSON. Failure or cancellation before RUN creation still yields a scheduler/raw archive with explicit missing RUN/evidence. No original is overwritten and no existing archive can be reused.

Every output, including a completed zero-exit job with all expected files, has `acceptance=false`, `independent_review_receipt=false` and `TERMINAL_RAW_ARCHIVED_NOT_ACCEPTED`. Original rank JSON may itself report GPU fixture PASS; the collector merely archives that original and does not adopt its assertion. Root's independent reviewer must validate Slurm/record/file bindings and actual numerical/live/cleanup content separately. `slurm-before.json` contains normalized `primary.authority` and terminal fields plus raw controller/accounting responses; its byte hash is in `file-manifest.json.archive_inventory` for an eventual separately authored review binding. Manifest self hash is emitted separately to avoid circular hashing.

Local tests:

```text
../fresh200-local-test-env/bin/python -m pytest -q reproduction/aligned-em200-deployment/test_collect_hardware_evidence.py
27 passed
```

These are CPU synthetic scheduler/file fixtures covering pending/running/completing/suspended rejection, terminal failed/cancelled/completed/timeouts, immutable raw copies and missing evidence, wrong job/token/controller command/root/node bindings, disabled sacct, purged controller/accounting absence, symlink escapes, duplicate archive refusal and scheduler transitions. The actual saved03:50 PENDING/Resources controller snapshot was also parsed locally with disabled accounting and remained unaccepted. No target collector execution, actual terminal transition, GPU allocation, remote operation, training or independent acceptance was performed by the implementation agent.
