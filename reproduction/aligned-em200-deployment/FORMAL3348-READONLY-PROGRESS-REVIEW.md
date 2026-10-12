# Formal3348 read-only progress request review

Result: CODE_STATIC_ONLY; no remaining monitoring must-fix found. Reviewer: alignment_generation_review. No remote execution, fixtures, training, release, submit or process action occurred.

Request: requests/read-formal3348-progress.py

Frozen SHA256: 258f83c17499659fa8a28add693bc54cf86dab485c561546e6e97e02c966623d

Static AST/read-through checked fixed h100-5/SOURCE080/job3348/token66cb18840a404586ab3333f8c393b8b7/RUN and canonical actual submission binding. Reads are limited to that receipt, own train.log/resource-watch.jsonl and bounded read-only scontrol/squeue queries. No training/module imports, mutation or final acceptance result is produced; scope explicitly says live snapshot, not full ledger/training acceptance. Slurm command exit/stdout/stderr remain factual, including failures.

File snapshots require canonical non-symlink ordinary files,256MiB maximum initial size, exact prefix read length, matching descriptor/path inode+device and no observed descriptor truncation. Appended bytes are explicitly excluded from the captured prefix. Each raw prefix is SHA-bound; an incomplete final line is excluded from JSON parsing but its exact byte count/SHA retained. Invalid complete UTF8/JSON lines fail instead of disappearing. Full completed metric, EM and debug-evaluation histories plus marker counts/latest records retain the observed structured train evidence; latest all-rank attempt is reported without claiming semantic/native verification or200 completion.

Resolved monitoring finding: reason_counts alone previously collapsed recovery/freeze events with absent reason intoNone. Current event_counts plus historical_notable_rows preserve every non-null event, pauseTrue, non-null error and non-within_threshold reason row. Later normal samples/cleanup cannot hide earlier abnormal/freeze/recovery evidence.

Limits: this is a captured live prefix, not an atomic snapshot of multiple files or a proof of all file bytes remaining immutable. It assumes the actual writers append; inode checks cannot detect arbitrary same-inode overwrite preserving size. It reports structured log markers rather than replaying full unstructured tracebacks, and absent logs remain absent entries. Completion markers and Slurm state are observations, not independent acceptance, successful final evaluation or cleanup certification. Consumers must inspect resource notable rows and command exits, and obtain raw diagnostics if failure occurs. Malformed rows, oversized files or command timeout abort observation rather than produce a success claim. Future actual execution/results remain separate evidence.
