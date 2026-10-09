# Fixed local original-tar reclamation

This independent Mac UID501 entry can remove exactly one regular file:
`opsd-online-launch-20261008/checkpoint-backup/protected-0-1-5-11-20261008/protected-checkpoints.tar` under the fixed workspace analysis root. Its accepted GNU bytes total 6,239,805,440 with whole SHA `0646f2da8b338108f753f880b30859a96831681cc99767cc6c47a50da93e9dd1`.

The fixed `5ad8a6d` XZ completion must already exist. It must bind the actual child wait rc0 and worker PID, independently decoded original whole SHA/bytes, one CRC64 XZ stream, retained original identity, final XZ metadata, historical 58-member receipt/inventory, and exact source publication/readback. The worker's old `.partial` path is historical; final inode/bytes/SHA/mtime/owner/mode/nlink continuity is checked while current final metadata includes its rename ctime. LIVE/incomplete compression cannot prepare a plan.

An independent root-observed actual **parent** rc0 is also mandatory: fixed `goal-continuation-20261009-27/recompression-actual-5ad8a6d/root-observed-parent-terminal.private.json`, schema `local-recompress-parent-terminal-root-observation-v1`, actual PID41057/UID501, `actual_returncode:0`, `status_not_inferred_from_absence:true`, pinned parent `start.private.json` SHA `5731582efd0e64e6d62840f657ef614ea7b7240307065f7912b198edae34bb2d`, exact completion ref, and original actual-wait evidence ref. All small refs require SHA/stat binding. Parent and child argv must differ only by the worker-only flag. Missing/nonzero parent proof rejects prepare; execute revalidates the same fixed parent proof through the immutable plan. Child rc0, completion existence and PID absence never substitute for actual parent return code.

No 58-member or model tensor hashes are repeated: exact original GNU bytes preserve the historical member acceptance. Only small proof/source SHA reads and current file metadata are used. The fixed published duplicate-reclaim helper is compiled from SHA-verified source bytes for its metadata, no-open, JSON/fsync and path primitives; its main/prepare/execute are never called. No Node, model, Torch, GPU, signal, training lock or queue dependency exists.

After this source is committed, pushed and independently read back, root supplies its publication refs:

```text
python3 -B reclaim.py prepare --source-manifest-sha256 SHA --publication PUB --publication-sha256 SHA --source-commit FULL40 --completion-sha256 ACTUAL_COMPLETE_SHA --parent-terminal-sha256 ACTUAL_PARENT_TERMINAL_SHA --output NEW_ABSOLUTE_PLAN_DIR
python3 -B reclaim.py execute --source-manifest-sha256 SAME_SHA --publication SAME_PUB --publication-sha256 SAME_SHA --source-commit SAME_FULL40 --plan PLAN_DIR/plan.json --plan-sha256 ACTUAL_PLAN_SHA
```

Prepare creates only a new immutable plan plus the original package's local nonblocking cooperative lease. Execute revalidates the complete chain, all current metadata and no-open observations, persists one intent and one pending record, checks the parent directory fd and exact immediate stat, unlinks that one basename, fsyncs, and records after/done/completion. A restart may report absence only with that exact durable pending and intent; it never infers an actual unlink from absence. Any changed source, path, inode, size, mode, owner, links, timestamps, XZ/proof identity, or uncertain open-file check rejects the operation.

XZ, all receipts, source/publication records, Node originals, directories, partials and other archives survive. Logical duplicated bytes are reported; actual APFS free gain is not presumed. Cooperative locks and `lsof` are observations, so the selected package must remain quiescent through the final check/unlink boundary.

One sandbox-only fixture covers incomplete/wrong decode proof, changed metadata, open-file rejection and a successful single-file unlink. It never touches the real tar. Run locally with `python3 -B -m unittest discover -s PATH_TO_THIS_DIR -p 'test_*.py' -v`.
