# Final original full-vocabulary collection (candidate)

This stdlib-only collector fills the Node collection input of the existing
`opsd_ablation_final_vocab_ledger_v1/close_ledger.py` (especially lines299–329).
It does not change or run that consumer. **Do not run on actual assets until
root independently reviews, commits, pushes and verifies the published source
bytes/FETCH_HEAD, and all seven original training writers have really closed.**
Running training or native evaluation does not wait for this future operation.
This implementation/its tiny fixtures do not establish current writer quiescence.

## Explicit input

The SHA-pinned UTF-8 request arrives on stdin; no request file is written on Node.
The schema is `opsd-ablation-final-vocab-collection-inputs-v1`:

```json
{
  "schema": "opsd-ablation-final-vocab-collection-inputs-v1",
  "study_root": "/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008",
  "study_id": "EXACT_ORIGINAL_STUDY_ID",
  "groups": [
    {
      "variant": "C0", "run_id": "EXACT_ORIGINAL_RUN",
      "config": {"path": "/EXACT_ORIGINAL_CONFIG", "bytes": 1, "sha256": "EXACT64HEX"},
      "status": {"path": "/EXACT/group-C0/runs/status.json", "bytes": 1, "sha256": "EXACT64HEX"},
      "terminal": {"path": "/EXACT_ORIGINAL_TRAINING/group-terminal.json", "bytes": 1, "sha256": "EXACT64HEX"},
      "outers": [{"path": "/EXACT/group-C0/runs/outers/outer-00000/training/outer-complete.json", "bytes": 1, "sha256": "EXACT64HEX"}]
    }
  ],
  "accepted_source_inventories": [
    {
      "variant": "C0", "run_id": "EXACT_ORIGINAL_RUN", "config_sha256": "EXACT64HEX",
      "reference": {"path": "/MAC/ACCEPTED/source-inventory.json", "bytes": 1, "sha256": "EXACT64HEX"},
      "content_base64": "EXACT_ORIGINAL_INVENTORY_BYTES_BASE64"
    }
  ]
}
```

The abbreviated example is not runnable: `groups` must be exactly ordered
C0,C1,M,R,W,P,J, with seven distinct original runs and ten ordered completion
refs per group. All refs pin original bytes/SHA. Fixed production root and
UID1005 are code constants with no CLI bypass. Output roots must be exactly
`STUDY_ROOT/group-V/runs`; every original closure ref and shared ledger must
stay under that fixed root. Terminal refs recursively supply the original
actual-child-exit and actual-release refs, read as pinned small JSON.

The gate requires each original ten-outer completion, TRAIN_COMPLETE10 status,
matching final checkpoint reference/contract/run/config, original foreground
exit0, separately read actual parent wait0 and both native/external clean
same-UID release receipts, anchored to the original controller identity.
Release blockers/owned GPU process lists must be empty. No process absence is
used as exit0. Common original study/source/runtime/initial/cohort/helper
identities and shared ledger root must agree. This narrow gate does not repeat
the final consumer's accepted-facts counters, coverage, math or finite checks.

Accepted inventories are embedded because their preserved Mac paths are labels,
not Node paths to read. Root must select **already accepted, pinned** inventories;
each embedded original byte string is SHA/length checked and must have a
`files` list. Each envelope explicitly binds original variant/run/config SHA;
inventory fields, when present, must agree. An inventory with `accepted:true`
records its own acceptance. Published generic `opsd-ablation-closed-raw-inventory-v1`
has no such flag: also embed `acceptance:{reference,content_base64}` for the
already accepted `opsd-ablation-closed-raw-compressed-local-backup-v1` receipt.
Its exact inventory SHA, original identity, actual SSH0 and original lossless
member verification flags must agree. Both pins are recorded in SHA origins;
the collector does not repeat decompression or archive hashes.
For every NPY, accepted original device/inode/bytes/mode/mtime_ns/ctime_ns must
exist and match current metadata exactly; nlink is also compared when present.
Missing/changed/conflicting accepted NPY metadata causes HOLD. No accepted
inventory is inferred from a filename or from a caller's `writer_quiescent` flag.

## Future invocation and outputs

After publication/readback and genuine closure only:

```text
python3 -B capture.py --request /ABS/final-collection-request.private.json \
  --request-sha256 EXACT_SHA --source-sha256 PUBLISHED_COLLECT_PY_SHA \
  --remote-collector /data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/source/PUBLISHED_DIR/collect.py \
  --output /ABS/NEW_PRIVATE_OUTPUT
```

The wrapper uses the current ordinary `hkust1` alias, BatchMode and an 8-second
connection limit. Node executes `python3 -B collect.py` with the request on
stdin, returns one JSON stdout, and writes no Node file. There is no helper
deployment, model/GPU/Torch/NumPy/HTTP, lock, signal, cleanup or producer execution.

After all closure gates pass, the collector reads the one original shared ledger,
enumerates **all** `*.npy` recursively below all seven `output_root/outers`, and
enumerates again. Failed/retry/partial subdirectories are included without a
success-name filter. Unknown NPYs are freshly SHA256-read in chunks and marked
`fresh_sha256_readback`; accepted unchanged source metadata reuses its pinned SHA
and records the exact inventory/source-file origin. No other large payload is
read. All original NPY and directory metadata must match across enumerations;
fresh hash fd/path stats, explicit small closure stats, and the ledger's exact
bytes/stats must remain stable. Symlinks, scope escapes, special files, missing
closure, or changes return exit2/HOLD; failed partial snapshots are not accepted.
Small closure reads are cached, not broad recursive JSON rehashes.

Exit0 stdout is the consumer's
`opsd-ablation-final-vocab-physical-inventory-v1`, including `ledger`,
`ledger_source_before/after`, seven `scanned_roots`/`run_ids`, `started_unix`,
`actual_read_returncode:0`, and every NPY's original path/bytes/SHA/device/inode/
mode/nlink/mtime_ns/ctime_ns. It also carries exact base64 ledger and small
closure bytes, SHA origins, and source/request identity. `finite_checked:false`
is unconditional: byte collection does not claim finite arrays or scientific
acceptance.

Mac always preserves actual Popen PID, `communicate()` wait return code, argv,
stdout SHA and stderr in `ssh-execution.private.json`. Only after actual wait0,
accepted collector output and exact decoded local bytes does it create
`physical_inventory_execution.private.json`, `ledger-copy-execution.private.json`
and `consumer-copy-refs.private.json`. The latter supplies consumer-compatible
document/ledger/inventory refs. `ledger-copy-execution` binds the original
before/after stats and collection start, exact Mac copy, and actual SSH receipt.
Nonzero SSH keeps stdout/stderr/execution evidence and creates no accepted copy
or physical-inventory execution receipt. PID disappearance never supplies rc0.

## Tiny CPU I/O fixtures only

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s ABS_THIS_DIR -p 'test_collection.py' -v
```

These use temporary tiny byte files and real local CPU subprocess waits. They
check missing closure/nonzero actual wait rejection before scanning, accepted
generic inventory's separate pinned acceptance,
metadata SHA reuse, additional unknown failed/retry NPY inclusion, symlink/NPY/
ledger changes and unknown metadata rejection, plus wrapper exit7/exit0 receipt
behavior and exact ledger/stdout copies. They do not parse NPY values, rerun
math/coverage tests, import the existing consumer, access Node or execute SSH.
