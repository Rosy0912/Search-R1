# Closed seven-group raw backups

One independent stdlib implementation replaces repeated hardcoded C1 copies. It accepts exactly one C0/C1/M/R/W/P/J run through an immutable JSON `--scope/--scope-sha256`. All ten original outer completions and immutable checkpoint pins plus the original group controller's actual foreground rc0 and owned-release proof are required before inventory. Missing closures are backup HOLD, never a training failure. There is no training lock, callback, GPU/model/Torch import, HTTP, signal, remote write, deletion or numerical optimizer audit. Root must publish and byte-readback all source before invoking it. No real transfer is claimed here.

`source-operation.py` provides inventory, USTAR stream and source-after modes. `transfer-gzip.py` runs only on Mac, using the existing direct HKUST1 SSH route, original Python environment, CVD empty and independent CPU process. `verify-archive.py` reads the saved compressed file completely, checks every original member SHA and exact set, gzip footer/CRC, terminal USTAR padding, original transfer rc0 and fresh source-after evidence, then atomically renames and creates `complete-sha-verified.json`. No bare local extraction is made. Failures keep partials and all Node originals. Disk admission reserves the exact USTAR upper bound plus1% and5GiB, without assuming compression.

The reused dependency is published `opsd_ablation_outer_backup_v2/backup.py`, exact SHA `d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57`. It supplies fixed-handle SHA/stat/symlink protection, safe USTAR names/streaming and fsync. Its actual source bytes are compiled directly, not imported through timestamp-valid pyc. The new source-operation bytes are similarly SHA-checked/compiled by both Mac entries. Scope JSON and inventory files are read through the same fixed-handle reader. Existing published helpers and production code stay unchanged.

The scope shape below uses placeholders; it cannot launch with unspecified references. References may include original metadata, which is checked whenever present. `outers` must contain exactly indices0..9 in order. Update counts are the actual immutable producer values; K0 may retain its update count if there is an accepted checkpoint for that same outer. A completion pointing to an earlier outer is unsupported/HOLD and is never relabeled.

```json
{
  "schema": "opsd-ablation-closed-raw-scope-v1",
  "accepted": true,
  "cluster": "HKUST",
  "variant": "M",
  "run_id": "EXACT32HEX",
  "config": {"path": "/EXACT/group-M/receipts/authorized-launch-20261009/config.json", "sha256": "EXACT64HEX"},
  "output_root": "/EXACT/group-M/runs",
  "closed_outers": [0,1,2,3,4,5,6,7,8,9],
  "outers": [
    {"outer_index": 0, "optimizer_updates": 11,
     "completion": {"path": "/EXACT/group-M/runs/outers/outer-00000/training/outer-complete.json", "sha256": "EXACT64HEX"},
     "checkpoint": {"path": "/EXACT/group-M/runs/checkpoints/update-00000011-outer-00000/manifest.json", "sha256": "EXACT64HEX"}}
  ],
  "status": {"path": "/EXACT/group-M/runs/status.json", "sha256": "EXACT64HEX"},
  "group_terminal": {"path": "/EXACT/STUDY/runs/02-M/training/group-terminal.json", "sha256": "EXACT64HEX"},
  "include_outer_raw": true,
  "global_files": [],
  "external_full_vocab_files": []
}
```

The actual study base and group paths are fixed to the current authorized HKUST1 study; `/EXACT` denotes those real paths, not a new configurable cluster. Inventory rechecks config/run/variant, all ten completion/CP SHA/counters, final `phase=TRAIN_COMPLETE`, the native nested parent wait identity/actual rc0, and original release. It does not infer rc0 from PID absence. Checkpoint manifest identity is read, not optimizer tensor payloads.

With `include_outer_raw:true`, all original regular files under ten outer roots are included: five sibling rollouts, failed attempts/retries, HTTP responses, student/teacher traces, all scalar and top32 token logs, and NPY files as original bytes. Only each outer's top-level `merged-model-cache` is excluded. Checkpoint payloads are outside these roots and have their independent compact queue. No failed attempt is recategorized as successful.

`global_files` can supplement this, or `include_outer_raw:false` can make an independent closed-global archive. Every listed file requires exact path/SHA/bytes; allowed locations are this group's phase-exits, fixed child-history/status/run-identity/config, dtype-recovery directory, and the terminal's closed training directory. There is no directory wildcard or mutable study driver/observer/evaluation scope. External shared repair files not in those locations remain separately backed up and are not claimed covered.

Actual trainer `VocabularyDump` writes NPY payloads inside the outer's training attempt/rank directory, so these original bytes are automatically covered. The study-wide `runs/full-vocabulary` root holds the mutable ledger/cap lock; those are excluded and remain pending until study closure. This version refuses nonempty `external_full_vocab_files` rather than pretending an unbound external full-vocab manifest exists. Native512 outputs remain an independent closed-evaluation scope.

Execution after publication uses one exact scope JSON on Mac. Its original bytes are base64-encoded and transmitted as a safely quoted argument; no scope/source file is written on Node. `--scope-base64` and an already-existing Node `--scope` path are mutually exclusive. The Mac transfer requires the inline route, so all three remote modes must use the same original base64 bytes and SHA. Python `-c` contains the published program, while stream/after inventory JSON is a separate stdin; the two inputs are never conflated.

1. Run Node `source-operation.py inventory --scope-base64 ORIGINAL_BYTES_BASE64 --scope-sha256 SHA` through the original read-only Python stdin/`-c` route; use `shlex.join` for the literal argument list and preserve stdout, stderr and actual rc separately.
2. On Mac, `python3 -B transfer-gzip.py --scope LOCAL_SCOPE --scope-sha256 SHA --inventory INVENTORY --destination NEW_UNUSED_DEST --source-operation PUBLISHED_SOURCE_OPERATION --operation-sha256 SHA --source-helper PUBLISHED_LOCAL_V2_BACKUP_PY`.
3. Run Node source-operation `after` with the original inventory bytes on stdin and the same literal base64 scope/SHA. Save its actual stdout/rc. It recomputes closures, exact noncache file set and every fixed metadata identity; stream mode separately hashes all actual payload bytes while emitting USTAR.
4. On Mac, `python3 -B verify-archive.py --scope LOCAL_SCOPE --scope-sha256 SHA --inventory INVENTORY --after-metadata AFTER --destination DEST --source-operation PUBLISHED_SOURCE_OPERATION --operation-sha256 SHA --source-helper PUBLISHED_LOCAL_V2_BACKUP_PY`.

The new destination is0700 and must not already exist. Output receipts prove original byte transport, not ten-outer science/tensor/effect acceptance. Old local raw copies are not imported; reuse/space reclamation is outside this minimal version. Training and native512 evaluation never wait for backup completion.

Five small CPU fixtures cover all10 identity/archive closure, cache exclusion, partial/changed config rejection, nonzero native wait/mutable-ledger rejection, exact literal scope byte/SHA transport and corrupt/truncated member verification. They do not run SSH, real payload hashing, a scientific model, or any production transfer.

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_ablation_closed_raw_backup_v1 -p 'test_*.py' -v
```
