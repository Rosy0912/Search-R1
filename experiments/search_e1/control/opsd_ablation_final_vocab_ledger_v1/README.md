# Final shared full-vocabulary ledger: local read-only join

This candidate is for one audit after all **70 native outers and seven training
writers have really exited and released their resources**. It does not collect
Node evidence or acquire any lock. Running W/P/J and their native512 evaluations
do not wait for it. Root must publish/read back this source before a real audit.

```text
python3 -B close_ledger.py --manifest /absolute/final-inputs.json \
  --manifest-sha256 SHA256 --output /absolute/new-private-directory
```

There is no model, Torch/NumPy, HTTP, SSH, signal, RNG, producer execution,
reservation refund or cleanup. The SHA-pinned existing stdlib `FixedBackups`
validates persistence metadata; accepted finite proofs are reused. This CLI does
not read NPY payloads, repeat token/clipmass scans or recompute KL. Missing array
finite/persistence evidence remains `PARTIAL`; it can be supplied later from the
existing published `npy_vector` reader in a separate authorized operation.

## Explicit final input

The manifest schema is `opsd-ablation-final-vocab-ledger-inputs-v1`. All references
use `{path,sha256,bytes}`; absolute nonsymlink local files are read with SHA and
before/after metadata checks. Native original Node paths are **labels only**.

- `study_id`, `aggregate_source` (published b7a264…), `producer_source` (f42fee…)
  and `producer_manifest` (original 871 manifest 8b22e6…). The producer is only
  read as bytes, never imported. Original source/runtime/cohort/initial identities
  must agree across all seven configs.
- `facts`: accepted original `opsd-ablation-outer-facts-v1`, exactly all70 complete
  with ten unique rows per group. Native completion counts and each committed
  fact update sequence are checked; this is not a tensor/gradient re-audit.
- `documents`: `[{original: NATIVE_FILEPROOF, local: LOCAL_FILEPROOF}, ...]`.
  The two SHA/byte counts must match. Include seven configs, the 70 original
  `sources.completion` documents from facts, seven status and foreground-terminal
  documents, and each terminal's original actual-child-exit and actual-release.
  No live original path is accessed. Standard original fields beyond
  path/SHA/bytes are preserved but are not confused with local inode identity.
- `groups`: exactly ordered C0,C1,M,R,W,P,J; each has `variant`, native `config`,
  `status`, `terminal` refs resolved through `documents`, and `coverage` below.
- `ledger`: `{original,local,copy_execution}`. `original.path` must be the one
  shared config root plus `/ledger.json`. The original fileproof includes
  device/inode/bytes/mode/mtime_ns/ctime_ns. `copy_execution` is a SHA-bound local
  actual execution receipt containing `actual_returncode:0`, exact `source` and
  `local` refs, `source_before`, `source_after`, `started_unix`. Both source stats
  must equal each other and the original fileproof. Copy start must follow all
  seven actual external release times, not process disappearance.
- `physical_inventory`: SHA-bound final snapshot with schema
  `opsd-ablation-final-vocab-physical-inventory-v1`, `accepted:true`,
  `actual_read_returncode:0`, `started_unix`, `source_metadata_stable:true`,
  `scanned_roots` exactly the seven `output_root/outers` paths,
  all seven `run_ids`, `ledger` original ref, and `ledger_source_before/after`
  identical to the stable ledger-copy stats. `files` enumerates **all** original
  NPY paths under those roots, including failed/retry/partial attempts, with
  path/bytes/sha256/device/inode/mode/nlink/mtime_ns/ctime_ns. Unknown NPYs cannot
  be omitted. SHA is either actual readback or an accepted source-inventory SHA
  with all original source metadata unchanged; this origin must be recorded in
  the collection proof. The reader crosschecks those accepted inventory stats.
- `physical_inventory_execution`: actual local receipt with `actual_returncode`,
  nonempty `argv`, and `stdout` ref exactly matching `physical_inventory`. A
  manifest `writer_quiescent:true` or `accepted:true` alone is insufficient.
- Optional `classifications`: `{output,token_index,kind,reason,evidence:[REF...]}`
  with preserved raw small failed/retry receipts. Kinds are `failed_attempt`,
  `retry_abandoned`, `committed_record_write_interrupted`, `inert_reserved`.
  These are declared interpretations, never a new exit0/finite/complete proof.
  Uncommitted complete and every residual reserved entry conservatively keep
  `artifact_complete=false`, even when accounting and writer release are sound.

The real native structures used here were inspected from R original closure
bytes (G30/R-closure-read-1349) and original 871/72 source: status phase
`TRAIN_COMPLETE`, outer_iterations10, original run/config/owner/finalCP; terminal
schema `opsd-ablation-group-foreground-terminal-v1`, actual_exit_code0; separately
read referenced actual wait0 and release bytes. Releases must have the original
clean same-UID audit/blockers[], actual controller anchor and no owned GPU
processes. These are historical release facts, not a new resource admission.

### Accepted coverage inputs

Every group `coverage` has `kind`, `config_sha256`, and `backups` exactly as the
published `FixedBackups` API (direct or accepted gzip specs).

For `kind:recorded`, supply `publication`, `details`, `derived_manifest` from
99e97… coverage; artifact/source SHA, run/config, deterministic selection and
NPY finite checks must match. The reader consumes `full_vocabulary_records`,
`NPY_checks` and accepted inventories, without re-running coverage.

For C0 only, `kind:C0_legacy` uses its earlier accepted `publication` plus G28
`finite_reader` stdout (5d70e7…), whose exact prior coverage/source/array refs
are verified. C0's prior header-only acceptance is insufficient on its own.

## Verdict and budget

`ledger.bytes = sum(complete.actual_bytes) + sum(reserved.reserved_bytes)` must
hold and be <=25,769,803,776. Actual physical bytes and reserved capacity are
reported separately. Each complete reservation has four independent named
FP32[151936] vectors. Counts derive from actual preselection/cap skips/retries;
there is no fixed array count. Equal teacher/target SHA is allowed but the files
remain distinct paths and physical identities; archive container inodes never
stand in for original NPY identities.

Outputs are `audit.private.json`, safe `summary.json`, and
`derivation-manifest.json`. Exit0 means this narrow artifact join is complete;
exit2 means `PARTIAL` with explicit missing/invalid evidence. Accounting, writer
release, committed coverage, persistence and residual entries stay separate.
Partial evidence is not a claim that training failed. Nothing is repaired or
deleted. This does not prove distribution normalization, numerical KL/gradient
semantics, GPU resume, or the separate 5632 formal evaluation requirement.

Tiny synthetic checks only:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s ABS_THIS_DIR -p 'test_*.py' -v
```

They check join arithmetic, equal-content independent files, capacity/duplicate/
unregistered/wrong-reference rejection and missing-proof partial behavior. They
do not audit actual final assets, execute a writer, or emulate CUDA/NCCL.
