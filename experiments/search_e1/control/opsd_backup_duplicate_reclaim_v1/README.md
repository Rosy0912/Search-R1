# Reclaim only proven loose duplicates from two old online backups

This independent Mac-only helper fixes the exact old online run and the two
already accepted packages in `reclaim.py`: protected updates0/1/5/11, with58
members, and pinned update89, with18 members. It keeps every original `.tar`,
accepted receipt, inventory, directory, failed partial and Node asset. No
compression, new transfer, model, Torch, CUDA, HTTP, training, process signal or
Node operation is performed. It does not use or modify the compact queue's
USTAR-only validator. This is a storage operation, never a GPU-study dependency.

Root must publish and verify exact source readback before either production
mode. Required publication is accepted branch`search-e1`, the old run binding,
matching source commit/remote readback and manifest SHA, and exact equal SHA/bytes
for all payloads plus this source manifest. A bare local commit is insufficient.

## Two separate invocations

```text
python3 -B /PUBLISHED/reclaim.py prepare
  --source-manifest-sha256 SOURCE_MANIFEST_SHA
  --publication /PRIVATE/publication.json --publication-sha256 PUBLICATION_SHA
  --source-commit ACTUAL_COMMIT --output /NEW/PRIVATE/independent-plan-directory

python3 -B /PUBLISHED/reclaim.py execute
  --source-manifest-sha256 SOURCE_MANIFEST_SHA
  --publication /PRIVATE/publication.json --publication-sha256 PUBLICATION_SHA
  --source-commit ACTUAL_COMMIT --plan /PRIVATE/plan-directory/plan.json
  --plan-sha256 ACTUAL_PREPARED_PLAN_SHA
```

`prepare` does not remove anything. It fresh reads each original tar in one pass,
checking its whole SHA, all member SHAs, exact58/18 inventory and declared byte
counts. It then independently fresh reads every corresponding loose file and
requires three-way historical/archive/loose SHA equality. The exact raw GNU
header format is accepted only for the protected package, and exact USTAR only
for CP89. Path escapes, symlinks, hardlinks, extensions, sparse/special/link
members, duplicate/extra/missing members and nonzero trailing/padding bytes
reject preparation. Complete/inventory JSON is pinned by known SHA. The emitted
plan includes device/inode/size/mode/UID/nlink/mtime/ctime for every loose file and
preserved archive/receipt. A new independent0700 directory is required.

`execute` uses the immutable plan at its explicit SHA, the same published source
identity and original small pinned receipts. It repeats only small metadata
identity checks, not multi-GB hashes. Existing intent/journal must match exactly.
A durable per-file pending receipt precedes unlink; a durable completion entry
follows unlink and parent-directory fsync. After a crash, absent files are accepted
only with their exact prior pending intent and recorded as absence after intent,
never fabricated new unlink evidence. Missing unplanned files, foreign added
files, replaced identities or reappearing completed files reject execution.
No directory is removed. Original archive and receipt metadata are checked
before each file deletion and after completion.

Only actual Mac UID/EUID501 can call production CLI. Both package-specific
nonblocking leases are held across each invocation. `lsof` must report no open
loose files or retained archives; missing capability, warnings, unknown outcomes
and open files reject. The prepared files must remain quiescent between prepare
and execute. These are observation/advisory controls, not an atomic filesystem
prohibition on another process opening or mutating a file after the final check.
Unlink uses a verified parent directory handle and immediate exact file metadata
check. Do not run alongside a reader/writer of these selected old backup files.

Returned logical duplicate bytes are not measured APFS free space. This does not
change or revalidate numerical Adam, training, effects or any active study.

## Necessary small CPU fixtures

```text
python3 -B -m unittest discover -s /PUBLISHED/opsd_backup_duplicate_reclaim_v1 -p 'test_*.py' -v
```

Fixtures use temporary tiny GNU/USTAR files. They cover both valid formats,
malicious member paths/links/extra files, loose aliases, replaced identities,
busy/uncertain inspection, absent-without-intent rejection and crash-after-intent
recovery. They neither read nor reclaim the actual76-member backups. Actual
multi-GB preparation and actual reclamation remain Root's separate operations.
