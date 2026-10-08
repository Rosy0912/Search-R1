# Lossless compact checkpoint backups

This Mac-only queue preserves the same seven-group × ten completed-outer targets
as the published V2 goal. Every job pins config, run, outer, actual Adam count,
checkpoint and outer-completion path/SHA. K0 retains its true unchanged Adam
count. This tool proves transport byte identity and recoverability; it does not
perform numerical optimizer or model acceptance.

The unchanged published V2 Node source provides read-only discovery, inventory,
and the exact USTAR stream. Its whole source inventory, goal and publication
remain bound. Local reuse compiles SHA-checked source bytes without reading pyc;
only V2's three module-loader statements are replaced, not its function bodies.
No compact source deployment on the Node is necessary. Use the already published
direct-SSH transport shim in PATH. No GPU/HTTP, experiment lock, signal, Node
mutation, checkpoint deletion or training dependency is introduced.

## Format and acceptance

`checkpoint.tar.xz` contains the original exact USTAR bytes in one SHA256-checked
XZ/LZMA2 stream with a 128 MiB dictionary. Compression's memory use exceeds the
dictionary size; decompression has a 256 MiB memory limit and a pinned output
bound. Actual compression ratio and throughput have not been measured on a real
checkpoint. No saving estimate is used as a capacity assumption.

Each package retains `node-inventory.json` and `compact-complete.json`. Acceptance
requires an independent disk read/decompression, exact original whole-tar SHA
and byte count, and every member's declared SHA/size. Only canonical regular
USTAR members are accepted; missing, duplicate, extra, escaping, special, hidden
extension, nonzero padding, concatenated XZ and truncated streams are rejected.
The compressed file is read back again after its final rename. Receipt and
directory are fsynced. Future transfers compress SSH stdout directly; no raw tar
or extraction copy is created. Failures retain partials and actual wait evidence.

Space is checked against the fixed inventory: padded USTAR upper bound plus 1%
and 1 MiB enforced compressed-writer headroom, plus at least 5 GiB free margin.
The writer refuses to exceed this reservation. Imports reserve the new compressed
package in addition to already existing files. Space failure is BACKLOG and does
not affect training. The tool does not delete unique or unrelated files to make
space.

## Old-package import and explicit reclamation

Every mode acquires the **existing V2 `.backup-queue.lock`**, nonblocking, before
creating or mutating output. Root must retire the old queue separately. A live
old queue rejects this tool; it is never stopped by this tool. Keep the old queue
root and its lock after migration. The new output root must be independent.

Existing V2 COMPLETE jobs are imported only after their exact completion,
inventory, tar SHA and extraction metadata validate. Original metadata is fresh
at import; the new receipt explicitly preserves that historical scope. New
COMPACT COMPLETE status is durable before any optional reclamation.

Default operation never reclaims. With explicit `--reclaim-legacy`, a second full
fresh decompression/member check must pass. An exclusive durable plan pins only
that package's exact tar and `verified-files` inventory. Their original metadata
must still match and hardlinks/symlinks are refused. Only these proven recoverable
duplicates and then-empty extraction directories can be removed. The old
completion, inventory, execution and history JSON files remain untouched, as do
all Node originals. A crash resumes that saved deletion intent; missing files
are recorded as previously planned absence, not fabricated new deletion proof.
`legacy-reclaimed.json` reports logical duplicate bytes, not measured APFS space.

After reclamation the old V2 queue cannot validate the removed files. Continue
with this new queue; do not relaunch V2 or present its historical extraction
metadata as fresh. The new queue verifies compressed-file metadata continuity
and small pinned receipts on routine polling. A continuity failure is HOLD, not
silent replacement. A durable accepted receipt left before a queue-status crash
is recovered only after a new full decompression verification. Reclamation
errors retain an independently complete compact package and never cause a
redownload.

## CLI

Root supplies actual immutable references after publication/readback. The V2
source manifest is `4f9ee158ec1c326498898d26ee5b6a1d3212e94d5ecbbd24c8ef7864fa079b69`;
its commit is `554c52c845561bcab00ec56d469c945878689890`. Keep its existing goal,
Mac/Node publication paths and SHA values unchanged.

```text
python3 -B /published/opsd_ablation_compact_backup_v1/queue.py queue
  --legacy-helper-root /published/opsd_ablation_outer_backup_v2
  --legacy-queue-root /existing/v2-queue
  --goal-identity /private/goal.json --goal-identity-sha256 GOAL_SHA
  --source-manifest-sha256 V2_MANIFEST_SHA
  --publication /private/v2-publication.json --publication-sha256 V2_PUB_SHA
  --source-commit V2_COMMIT
  --remote-helper-root NODE_V2_ROOT --remote-publication NODE_V2_PUBLICATION
  --remote-goal-identity NODE_GOAL
  --compact-manifest-sha256 NEW_MANIFEST_SHA
  --compact-publication /private/compact-publication.json
  --compact-publication-sha256 NEW_PUB_SHA --compact-source-commit NEW_COMMIT
  --destination /new/compact-queue --host hkust1 --margin-gib 5
```

Add `--reclaim-legacy` only for explicitly authorized duplicate reclamation;
`--once` performs one serial pass. New source publication must bind all payloads
and its own manifest, with accepted search-e1 remote readback, exact commit and
authorized run IDs. The command's new destination cannot contain or be contained
by the legacy queue root. Queue state pins each outer once and never follows
mutable latest. All seventy completion states require intact pinned receipts and
a successful final discovery.

`verify` uses the same binding flags and queue lease, replacing `--destination`
with `--complete-receipt PATH --complete-receipt-sha256 SHA`. It independently
decompresses and verifies all members without extraction or reclamation. For a
future recovery, standard `lzma.open` can recreate the exact tar at a new private
path; verify its recorded whole SHA and use the published V2 `extract_verified`
against the stored member inventory. Reserve space for both recovered tar and
extraction before recovery. No recovery has been executed by this implementation.

## CPU verification scope

```text
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_ablation_compact_backup_v1 -p 'test_*.py' -v
```

Six focused fixtures cover exact published V2 USTAR reconstruction, truncation,
member/whole-tar identity tampering, special/escape members, concatenated XZ, and
refusal to acquire an active old queue lock. Fixtures explicitly use a 1 MiB
dictionary and temporary toy data; they are not a real 128 MiB checkpoint smoke,
transfer, migration, deletion, compression-ratio or training acceptance. Root
performs the first real pinned-CP smoke only after independent source review,
publication and byte readback.
