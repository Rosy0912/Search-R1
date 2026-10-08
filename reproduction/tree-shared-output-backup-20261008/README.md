# Retained18 one-shot shared output backup

This entry preserves one latest complete native Tree-GRPO state from the retained
h100-10 task to its existing NFS11 Tree namespace. Training code and job3341 are
never changed. Native/model bytes move only from h100-10 to the mounted same-H100
NFS; the management route transfers this small code and status metadata only.

The deployed r2 entry SHA256 is
`1721d96d21df5a2f36dfa0793a1c099df0ddd31722ebe10ffeb207ecc9e21c22`.
Its single-launch request is `retained18-native-once-20261008-r2`; the launch PID is
489625 and its start ticks are 1850970667. It pinned native3341 outer13, original
manifest SHA256 `fb4bdd43440f6452ec50307b387df443880ca6932f1048461462d03f58d708fb`.
These are deployment identities, not proof that the full copy has completed.

`deploy_backup_once.py` writes an exclusive launch journal. Rerunning it only
inspects an existing request; an unknown launch result must be audited before any
new request is authorized. The initial request ended before opening checkpoint
files because its mount assertion incorrectly selected an autofs wrapper. This
known failure was fixed with an actual NFS4 mount filter before the explicit r2
launch; the original journal and log remain.

`backup_native_once.py` verifies the complete manifest, eight rank states, driver,
frozen run/code/config/data/model and scheduler identities. It opens all native
files first, then streams from their descriptors, so rolling cleanup cannot remove
the bytes underneath the copy. Only one native checkpoint can be pinned. It runs
at nice19 and ionice3, has one ordinary CPU thread, and caps each sequential copy
and independent readback pass at 128MiB/s. It computes source SHA256 during copy
and checks every destination byte in a second pass against the original manifest.
Only then can Linux RENAME_NOREPLACE atomically publish the task's `.partial`
directory with its `.complete` receipt. Existing shared assets are never deleted
or replaced. Failures release only this backup's descriptors and preserve partial
evidence; source outputs are retained.

Validated config, split and scheduler evidence and the package's 767 source files
are saved with their hashes. Optional growing logs and score metadata have capture
times and explicit byte ranges; they are labelled mutable source snapshots.
`supplement_external_provenance.py` additively validates the frozen package
manifest SHA256 and its four external runtime source files without restarting the
native copy. Its separate small receipt is preserved alongside the native backup.

`poll_backup_once.py` reads the own small status, PID/start identity, launch
journal, actual destination native byte count and eventual completion receipt.
It saves the result as local `metadata/progress.json`. Until the status is
`COMPLETE_SHARED_NATIVE_BACKUP` with a published `.complete` receipt, the copy is
in progress and is not a verified full replica.

Remote control namespace:
`/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/.shared-output-backup-20261008`.
Destination namespace:
`/mnt/shared/workspace/h100-11_from_h100-9/shaoshitong/search-r1-rl/tree-grpo/retained18-backup-20261008`.
Verified NFS source is `10.32.0.11:/dataset/from_h100-9`; the provider is scheduled
for release on 2026-10-21, exact time unknown. Retained18 originals must remain and
the replica needs further same-H100 durable storage before that deadline.
Native manifests retain `automatic_resume_supported=false` and
`exact_executor_resume_verified=false`; copying does not upgrade those limits.

## Existing r2 publication recovery

The r2 copy finished both the source-stream and full independent destination SHA
passes for all nine native files (49,376,545,288 bytes) at 08:58:20 UTC. Its actual
failure was `renameat2(RENAME_NOREPLACE)` returning EINVAL on this NFS mount;
directory fsync had succeeded. The original request remains FAILED in its original
status, log and journals, and its prepublication `.complete` bytes are archived.

`finalize_existing_backup.py` validates the existing native13 manifest and fixed
inventory hash, nine successful target SHA receipts, sizes and unchanged target
stats, then publishes the existing copy. It uses the same own-writer flock, an
exclusive new request and an exclusive empty final-directory reservation. Ordinary
atomic NFS rename replaces only this newly reserved empty directory. Every
previously existing final path is rejected. File and directory fsync errors remain
fatal. A new completion receipt is written only after the rename succeeds.

The isolated finalize request `retained18-native-finalize-r2-20261008` completed at
09:09:01 UTC, using entry SHA256
`f1f0b524a1a83be194118b9cdd201b504c84b9c54a685266dec0cc6051f62863`.
It copied zero new native bytes and performed no repeated large native hash pass,
reusing the frozen prior full verification evidence. Its new published `.complete`
SHA256 is `b8009471f0dff57fcff35ac2c950290ef0077e64b3d94756d5c1cc9f22033309`.
The receipt explicitly distinguishes recovery success from original r2 failure and
retains the native13/260 Adam-call identity, independently of later live training.
It records successful file/directory synchronization calls without claiming a
server power-failure guarantee or retention after the provider's release date.

`deploy_finalize_existing.py` deploys this small helper and journals its one launch.
It refuses to relaunch any existing finalize intent. Read both the original failed
status and the new recovery status/complete receipt when auditing this boundary.

Local entry wrappers use the pre-existing strict `../tree-speed-repair-20261008/ssh_retained18.py`
management helper. No credentials, model weights, data or environment are included
in this source publication.
