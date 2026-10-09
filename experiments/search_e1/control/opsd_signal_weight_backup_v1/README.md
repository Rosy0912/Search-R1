# HKUST-only pinned checkpoint backups

This helper is independent of the running trainer, controller and memory
observer. Publish/read back both peer and transport code before deployment.
Deploy the identical published `peer.py` bytes on HKUST4 and HKUST1; bind the
actual absolute peer paths and its explicit file SHA in every invocation.
Keep deployments independent: source
`STUDY/source/backup-<published-commit>/opsd_signal_weight_backup_v1/peer.py`;
target `BACKUP/source/backup-<published-commit>/opsd_signal_weight_backup_v1/peer.py`.
Use the approved K4 scientific environment's Python and the verified K1
`/usr/bin/python3` (3.12.3); the helper needs only Python>=3.9 and the standard library.
No GPU calls, Torch imports, training locks, process signals, credentials or
model downloads occur. Backups can run concurrently with training. The original
production ownership scanner classifies these `--checkpoint`/tar `-C` paths as
readonly inputs, not production execution/output paths. Peer calls force an
independent marker and empty CUDA visibility; they never borrow the trainer run.

The source is fixed to HKUST4 UID1001,
`/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010`.
The target is fixed to HKUST1 UID1005,
`/data/shaoshitong/search-e1-opsd-signal-weight-backup-20261010`.
Only an explicitly pinned accepted four-rank checkpoint at true Adam1/5 or a
complete outer-end cursor may be backed up. Pure JSON consistency checks bind
runtime/source100/science/common-initial identities, four declared adapter/Adam
digests, scaler/RNG/scheduler claims, cursor/exposure and exact distinct rank,
shared optimizer and adapter inventory. These remain declaration/byte checks,
not numeric tensor or RNG restoration. All source files are immutable and
read-only. The Mac controller joins two remote SSH pipes; it never reads the
checkpoint stdout, creates a tar/payload file, or loads weights. Only config and
small inventory/terminal/audit JSON passes through local memory.

Run locally with verified lowercase `hkust4` and `hkust1` SSH aliases. Uppercase
aliases are rejected because the actual Host patterns do not match them:

```text
python -B transfer.py
  --variant G_POS
  --config REMOTE_K4_CONFIG --config-sha256 CONFIG_SHA256
  --checkpoint REMOTE_K4_MANIFEST --checkpoint-sha256 MANIFEST_SHA256
  --expected-updates ACTUAL_ADAM_COUNT --expected-outer ZERO_BASED_OUTER
  --source-python ABSOLUTE_K4_PYTHON --source-script ABSOLUTE_K4_PEER
  --target-python ABSOLUTE_K1_PYTHON --target-script ABSOLUTE_K1_PEER
  --peer-sha256 IDENTICAL_PUBLISHED_PEER_FILE_SHA256
  [--audit-output NEW_SMALL_LOCAL_OR_REMOTE_SUMMARY_JSON]
```

Every shell argument is quoted with `shlex.join`. Neither alias can name another
cluster. Every call forces `-B en0 -o ProxyCommand=none -o ProxyJump=none
-o ControlMaster=no -S none -o BatchMode=yes -o ConnectTimeout=10`; there is no
proxy, jump host, persistent master or inherited multiplex socket. The actual
direct options and lowercase aliases are recorded and revalidated at target
commit and later accepted-backup probes. Explicit source config/checkpoint SHA, variant, run, filename, true
Adam/outer and exact complete payload inventory must agree. The complete source
directory is rehashed before and after the actual USTAR sender; only its real
`wait()==0` creates the source success terminal. The target stores the stream
under a unique remote `attempt-*/checkpoint.tar.partial`; strict regular-file
USTAR validation rejects missing zero terminators, unsafe/unknown/duplicate
members, symlinks, unexpected sizes, nonzero padding/trailing content and SHA
mismatches. Files are extracted exclusively into that new remote attempt.

Before reception, the target checks actual free space for both retained tar and
extracted payload plus a100GiB reserve; it cannot consume another job's reserve.

Receiving/extracting alone produces **pending** evidence. The target writes
`complete-sha-verified.json` only after the actual source tar terminal and both
local SSH waits are0 and all target member SHA are freshly checked again.
Existing accepted backups get a fresh target SHA probe and are reused without
another stream. Source assets are never deleted. Failed or interrupted remote
attempts remain intact. An interrupted commit may recheck and reuse only the
same pinned regular-file payload/config bytes; different or damaged content is
held and never overwritten. Target accepted probes also reject extra files or
symlinks and revalidate actual tar/direct-SSH wait evidence.
Transport failures preserve their actual nonzero SSH codes in the small error
audit. Exit0 means this pinned byte backup was accepted, not numerical Adam,
effect evaluation, full-study or resume acceptance.

Source diagnostics and target control stdout/stderr are drained continuously in
separate bounded readers while the unread tar pipe connects the two peers.
This prevents stderr backpressure from blocking tar/receiver EOF. Captures are
limited to128KiB each; all diagnostic bytes are still drained and SHA-counted.
Overflow or reader errors preserve evidence and reject completion. Checkpoint
payload bytes are never consumed by these control readers.

The backed-up manifest/config bytes retain their original HKUST4 absolute paths.
Restoring the original directory layout and its separately preserved inputs,
runtime, source100, debug/trajectory evidence is required for a genuine resume.
This backup does not assert those dependencies or an independently deserialized
optimizer have all been restored. Target tar and extracted payload are both
retained; disk usage and any later staging cleanup need separate observation.

CPU tests use only tiny invented bytes, including an actual local USTAR command,
exact source/target SHA, bad/truncated streams, wait-code gates, quoting and a
pipe whose local `read()` deliberately raises. No test connects to a node or
copies production tensors. On a separately deployed test tree, set
`OPSD_SIGNAL_TEST_PRODUCTION_HELPER` to the actual published production helper
directory to reuse its readonly `matched_roots` contract.
