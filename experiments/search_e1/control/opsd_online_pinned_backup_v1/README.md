# One pinned online checkpoint → private Mac backup

This independent, stdlib-only tool backs up one explicitly identified retained
checkpoint from the authorized HKUST1 online run. It does not modify the old
backup tool, training, evaluator, runtime, routes, or node files. This directory
contains source and synthetic CPU tests only; no private question, trajectory,
token, weight, environment, or credential data belongs in Git.

Current target is protected update89/outer8, manifest SHA
`325c9f41f5a44d5e16a32578c5fa03dfaecec92bbdbd87a4f5e289c81c9bf75f`.
The same entry can later back up the explicitly pinned raw outer199 endpoint.
It does not discover “latest”, transfer an unpinned checkpoint, or re-transfer
the already backed-up CP0/1/5/11. Final outer199 inputs are not ready merely
because this code is ready. Transfer acceptance does not certify tensor
finiteness, equal Adam moments, model effects, or completion of all 200 outers.

## Execution and publication boundary

There is one source entry, `backup.py`, and no experimental module imports.
Thus it never uses an older timestamp-valid experimental `.pyc`, imports the
old backup module, mutates its globals, imports Torch, initializes a model,
creates an optimizer, queries a retriever, signals a process, or writes to the
node. Standard-library imports remain ordinary Python imports.

Root must review, commit, push, and verify remote Git/source readback before
calling any production mode. Both the Mac copy and the separately deployed
HKUST1 copy validate their exact source directory, source-manifest SHA, and
the same publication receipt before use. The source directory contains only
`backup.py`, `test_backup.py`, `README.md`, and `source-manifest.json`; even an
unexpected `__pycache__` is rejected. The entry is called with `python3 -B`.

Required publication JSON follows the already used interface:

```json
{
  "accepted": true,
  "branch": "search-e1",
  "run_id": "1d1d67b60f0e46a2afc0536a36ab69b1",
  "source_commit": "<40-hex actually published commit>",
  "remote_readback_sha": "<same actual remote Git commit>",
  "source_manifest_sha256": "<new source-manifest SHA>",
  "files": [
    {
      "path": "experiments/search_e1/control/opsd_online_pinned_backup_v1/backup.py",
      "equal": true,
      "bytes": 0,
      "sha256": "<actual payload SHA>"
    }
  ]
}
```

`files` must contain one exact equal/SHA/bytes record for each of the three
payloads **and** `source-manifest.json`. The placeholders above are not an
accepted receipt. No publication receipt is manufactured by this helper.

## Modes and explicit inputs

All modes require the exact original config path/SHA, exact checkpoint manifest
path/SHA, expected run/update/outer, this helper's source-manifest path/SHA,
publication path/SHA, and actual source commit. The original training run ID is
fixed to `1d1d67b60f0e46a2afc0536a36ab69b1`. Production remote modes additionally
require actual UID/EUID1005 and the authorized `/data/...20261008` config and
checkpoint root. `--host` is restricted to the existing `hkust1` SSH alias.
The original immutable config SHA currently is
`d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17`.

* `inventory`: HKUST1 reads and verifies the pinned identities/files and prints
  private JSON to stdout. It writes no node file.
* `stream`: HKUST1 reads the previous inventory JSON from stdin, independently
  re-reads the exact config/manifest and all eighteen file SHAs/stats, then
  streams an uncompressed USTAR archive to stdout. It writes no node file.
  Source SHA/stat and publication checks run before and after streaming; any
  failure yields a nonzero actual SSH return code, never a completion record.
* `backup`: Mac validates its published source, runs those two published remote
  modes through SSH, and writes a **new** independent destination. No arbitrary
  inline Python or shell program is sent to the node. All remote argv fields
  are individually quoted with `shlex.join` and the shell string is not
  evaluated locally.

Example after publication and exact remote deployment/readback, with real
paths and actual SHAs supplied by Root:

```sh
python3 -B /ABS/MAC/opsd_online_pinned_backup_v1/backup.py backup \
  --config /data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/receipts/config.json \
  --config-sha256 d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17 \
  --checkpoint /data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/runs/checkpoints/update-00000089-outer-00008/manifest.json \
  --checkpoint-sha256 325c9f41f5a44d5e16a32578c5fa03dfaecec92bbdbd87a4f5e289c81c9bf75f \
  --expected-run 1d1d67b60f0e46a2afc0536a36ab69b1 \
  --expected-updates 89 --expected-outer 8 \
  --source-manifest /ABS/MAC/opsd_online_pinned_backup_v1/source-manifest.json \
  --source-manifest-sha256 ACTUAL_MANIFEST_SHA \
  --publication /ABS/MAC/pinned-backup-publication.json \
  --publication-sha256 ACTUAL_PUBLICATION_SHA --source-commit ACTUAL_COMMIT \
  --remote-helper-root /ABS/HKUST1/published/opsd_online_pinned_backup_v1 \
  --remote-publication /ABS/HKUST1/pinned-backup-publication.json \
  --destination /ABS/MAC/private-checkpoint-backup/pinned-cp89-new \
  --margin-gib 5
```

The same publication receipt bytes must be available at both explicit
publication paths. Source-manifest bytes must likewise be identical. Existing
destinations are rejected; there is no automatic partial resume or deletion.
The destination parent must already exist and have no symlink component; use
the real absolute Mac path (for example `/Users/...`, not a symlink alias).
New destination and extraction roots are created with mode0700; the helper
does not change permissions of existing backups or partial directories.

## What is verified and copied

The checkpoint must declare the actual run/config/runtime/source identities,
expected update and outer, all eight ranks, equal-step declaration, and an
immutable completed `outer_end` cursor. These declaration checks are **not**
the separate CPU tensor/Adam numerical audit.

The exact eighteen unique, non-hardlinked members are:

1. the checkpoint manifest;
2. eight exact `rank-N-state.pt` files;
3. `adapter/README.md`, `adapter_config.json`, and `adapter_model.safetensors`;
4. the original `training_input` outer manifest, pairs, order, and encoding;
5. the original config and `runs/run-identity.json`.

Original pairs may be in the original sampling-attempt directory, as the
producer actually records. All four inputs must be inside the exact outer
directory. The outer manifest's run/config/outer/runtime/source/pairs binding
is checked. Actual checkpoint files must match the declaration exactly. A
symlink in **any** source component, hardlink alias, path escape, special file,
missing declaration, changed SHA/bytes/stat, or extra checkpoint file rejects
the inventory. Fixed-handle reads check path/device/inode/size/mtime/ctime/mode
before and after the read; streaming additionally recomputes each file SHA.

The three config-bound scientific identity-manifest JSON files are freshly
SHA-read and included in inventory provenance. The original 305 source tree,
source100 weights, helper payloads, and checkpoint tensors are not independently
numerically validated or included as additional archive members by this tool.
Their existing independent audit/publication remains separate evidence.

## Local durability and failure behavior

The free-space gate covers the calculated uncompressed archive upper bound,
the extracted file bytes, and at least 5GiB margin. Archive, record, and file
writes are flushed/fsynced. The archive is first `checkpoint.tar.partial`.
Manual extraction rejects noncanonical names, traversal, duplicate/unexpected
members, directories, all links/devices/FIFOs, PAX/GNU hidden extensions, missing members,
wrong sizes/SHAs, truncated end markers, and nonzero/second archives after the
first end marker. It never calls `extractall`.

After stream SHA verification, it performs a separate full local disk read of
every extracted file with stable-stat/SHA/bytes checks and exact directory
inventory. It independently re-reads the archive SHA, syncs directories, and
only then renames the archive and writes `complete-sha-verified.json` last.
The record has actual transfer return code0 and
`independent_local_disk_readback:true`, with eighteen local file proofs.
Failure writes `failed.json` with `accepted:false`, retains any `.partial`,
partially extracted bytes, logs, and inventory, and leaves node originals
untouched. A local evidence-write failure can leave the failure JSON absent;
it still cannot create an accepted completion record. No automatic cleanup,
node release, source removal, or disk-space reclamation is performed.

This is an ordinary failure-detection/durability boundary, not adversarial
protection against a privileged process rewriting bytes and restoring metadata
between observations. Full streaming and readback SHA checks detect actual
bytes read. Abrupt host power loss still requires inspection of the retained
partial directory; no claimed automatic recovery is provided.

## CPU tests and remaining production gates

```sh
python3 -B -m unittest discover \
  -s experiments/search_e1/control/opsd_online_pinned_backup_v1 \
  -p 'test_*.py' -v
```

Synthetic fixtures cover the real producer's eighteen-member structure,
identity/count/metadata failures, source path escapes/symlinks, mutation before
and during streaming, tar malicious/special/duplicate/missing/trailing members,
independent local readback corruption, sufficient space, publication and remote
commit binding, SSH argv quoting/HKUST-only scope, and failed-partial retention.
Passing fixtures do not mean the actual CP89 has been transmitted. Production
source publication/deployment/readback, fresh Node inventory, actual transfer,
actual disk readback, and actual completion receipt are still required. No
actual backup or final200 acceptance is prewritten by source preparation.
