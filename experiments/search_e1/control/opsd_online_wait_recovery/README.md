# External wait recovery, same online OPSD run

This independent CPU-only entry dynamically imports the immutable original
`opsd_online/controller.py` and subclasses **only `wait`**. `run`, `sample`,
`spawn`, checkpoint recovery, native mining, sampling and scientific training
remain inherited. No old source/config/runtime is rewritten. No process signal,
automatic kill, tmux or CUDA initialization exists here.

The original 0-Adam failure has eight accepted worker-complete receipts but **no
persisted actual child wait return codes**. Those receipts do not prove rc0. The
original terminal proof must retain `old_wait_returncodes=NOT_PERSISTED_UNKNOWN`.
Because no `sampling-complete.json` was committed, original `sample` creates
`sampling-attempt-00001` and freshly generates all 512 TRAIN questions ×5.
Attempt00000 and its raw trajectories/logs remain untouched. Later complete
checkpoints still resume through the unchanged original implementation.

Each new phase writes exclusive private files under
`BASE/receipts/wait-recovery/UUID/`: fsynced `events.jsonl`,
`actual-terminal.json`, and `phase-complete.json`. Actual `Popen.wait()` values
are fsynced **before** error checks. Finished actual logs are SHA-bound.
Evidence write/hash/status I/O failure also retries while all leases remain
held. A failed partial JSONL append is truncated back to this sole writer's
last durable boundary before retry, so no partial/duplicate event is accepted.
Terminal/summary JSON is fsynced to a private pending file and atomically renamed;
an identical completed file may be verified after an interrupted fsync.
Admission read exceptions are recorded, without sticky worker-failure status.
Real nonzero child exits remain hard failures, after natural child termination
and strict release. Terminal success requires current phase admission, original
GPU release, CPU admission, original full-owned-process audit, and CPU admission
again. Persistent/opaque/failed release keeps all original leases held and waits
indefinitely; no next CUDA phase starts. The original main containment is used
for exceptions after successful release.

Startup requires the original immutable config SHA
`d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17`,
run `1d1d67b60f0e46a2afc0536a36ab69b1`, `--resume`, helper manifest
`66cb43dd0830b9a7776bd644a0ef5bbf3d1661a3db3b159cc9e781db582c5233`,
and original commit `7cf285308572af645676ae85a48fc8359f8c8ccb`.
All original helper and new recovery files are freshly checked, including full
inventory equality and import path. External publication must contain:

```json
{
  "accepted": true,
  "branch": "search-e1",
  "source_commit": "40-lowercase-hex-new-commit",
  "remote_readback_sha": "same-new-commit",
  "source_manifest_sha256": "recovery-manifest-SHA256",
  "run_id": "1d1d67b60f0e46a2afc0536a36ab69b1",
  "config_sha256": "d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17",
  "files": [{"path": "experiments/search_e1/control/opsd_online_wait_recovery/recovery.py", "sha256": "actual-node-and-Git-readback-SHA256", "bytes": 0, "equal": true}]
}
```

`files` must contain exact SHA/bytes readbacks for **all** files in this directory,
including its source manifest. Additional publication entries are allowed but
duplicate paths are rejected. The external source manifest itself inventories
all recovery files except `source-manifest.json` and Python bytecode caches.

Original terminal reference is `BASE/receipts/wait-recovery-original-terminal.json`
(expected SHA `b8978990c3c276570383e82f280eef0d52a7f39d3b73992f8e11e595e8a0484a`).
Schema `opsd-online-failed-original-owner-terminal-v1` requires accepted, run_id,
config_sha256, nine old_owned_processes `{pid,uid,starttick}` (owner
612077/1005/924207665 plus eight original workers), unknown old wait codes,
and old_controller_tmux_exit_status1. **Each old PID must be absent freshly at
startup**, twice; the historical proof alone never substitutes for current
launch/resource gates and original base/four shared lease acquisition.

Root publishes and reads back this new independent directory before deployment.
No original source manifest/config or original observer binding needs alteration.
Example (all paths/SHA filled from actual publication):

```sh
E1_RUN_ID=1d1d67b60f0e46a2afc0536a36ab69b1 PYTHONDONTWRITEBYTECODE=1 \
  ORIGINAL_PYTHON NEW_RECOVERY_DIR/recovery.py \
  --config BASE/config.json \
  --config-sha256 d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17 \
  --run-id 1d1d67b60f0e46a2afc0536a36ab69b1 --resume \
  --recovery-manifest NEW_RECOVERY_DIR/source-manifest.json \
  --recovery-manifest-sha256 RECOVERY_MANIFEST_SHA \
  --publication BASE/receipts/wait-recovery-publication.json \
  --publication-sha256 PUBLICATION_SHA \
  --original-terminal BASE/receipts/wait-recovery-original-terminal.json \
  --original-terminal-sha256 b8978990c3c276570383e82f280eef0d52a7f39d3b73992f8e11e595e8a0484a
```

Retain the original approved bootstrap numerical environment (including
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, XFORMERS and utilization0.85). This entry changes
no sampling, optimizer, checkpoint or logging budgets. Existing observer remains
bound to the same immutable config/run; an accepted CPU fixture is not a live
GPU/five-update acceptance.

CPU test entry, from the publisher:

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_online_wait_recovery -p 'test_*.py' -v
```

Fourteen CPU fixtures cover transient vs persistent gates, owned-release uncertainty,
durable real nonzero, only-wait inheritance, actual inherited new512×5 sampling
attempt with old poison receipts preserved, fresh old PID absence/unknown rc,
exact source/remote publication refusal, partial journal append recovery,
persistent evidence I/O with a live child, status/terminal fsync recovery, and
log-hash read failure without another reap or signal.
