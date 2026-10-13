# Native manifest history watcher

This packet audits only `sst-tree-nested-branch-fresh200-ba9f6627.service` and its owned ba9f6627 run on H10018. It does not control or modify training. The frozen training source remains ba9f6627ec9a4e9749830d7b6acca53f6feec6b4. This watcher has its own Git identity and is published separately.

Every 60 seconds it discovers retained complete rolling manifest JSON, runs the existing collector's hash-bound validation and atomic no-replace publisher, and indexes archived receipts. The original manifest bytes and SHA are retained. Eight-rank finite/Adam/scheduler fields remain **the original native writer's evidence**, not independent recomputation. PT files are only stat'ed at capture time; no PT/tensor bytes, hashes or Torch loading. Pruning races retry on the next poll. Existing same-SHA entries are idempotent; conflicts are audit errors and never overwritten.

History is inside the same run at `output/native-manifest-history`. Watcher state is a bounded 100KB `watcher-ba9f6627/state.json`, including archive coverage/gaps, errors, costs, training unit observations, code/collector/release hashes, and explicit evidence limits. It keeps no raw rollout/log stream. Errors do not send training signals. A valid terminated unit (inactive/failed, no live main PID, recorded main exit) triggers the last collection, a small immutable `final.json`, then normal exit; failure of training is recorded without claiming successful acceptance. Unit-query errors/missing unit do not count as terminal. A different nonempty training InvocationID is an audit scope error, never permission to follow a new training execution.

The service has MemoryMax=256M, CPUQuota=25%, TasksMax=16, RuntimeMaxSec=7d, Restart=no, a read-only filesystem except its owned history, and AF_UNIX-only access for local systemd queries. A service resource failure can stop the audit alone; training is unaffected. It cannot recover manifests already pruned before any collector observed them; `missing_outers_since2` exposes gaps instead of claiming 200-round coverage. Parent outer1 remains the existing separate receipt; this watcher scans only the child run's outer>=2 snapshots.

## Prepare code only on the Mac

After commit/push verification, run `prepare_release.py --repo ISOLATED_REPO --output NEW_PACKET_OUTSIDE_REPO` using local Python. This copies only the two small committed Python files, release descriptor, systemd unit and packet receipt. It does not SSH, upload, read PT/model assets or start anything. The descriptor pins collector and watcher SHA; ExecStart also pins release SHA. The deployment directory contains the separate watcher Git commit, so it cannot overwrite frozen training code.

## Mainline deployment and one start on H10018

Only the mainline executes these steps; implementation does not execute remote commands.

1. Transfer the code-only packet to its receipt's `remote_code_directory` using the existing retained18 route. Reject an existing destination. Read back SHA for each packet file and compare with `packet-receipt.json`; record actual pushed remote Git SHA separately.
2. Create only the ba9f run's `output/native-manifest-history` directory, if absent. It must belong to this exact run. The training source, PT files, optimizer, environment and unit are untouched.
3. Confirm `sst-tree-native-history-watch-ba9f6627.service` has not previously been installed/started. Install **only** that packet's audit unit under `/etc/systemd/system/sst-tree-native-history-watch-ba9f6627.service`. Run `systemctl daemon-reload`, then **once** `systemctl start sst-tree-native-history-watch-ba9f6627.service`. Do not enable/restart it or run training stop/start/restart operations.
4. Mainline performs two actual measurements separated by at least one polling interval: read its small state and archived receipt counts; confirm `polls` and costs/observed timestamps advance, and new completed native manifests are archived if training finishes another outer. Check the watcher journal/errors and service exit state. Process existence alone is insufficient. Preserve both measurements as small local JSON.

This implementation performs minimal syntax validation only, with no new review/test or training gate. Training continues toward 200 outers and final 51713-row evaluation; full native/HF cold reload remains an independent final acceptance step, not supplied by this watcher.
