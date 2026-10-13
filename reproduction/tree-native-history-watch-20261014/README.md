# Native5 child native-manifest metadata watcher

This small extension of published watcher60f0954 is prepared separately from training. It audits the same H10018 node and **one actual new native5 recovery child**, with immutable source/run/unit/InvocationID binding supplied only after those identities exist. Training does not wait for this watcher. No source identity is guessed from a date, branch HEAD or previous unit.

The packet builder requires `--training-source` (actual40hex recoverycommit), `--training-unit` (actual own unit name), `--run-directory` (actual own systemd run), and `--training-invocation-id` (actual32hex systemd InvocationID). The release descriptor pins these, interval60 seconds, first child outer6, last outer200, plus collector and watcher byte hashes. ExecStart pins the release SHA. Source/run paths use the existing same-node nested-branch namespace, and the owned unit name must contain the actual source's short SHA.

Each poll reads only the exact bound unit with `systemctl show`, requires its actual InvocationID, and scans only that child's retained complete `global_step_6..200/manifest.json`. The byte-identical original collector's manifest validator, PT stat checks and atomic no-replace publisher are reused; its SSH CLI and default multi-run collector are never called. Raw manifest JSON bytes and SHA remain immutable, same-SHA retries are idempotent, conflicts record audit errors without overwriting. No tensor/PT contents, Torch, GPU queries or training signals.

Original0bef outer1 and ba9 outer2..5 metadata histories remain in the existing **ba9 run's** `output/native-manifest-history` (the original collector saved both source identities there). The new descriptor lists these two inherited source/range entries. They are not moved, copied, rewritten or counted as new-child captures; their original source-qualified names remain intact. New-child6..200 archives live under the new run's own history. This is lineage metadata preservation, not independent native/HF cold reload or proof of unread pruned tensor bytes.

The bounded state at `output/native-manifest-history/watcher-native5-SHORT/state.json` records code/release hashes, expected InvocationID, child coverage/missing steps, inherited locations, errors and cumulative CPU/wall/JSON costs. State is at most100KB; there is no accumulating raw log. Training terminal is recognized only when the exact pinned InvocationID's unit is inactive/failed, main PID zero, and main exit recorded. That matching final poll performs the last collection, writes immutable `final.json`, and exits normally. Missing/changed identity or audit errors never send a signal or create permission to follow another execution.

The separate audit service retains MemoryMax256M, CPUQuota25%, TasksMax16, RuntimeMaxSec7d, Restart=no, AF_UNIX-only systemd access, and read-only filesystem except its own history. Runtime/resource failure affects the audit alone. A60s poll cannot recover a manifest pruned before any collector observed it; gaps remain explicit.

## Prepare code only, after the new training unit starts

Run the committed `prepare_release.py` locally with:

```
--repo ISOLATED_WATCHER_REPO --output NEW_PACKET_OUTSIDE_REPO
--training-source ACTUAL_NEW_RECOVERY_COMMIT
--training-unit ACTUAL_NEW_TRAINING_UNIT
--run-directory ACTUAL_NEW_RUN
--training-invocation-id ACTUAL_SYSTEMD_INVOCATION_ID
```

The builder copies only two small committed Python files, release.json, the separate watcher service and a packet receipt. It does not inspect remote state or start anything. The new actual training identities come from mainline's normal unit observations; this adds no training review/test/hold gate.

## Mainline deployment and one start only

1. Transfer the code-only packet via the existing retained18 route to its commit/source-qualified `remote_code_directory`, rejecting an existing destination; compare all packet file SHA readbacks with its receipt. No credentials, training environments, checkpoints, weights or raw rollout are transferred.
2. Create only the new bound run's `output/native-manifest-history`, if absent. Install only the new `sst-tree-native5-history-watch-SHORT.service`. Confirm this audit unit has not previously been installed/started; run `systemctl daemon-reload`, then **once** `systemctl start sst-tree-native5-history-watch-SHORT.service`. No enable/restart or training stop/start/restart operation is part of these instructions.
3. Mainline observes two actual cycles at least60 seconds apart: verify state polls/timestamps/cost advance and a newly completed manifest is captured when available. Check audit errors/service exit. Record small measured receipts. Process liveness alone is not progress.

Implementation performs minimal syntax parsing only, no tests, review or remote deployment. The mainline remains responsible for actual startup/measurements and completing200 outers plus51713-row final evaluation and independent cold-reload acceptance.
