# Four-arm supervisor foreground terminal capture

This independent CPU-only helper closes the gap between “the supervisor PID is
absent” and “this exact supervisor actually returned zero”. It never imports the
producer or Torch, launches a model/job, signals a process, takes resource locks,
changes tmux settings, or writes Node output. JSON goes to stdout. The caller
saves the audit output in an exclusive file on the authorized HKUST storage.

The live target is the already running evaluation-recovery31 supervisor, UID1001,
PID1811163/starttick689575157, marker
`signal-study-eval-recovery-4e1e70d464834408b43cf00e9fd030a5`, tmux session
`e1-signal-eval-recovery-31e2d74-20261010`, sessionID `$46`, pane `%46`, server
PID1100668. The pane PID is the driver itself: the launch shell has already
executed the Python driver with the same PID. No surviving Bash wrapper is
required. This is a named capture for this current study, not a generic resume
or authorization mechanism.

## Immutable input request

`opsd-signal-weight-terminal-request-v2` has `accepted: true` and these fields:

```json
{
  "schema": "opsd-signal-weight-terminal-request-v2",
  "accepted": true,
  "study_manifest": {"path": "<exact original remote study path>", "sha256": "86db5c8e1a86f6b5fd27f880ca062d193b4623e29869e8fa2aa3dba0e8e46c8e"},
  "recovery_manifest": {"path": "<exact evaluation-recovery-input-31e2d74.json remote path>", "sha256": "ff44d73effa3a40ff5ed1f3d6b5049002b6b8937aa491a761b00266d74f1863e"},
  "launch_script": {"path": "<study>/receipts/launch-evaluation-recovery-31e2d74.sh", "sha256": "0f50705a162a43c7138f03916dcb4063455b9a3f232270799f6caff880a57772"},
  "launch_input_publication": {"path": "<published input readback receipt>", "sha256": "<actual SHA>"},
  "owner": {"pid": 1811163, "uid": 1001, "starttick": 689575157},
  "supervisor_run_id": "signal-study-eval-recovery-4e1e70d464834408b43cf00e9fd030a5",
  "log_path": "/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/logs/evaluation-recovery-31e2d74.log",
  "tmux": {
    "session": "e1-signal-eval-recovery-31e2d74-20261010",
    "session_id": "$46", "pane_id": "%46", "pane_pid": 1811163,
    "server_pid": 1100668,
    "start_command": "<exact recorded original pane_start_command>"
  }
}
```

The helper reconstructs the new actual start command from the verified script/log:
`bash -lc 'tmux set-option -w remain-on-exit on; exec bash <script> > <log> 2>&1'`.
It checks all four original config bytes, original00fec71 source inventory and
publication, evaluation-recovery31 inventory/publication, launch input publication, and
its own published source inventory. It rejects duplicate/missing source files,
extra commands, another study/recovery owner, SHA drift, and symlink inputs.
The four original scientific configs are not rewritten.

## Live capture

Run in the original HKUST4 Python environment, `CUDA_VISIBLE_DEVICES=''`,
`PYTHONDONTWRITEBYTECODE=1`, UID1001:

```text
python -B capture.py --mode live
  --request REQUEST --request-sha256 REQUEST_SHA
  --source-manifest MANIFEST --source-manifest-sha256 MANIFEST_SHA
  --publication PUBLICATION --publication-sha256 PUBLICATION_SHA
```

The output schema is `opsd-signal-weight-driver-live-binding-v2`,
`accepted=true`, `mode=live`, **`completion=false`**. It binds actual `/proc`
argv (including the Python executable), UID/eUID/starttick/parent/processgroup,
run marker and empty CUDA, original source/input references, actual stdout and
stderr log, tmux sessionID/pane/PID/start command, retained-pane setting, the
actual server UID/starttick and socket device/inode/UID/ctime. It verifies the
session has exactly one pane. A live capture does not claim any exit code,
training completion, token coverage, or evaluation result.

Preserve the exact output bytes and SHA. Do not reformat this JSON before the
terminal check. This small binding can also be retained on Mac; all raw results
and checkpoints remain remote.

## Terminal capture

Only after the same driver has finished, pass its **exact original live bytes**:

```text
python -B capture.py --mode terminal
  --request REQUEST --request-sha256 REQUEST_SHA
  --source-manifest MANIFEST --source-manifest-sha256 MANIFEST_SHA
  --publication PUBLICATION --publication-sha256 PUBLICATION_SHA
  --binding BINDING_OR_DASH --binding-sha256 EXACT_BINDING_SHA
  --native-completion NATIVE_FOUR_GROUP_RECEIPT
  --native-completion-sha256 NATIVE_FOUR_GROUP_RECEIPT_SHA
```

`--binding -` reads the raw binding bytes from stdin. Terminal inputs must be
explicit immutable references, not mutable `study-status.json` or guessed
`latest`. The helper requires the same server/starttick/socket and original
retained pane/start command, now `pane_dead=1`, **`pane_dead_status=0`**. A
missing PID, live pane, empty/dead unknown status, nonzero/signal exit, reused
server/socket/pane, or reserialized binding is rejected. PID absence is only a
separate release check and never supplies the return code.

The native four-group receipt must bind the same supervisor's explicit kernel
identity and original study. Each of four groups must have the actual
controller parent wait0, its bound terminal file, successful original owned
descendant audit, and TRAIN25 ledger with exactly ordered outer0..24 rows and
original group run/config SHA. Final fresh GPU inventory on **owned GPU4–7**
must show no non-oracle process. The only allowed GPU share is the original
UID1001/PID/starttick/source-bound K4 GPU7 oracle at ≤1024MiB. Foreign GPU0–3
are not claimed as owned or altered. A same-UID `/proc` scan rejects remaining
study/recovery/group-marked descendants, including original observers, and
opaque owned identities do not count as absence. No signal is sent.

Output schema: `opsd-signal-weight-driver-terminal-v1`:

- `accepted=true`, `mode=terminal`;
- `actual_supervisor_wait_returncode=0` is the normalized actual retained-pane
  status, with `driver_exit_evidence.source='same-retained-tmux-pane'`,
  `pane_dead=1`, `pane_dead_status=0`, and
  `no_exit_code_inferred_from_absence=true`; it is not a fabricated Python
  `Popen.wait` on a process this checker did not create;
- `study_manifest`, `native_study_completion`, `live_binding_sha256`,
  `actual_cpu_supervisor`, and all source/input/publication refs;
- `completed_groups=4`, `completed_outers=100`,
  `full_training_em_rows=100`, `groups` containing each original config,
  TRAIN ledger, actual controller terminal/wait0 and release evidence;
- `owned_gpu_release.accepted=true` with actual GPU UUID/free memory and
  oracle identity, non-oracle owned GPU process list empty;
- `analysis_still_required=true`, `full_numeric_and_token_audit_not_claimed=true`.

This terminal proof is a prerequisite for independent analysis/resource reuse.
It does **not** substitute for the separate100outer native token audit,
four-rank final Adam/RNG numerical reader,8formal512+commonzero slot audit,
heldout replay, loss/probability-versus-EM analysis, or backup acceptance.
Old failed driver/controller exit1 remains preserved; it is not relabeled0.

## Exact recovery and preserved-failure chain

This namespace fixes bindings only. The unchanged original source is00fec71;
the current external recovery source is31e2d74 with manifest
5ba5b8d93cc02b93fe1a3f62f4fe3038a5080a10cda27f5590ddc95e6745dcbc.
The launch input publication has its actual
`input_publication_commit == remote_readback_sha == be8b41ce40c66ffd556135c4d26a10f0d2889a0b`
and `deployed_source_commit == 31e2d74a217aa7749c7de89d7c4a4bb85215c716`.
Every launch/input publication row must have exact SHA/bytes and
`remote_equal=true`. The repair schema is the exact
`opsd-signal-weight-evaluation-recovery-manifest-v2`, scope
`evaluation_hold_retry_and_existing_named_outer_old_fp32`. It is not a generic
resume or identity override.

`verify_failure_chain` binds the original a337 released-failure metadata, the
755441 preserved-tree mapping, CP150 fda38 manifest, actual old evaluator and
controller terminal1 files, and all four preserved mutable metadata copies.
The original `$45/%45` pane is recorded as exit1, three actual failed waits stay1,
and eight missing orphan-worker waits stayUNKNOWN. New `$46/%46` is independently
bound; its later exit0 cannot replace or relabel the old failure. Failed-tree
raw file headers remain the previously accepted preservation action's SHA
evidence; this capture does not rehash the raw tree or read checkpoint payloads
or tensors. Only the four small preserved metadata JSON copies are freshly read.
The output records that limit explicitly.

To preserve the existing7d completion assembler contract, the terminal output
**continues to use `opsd-signal-weight-driver-terminal-v1`** and its existing
required fields. Only the request/live-binding schemas become v2; their
exact-source/owner/pane checks reject the old live binding. Extra provenance is
under `originals.prior_failed_evaluation`. The native four-controller actual0,
ordered TRAIN100 and owned release algorithms are copied unchanged from515.
The44 fixtures include the original23 checks plus21 exact new binding/old
failure negative cases. Metadata topology fixtures explicitly stub file/tree
reads using synthetic records; they do not claim real Node publication or
production completion. Separate original file/source tests verify SHA drift.

## CPU validation

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover
  -s experiments/search_e1/control/opsd_signal_weight_terminal_v2 -p 'test*.py'
```

44 synthetic CPU fixtures cover pinned files and source publication bytes,
single foreground exec, native actualwait0 versus UNKNOWN/nonzero, full ordered
TRAIN ledgers, same-PID live exec, strict original binding bytes,
server/socket/pane reuse, missing or nonzero retained statuses, and the
GPU7-only admitted oracle boundary. They perform no remote query, tmux command,
GPU operation or model loading. Independent review and actual Node live capture
must be recorded separately after publication.
