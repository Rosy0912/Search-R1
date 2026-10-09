# Four-arm supervisor foreground terminal capture

This independent CPU-only helper closes the gap between “the supervisor PID is
absent” and “this exact supervisor actually returned zero”. It never imports the
producer or Torch, launches a model/job, signals a process, takes resource locks,
changes tmux settings, or writes Node output. JSON goes to stdout. The caller
saves the audit output in an exclusive file on the authorized HKUST storage.

The live target is the already running recovery98 supervisor, UID1001,
PID1554337/starttick687293806, marker
`signal-study-recovery-d96c7fc1e0f14ea2bf3edcb01a22bfa8`, tmux session
`e1-signal-four-recovery-98b9326-20261010`, sessionID `$45`, pane `%45`, server
PID1100668. The pane PID is the driver itself: the launch shell has already
executed the Python driver with the same PID. No surviving Bash wrapper is
required. This is a named capture for this current study, not a generic resume
or authorization mechanism.

## Immutable input request

`opsd-signal-weight-terminal-request-v1` has `accepted: true` and these fields:

```json
{
  "schema": "opsd-signal-weight-terminal-request-v1",
  "accepted": true,
  "study_manifest": {"path": "<exact original remote study path>", "sha256": "86db5c8e1a86f6b5fd27f880ca062d193b4623e29869e8fa2aa3dba0e8e46c8e"},
  "recovery_manifest": {"path": "<exact recovery-input-98b9326.json remote path>", "sha256": "73a5594a0dae56125cdb5f7a28f8f108454df5c8c1761d8fa629556b0cac9f4e"},
  "launch_script": {"path": "<study>/receipts/launch-recovery-98b9326.sh", "sha256": "87735981dd1a9f60c162d096ee0fafdf43d9fb5a7b12256e1ecdc3c4438560fb"},
  "launch_input_publication": {"path": "<published input readback receipt>", "sha256": "<actual SHA>"},
  "owner": {"pid": 1554337, "uid": 1001, "starttick": 687293806},
  "supervisor_run_id": "signal-study-recovery-d96c7fc1e0f14ea2bf3edcb01a22bfa8",
  "log_path": "/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/logs/recovery-98b9326.log",
  "tmux": {
    "session": "e1-signal-four-recovery-98b9326-20261010",
    "session_id": "$45", "pane_id": "%45", "pane_pid": 1554337,
    "server_pid": 1100668,
    "start_command": "<exact recorded original pane_start_command>"
  }
}
```

The helper reconstructs the exact start command from the verified script/log:
`bash -lc 'tmux set-option -w remain-on-exit on; exec bash <script> ><log> 2>&1'`.
It checks all four original config bytes, original00fec71 source inventory and
publication, recovery98 inventory/publication, launch input publication, and
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

The output schema is `opsd-signal-weight-driver-live-binding-v1`,
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

## CPU validation

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover
  -s experiments/search_e1/control/opsd_signal_weight_terminal_v1 -p 'test*.py'
```

23 synthetic CPU fixtures cover pinned files and source publication bytes,
single foreground exec, native actualwait0 versus UNKNOWN/nonzero, full ordered
TRAIN ledgers, same-PID live exec, strict original binding bytes,
server/socket/pane reuse, missing or nonzero retained statuses, and the
GPU7-only admitted oracle boundary. They perform no remote query, tmux command,
GPU operation or model loading. Independent review and actual Node live capture
must be recorded separately after publication.
