# Stats recovery handoff

This independent stdlib entry waits on the Mac. Every 15 seconds it makes one
direct read-only HKUST1 SSH probe. No long-lived watcher is added to the Node,
no training lease is acquired, and no process is signalled. Backups do not gate
this handoff. It does not change the original training/evaluation source,
configuration, process, tmux window/options, or logs. It only configures its
newly created own window to retain exit status before submitting the new exec.

The first successful live observation binds the actual old driver-start/input,
tmux server UID/starttick, socket device/inode/owner/mode, session/pane identity,
actual foreground Python UID/starttick/full argv/run, and source publication.
The actual old pane has literal `pane_start_command='bash'`, but its same PID
has already execed the exact Python driver. This is allowed only by the
`exec_same_pid` proof: pane PID equals original/actual Python PID, PPID is the
bound tmux server, PGID is the pane PID, source/manifest are fresh SHA-checked,
CUDA visibility is empty, and actual stdout FD1 names the fixed old input log.
The live binding is saved separately and referenced by both the parent
terminal proof and `completed_c0`. Literal `bash` alone cannot pass.

The helper also supports the previously specified single foreground env
command with an exact live Bash wrapper. A missing/reused process or pane is
UNKNOWN. The watcher requires the same retained pane to actually become dead
with status 2, the exact old Python (and wrapper if any) to disappear, and C0's actual terminal to
report exit 0 plus its unchanged TRAIN_COMPLETE/10 status, actual child wait 0,
release, and raw outer-9 checkpoint. It never derives exit 0 from absence.

Only after this boundary, a single remote `finalize` call writes new independent
JSON inputs under the same study. Seven training configs, science budgets,
data, seed, dtype-repair and checkpoint identities stay unchanged. The eight
evaluation templates and driver receive explicit new run IDs/output roots and
the separately published stats source. Publication receipts preserve the
actual remote source readback facts, with a new run binding. Storage receipts
retain the actual human seven-group HKUST-local exception and Mac background
backup policy, with an explicit derived-from reference and new output binding.
The output is never described as shared storage.

The new source's `validate_manifest`, `completed_c0.preflight` and current
`check_dead_parent` run from freshly SHA-verified source bytes, bypassing pyc.
All eight evaluator source/storage gates are also checked before starting the
new study driver. The native driver performs full ten-outer/science/resource
admission again and C0's no-training-spawn reuse, immediate evaluation, then
the other six groups. This helper's terminal checks alone do not numerically
accept a checkpoint or complete the 70 outer/5632 evaluation goal.

## Explicit plan

The plan is immutable JSON, passed by path/SHA on the Mac and exact bytes over
SSH stdin. Example values below are placeholders; the Root executor binds the
actual published refs and fresh run IDs before use.

```json
{
  "study_root": "/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008",
  "old_study": {"path": "OLD_STUDY_MANIFEST", "sha256": "EXACT_SHA"},
  "old_session": "e1-reverse-seven-observer-1a1850f7-20261009",
  "old_pane": "%73",
  "live_foreground_binding": {"path": "OPTIONAL_ROOT_ACCEPTED_LIVE_PROOF_ON_NODE", "sha256": "EXACT_SHA"},
  "new_source": {
    "helper_root": "N/source/NEW_STATS_V2_COMMIT/opsd_ablation_eval_v1",
    "manifest": {"path": "STATS_MANIFEST", "sha256": "EXACT_SHA"},
    "publication": {"path": "GENERIC_ACTUAL_READBACK_RECEIPT", "sha256": "EXACT_SHA"},
    "commit": "FULL_PUBLISHED_STATS_V2_COMMIT"
  },
  "destination": "N/receipts/NEW_HANDOFF/inputs",
  "output_root": "N/NEW_STUDY/runs",
  "new_session": "e1-reverse-seven-stats-NEW",
  "run_ids": {"driver": "32hex", "zero": "32hex", "C0": "32hex", "C1": "32hex", "M": "32hex", "R": "32hex", "W": "32hex", "P": "32hex", "J": "32hex"},
  "eval_bases": {"zero": "N/NEW-zero", "C0": "N/NEW-C0", "C1": "N/NEW-C1", "M": "N/NEW-M", "R": "N/NEW-R", "W": "N/NEW-W", "P": "N/NEW-P", "J": "N/NEW-J"}
}
```

`N` must be replaced by the complete absolute study path. Every new output
must be absent, distinct, below that study and outside protected old outputs.
The optional already accepted live binding is needed if the watcher starts
after the parent exits. It must have the exact exec schema/accepted/route and
is checked against the actual retained server/socket/pane and source identity;
an absence observation cannot create it. Without that ref, the watcher must
first capture the real live process itself.
The stats source must be the independent `opsd_ablation_eval_stats_recovery_v2`
publication; the earlier v1 source lacks the actual exec-binding contract and
is rejected. The publication generic basis need not have a run ID. It must bind branch
search-e1, exact commit/readback, the source manifest and every source file.

## CLI after Root publication and Node readback

```sh
python3 -B PUBLISHED_LOCAL_HELPER/handoff.py watch \
  --plan PRIVATE_PLAN --plan-sha256 PLAN_SHA \
  --local-output NEW_PRIVATE_WATCH_OUTPUT \
  --remote-python ORIGINAL_PYTHON --remote-helper NEW_NODE_HELPER/handoff.py \
  --self-manifest NEW_NODE_HELPER/source-manifest.json --self-manifest-sha256 SHA \
  --self-publication NODE_HANDOFF_PUBLICATION --self-publication-sha256 SHA \
  --self-commit FULL_HANDOFF_COMMIT --ssh-host hkust1
```

The direct route is explicit `/usr/bin/ssh -B en0 -oProxyCommand=none
-oProxyJump=none -oControlMaster=no -S none`. Each probe checks this helper's
own published manifest and actual source bytes before reading Node state.
`probe` and `finalize` are remote subcommands consuming the transport envelope
on stdin; their payload includes the exact plan bytes/SHA and live binding.
Remote handoff processes explicitly unset `E1_RUN_ID`, keep CUDA invisible,
and import no Torch. Their arguments name only independent helper/publication
refs (the plan is stdin), and they are not a native phase-child ancestor. The
published native same-UID scanner permits such readable independent CPU
processes only after known/run/source/descendant checks; this helper adds no
process exclusion or weaker gate. Launch intent records the actual bootstrap
identity. If the real native scanner cannot establish this, it still holds.

The new foreground launch is a single `exec env CUDA_VISIBLE_DEVICES= E1_RUN_ID=NEW
PYTHONDONTWRITEBYTECODE=1 ORIGINAL_PYTHON -B NEW_SOURCE/study_runner.py ... >
FIXED_NEW_LOG 2>&1` command. The server's global option may be off. This helper
first creates its independent bash session, binds the actual session/window/
pane PID/UID/starttick, sets `remain-on-exit=on` only on that new window and
reads it back. Only then does it send the literal exact exec command followed
by Enter. The global option, old pane and all other windows stay unchanged.
The new pane metadata consequently records `bash`; its actual Python process
uses the same-PID exec route, so future terminal verification needs a live
binding too. It saves a launch intent before tmux, actual creation return code
and the window-before-exec proof, then the command-submission receipt. Any finalization transport
failure, existing new output/session, or different existing JSON is preserved
and fails rather than retrying an uncertain launch. A successful tmux return
and command submission mean only the new foreground command was submitted,
not that its Python exit code or evaluations have passed.

Mac probe logs and raw stderr are private. The new inputs, launch intent and
launch receipt are Node artifacts. Raw questions, trajectories, weights,
tokens and environment credentials are not emitted by this helper or committed.

## Focused CPU checks

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_ablation_stats_handoff_v1 -p 'test_*.py' -v
```

Two fixtures verify (1) missing/uncertain terminal evidence never imports the
new source, writes inputs or spawns; (2) the successful boundary preserves
original seven training/science refs, derives independent eight evaluator
inputs, checks source/storage before one spawn callback, and accepts the exact
live same-PID exec binding while rejecting a different PID. The success
fixture also checks new-window-only retention before literal exec/Enter. They do not claim
real Node terminal, SSH, tmux, GPU, checkpoint numerical or effect acceptance.
