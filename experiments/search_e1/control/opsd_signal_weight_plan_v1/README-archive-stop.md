# Narrow HKUST4 archive replacement stop

This helper implements the latest human instruction to stop the old collapsed
HKUST4 archive so its four GPUs can be reused. It does not start a new experiment.
It must be reviewed, published, and deployed in a new directory before execution.
The old source, logs, checkpoints, retriever and monitoring services remain intact.

Run only on HKUST4 (`xielab5`, UID 1001). Required private inputs are the exact
`K4-human-resource-readonly-derived-v1` compact snapshot, its SHA-256, the raw
latest human stop authorization, its SHA-256, and the time that authorization was
observed. The fixed old run is `f04fe593830c4831918ede872812c016`; the bound
supervisor and worker PIDs/startticks cannot be substituted by CLI options.
Effective UID is read from `/proc/status` before any environment access. Unrelated
opaque processes are ignored; an opaque identity/environment in the selected
scope yields HOLD. Ray actor process titles may legitimately change. Their actual
titles are recorded while effective UID, starttick, old run marker and runtime
cwd remain mandatory; static supervisor/protected argv identities stay exact.

```text
python3 -B archive_stop.py \
  --snapshot PRIVATE_COMPACT_PATH \
  --snapshot-sha256 EXACT_COMPACT_SHA256 \
  --authorization-file PRIVATE_HUMAN_AUTHORIZATION_PATH \
  --authorization-sha256 EXACT_AUTHORIZATION_SHA256 \
  --authorization-observed-utc ACTUAL_TIME_WITH_TIMEZONE
```

The command above validates without signals or Node writes. Add `--execute` only
after the review/publication/deployment identity checks. Python's actual Linux
`pidfd_open` and `pidfd_send_signal` APIs are required. No process-group signal,
name-based kill, Torch import, model update, or Ray RPC is used.

Execution first quiesces `prioritize_formal_training`, `pipeline.sh`, and
`run_checked.py`, preventing automatic evaluation/OPSD/second GRPO stages. It
then quiesces the exact old run and attributed descendants, checks the protected
retriever/observers, hashes existing checkpoint directories, and copies source,
frozen runtimes, logs and receipts into a new private `preserved/user-stop-...`
directory. The original directories are never removed or rewritten. Checkpoint
manifests are fresh full-file hashes. Twelve filenames are insufficient: pure CPU
safetensors header/dtype/shape/offset/EOF and index-to-three-shard key coverage are
checked alongside stable directory/hash evidence. Only such endpoints are labeled
layout-valid; this does not newly accept all tensor values or optimizer state.
A partially written step200 is retained and labeled incomplete. The layout-valid
150 or200 weights **do not include live199 Adam,
scheduler or RNG**, and are not advertised as complete-state resume checkpoints.

Protected observers continue writing. Their receipt copies are independent
per-file byte snapshots with source inode/time and any subsequent replacement
recorded; this is not a simultaneous snapshot of all continuously written files.
Checkpoint files use stricter before/after stability checks. Symlinks are retained
as symlinks, without following external model/environment assets.

After preservation, exact pidfds receive TERM. CONT is sent only to exact targets
quiesced by this invocation so pending TERM can be delivered. Remaining approved
processes receive exact KILL after twenty seconds. Current scope is unioned with
every captured exact historical identity, so a helper reparented to PID1 cannot
escape termination just because its parent died. Reused PIDs are never signaled
as the former owner and are explicitly reported. GPU4–7 must have no old archive
compute contexts or foreign compute contexts. GPU7 retains only protected oracle
PID1749337 (at most1024MiB). Direct `memory.free` is checked alongside
`memory.used`; driver-reserved memory is recorded as `total-used-free`, with a
64MiB residual context allowance. Driver-reserved MiB must be within0–1024.
No GPU0–3 process is signaled or allocated.

Every outcome prints JSON. `accepted=true` requires preservation, old-process
termination, actual GPU release and unchanged protected PID/UID/starttick/argv
identities. Any failure returns a HOLD with `accepted=false`; preservation failure
leaves owned targets quiesced instead of resuming a supervisor that could start
OPSD. The parent must resolve a HOLD using its exact receipt. Reinvocation is not
a resumption of old training. A result receipt write failure also prevents success.

CPU fixtures use mocks and temporary files only:

```text
python3 -B -m unittest discover -s . -p test_archive_stop.py -v
```

They cover foreign runs, protected services, PID reuse at pidfd boundaries,
descendant attribution, original and partial checkpoint preservation, driver
memory accounting, validation-only behavior, ancestor-first stop ordering and
failure without automatic supervisor resume. They do not claim an actual Node
stop or GPU-release acceptance.
