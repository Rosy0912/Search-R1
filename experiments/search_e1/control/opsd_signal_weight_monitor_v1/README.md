# Four signal arms: independent read-only monitor

This namespace is published and deployed independently. It does not edit the
production namespace, start/resume training, signal a process, acquire its GPU
leases, import Torch, load a tokenizer or deserialize tensor/optimizer files.

Invoke in a fresh HKUST4 UID1001 process, `CUDA_VISIBLE_DEVICES=''`, with:

```text
python -B monitor.py
  --study-manifest ABSOLUTE_PATH --study-manifest-sha256 SHA256
  --config G_POS ABSOLUTE_PATH SHA256
  --config G_NEG ABSOLUTE_PATH SHA256
  --config H_POS ABSOLUTE_PATH SHA256
  --config H_NEG ABSOLUTE_PATH SHA256
  --source-publication ABSOLUTE_PATH --source-publication-sha256 SHA256
  [--audit-output NEW_ABSOLUTE_SMALL_JSON_PATH]
```

All four config references must match the immutable study manifest. The explicit
publication must be its study-wide publication. These are production bindings;
this independent monitor's own source must also be published/read back before
deployment. Stdout is a compact private snapshot. The optional audit uses an
exclusive new file, never an overwrite, with a 2MiB maximum. Exit0 means the
monitor found no current evidence issue; it does not mean the study has ended.
Exit2 is attention or unknown evidence, not an instruction to stop a job.

The monitor distinguishes native completed outers, controller-accepted outers,
committed Adam updates, actual tail/pair exposures, preupdate2560-slot TRAIN EM,
partial heldout512 slot progress, and accepted zero/15/25 heldout512 with actual
parent wait/release evidence. A missing process yields ABSENT/UNKNOWN wait; no
monitor converts disappearance to exit0. Status HOLD/FAILED and an absent active
controller are explicit attention items. Logs expose keyword markers and line
SHA only, without raw question/answer text; a log marker is not a return code.

It rehashes the production helper inventory, all305 runtime files, two data
inputs, all small committed checkpoint/update references and the latest complete
checkpoint's adapter/four-rank state/shared-Adam payload bytes. It does not
deserialize numeric Adam tensors or claim tensor equality independently. The
source100 twelve payload files get current length checks and a freshly hashed
manifest; their fresh complete weight SHA acceptance belongs to the separately
recorded launch asset gate. No full-history token/logit/loss reconstruction is
claimed. Stored native512 strict counts are checked against all512 complete rows;
the monitor does not regenerate or re-score full traces.

Direct memory.free and compute ownership inspect only authorized GPU4–7. Other
users on GPU0–3 are ignored. The original bound18003 oracle must pass its actual
published numeric/proc/source/health gate and remain only on GPU7 at<=1GiB.
Current owned-card free must remain>=6GiB. Live observer/source/age and natural
terminal receipts are reported separately. No check changes the routing or
restarts the oracle.

CPU tests use tiny synthetic identity files, never production weights or raw
questions. Run `python -B -m unittest discover -s THIS_HELPER -p test_monitor.py`.
When deployed separately on a node, provide
`OPSD_SIGNAL_TEST_PRODUCTION_HELPER=ACTUAL_PUBLISHED_PRODUCTION_HELPER` for the
test's actual readonly checkpoint contract. That fixture checker performs byte
and metadata checks only; tests are not GPU or numerical Adam acceptance.
