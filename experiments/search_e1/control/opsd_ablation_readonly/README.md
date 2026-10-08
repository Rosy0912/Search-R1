# Archived read-only resource snapshot

`resource_snapshot_v1.py` is the exact source used for the single HKUST1 CPU
resource observation on 2026-10-09 at 00:50–00:51 CST. Its source SHA is
`88547e96a69e370c8045ee42e78b8dd3445fc0447be0e379dc87fee3daaad62b`.
The original source, input, command, stdout, stderr and actual return code remain
in the private `work/search-e1-analysis-output/opsd-ablation-execution-20261008/
turn2-resource-readonly/` directory. The original successful return code is 0;
no new execution or test accompanied this archival copy.

The literal `INPUTS` contains this observation's paths, study/config/source
identities, expected file hashes and metadata. It contains no credentials,
questions, gold answers, token text or model weights, and does not grant human
authorization. The code reads the published SHA-bound training `gates.py` and
uses its CPU process/GPU inventory and original 18004 `/health` admission checks.
It reads small checkpoint-preservation metadata, not model or optimizer payloads.
It retains only the selected run marker from readable process environments;
unreadable environments are recorded as opaque, never inferred absent.

Execution uses empty `CUDA_VISIBLE_DEVICES`, system Python and disabled bytecode
writes. The script writes only stdout, sends no signals, takes no leases, starts
no observer/model/GPU job, and sends no retrieval request. Its subprocesses are
read-only `nvidia-smi` and `findmnt` queries. Snapshot stdout and raw evidence are
private artifacts and must not be committed to Git.

`accepted_evidence` means the identity-bound observation was collected; it is
not storage permission, a training-success claim, a full CPU foreign-process
clearance, or future launch admission. `foreign_compute_none` is limited to the
instantaneous GPU inventory. This version requires the seven candidate configs
to retain their pending-storage state and refuses changed config/source hashes.
Old online200 storage permission is not inherited. New execution must verify
fresh resource/storage/observer/locks separately, and a changed oracle PID or
source requires a new full numerical/HTTP gate.

This source is an archival snapshot tied to its immutable input identities, not
a generic launcher. Any future node-side auxiliary source must first be saved,
reviewed, committed, pushed and read back by the root executor before execution.
