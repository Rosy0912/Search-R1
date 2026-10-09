# Signal-weight OPSD: independent named-adapter storage recovery

This namespace overlays only the immutable00fec71 four-card trainer's named
`outer_old` loading. Original config/run/source bytes, completed common-zero512,
the2560 sampled trajectories, miner/reference/selected D, optimizer settings,
all-five scorer, actual forward objective and resume cursor remain unchanged.
No old source is rewritten, and no question is resampled because of this repair.
The original runtime305 and model/data manifests remain original inputs.

PEFT0.13.2 creates a named adapter in the BF16 base's storage dtype. Its native
state-dict setter copies the FP32 source into that BF16 storage, discarding low
bits. The original exact `SharedTargets` tensor digest correctly rejects this
rounded outer-old adapter. Casting after the setter cannot recover those bits.
The wrapper promotes only the504 named outer-old parameter storage to FP32
**before** calling the original setter with the original FP32 tensors. It then
checks every named tensor's exact input digest, unchanged current adapter,
unchanged non-old parameter objects/versions/dtypes/trainable flags and the
original setter return. It does not lower the original exact digest guard or
alter parameters in the current optimizer. Original target freezing and mode
switching still run in the original trainer.

`recovery_common.bind` verifies all four original configs, the exact old source
manifest and controller/trainer hashes, and the separately published/read-back
new recovery manifest. Original modules are compiled from freshly checked bytes,
never an old timestamp-valid pyc. A fresh worker process must not import or shadow
those modules before the loader is installed. `trainer_launcher.run` patches
only `peft.set_peft_model_state_dict` while calling original `trainer.main` and
restores the setter on every exit. Its default-adapter calls are exact passthrough.

The worker entry needs the unchanged original trainer arguments and:

```text
E1_SIGNAL_RECOVERY_MANIFEST=ABSOLUTE_NEW_RECOVERY_MANIFEST
E1_SIGNAL_RECOVERY_MANIFEST_SHA256=EXACT_NEW_SHA256
E1_RUN_ID=ORIGINAL_GROUP_RUN_ID
WORLD_SIZE=4
RANK=0..3
```

The immutable recovery JSON has schema `opsd-signal-weight-recovery-manifest-v1`,
`accepted=true`, `scope=named_outer_old_fp32_before_setter_only`, `study_id`,
`helper_root`, `source_manifest`, `publication`, `source_commit`,
`original_configs` (four original path/SHA refs), `original_helper_manifest` and
an explicit `supervisor_run_id`. Additional failure/zero/outer/sampling/actual
terminal refs belong to the separate CPU dispatch entry, which must preserve
actual argv identity rather than claim to be the old controller source.
`bind(..., cpu_supervisor=True)` verifies that independent supervisor marker,
UID1001, empty CUDA visibility and no Torch import; GPU workers still require
the actual original group marker and four-rank world.

Each actual trainer rank writes a unique small receipt under original group
`receipts/signal-recovery/RECOVERY_MANIFEST_SHA/`. Schema is
`opsd-signal-weight-recovery-execution-v1`. It records original config/outer/start,
new source/publication, before/after storage dtypes, exact504 tensor digests,
current unchanged and original-guard preservation. These records are loading
evidence, not Adam updates, five-update acceptance or EM benefit.

Local CPU regression:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover
  -s THIS_HELPER -p test_trainer_recovery.py -v
```

Eight fixtures use only tiny synthetic Torch values. They reproduce the native
BF16-copy failure through the exact old `SharedTargets` code, prove post-load
upcasting insufficient, preserve default optimizer parameter objects and RNG,
and reject wrong digest, NaN, incomplete inventory, changed shape or unintended
current mutation. `OPSD_SIGNAL_RECOVERY_TEST_ORIGINAL` can bind the actual old
helper directory on a node. Local absence of PEFT does not count as a skipped
or passed real-PEFT test.

After publication, run the separate actual-dependency fixture on HKUST4 with the
original scientific environment and empty CUDA visibility:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 ORIGINAL_ENV_PYTHON -B
  node_peft_cpu_fixture.py
  --original-helper-root ORIGINAL_00FEC71_HELPER
  --original-helper-manifest ORIGINAL_SOURCE_MANIFEST
  --original-helper-manifest-sha256 af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314
```

It constructs a tiny Qwen2 (hidden8,36 layers) locally from configuration, uses
actual PEFT0.13.2/Transformers4.46.3/Torch2.4.0, checks504 LoRA tensors and
requires the original named-adapter BF16 failure to reproduce. It then checks
the repaired native setter and native target guard, exact input/old FP32 digest,
current/optimizer objects, no RNG consumption and no CUDA initialization. No
real model/checkpoint, download, tokenizer, data or GPU is used. Its accepted
stdout is `opsd-signal-real-peft-cpu-storage-fixture-v1`; this is a CPU fixture,
not the actual resumed four-rank production acceptance.

The supervisor entry CLI is deliberately exact and contains no `-B` option
(the no-pyc policy is the environment variable). Set `CUDA_VISIBLE_DEVICES=''`,
`PYTHONDONTWRITEBYTECODE=1`, `E1_RUN_ID=repair.supervisor_run_id`, and run:

```text
ORIGINAL_ENV_PYTHON entry.py --mode study
  --recovery-ref ABSOLUTE_NEW_RECOVERY_MANIFEST --recovery-sha256 EXACT_SHA
  -- --study-manifest ORIGINAL_STUDY_PATH --study-manifest-sha256 ORIGINAL_SHA --resume
```

The recovery JSON also requires original `study_manifest`, `first_failure`,
`common_zero`, `first_outer` and `sampling_complete` immutable path/SHA refs.
The first supervisor checks the original explicit nonzero parent/pane exits,
all previous owned processes released, zero checkpoints/Adam, all2560 sampled
slots/1280 strict successes and327 original pairs. It verifies and reuses the
full accepted common-zero parent/eight-worker wait/release proof. It will not
reinterpret a later fault or trained checkpoint as the first-Adam failure.

The CPU external dispatch replaces only exact whitelisted controller and
four-rank torchrun trainer script commands. All original arguments, environment,
Popen handles, actual waits and original config/run IDs remain. Sampling,
evaluation and teacher commands are untouched. The scoped driver role verifies
new entry/source/publication and actual `/proc` argv separately from original
study code; it never pretends the new process executes the old argv. Original
observer source/config/PID/marker remains verified. The original external-release
`roles`/identity API and single SHA-checked JSON parse are preserved. Fourteen
additional CPU fixtures check dispatch rejection, environment/return preservation,
real-argv overlay proof, source/publication, foreign GPU rejection, external
release API, immutable read and first failure gates.
