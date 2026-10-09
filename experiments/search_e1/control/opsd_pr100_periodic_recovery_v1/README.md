# PR100 periodic evaluation recovery: reap finished CPU mirrors

The first original training segment actually completed ten online outers and99
Adam updates with accepted parent wait0/full release. The first evaluator failed
with actual2 before GPU/any512 work. The blocker was an inert checkpoint-mirror
zombie owned by the original supervisor: its Popen object was not waited until
outer100. This is an external child lifecycle error, not an OPSD loss, optimizer,
model or resource failure. Original training/scientific source/config/checkpoints
and the failed evaluation are preserved.

This independent source changes only mirror lifecycle and explicit continuation.
Evaluation gates and the frozen ownership scanner are byte identical to the
original apart from the new publication namespace. No PID-filter, directory
exemption or relaxed descendant/opaque rule is added.

Each new async mirror has a background CPU reaper calling its actual Popen.wait.
The supervisor harvests completed return codes and durable receipts before eval
launch and every active admission/release observation; live mirrors are never
waited in a GPU-start path. Harvested results are reused at final100. Failed copy
return codes remain explicit and do not block training/evaluation startup. Final
persistence completion still requires actual successful copies.

## Preparation and exact prior terminal contract

```sh
python -B prepare_recovery.py \
  --previous-pipeline-config ORIGINAL_PIPELINE_CONFIG \
  --previous-pipeline-config-sha256 ORIGINAL_SHA \
  --base B/periodic-recovery-NEW --helper-root B/source/RECOVERY/opsd_pr100_periodic_v1 \
  --output B/periodic-recovery-NEW/receipts/pipeline-template.json
```

Runtime final directory name remains `opsd_pr100_periodic_v1` beneath a new
independent revision so the unchanged training gates can source-bind its exact
supervisor role. Repository/publication namespace is
`experiments/search_e1/control/opsd_pr100_periodic_recovery_v1`. Root publishes
and fills source/readback/storage for ten fresh eval templates + one new pipeline
config. The training config is identical, not regenerated or scientifically edited.

`recovery` contains immutable path/SHA refs:

- `previous_pipeline_config`: original frozen pipeline.
- `previous_supervisor_terminal`: root actual normalized technical terminal,
  accepted=true, technical_recovery=true, old run_id/config_sha256/controller
  exact original supervisor-start identity; actual_exit_code=null and
  actual_wait_status=UNKNOWN. Require pidfd_signal_sent=15,actual_dead_pane=true,
  terminated_owner_absent=true,actual_signal_is_not_wait_returncode=true plus
  at least four preserved path/SHA evidence refs (beforeproc/config/pidfd action/
  after actual pane and full GPU release). Signal15 is never represented as
  actualwait-15 or a successful return. This receipt never calls the old parent successful.
- `completed_training_terminal`: original first10 accepted train foreground0,
  status snapshot and full-release refs.
- `completed_boundary`: original training `runs/boundaries/outer-00010.json`.
- `failed_evaluation_actual_exit`: original first evaluator's actual2 proof.
- `previous_cpu_registry`: immutable saved original role-registry bytes; the
  current dynamic registry must still equal it before migration.
- `observer_binding`: original immutable live memory observer binding.

The old mirror actual return is UNKNOWN if its Popen parent was technically
stopped without waiting; disappearance or an existing copy manifest is not rc0.
The new supervisor preserves that unknown and creates independent new mirror
outputs. It reuses the exact source/config-bound live original CPU observer,
recording actual PPID reparenting in the new registry without modifying its
original binding/report/source.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 E1_RUN_ID=NEW_PIPELINE_RUN \
  python -B supervisor.py --pipeline-config FINAL_CONFIG \
  --pipeline-config-sha256 SHA
```

First action is fresh native512 on the existing accepted raw first10 CP99,
without new sampling, Adam or training0. After real512 completion, the same
original training run continues `--resume --until-outer20`, then eval20, through
100/ten evaluations. Each eval uses the unchanged HF SDPA BF16 base + FP32 raw
adapter ordinary greedy protocol and the accepted same-condition source239
baseline. No FP16 extra baseline, best fallback, GRPO or scientific change is
introduced. All raw validation/log/state/copy payloads remain HKUST-only.

Only four focused CPU regressions cover completed reaping/reuse, no waiting for
a live copy, actual failed copy preservation, and unchanged foreign-descendant
rejection. Preparation/test success is not GPU/EM acceptance.
