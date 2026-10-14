# Recursive ablation150 execution

This independent publication uses the actual resolved A configuration SHA
`43e28cc7ecc0d38a93324bf480aafb23a0c1ecdb979e9ea587d0358ae095233d`.
No original production file is edited. `bootstrap.py` lazily installs the new
planner in the entry and Ray workers and retains the original optimizer and
training implementation. Dataloader initialization stops training at150 while
the optimizer's scheduler retains its original200 scale and57 warmup.

## Launch prerequisites

- Published full source checkout/archive containing `recursive_ablation.py`.
- Existing exclusive task-owned eight-H100 allocation, inherited Slurm identity.
- Original resolved YAML and the existing model/data paths in it.
- Existing CPU retriever on localhost18086 with its own unchanged numeric gate
  and private control-token path. The launcher does not recreate or stop it.
- A actual post-evaluation150 manifest, for matched baseline reuse.

```sh
export RBA_ABLATION_BASE=/dataset/shaoshitong/recursive-ablation150-20261014
export RBA_ABLATION_SOURCE_COMMIT=ACTUAL_PUBLISHED_FULL_COMMIT
bash release/recursive_bran_a/ablation150_v1/study.sh
```

The study runs C1 then C2, then the unique single-factor conditions. Eleven
requested labels require seven new150-round trainings: budget10/uniform/depth4
reuse A, depth1 reuses C1. Original target5/root-included acceptance cannot
produce generation5 before stopping, which makes depth4 exactly nonbinding.
No ordinary independent-sampling control is introduced.

`run.sh` can run one already prepared condition. `prepare.py` prepares a new
condition once; existing configurations are never overwritten. Failed runs
require an explicit accepted full native manifest and SHA:

```sh
RBA_ABLATION_BASE=/dataset/shaoshitong/recursive-ablation150-20261014/C1-root-only \
  bash release/recursive_bran_a/ablation150_v1/run.sh \
  --resume-manifest /remote/this-condition/checkpoints/native-rolling/global_step_N/manifest.json \
  --resume-sha256 ACTUAL_MANIFEST_SHA256
```

## Outputs and acceptance

Each round records small finite scalar/time/Adam summaries immediately. Rolling
state retains two full native model/Adam/scheduler/RNG/cursor snapshots without
per-round HF weight exports. HF keypoints, actual dev512 and detailed trees are
only5/50/100/150. Actual post-evaluation RNG snapshots remain at those keypoints.
The native loop's terminal151 validation reuses the actual scheduled150 metrics;
there is no extra dev run or51713 scan per screening condition.

Follow `logs/training.log`, `receipts/current-entry.json`, the indicated attempt
`progress.json` and `completed-outers.jsonl`. Initial five-round acceptance needs
five genuinely complete records and observable sixth-round sampling growth;
the driver automatically continues without a blocking acceptance gate.
`training-complete.json` proves native150/fourdev results, not process exit.
The study additionally records actual parent wait0 and confirmed GPU-context
release before the next condition. Normal matched numerical guards retain the
original consumed-round/separate-Adam accounting; programming errors fail visibly
and preserve committed native state for an independent repair.

The isolated entry also installs the receipt-bound checkpoint namespace repair
from `ablation150_guard_recovery_v1`: native audit output lives in
`base/output/attempt-*`, while native checkpoints live in `base/checkpoints`.
This changes no optimizer, objective, guard threshold, or sampling contract.

`receipts/explicit-recovery.json` is an owner-bound operational request. A
`ZERO_WRITE_SAVED_BATCH` request accepts only the proved first namespace refusal:
all eight workers recorded zero Adam writes and scheduler0, and the original
prepared2560-row actor batch and consumed driver snapshot remain intact. The
independent entry reloads the configured official model and observes empty Adam,
restores the saved driver/cursor/tree RNG, and executes that prepared actor batch
once before advancing to the next cohort. It does not claim restoration of
unsaved worker CUDA RNG or vLLM executor state. A later interrupted run requires
its own explicit complete native boundary, not this one-time zero-write scope.

Current report scope follows the human override: aggregate EM only; no A
checkpoint replay, question-pair comparison, or bootstrap inference.
