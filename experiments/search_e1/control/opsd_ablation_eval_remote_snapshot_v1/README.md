# Closed native evaluator copies on HKUST

This helper makes independent inode copies on the same HKUST node. Raw
predictions, generated token IDs, original native traces, retrieval HTTP bodies,
worker arguments, rank64 completion, phase actual waits/releases, original
evaluation configurations and scalar model/data/runtime manifests remain remote.
No payload is downloaded to the Mac. Protected source weights, checkpoints,
optimizer state and data remain in their original remote locations. This is
persistent independent file evidence, **not physical disaster recovery**.

`native_scope.py` is derived from the already published closed-native V3
`source-operation.py`, SHA `086c15e619a19ba646232b5aa62653a43b316252b1862b1ba7cd4291086d5311`.
It extends that exact actual parent/8worker wait0/strict recorded512 coverage
contract to the existing `common-zero` evaluator. The ordinary group contract is
unchanged. It records the native producer's scoring and identity; it does not
run the model, recompute QA scores or repeat numerical tests. `zero` has the
recorded C0 variant and `common_zero` role with BF16zero/FP16zero versions.

The file copy and complete SHA/metadata primitives are loaded only from the
published `opsd_ablation_remote_raw_snapshot_v1/snapshot.py` whose exact SHA is
`40e05812069c118b14df6605128204bd1a013286674c4cacab768a91ec48c211`.
The native scope retains the exact published V2 primitive dependency at
`N/source/outer-backup-v2-554c52c/opsd_ablation_outer_backup_v2/backup.py`, SHA
`d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57`.
The new own manifest pins both Python files before either mode executes.

## Copy an already closed subset

Set `N=/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008` in the
calling environment without changing any system variable. Use the original
HKUST environment Python with `-B`, `CUDA_VISIBLE_DEVICES=` and
`PYTHONDONTWRITEBYTECODE=1`. Publish and verify this helper before execution.

```text
python -B NEW_HELPER/snapshot.py snapshot \
  --copy-primitives EXACT_REMOTE_40e058_SNAPSHOT \
  --groups zero,C0,C1,M,R,W,P \
  --output N/analysis/NEW_UNIQUE_REMOTE_EVAL_COPY_BATCH
```

After J's original native controller and its parent wait have truly completed,
copy `--groups J` into another new output. A group that lacks its immutable
controller completion, parent actual wait0, accepted native summary/result,
both required phases or exact8×64 recorded predictions is rejected. A live
or partial evaluator is never copied. Copies do not acquire any training lock
and never delay another GPU stage.

Each included source file is fully read and SHA hashed during copying. Its
destination is independently read and SHA hashed. Source before/open/after
metadata and all destination metadata must remain exact; inode+device must
differ. No hardlinks, symlinks, special files, aliases, overwritten artifacts or
implicit resumes are allowed. Capacity reserves payload bytes plus10GiB.
After copying, the exact native closure and source file set are reread, then
all pinned producer metadata is rechecked.

The exact include file list and hash are retained in
`GROUP/source-inventory.private.json`; destination members and hashes are in
`GROUP/copy-accepted.private.json`. Original producer references are retained in
`GROUP/complete.private.json`. The group includes the full closed native result
tree and its original closed parent `evaluation` tree, plus selected scalar
identity manifests and original training configuration. Two subtrees are
excluded when applicable: `fp16_zero/cpu-merge/model` and
`raw10_fp16/cpu-merge/model`. Their original CPU merge request, actual wait,
FP16 model manifest, tensor digest and load identity are kept; the reproducible
large model payload is excluded. Unknown `model/models/cache/checkpoints`
subtrees reject enumeration rather than silently disappear.

Identity manifests preserve the already accepted original source/data/adapter
payload hashes. This helper does not copy or independently rehash those large
protected payloads and does not claim to do so. The actual evaluation payload
files it does copy all have fresh complete source/destination SHA. Mac should
receive only the final small public summary and small source/publication audits,
not the private inventories or raw payload trees.

## Final eleven-model persistence acceptance

Final mode consumes an explicitly prepared remote JSON file with schema
`opsd-ablation-remote-eval-final-copy-inputs-v1`, `state: "READY"` and exact
`{path,sha256}` references:

```text
study_manifest
driver_start
driver_complete
foreground_terminal
actual_wait_evidence
full_owned_release
copy_batches: [
  {complete: {path,sha256}, actual_execution: {path,sha256}},
  ...
]
```

Each external copy execution receipt is captured from its real returned
process, with `actual_wait_returncode: 0` and `complete` pointing to that exact
batch `complete.private.json`. The two batches remain separately waited,
without invented shared process or absolute cross-host ordering.

The root's actual foreground terminal has the existing normalized fields
`accepted`, `actual_exit_code:0`, `signal_termination_not_exit0:false`, `owner`,
`run_id`, `input_manifest` and `actual_wait_evidence`. The raw actual wait has
`identity` equal to original driver-start owner, `actual_wait_returncode:0`, and
`exit_status_source` equal to `Popen.wait` or
`retained_same_pid_foreground_pane`. A retained original pane must first have
actually returned0 on the bound same server/socket/session/pane; process
absence or a CPU helper/wrapper0 is insufficient. The root's separate full
release ref must have `accepted_no_owned_or_opaque:true` or
`accepted_no_owned_or_foreign:true`. Those facts are producer evidence, not
values to fill before the driver ends.

```text
python -B NEW_HELPER/snapshot.py finalize \
  --copy-primitives EXACT_REMOTE_40e058_SNAPSHOT \
  --final-inputs REMOTE_ACTUAL_FINAL_INPUT_JSON \
  --final-inputs-sha256 ACTUAL_SHA \
  --output N/analysis/NEW_UNIQUE_FINAL_EVAL_PERSISTENCE
```

Final mode pins original PID1894801/UID1005/starttick927990515/run697c6bdf…,
the exact current study manifest and driver-start. It requires original
driver-complete11/5632, eleven native accepted models, successful original
foreground actual wait and full release, exactly eight evaluator roots across
explicit copy batches and total5632 predictions. It rereads the group closures
and all original/copy metadata, then independently copies the immutable named
final input/driver/summary/terminal/release records. It does not scan mutable
observers or a running driver directory. Only then is the small
`public-summary.json` emitted with `accepted:true`,11 models and5632 slots.

The root must preserve real execution/source/push/readback evidence for each
invocation. Errors keep partial remote output and originals, never signal or
restart the active pipeline. There are no fixtures or new reviewer rounds;
source syntax parsing is the only local implementation check.
