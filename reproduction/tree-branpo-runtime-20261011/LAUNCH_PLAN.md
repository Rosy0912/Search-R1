# Fresh BranPO tail runtime, retained18

Local preparation only. No remote read, deployment, scheduler submission or model execution was performed by this worker. Root's latest remote observation reports 5.78 TB f_bavail; scripts remeasure at submit, allocation startup and throughout training. Minimum real available capacity is 500 GB, with actual fsync write probe. No reserved-block change or asset deletion.

Complete owned service stack copied from reviewed V14 resource wrapper and stripped of current-chain recovery: same h100-10 allocation, UUID8 initial whole-card preflight, held lease/OFD identity and own cgroup, foreign memory/SM <=15%, GPU E5/full sharded FP16 FAISS HTTP127.0.0.1:18082, Ray127.0.0.1:16382, direct fresh actor/ref official aa8e725, emptyAdam/scheduler rank startup validation, unchanged validation/mu/PPO/rolling/memory observers. Default large output/cache/checkpoints remain remote through verified run-root and owned runtime symlink. One5-leaf batch512 per credited outer, original20 opportunities/rank,200 credited outer, dev512 every5, milestones5/50/100/150/200 and rolling native boundaries, final51713. Step5 is a user-authorized same-instance continuation, no baseline/extra calibration batch/stop5.

Runtime code must be committed with implementation into reproduction/tree-branpo-runtime-20261011 before packaging. Root independently reviews source/runtime and creates pinned release afterward; build/deploy/renderer never submit. Once journal SUBMITTING/UNKNOWN/SUBMITTED denies resubmission and binds exact SBATCH bytes. Never delete intent to retry.

## Build locally after commit/push

```sh
python3 work/tree-branpo-runtime-20261011/build_package.py \
 --repo work/tree-speed-repair-20261008/branpo-source \
 --output work/tree-branpo-package-COMMIT \
 --instruction 'Paste exact valid human fresh200 BranPO-tail execution instruction and time here'
python3 work/tree-branpo-runtime-20261011/deploy_package.py \
 --package work/tree-branpo-package-COMMIT --target h100-10
```

Second command is dryrun by default and uses the existing ssh_retained18.route(command) management hop, including code-only tar streaming stdin; no new SSH alias/config/key is assumed. Root uses --execute only after independent review and fresh resource observation. It copies only committed code/config/small manifests, then verifies all target file hashes. Package-build.json provides actual commit, remote_package and manifest SHA. No tensors/checkpoints/weights downloaded to Mac.

## Root's pinned release

Use reviewed `make_release.py` after actual push equality and review are observed, without self-reference:

```sh
python3 work/tree-branpo-runtime-20261011/make_release.py \
 --package work/tree-branpo-package-COMMIT \
 --review outputs/tree-branpo/CODE_REVIEW.md \
 --remote-source-sha ACTUAL_PUSHED_SHA --verified-at ACTUAL_UTC_TIME \
 --output work/tree-branpo-release-COMMIT.json
```

This pins accepted independent review bytes, actual remote source SHA, package manifest, authority and dispatcher hashes; never implies empirical training acceptance. Copy this small release through the same route to ROOT/instruct300/branpo-release-COMMIT.json before renderer invocation. Release remains outside package manifest to avoid self-reference.

## Route root commands to retained18

Use `python3 work/tree-branpo-runtime-20261011/ssh_retained18.py 'REMOTE_COMMAND'` for render/submit. The following blocks are the remote command text, not local GPU operations.

## Render on retained18 (no submission)

```sh
CUDA_VISIBLE_DEVICES='' /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python PACKAGE/render_dispatch.py \
 --package PACKAGE --manifest-sha MANIFEST_SHA \
 --release-path RELEASE --release-sha RELEASE_SHA \
 --wall-hours 168 --output ROOT/instruct300/branpo-COMMIT.sbatch
```

Renderer prints actual script SHA. Fresh source/import/runtime guard checks run from packaged modules; immutable manifest hash and actual config SHA bind driver/workers/observers. Config name ppo_tree_branpo_fresh200, branch_sampling=branpo_tail_v1, continuation_budget_per_question=3. No current_chain_runtime import or parent manifest in fresh path.

## Submit exactly once on retained18

```sh
CUDA_VISIBLE_DEVICES='' /dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python PACKAGE/submit_once.py \
 --package PACKAGE --manifest-sha MANIFEST_SHA \
 --release-path RELEASE --release-sha RELEASE_SHA \
 --script ROOT/instruct300/branpo-COMMIT.sbatch --script-sha SCRIPT_SHA
```

Run identity tree-branpo-fresh200-JOB; direct source/config/run/model/split identities recorded at startup. Before reporting launch complete root takes at least two real advancing measurements, confirms no failure/exit, five consecutive credited iterations and iteration6 start. A RUNNING allocation or process alone is insufficient.

Final supervisor JSON accounting binds all200 actual attempt/rank records, forty dev512 reports, final51713, five native/HF milestone manifests and fixed fresh identity. Full HF/tokenizer reload and final independent acceptance remain required root work after training, expressly unclaimed by JSON accounting. Root continues final validation through completion and only then own-resource cleanup/acceptance. Supervisor cleans only its own cgroup child identities; never bare scancel/foreign signals.
