# Random nested backtrack fresh200 launch preparation

Status: LOCAL_PREPARATION_ONLY_PENDING_SOURCE_AND_RUNTIME_REVIEW. No remote access, deployment, Slurm submission, model execution or modification of3360 occurred. These files are not a receipt that the new mechanism ran. New source SHA remains unset until implementation, tests, independent review, commit and push equality are complete. Do not package dirty source or use the prior8b6 source SHA as the new experiment identity.

The canonical primary configuration is `reproduction/tree-nested-branch-runtime-20261012/config/ppo_tree_nested_branch_fresh200.yaml`. Supervisor initializes Hydra from this independent directory and searches only the packaged committed source's `verl/trainer/config` for the existing official fresh base. New runtime config, inherited source configuration and all modules are included in the code-only package inventory. `independent_config.path` and SHA bind the primary file to the new source commit. No old config is edited.

The only sampler is `random_nested_backtrack_v1`, objective `reward_selected_nested_tree_grpo_surrogate_v1`, ordered leaves A/A1/B/B1/B2 and quotas[2,3]. Valid action starts, initial exclusion, single unconditional fresh initial fallback, inherited remaining budget and absolute reward tolerance0.0 are explicit. B2 uses B1's actual path and reward. Failed candidates cost generation/retrieval/scoring but never become training leaves. Legacy continuation_budget_per_question=3 denotes three retained children, not a total candidate budget; extra failed attempts are separately counted. Token-level actual mu and PPO ratios do not correct reward-selection probabilities; selection_unbiased_claim=false. Advantages/token weights/global row reduction remain the prior objective.

Keep official Qwen revision aa8e72537993ba99e69dfaafa59ed015b17504d1, common-data SHA0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24, train169028/dev512/final51713,512 questions×5 leaves, mini128/micro16, rollout old/ref scoring32, LR1e-6/warm57, BF16 weights/FP32 reductions, unchanged prospective safety values and global20 opportunities credit semantics. One fresh instance runs continuously5→200; no repeated initial baseline, calibration-only0–5 run or stop5. Dev every5, native/HF saves5/50/100/150/200 and rolling native boundaries remain. Final51713 and HF/tokenizer acceptance are still required.

Resource wrappers are copied into this separate directory: retained18's h100-10,8 exclusively allocated GPUs, fixed UUID8, own held lease/cgroup, foreign memory and SM independently<=15%, sharded GPU retriever HTTP127.0.0.1:18082 and Ray127.0.0.1:16382. All weights/checkpoints/cache/large outputs remain on verified remote storage.500GB real available plus actual fsync probe is required at admission/start/during run. Never resume3360 or any earlier native/optimizer state. Never kill, modify or remove3360 or foreign processes.

## Predecessor gate before admission and startup

`predecessor_gate.py` reads existing3360 schemas; it invents no new future receipt. It requires exact3360 run-result exit0/actual200, bound terminal accounting and actual final51713 validation/ledger, native200 manifest and retained HF member inventory, plus the existing independently reviewed cold-HF wrapper's actual identity-bound result and full/tokenizer report hashes. The old package/source/config pins are in manifest.template.json. Missing or incomplete evidence refuses dispatch. This gate is read-only and never stops3360. Root must finish actual final independent acceptance and preservation before approving new release; JSON accounting/cold reload alone do not assert independent acceptance. Gate emits small hashes suitable for the new submission journal.

## Packaging after new implementation review and publication

Use the isolated source checkout and this runtime only. Commands below are plans, not executed commands:

```sh
python3 work/tree-speed-repair-20261008/nested-branch-source/reproduction/tree-nested-branch-runtime-20261012/build_package.py \
 --repo work/tree-speed-repair-20261008/nested-branch-source \
 --output work/tree-nested-branch-package-NEW_SOURCE_SHA \
 --instruction 'Exact valid human fresh200 instruction and original time'
python3 work/tree-speed-repair-20261008/nested-branch-source/reproduction/tree-nested-branch-runtime-20261012/make_release.py \
 --package work/tree-nested-branch-package-NEW_SOURCE_SHA \
 --review outputs/tree-nested-branch/CODE_REVIEW.md \
 --review-manifest outputs/tree-nested-branch/CODE_REVIEW.json \
 --remote-source-sha NEW_PUSH_VERIFIED_SOURCE_SHA --verified-at ACTUAL_UTC_TIME \
 --output work/tree-nested-branch-release-NEW_SOURCE_SHA.json
```

Release requires accepted source/runtime review and actual pushed SHA equality. Remote package namespace is `ROOT/instruct300/candidates/nested-branch-fresh200-NEW_SOURCE_SHA`; outputs/run namespace is `ROOT/runs/tree-nested-branch-fresh200-JOB`. build/deploy/render never submit. `deploy_package.py` defaults to dryrun; any later authorized execution uses the unchanged retained18 management route and verifies every code-only hash. No model/tensor download to Mac.

## Exactly one submission and unknown resolution

After3360 terminal/final acceptance, static review, publication, storage and resource checks, root may deploy pinned code/release and render168h SBATCH with `render_dispatch.py`. Submit only exact rendered bytes/SHA via `submit_once.py`. Global fresh200 lock plus new `SUBMISSION_ONCE-nested-branch-fresh200-SHA.json` records SUBMITTING before sbatch and exact base64 script bytes/hash, command, timestamps, source/config package/release pins and predecessor hashes. Existing SUBMITTING/UNKNOWN/SUBMITTED prevents any automatic repeat. Never delete or overwrite the intent journal to retry.

If sbatch times out or returns ambiguous output: retain UNKNOWN; make read-only squeue/sacct/scontrol observations on the same local cluster and correlate job name, user, submit time, node, workdir/command/script SHA and source/package/run identity. Save observations in a separate reconciliation audit. Absence in one queue query is not proof of no submission. If identity is ambiguous, remain blocked for launch; do not resubmit. No reconciliation cancels an old/foreign job or edits a submitted script.

Before ending a later execution turn, observe at least two real advancing measurements, one completed credited iteration and absence of errors/exit; first5 and iteration6 continuation are checked within the same run. Continue through200/full51713 and final acceptance when authorized. Slurm RUNNING/process alive/first log do not prove effective progress.

## Files and pending checks

Entry/config/binding: config/ppo_tree_nested_branch_fresh200.yaml, manifest.template.json, build_package.py, make_release.py, render_dispatch.py, submit_once.py, predecessor_gate.py, supervisor.py, launch.sbatch, run_tree.sh. Copied service/storage/UUID/lease/accounting/mu/validation/native-state modules live alongside these files and are included in the package.

Pending: implementation CPU tests, independent source and runtime review (including Hydra compose of this canonical primary configuration), new source commit/push equality, exact actual3360 final acceptance and preservation, remote code-only deployment verification, release binding and fresh resource observation. No commit or launch was done by this preparation worker. Local static AST/shell/manifest checks are syntax/contracts only and never imply GPU training acceptance.
