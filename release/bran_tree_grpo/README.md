# Bran Tree-GRPO release

This is Tree-GRPO with BranPO-inspired finite tail-first sampling, not the complete BranPO loss. The training code matches job3360 frozen source `8b6f99d1b3353145200eb3fa7c16a579a79a987c`. Five finite leaves cannot guarantee distinct rewards.

Two roots retain quotas [3,2] (five leaves total). Each original root is retained; three new continuations are sampled per question. First branch is before the last original model action. If root0's first continuation has the same reward as its original, its remaining slot backs off one original action, clamped at the earliest state. Every sampled leaf is retained. No rejection sampling, extra candidates, answer injection, or changed objective.

Original Tree-GRPO question/inter-root and within-root normalized reward components remain combined; leaf mass0.2, root masses0.6/0.4. Population variance of pure within-root advantage, raw reward variance, mixed/zero-variance rates, actual total advantage variance, tree structure and prefix inheritance are recorded every iteration.

## Exact hyperparameters

| Item | Setting |
|---|---|
| Actor/reference | official Qwen/Qwen2.5-3B-Instruct, revision aa8e72537993ba99e69dfaafa59ed015b17504d1; fresh Adam/scheduler |
| Dataset identity | 0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24 |
| Dataset rows | train169028; dev512=NQ256+Hotpot256; final51713, unchanged split |
| Batch | 512 questions ×5 leaves=2560 trajectories; PPO mini128, epochs1 |
| Microbatch | actor global16/local2; old/ref global32/local4; 8 GPUs |
| Actual update budget | 200 complete outers;20 optimizer opportunities/rank/outer; successful writes and skips reported separately |
| Optimizer | Adam; LR1e-6; linear warmup57 credited outers, then constant; scheduler once/credited outer |
| Precision | BF16 forward, FP32 master parameters and Adam state |
| PPO | clip0.2, gradient clip1, entropy0; reference KL coefficient0, low_var_kl |
| Safety | DualClip3; sampling-policy KL brake0.2; negative-advantage tail limit0.01; exponent log bound30; finite checks |
| Sampling probabilities | actual behavior μ correction enabled; native sampled-prefix token preservation enabled; repeat-actor tolerance0 |
| Sampling | temperature1, top_p1, top_k−1, best_of1; vLLM TP1, GPU memory utilization0.4 |
| Search | max4 turns, retrieved top3; start/prompt/response/observation lengths2048/4096/500/500; no forced final answer |
| Cadence | no initial baseline; dev512 every5; checkpoints5/50/100/150/200; rolling native state every outer; final51713 |
| Infra | H10018/h100-10/10.33.0.18;8 GPUs, exclusive Slurm,32 requested CPU,256GiB,168h,no-requeue |
| Services | Ray127.0.0.1:16382; GPU E5/full8GPU FP16 FAISS HTTP127.0.0.1:18082 |

`production_overrides.json` is copied directly from the production manifest and is the authoritative full override list. Hydra base config is `verl/trainer/config/ppo_tree_branpo_fresh200.yaml`; invoking only that base without overrides is insufficient.

An outer is credited only after all20 opportunities are processed with at least one successful write. All-skipped or abandoned attempts do not advance the outer/warmup. Partial successful writes remain separately accounted. Reference KL0 does not disable behavior-policy brakes.

## Launch on the existing retained18 installation

This is a reproduction command for a **new** fresh run, not a resume command. The launcher refuses while another Tree GPU job exists; job3360 is left running. It is deliberately tied to the verified retained18 installation, not a portable one-command environment installer.

Prerequisites: the existing Python environment at `/dataset/shaoshitong/search-r1-rl/tree-grpo/env/bin/python`; official model/common-data/wiki index/E5 assets already on this same node; storage symlink points to `/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008`; the external retriever and owned-runtime helper have the exact hashes pinned in `manifest.template.json`. The helper is included as `tree_shared_owned_runtime_fixed.py` here for source backup. No assets, credentials or environments are included in Git. Never overwrite an active installation to launch this release.

On h100-10, in a separate remote source directory:

```bash
git clone --branch codex/bran-tree-grpo-release-20261011 --single-branch   git@github.com:shaoshitong/Search-R1.git bran-tree-grpo-release
cd bran-tree-grpo-release
export BRAN_TRAINING_INSTRUCTION='Start a fresh official Bran Tree-GRPO 200-outer run using this published protocol; continue the same instance through dev and final51713.'
bash release/bran_tree_grpo/launch_retained18.sh
```

The script verifies the checkout against the remote branch, creates a fresh immutable package, pins the source/review/manifest, renders8GPU Slurm and invokes `submit_once.py`. Existing/unknown submission journal prevents retries. Missing prerequisites refuse launch; never weaken the guards. Source/config/split identities, UUID lease, own cgroup and foreign memory/SM15% guard remain enforced.

## Release cleanup and evidence scope

Historical per-job outputs, probes, copied author code, old cluster bootstrap/smoke scripts and obsolete review/debug receipts are absent from this branch tree. Original local checkout and historical branches remain available. Git history is retained; this is not a force-push or history purge. `CLEANUP.json` records the removed paths.

Core training sources, meaningful regression tests, native recovery/state handling, resource guards and cost/variance observers are retained. The required regular-file/hash reader previously named `build_package_draft.py` is now `pinned_files.py`; callers are updated. This release changes packaging and documentation only, not current job3360.

At the checked milestone50:50 actual complete updates,1000 Adam writes/rank,0 skips, dev512 F1 46.972544%/EM39.062500%. This is an intermediate result;200-outers/final51713/full HF-tokenizer cold reload are not complete acceptance.
