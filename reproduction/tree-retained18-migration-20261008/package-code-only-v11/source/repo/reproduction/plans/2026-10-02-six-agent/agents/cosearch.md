# Agent handoff: cosearch

Read ../PLAN.md first. Own branch `reproduce/cosearch`, starting code0723dca. Node `h100-16`,8 H100 GPUs0-7. Planned candidate assignment, not an existing reservation. Recheck utilization<=15% AND memory<=15%; existing small processes are allowed by the user and must remain untouched. First merge the reviewed common preflight adaptation; current code still rejects all processes. Never bypass leases or kill original processes. Obtain the proper Slurm allocation.

Root: `/dataset/shaoshitong/search-r1-rl/cosearch`. HTTP=18086,Ray=16386. Follow shared environment/assets/CPU retriever preparation and Ray initialization in PLAN. Common setup owner is the GiGPO agent; all six must use the SAME reviewed common commit. Do not alter other checkouts or main/search-r1/search-g1.

Set COSEARCH_RANKER to the base asset path for a separate CPU FP32 ranker/AdamW instance; disclose this warm-start versus author settings. Verify real top50 retrieval,valid3 document IDs,query grouping,relevance/main-reward gating and nonzero ranker updates. Measure50-document prompt length; if context overflows, stop and ask the user about passage clipping or author-ranker changes, never silently reduce candidates or use gold sorting.

After PLAN section6 exports AGENT_ROOT,SEARCH_R1_ASSETS,unique persistent output/runroot,CUDA_VISIBLE_DEVICES and RAY_ADDRESS, and the reviewed CPU retrieval service is healthy, launch:

```bash
export COSEARCH_RANKER="$SEARCH_R1_ASSETS/assets/qwen2.5-3b"
bash reproduction/baselines/run.sh cosearch \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  retriever.url="http://127.0.0.1:18086/retrieve"
```

CPU gate:

```bash
python -m unittest discover -s tests/baselines -v
python -m unittest discover -s tests/reproduction -v
```

GPU smoke uses a NEW run-id and the same command with `trainer.total_training_steps=3 trainer.val_before_train=false`; verify two actual outer updates in logs. This is only an update-budget check, not proof all expensive validation is disabled. If final full evaluation remains enabled, common owner must add/review a distinct smoke-only flag; do not remove formal final evaluation. Then5-step checkpoint/loading validation before500-update formal run. Keep train512/mini256,8-card globalmicro8,SP1,model/data/search unchanged. If OOM or algorithm/task conflict requires protocol change, explain and ask user; context mismatch is explicitly retained.

Prove finite losses/logprobs,nonzero intended parameter changes,correct masks/normalization,path-independent split content identity and output SHA (PLAN section8) and actual production3B inference/search. Record auxiliary and retrieval costs. Full exact optimizer/RNG/dataloader resume is currently unsupported; use tmux and attach to the same process rather than duplicating jobs.

After changes, start an independent subagent for strict paper+author-code comparison of code/math/logic/infra. Fix findings and revalidate. Push only `reproduce/cosearch` and report fixed commit/common commit,config/model/data hashes,environment freeze,GPU UUID,snapshot,logs,metrics,cost and deviations. Sync final REPORT to `/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/cosearch/REPORT.md`. Do not commit keys/auth/checkpoints/sensitive logs.

Mandatory shared review corrections: PLAN section8 defines content-based split identity (absolute paths differ), a new holdout-only checkpoint selector (not currently implemented), and TOP-D ordering. TOP-D cannot run algorithm smoke before the trained teacher exists. GRPO selector candidates:actual updates5/100/200/300/400/500; maximize2% holdout mean answerF1, ties prefer earliest update; never inspect final test for selection. Actor save already includes a complete HF directory.
