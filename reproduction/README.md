# Reviewed Search-R1 reproduction

This fork retains the full upstream history and Apache-2.0 attribution. The baseline is [upstream commit 598e61bd](https://github.com/PeterGriffinJin/Search-R1/tree/598e61bd1d36895726d28a8d06b3a15bed19f5d3); the experiment reference is [Search-R1 paper v5](https://arxiv.org/html/2503.09516v5).

The reviewed profile reproduces the **GRPO variant**, not the paper's default PPO main-table experiment. It uses Qwen2.5-3B base, NQ+HotpotQA, five trajectories per prompt, masked observations, exact-match outcome rewards, top-p/temperature 1, top-3 E5/Wikipedia retrieval, and four total actions. The original NQ-only launcher is retained as `h20/train_quickstart.sh` and must not be described as the paper experiment.

The deployment uses eight H20-3e GPUs, CPU FAISS FP32 and a GPU E5 encoder. H20 versus H100 and CPU FP32 versus the example's GPU FP16 search are numerical/hardware departures. No bitwise ranking equivalence or paper score reproduction is asserted. Complete training and seven-dataset EM remain required.

## Corrections

- Preserve sampling and validation metadata through filtering and GPU-size padding.
- Keep validation tail batches and report each dataset's evaluated count.
- Parse outcome answers only from contiguous model-authored spans; exclude prompts, observations and tags spanning an observation.
- Save the final updated policy, including updates off the regular save interval.
- Keep quickstart's extra final generation by default, but disable it for the paper's four-action budget.

The trainer's unchanged counter starts at one and terminates after incrementing. Setting its threshold to 501 yields 500 update calls and saves `global_step_500`. Its constant warm-up uses 142 steps for both a nominal 500-step and 501-step horizon. The first call uses zero learning rate under the upstream schedule; this behavior is recorded rather than silently changed.

## Deployment and acceptance

Follow [environment setup](h20/environment-setup.md). Copy the checked-in `h20` scripts to `/data/search-r1`, alongside the checkout at `/data/search-r1/repo`. All caches, data, environments, logs and checkpoints go under that project directory.

```bash
bash /data/search-r1/bootstrap.sh
/data/search-r1/env/bin/python /data/search-r1/download_assets.py
cd /data/search-r1/repo
SEARCH_R1_REQUIRE_REAL_DATAPROTO=1 SEARCH_R1_ADAPTATION_DIR=/data/search-r1 \
  /data/search-r1/env/bin/python -m unittest discover -s tests/reproduction -v
bash /data/search-r1/run_pipeline.sh
```

The pipeline holds a job lock, rejects other GPU compute processes, checks immutable assets, tests NCCL/BF16 and real-model generation, builds mixed data with globally unique group IDs, tests retrieval, then runs a two-step mixed-data smoke with a 9-example validation set to exercise the 8+1 tail. It checks complete, fresh checkpoint shards and actual parameter changes. Training launchers require the supervisor's lock.

Set `SEARCH_R1_FULL_RUN=1` on the pipeline to continue into the full paper-GRPO profile after those checks. Final acceptance requires the fresh step-500 checkpoint, final EM and the complete counts for all seven datasets. A running task or a passed short test is not a reproduced paper score.

Credentials, downloaded models/corpus, environments and raw training logs are not versioned. See [independent reviews](review/) and the [acceptance record](acceptance.md).
