# Seven-group small status monitor

Read-only HKUST process identity, original completed-phase evidence, worker row counts and direct GPU memory. Output includes aggregate statistics and SHA/path references only. It never exports full prediction rows, model states or raw trajectories, writes Node files, imports Torch, sends signals, acquires training locks or runs inference. Missing or unreadable evidence is reported explicitly; a row count is not a completed evaluation. The original driver remains the only executor.

Run exact published status.py through the original remote Python with -B and CUDA_VISIBLE_DEVICES empty, supplying --base /data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008. Save only its small stdout audit on Mac. Existing scientific source and running processes are unchanged.
