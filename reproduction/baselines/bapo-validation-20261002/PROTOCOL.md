# Agent 3: BAPO on Search-R1 validation

Execution scope is the approved six-agent plan at `1620c2e`, with algorithm code parent `0723dca0c1458482ac3e78f8ed43af04893955d2` and baseline ancestor `9ab694499ad3913b67499f7a4ea13da6b544d459`.

Branch: `reproduce/bapo`. Target: `h100-5`, eight H100 GPUs, persistent root `/dataset/shaoshitong/search-r1-rl/bapo`. HTTP port 18083; Ray port 16383. Shared deployment changes must come from the independently reviewed `reproduce/common-setup` commit owned by Agent 1. Do not duplicate or bypass its adaptation.

Retain Qwen2.5-3B base, original Search-R1 prompts, Wiki18/E5 assets and search settings. Batch512, mini256, global micro8, SP1, TP1, lr1e-6, warmup .285, five trajectories per question. Retain existing context mismatch behavior and report it. No extra IDK prompt, force-final, new retrieval, or test-guided stage selection.

BAPO reward is categorical-safe answer F1, format failure -1, legal IDK bonus .5 only in all-incorrect groups. Early gate uses batch IDK ratio<.05; late gate uses answer diversity<G/2. Validate on common disjoint2% holdout after every actual actor update; consecutive5 non-improvements switch permanently to late stage. Late all-wrong/no-IDK groups receive at most2 additional full-group rollout attempts. Include discarded rollout cost.

Validate CPU regression and independent paper/author-code comparison first. Once common setup freezes, merge exactly its commit, verify source/data/split content identities, deploy an isolated environment, and request proper Slurm allocation. Preserve original low-memory processes; apply utilization<=15% AND memory<=15%, retain UUID leases and checks. No global process termination.

GPU stages:8-card BF16/NCCL/flash/vLLM and real CPU retriever parity;2 actual-update smoke;5 actual-update checkpoint/load acceptance;500 actual-update formal run and full51,713-question evaluation. Every stage has a new run ID. Smoke may disable final evaluation only through the reviewed common smoke setting, while BAPO stage validation frequency remains1. Training losses, logprobs, gradients, parameters and intended masks must be checked on real production inputs.

Deliver exact code/common commit, resolved config and environment freeze, model/data/split hashes, GPU UUID snapshots, raw logs, F1/EM/format rate, IDK/coverage, search/token/sample/update totals, phase-validation overhead, retriever CPU/RAM and total compute cost. New GRPO control is the comparison; old100%-train GRPO is not a fair control. Do not claim performance success before real training and final evaluation. Exact optimizer/RNG/dataloader resumption remains unsupported.

Author comparison sources: [paper v2](https://arxiv.org/html/2601.11037v2), [fixed author source](https://github.com/Liushiyu-0709/BAPO-Reliable-Search/tree/8ddbb4fa23f105d52d3621d232f839212e895c3f). This is a controlled Search-R1 adaptation; original BAPO uses a different prompt/model/data/rollout recipe, and source counters/diversity differ from the chosen paper semantics.
