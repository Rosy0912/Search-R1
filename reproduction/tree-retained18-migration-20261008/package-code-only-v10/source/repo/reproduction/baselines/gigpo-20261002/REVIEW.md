# Independent GiGPO review — 2026-10-02

Independent reviewer: review_common. Core advantages, metadata transport and policy objective pass static/CPU checks; no GPU or production metric acceptance.

One rollout: sample five trajectories per question, record observation hashes and action IDs, retain the whole empty/nonempty action timeline. The driver centers each trajectory return within question, recursively discounts returns along steps, centers step returns within exact (question UID, observation hash) groups, then adds episode and step advantages (gamma=.95, omega=1, mean-only). Observation/pad tokens do not receive policy loss.

Action ratio is exp(sum of token logratios), followed by action clipping; nonempty actions and episodes receive equal weights. Driver-computed weights survive row reordering; microbatch sums and DP averaging recover the intended mini objective. Global empty supervision skips AdamW steps; local empty ranks retain collective/backward participation.

Original sources reviewed:
- Paper v2: https://arxiv.org/html/2505.10978v2
- Author core: https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/gigpo/core_gigpo.py
- Author Search configuration: https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/examples/gigpo_trainer/run_search.sh
- Author actor: https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/verl/workers/actor/dp_actor.py

Explicit approved differences: exact observation groups and mean-only versus author similarity=.9 and standard-deviation normalization; episode statistics once per trajectory instead of repeated step rows; joint action clipping versus author token PPO/dual clipping; Search-R1 base model/EM reward/entropy=.001 without author invalid-action penalty. Reference KL and entropy are action-weighted token means, not complete sequence-distribution KL. Original rolling generation/full-trajectory scoring context mismatch is retained and must be measured.

Found and fixed metric reduction: paper losses were already normalized by full-mini denominator but averaged again across micros; KL logged only last micro. The correction accumulates paper PG/entropy/reference KL per mini and weights clipfrac/PPO KL by global-mini weight mass. Loss/backward expressions remain unchanged; GRPO token objective logging is unchanged. Regression uses the real actor, comparing micro1 and micro2 for identical metrics and parameter updates; actual Gloo/DDP runs retain empty-rank and unequal-row invariance. Independent nonuniform GiGPO/TOP-D clipping and unequal-micro KL probes pass.

Evidence: complete branch CPU gate 140 passed / 1 skipped; focused advantages/objectives/driver/numeric/DDP set 34 passed before metric fix; metric/DDP rerun 2 passed after fix. Counts overlap and must not be added. Linux procfs is skipped on macOS.

Still required: eight-card NCCL/FSDP/Flash/vLLM, immutable real assets, CPU vs original GPU retrieval top3/top50 consistency, exact split identities, real 3B two-update smoke and five-update checkpoint load, complete new GRPO control + holdout teacher export, complete GiGPO run, seven-set final F1/EM/format/search/token metrics and measured cost/peak memory. Long-action joint-ratio finite stability needs real rollout validation. CPU pass does not establish effect improvement or paper-metric reproduction.
