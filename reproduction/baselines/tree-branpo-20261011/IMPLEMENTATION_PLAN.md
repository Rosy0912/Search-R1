# Tree-GRPO + BranPO-inspired sampling

Isolated branch `codex/tree-branpo-tail-20261011` starts from ebef786. The historical checkout and running source are untouched.

Use two original root trajectories and exactly three continuations per question: root 0 receives two, root 1 receives one. Each first continuation starts immediately before that root's final model action, inheriting all prior sampled tokens and retrieval observations. Root 0's second continuation backs up one original-root action state only if its first branch has the same actual training reward as the original root; otherwise it reuses the tail state. Clamp at the earliest available action. Retain every sampled trajectory, including failed and same-reward trajectories. No correctness-based stopping, hidden extra samples, or finite-budget guarantee of different rewards.

Reward decisions use the existing trainer reward callback on composed trajectories and source metadata; answers never enter prompts. Keep Tree-GRPO's existing sum of question-normalized and root-normalized advantages, all token masks, behavior-mu correction, PPO safety, and leaf weighting.

Record each branch's depth, backoff reason, inherited token/observation proof, and rollout costs. Each outer reports population raw reward variance, within-root and between-root actual scalar advantage variance, five-leaf group variance, identical-reward rates, and all-correct/all-wrong/mixed counts, including root-slot summaries. Normalized component variance is diagnostic and not evidence that sampling improves learning.

CPU verification covers exact quotas, token/observation inheritance, bounded all-zero/same reward behavior, tail/backoff choices, and independently checked metric formulas. Keep official fresh200 model/split/batch/LR/search/checkpoint contracts; fresh launch never inherits old optimizer. Remote launch follows independent code review and storage/capacity checks; this implementation task does not launch training.
