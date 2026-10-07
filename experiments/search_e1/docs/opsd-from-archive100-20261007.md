# Eight-card OPSD from archive GRPO checkpoint100

Latest human instruction on2026-10-07 supersedes the older first-GRPO200 requirement for the replacement eight-card experiment. Stop only the owned degraded aligned8 run after preserving complete eight-rank weights, Adam, scheduler and RNG. Keep archive4 training and every older stopped run unchanged. The exact human message timestamp is unavailable; remote authorization receipt records the live-turn receipt time and original text.

Use archive4 global_step_100 model weights, verify every transferred file SHA256, and start OPSD directly. No additional GRPO and no GRPO optimizer restoration. The display100→105 means GRPO source checkpoint100 plus five real OPSD updates; source and OPSD counters remain separate.

OPSD already computes valid-token mean within each trajectory then global trajectory mean. Its original pair_loss and encoded_pair functions are unchanged in the new runtime. Preserve lr1e-5, rank16, alpha32, clip10, chunk32, max_context8192, seed42, epochs1 and global trajectory batch32 (eight ranks × accumulation4). GRPO entropy, micro64, KL coefficient and scheduler do not apply to OPSD. Do not add a GRPO entropy term to imply preservation.

Full training data169103 questions, five fresh rollouts each, original mining,512 heldout validation and seven-benchmark final evaluation. Full generation and mining precede training; initialization, synthetic updates, shortened data or baseline evaluation are not acceptance. No extra GRPO is included after this OPSD stage without new human authorization.

New HKUST1 root: /data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007. Source: HKUST4 /weka_3/shaoshitong/search-e1-config-study-20261004/runs/archive/round-1-grpo/actor/global_step_100. New tmux e1-opsd-from-archive100-20261007; logs/pipeline.log, logs/rollouts.log, logs/opsd.log; receipts/pipeline-stage.json, receipts/archive100-transfer-accepted.json, receipts/opsd-runtime.sha256.json. Separate memory observer e1-opsd100-memory-20261007.

Instrumentation observes each real Adam update, finite trainable parameters and gradients, changed LoRA_B weights, actual per-rank Adam counters and GPU allocator peaks. Save actual adapter and optimizer/RNG state at OPSD update1 and5 without stopping or resetting the one continuous run. Acceptance requires real updates1–5, same source model, finite changed weights, matching Adam counters on all8 ranks and healthy observed GPU usage. Do not label GRPO metrics or source checkpoint as new OPSD updates.

Current setup status: source checkpoint transferred and12 files hash-verified. Old state capture queued behind active update; new pipeline waits for full-complete.json before the narrow stop. Five OPSD updates have not yet been executed; no acceptance claimed. Original frozen runtime retained; instrumentation modifies only independent runtime/opsd copy. Hourly automation updated to new human scope; never reactivate replaced aligned8 GRPO.
