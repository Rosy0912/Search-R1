# CUDA/FSDP tiny smoke 审查

日期：2026-10-02。范围：只读审查 `reproduction/baselines/gpu_smoke.py` 和已经保存的 A800 run-1/run-2 证据；仅新增本审查报告，没有修改算法、启动GPU任务或操作原Search-R1训练。

**结论：PASS，原P2验收弱点已关闭。** 已读取独立run-2结果，exit code 0，五种actor两rank均记录两次finite且>0梯度范数、weight_decay=0和非零参数变化；ranker也在decay0下更新。原run-1保留，GPU任务已自然结束；结束快照GPU0/1均0%/4MiB。

## 原P2发现与修复

原脚本actor及ranker使用AdamW默认weight decay。仅断言参数L1变化大于零，不能排除零梯度情况下单纯衰减导致的假通过。

当前修订：

- actor `AdamW(..., weight_decay=0.)`，保存每方法的weight_decay=0；
- `actor.update_policy` 返回的每项 `actor/grad_norm` 必须finite且>0，并保存grad_norms；实际optimizer_steps必须为2，参数变化>0且更新后所有参数finite；
- ranker optimizer全部param_groups显式decay0，在新optimizer上执行实际update，断言lm_head变化>0及training_events=3，保存decay0；
- run-2另存，不覆盖run-1。

因此新增门禁排除了原发现的decay-only假阳性。ranker没有单独记录grad_norm，但全param-group零decay、全新AdamW无历史动量，配合实际参数变化检查能够证明该次optimizer存在有效梯度更新。后续应保留完整run-2日志与结果，不将旧证据标成新门禁已通过。

## GPU选择与不干扰边界

脚本在CUDA初始化前检查GPU物理index/UUID、利用率≤15%、显存比例≤15%，并拒绝该UUID存在任何compute process；约1秒后复检，CPU tiny模型准备后、spawn之前再次检查。设置CUDA_VISIBLE_DEVICES，仅使用用户选择的GPU；每worker设置10%的PyTorch allocator预算；未发现kill、重启他人任务、覆盖用户模型或改系统环境的操作。

run-1保存的GPU0/1开始均0%/4MiB，结束仍0%/4MiB；高显存占用GPU7没有选入。该证据支持所选两卡当次任务结束后已释放。

锁是同output父目录下的advisory flock，只约束遵循同一路径的本脚本实例，不能阻止其他应用/调度器申请GPU；最后一次检查到进程启动之间仍有竞争窗口。故“空闲检查、不结束他人任务”是best-effort，不能称获得全局独占调度权。10% allocator预算也不等于含NCCL/CUDA其他分配在内的硬性显存隔离。正式共享集群应另用调度器预约资源。

## 已覆盖的运行接口

脚本构造tiny随机Qwen2权重和本地小tokenizer，使用脚本化固定搜索/答案token、observation mask、两步step_ids和受控reward：

- 真正初始化CUDA/NCCL两rank和FSDP，实际调用production `DataParallelPPOActor` 或 `DistillationActor` 的compute_log_prob/update_policy，执行反向和optimizer；
- GiGPO/Tree/BAPO/TOP-D优势进入真实CUDA actor路径；OPSD使用真实冻结tiny教师及student散度路径；
- FSDP FULL_STATE_DICT collective并写torch state文件；ranker在GPU0实际HF模型评分、AdamW更新及保存，GPU1等待；
- 子进程join完成后读取各rank结果，全部成功才写status=passed；异常不伪造通过。

run-1独立result报告五种actor各2次optimizer step，ranker1次，所有记录的参数delta为正，两rank峰值PyTorch allocated约20.36MiB。此结果证明当时CUDA/FSDP与小模型optimizer路径执行成功；它不包含新增run-2梯度范数验证。

## 严格证据限制

- 没有真实在线search rollout、HTTP retriever、tree continuation、Controller或Ray trainer完整链；固定token轨迹不能验证生成时策略、实际uid分组与检索交互。
- TOP-D的teacher_logprob人为构造为old logprob加受控非恒定偏置，不是正式教师推理验收；不能借此宣称TOP-D蒸馏效果或论文理论条件得到实证。
- ranker事件及ranking token脚本构造；不覆盖真实ranker generation、候选召回、解析失败及联合系统训练。
- 不覆盖正式Qwen2.5-3B、vLLM rollout/权重同步、长上下文采样与评分前缀一致性、8卡资源或吞吐。
- full-state权重序列化不等于训练恢复：没有checkpoint reload/optimizer恢复/学习率scheduler恢复验证。
- tiny随机模型不使用正式训练/评测数据，**无论文EM/F1、收敛或性能复现指标**。正式算法结果仍需独立训练、checkpoint/日志与一致评测。
- A800 smoke环境（run-1为Torch2.9.0+cu128）不同于原H20工程Torch2.4/vLLM0.6.3；跨机器环境兼容性不能由本smoke推出。

本审查未发现尚未解决的critical/major脚本问题。已读并校验run-2 result/rank0/rank1/allocation/post-run快照SHA256与manifest一致；两rank五方法的grad_norms长度均为2且finite>0，optimizer_steps=2，所有方法包括ranker均decay0且参数delta>0。run-2 peak allocated约20.13MiB/rank。证据目录：`evidence/gpu-a800-20261002-run2`；结果哈希：`1e04b94aa98b27bea14523686aead344a303c14100078ec560bc6cbedec09fdc`。通过范围严格限定为上述tiny CUDA/FSDP optimizer与序列化，不扩展到真实rollout或论文结果。
