# 第6轮故障与 native5 恢复边界

当前 ba9 服务已终态失败：MainPID=0，Result=exit-code，ExecMainStatus=1。stage 实测 Ray/Boost 创建线程报 `Resource temporarily unavailable [system:11]`；记录为线程 EAGAIN，没有 GPU OOM 证据。精确耗尽的线程/pids限制为 unknown。

已完成有效进度仍为 **5/200、100 Adam/rank（8rank）、scheduler5**。native5 manifest 状态为 COMPLETE_SAME_WORLD_STATE_SAVED，8rank full_worker_state_finite=true。有效来源分别是母0bef第1轮、ba9第2–5轮；失败母2和失败ba9第6轮不计有效轮。

最新完整固定512评测：F1=0.23584841611780016（23.5848416%），EM=0.18359375（18.359375%）。

失败第6轮 rollout 已完成，wall=576.467779420316秒，emitted tokens=950630（含原账本padding）。这不证明 actor update完成；未记录的部分Adam更新为 **unknown**，有效轮信用0。

ba9整个服务段实测 CPUUsageNSec=29067275364000，run-result wall=5552.515588521957秒，包含启动、有效第2–5轮及失败第6轮，不冒充失败第6轮独占成本。

raw快照的ControlGroup为空，错误回退到cgroup root；其中memory/cpu/procs统计不适用于该unit，不用于故障定量。终态快照明确该unit实际cgroup目录已不存在，pids/memory事件为unknown。

下一恢复请求边界是完整native5，继承实际100/rank后从第6轮重算。新child实际冷恢复尚未验证；不声称exact executor equivalence。目标仍为200，goal未暂停。摘要只用本地已存元数据，未操作训练、远端或tensor。

原native5 manifest：`/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:ba9f6627ec9a4e9749830d7b6acca53f6feec6b4/checkpoints/training_state/global_step_5/manifest.json`，SHA256 `b112caffda6514c34a07f2f6b6ecc62096c0717eb09ebae46aac71c6b9dfe09a`。JSON保留8rank、driver身份/计数/hash与原始成本口径。原PT当前可用性为unknown；记录原保存标志不等于新child已cold reload。
