# H10018 已完成 rollout 候选成本摘要

实测：2026-10-13T16:53:44.353339+00:00；源码 `ba9f6627ec9a4e9749830d7b6acca53f6feec6b4`。
已完成 rollout：1；详细轮次：2。

来源角色：child_native1_continuation；本文件仅汇总 `/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:ba9f6627ec9a4e9749830d7b6acca53f6feec6b4` 内完成的 rollout，不合并母链或失败历史成本。

每轮 512 题、1024 原始根轨迹与 1536 accepted continuation，共 2560 训练叶。fallback 是 accepted 的子集；discarded 额外产生费用，不进入训练叶。

| rollout / stage | 尝试 | accepted | discarded | fallback（accepted内） | 新生成token（候选记录） | 检索 | 评分 | 继承response / observation token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 / A1 | 512 | 512 | 0 | 1 | 117007 | 287 | 512 | 164788 / 138346 |
| 2 / B1 | 1005 | 512 | 493 | 465 | 233907 | 613 | 1005 | 340522 / 286956 |
| 2 / B2 | 1014 | 512 | 502 | 457 | 239739 | 604 | 1014 | 359365 / 302235 |

rollout 2：候选 2531 = accepted 1536 + discarded 995；accepted 中 fallback 923。

| rollout / outcome | 候选数 | 新生成token | 新prefill token | 检索 | 评分 | 继承response / observation token | 生成 / 评分分摊秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 / accepted | 1536 | 448147 | 1554062 | 1327 | 1536 | 229920 / 192431 | 264.140 / 13.707 |
| 2 / discarded | 995 | 142506 | 1017281 | 177 | 995 | 634755 / 535106 | 93.655 / 8.691 |
| 2 / fallback | 923 | 318635 | 958717 | 1019 | 923 | 0 / 0 | 205.740 / 8.539 |

| stage | 候选轮数 | 共享生成elapsed秒 | 共享评分elapsed秒 | ledger生成token（含padding） | ledger prefill token（含padding） |
|---|---:|---:|---:|---:|---:|
| A1 | 1 | 49.335 | 4.289 | 118215 | 492096 |
| B1 | 4 | 159.538 | 9.138 | 244716 | 1094522 |
| B2 | 4 | 148.922 | 8.971 | 243134 | 1091729 |

候选记录新生成 590653 token；候选轮ledger 606065 token、padding rows 100；差额 15412 token未向accepted/discarded/fallback分摊。
候选并行轮生成elapsed共 357.795 秒、评分elapsed 22.398 秒，每轮只计一次。完整rollout wall 539.1262151226401 秒（controller ledger原值）。

母链 0bef 第1轮原文件已原样保留为 `PARENT0BEF_CANDIDATE_COST_COMPACT.json/md`；继承1轮/20 Adam不算作本文件新rollout。母链失败第2轮单列为历史成本，child第2轮是重算结果，两者不叠加为有效第2轮。
失败母链第2轮：源码 `0befbaf99fc44942105831b3fbec94e4bf1b8ccb`；rollout wall 581.9643383733928 秒；emitted token 960898；这里只列历史费用，不纳入本文件rollup。

| 已有stage指标 | step | 原值 |
|---|---:|---:|
| timing_s/step | 2.0 | 959.3291499428451 |
| timing_s/update_actor | 2.0 | 261.0409162379801 |
| timing_s/stage_testing | 2.0 | 34.21468160673976 |
| tree_variance/group_raw_reward_population_variance | 2.0 | 0.06749999999999991 |
| tree_variance/within_root_raw_reward_population_variance | 2.0 | 0.046440972222222134 |
| tree_variance/intra_adv_population_variance | 2.0 | 0.12369750731128847 |
| tree_variance/group_actual_adv_population_variance | 2.0 | 0.7414528663652973 |

完整的 A1/B1/B2 × accepted/discarded/fallback 交叉成本、轮次明细、字段覆盖数和已有 tree_variance/timing_s 指标见同名 JSON。未新增数学审核。

字段限制：逐候选有生成、retained、prefill、检索、评分、继承前缀及分摊耗时；没有候选级实际独占wall、padding分摊、KV/FLOPs/缓存节省或货币费用。继承前缀是重复继承事件，不能计为新生成token。

只在原 H10018 远端读取本任务JSON并返回小型标量摘要；未搬运raw rollout/tokens/PT/model/evalarrays，未更改训练源码、服务或主线STATUS/REPORT。
