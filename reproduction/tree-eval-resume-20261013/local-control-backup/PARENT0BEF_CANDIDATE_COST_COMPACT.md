# H10018 已完成 rollout 候选成本摘要

实测：2026-10-13T16:00:16.780197+00:00；源码 `0befbaf99fc44942105831b3fbec94e4bf1b8ccb`。
已完成 rollout：1；详细轮次：1。

每轮 512 题、1024 原始根轨迹与 1536 accepted continuation，共 2560 训练叶。fallback 是 accepted 的子集；discarded 额外产生费用，不进入训练叶。

| rollout / stage | 尝试 | accepted | discarded | fallback（accepted内） | 新生成token（候选记录） | 检索 | 评分 | 继承response / observation token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / A1 | 512 | 512 | 0 | 2 | 113146 | 319 | 512 | 149204 / 126006 |
| 1 / B1 | 1016 | 512 | 504 | 470 | 232164 | 603 | 1016 | 346422 / 290438 |
| 1 / B2 | 1001 | 512 | 489 | 454 | 226442 | 586 | 1001 | 350049 / 294389 |

rollout 1：候选 2529 = accepted 1536 + discarded 993；accepted 中 fallback 926。

| rollout / outcome | 候选数 | 新生成token | 新prefill token | 检索 | 评分 | 继承response / observation token | 生成 / 评分分摊秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 / accepted | 1536 | 428436 | 1517576 | 1343 | 1536 | 213888 / 180663 | 287.155 / 13.556 |
| 1 / discarded | 993 | 143316 | 997613 | 165 | 993 | 631787 / 530170 | 104.520 / 8.551 |
| 1 / fallback | 926 | 305709 | 932357 | 1012 | 926 | 0 / 0 | 226.492 / 8.631 |

| stage | 候选轮数 | 共享生成elapsed秒 | 共享评分elapsed秒 | ledger生成token（含padding） | ledger prefill token（含padding） |
|---|---:|---:|---:|---:|---:|
| A1 | 1 | 51.097 | 4.076 | 114806 | 492320 |
| B1 | 4 | 170.265 | 8.703 | 249265 | 1108154 |
| B2 | 4 | 170.313 | 9.327 | 234894 | 1074768 |

候选记录新生成 571752 token；候选轮ledger 598965 token、padding rows 135；差额 27213 token未向accepted/discarded/fallback分摊。
候选并行轮生成elapsed共 391.675 秒、评分elapsed 22.106 秒，每轮只计一次。完整rollout wall 574.796056818217 秒（controller ledger原值）。

完整的 A1/B1/B2 × accepted/discarded/fallback 交叉成本、9轮明细、字段覆盖数和原有 tree_variance/timing_s 指标见同名 JSON。未新增数学审核。

字段限制：逐候选有生成、retained、prefill、检索、评分、继承前缀及分摊耗时；没有候选级实际独占wall、padding分摊、KV/FLOPs/缓存节省或货币费用。继承前缀是重复继承事件，不能计为新生成token。

只在原 H10018 远端读取本任务JSON并返回小型标量摘要；未搬运raw rollout/tokens/PT/model/evalarrays，未更改训练源码、服务或主线STATUS/REPORT。
