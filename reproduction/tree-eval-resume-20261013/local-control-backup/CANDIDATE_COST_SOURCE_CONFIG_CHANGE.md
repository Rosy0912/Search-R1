# 成本摘要来源配置

`candidate_cost_summary.py` 改为默认读取当前 `outputs/STATUS.json` 的完整 source commit / 实际 run / unit，选择相同 source+run 的最新本地 poll，拒绝混用来源。可用 `--status`、`--source-commit`、`--run-directory`、`--unit`、`--poll` 明确指定旧源，沿用原 SSH 管理路线。

只在远端聚合已稳定完成的 tree-audit；原有 stage 方差/耗时指标直接读取发出的标量，不重算模型或奖励。未发出的方差指标保留 pending。每个 candidate round 的并行 elapsed 只计一次；候选耗时是已有分摊字段，padding差额不按outcome推分。

`--reuse-summary` 可复用已绑定同一source/run的成本摘要，仅由明确指定的本地poll更新进展及方差标量，不产生新远端查询。本次step2的原有方差与完整959.329秒轮次耗时已按 `RECOVERY_POLL_17.json` 原值补齐。

母0bef第1轮原字节保留为 `PARENT0BEF_CANDIDATE_COST_COMPACT.json/md`；child ba9第2轮单独汇总，继承母1不计入child新增rollout。原母2/40Adam保留失败来源及小型rollout-cost JSON标量，`included_in_current_rollout_rollup:false`，不双计有效第2轮。

本次没有测试、code review、模型/奖励重算、训练操作或主线STATUS/REPORT写入。单次实际成本汇总成功，远端只返回小型标量，当前摘要JSON小于100KB。
