# 基于 search-r1 的算法分支迁移独立复审

日期：2026-10-02。只读实现复审；新增迁移回归测试，未改算法、训练入口或原 Search-R1 运行。

## 结论

**未发现本次迁移新增的阻断缺陷。** 当前 `search-rl-algorithm` 的共同祖先是用户指定的
`origin/search-r1` 提交 `9ab694499ad3913b67499f7a4ea13da6b544d459`。
六算法来源为此前修复后的 `68184965c532fe30fef0d89c3ea19f976ac3bd63`。
结论针对迁移完整性与 CPU 控制流；不等价于六算法已通过正式 GPU 全链路或论文效果验收。

## 静态核对

1. `advantages.py/actor.py/bapo.py/cosearch.py/distillation.py/objectives.py/rollout.py`
   与旧修复提交一致；shared PPO、masked_mean、actor 与 fsdp worker 的数值和空更新修复一致。
   迁移没有丢失先前的目标函数、奖励规则或 Tree 合批修复。
2. 新基座 `_prepare_final_validation` 完整保留，与 CoSearch `_baseline_validation` 装饰器共存。
   step 5 额外 checkpoint、500 次更新后的终端保存、完整最终验证与 JSON 指标保留。
3. `ppo_baselines.yaml` 采用周期 `validation-64.parquet`、终端 `test.parquet`、val batch 64，
   acceptance_step 5、训练前验证；与新基座 `reproduction/h20/train_paper_grpo.sh` 相符。
   原主模型、窗口、四动作、top3、KL、offload 和训练调度保留。
4. 共同 holdout 的来源摘要/问题排除覆盖周期验证与完整 final 文件。
   BAPO 路径别名检查和题目重叠检查覆盖 final_val_files；preflight 要求最终集存在。
   checkpoint state 同时记录新基座与旧算法来源，不再把旧基座 SHA 当作当前基座。
5. `.gitignore` 保留新 search-r1 全部规则，只补 `.venv-cpu/`。
   `reproduction/baselines/` 之外的复现文件相对新基座无修改。
   没有移植 search-g1 的独立 reward/probe/judge 组件；新基座不是 search-g1。

## 独立运行证据

命令：

```bash
.venv-cpu/bin/python -m pytest tests/baselines/test_branch_migration.py tests/baselines/test_models_and_driver.py tests/reproduction/test_final_checkpoint.py -q
```

**20 passed**，日志：[evidence/independent-review-tests.txt](evidence/independent-review-tests.txt)。
其中新迁移测试 9 项：

- 六算法分别执行实际 `fit()` 终端语句：周期验证、实际 BAPO stage_validation 的暂存恢复、
  checkpoint 与完整最终集切换的先后顺序。最终验证不再误用阶段集或周期 64 题。
- 实际 `_prepare_final_validation`、真实临时 parquet、实际 RLHFDataset 与 DataLoader：
  65 题产生 64+1 尾批，七源与全部唯一 index 都保留，最终评测忽略周期 val_data_num。
- CoSearch 验证失败时 flag 在原为 true/false 两种情况下都恢复，不留下 validation 状态。

另包含已有六方法 actual-fit（GPU transport mock）、离线 tiny Qwen teacher/ranker 更新、
step 5 不停止与终端 off-boundary checkpoint 测试。实际终测使用临时数据，不能代替
真实 51,713 题七集验收，也不能证明 GPU/vLLM/remove-padding/offload 的生产运行。

## 保留边界

用户已决定保留 Search-R1 上下文不一致；本次未改变该规则。
2% 阶段验证划分是所有新算法和新 GRPO 的共同控制变量，旧完整训练集历史指标不能直接作同条件比较。
此前报告中的生产验收、远端资产、完整精确 resume 限制仍有效；原历史报告保留其来源 SHA。
本复审未执行 GPU、未修改原复现 agent checkout/远端任务、未推送分支。
