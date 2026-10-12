# BAPO 修复独立复审

2026-10-02，只读实现；审查 `bapo.py`、`holdout.py`、`controller.py`、入口、Ray trainer阶段hook、配置及相应测试。审查时工作树持续整合，本报告描述读取时版本；不自动代表后来未读的修改通过。

## 结论

作者 scorer 差异、固定共同2%划分、独立阶段验证及DP非零/整除检查的主要修复正确。**存在两个边界问题需收尾，不能把全部配置路径记为无问题。** 原真实169615题资产及生产阶段生命周期未测试。

## 核证

- 固定作者 `Liushiyu-0709/BAPO-Reliable-Search@8ddbb4fa23f105d52d3621d232f839212e895c3f` 的 `get_f1_score` 双侧yes/no/noanswer判等、zero-overlap=0，与新 `answer_f1` 一致。普通CoSearch F1独立，不误施加分类判等。
- prompt序列化整组抽样，重复prompt同侧；hash(seed+prompt)使抽样顺序无关。holdout排除最终测试相同prompt，训练与阶段集互斥；全算法入口共享同源哈希/specification、2%/seed42和不可变产物。2%为独立问题数比例，不保证重复题情况下恰好2%行数，已文档说明。没有修改原源文件。
- hash校验、进程文件锁、临时目录到产物目录的rename保证新入口复用；阶段Controller再次检查阶段prompt与训练/最终集交集。该检查针对精确序列化prompt，没有声称消除语义近重复或原训练/终测历史污染。
- `stage_validation_freq`与最终test_freq解耦，使用独立stage loader与EM reward，finally还原。`weighted_validation`按各数据源**实际评估题数**加权，独立探针2题*.5+8题*1得到.9，未错误算成.75。五次连续非改善切换stage、不停止训练。
- global mini/micro除DP、mini整除micro、world除SP，OPSD要求SP1和globalmicro=DP，避免初始化模型后micro floor0。训练question batch/world整除是额外保守配置约束，无错误floor。
- 本轮实际命令 `.venv-cpu/bin/python -m pytest -q tests/baselines/test_bapo_fixes.py tests/baselines/test_models_and_driver.py`，**9 passed**。包含真实fit的BAPO stage验证/最终验证关闭/还原，不能替代真实3B/vLLM测试。

## 需修复边界

1. **P2 / 缺extra_info绕过UID校验**：`holdout.split_frames`仅在列存在时校验index。100个不同prompt但无extra_info的DataFrame实际成功拆为98/2；`RLHFDataset.__getitem__`对缺extra_info默认index=0，所以所有问题会静默并组。应明确要求每题合法index并校验，或在共同split里生成稳定UID（需文档固定）。实际生产文件是否缺该列未读取，不声称已发生。
2. **P2 / 未更新仍观察阶段**：Ray hook位于critic_warmup条件块外，且未根据新空mask skip的 `actor/optimizer_steps` 判定。配置warmup>当前step或全mask mini被跳过时，模型没更新但阶段仍累加非改善次数，可能提前切plateau。默认warmup=0且正常轨迹时没有该问题。若语义定为“每个actor update后验证”，需只在真正actor optimizer step后observe；若刻意按iteration调阶段，应改文档并说明无学习窗口的影响。

以上已通知root，未自行修改实现。共享数值修复、保留Search-R1上下文、完整resume能力与资源验收由主任务整合。
