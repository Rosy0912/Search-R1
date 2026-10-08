# BAPO 与共同训练划分修复

2026-10-02；仅独立 Search-R1-RL-Baselines checkout，未提交git，未触碰原训练进程。

## 修复

- `bapo.answer_f1` 补齐固定作者 `re_search_faith.py` 的 yes/no/noanswer 分类答案特判：任一侧为分类答案且两者不一致，F1=0。无共同token（包括两侧归一化都为空）为0。非法答案仍-1，IDK仍0。普通F1另保留 `ordinary_answer_f1` 给CoSearch，不把BAPO分类规则套在CoSearch上。
- 分类答案反例不再错误阻断全错组的IDK bonus或plateau整组重采样。
- BAPO阶段验证单独 `stage_validation_freq=1`，每个actor update后验证，与最终七集 `test_freq` 和 `val_reward_fn` 开关解耦。仍使用基座EM衡量holdout能力；临时换独立loader/reward并在finally还原，五次连续非改善切plateau。不是终止训练。第一观测建立best，下一连续五次非改善使第六次观测切阶段。
- 新共同划分模块 `holdout.py`：默认按问题prompt组固定seed42抽2%问题，98%仍训练，同prompt重复行整体同侧；排除最终测试题后选择holdout。存在重复prompt时行比例可能略不同于2%，manifest记录真实数量。若可独立抽取的题不够则拒绝运行，不静默泄漏。
- 所有六算法与新增 `grpo` 入口共用同样源文件+seed+比例的不可变划分。源和产物SHA256+manifest；文件锁与临时目录原子发布，重复运行验证哈希、防修改。原源parquet不修改。新GRPO使用原rollout/advantage路径，不创建算法Controller。
- 入口验证extra_info.index跨题冲突，防止不同问题静默并组。所有算法共同检查正DP/SP/mini/micro、可整除关系；OPSD SP1严格globalmicro=DP ranks，使每rank micro=1，模型加载前拒绝micro floor0。

## 验证

- `tests/baselines/test_bapo_fixes.py`：5 tests通过，涵盖奖励/重采样反例、重复题/终测无泄漏、稳定划分与篡改拒绝、关闭最终验证仍phase切换、验证异常恢复、DP/OPSD参数拒绝。
- `test_models_and_driver.py`：4 tests通过。真实fit包含BAPO update后的holdout验证，即使 `test_freq=-1`、最终 `val_reward_fn=None`，并确认loader/reward还原。
- 从审计缓存固定作者源码AST提取normalize_answer/get_f1_score，8个差分用例全部一致（含多alias、空token、分类答案、标点）。未安装作者训练依赖。

## 边界

用户明确保留Search-R1上下文不一致，未更改截断/评分。数值空mask修复由root统一处理。
尚未对真实169615题文件生成划分，也未在3B/vLLM/真实检索上跑阶段生命周期；当前是可复现代码与CPU验证，不能宣称正式训练通过。
默认2%约3393题，每update的阶段验证有额外推理成本；可显式调大stage_validation_freq，但五次非改善将对应五个验证间隔，需报告该实验配置。
历史完整训练集结果与新的98%训练GRPO不属于相同数据条件；新的算法比较必须使用共同划分GRPO。完整resume仍未实现/未验证，保留resume_supported=false。
