# CoSearch 修复独立只读复审

2026-10-02；审查当前工作树 `cosearch.py`、六项专属测试及修复记录；没有改算法/启动远端/提交git。

**结论：本轮 CoSearch 标签、分组、无信号组和序列聚合修复通过CPU复核；正式规模验收仍未完成。未发现新的P1实现缺陷。**

依据：[论文v2](https://arxiv.org/html/2604.17555v2)，固定作者 `snap-research/CoSearch@763bf8c4d7caa363ad845d39ddfd53b81ae377bd`，本地缓存的 `co_search_tool.py`、`answer_match_reward.py`、`uid_group_functions.py`、`score_assign_functions.py`。本轮重新读取原码与论文，不仅复述修复报告。

## 核对结果

- 文档matcher：NFD、Unicode连续词/标点token、lower、精确子序列，与作者算法一致；没有沿用QA去冠词。修复证据文件12个反例本地/作者一致。
- trivial判定：所有别名均yes/no/true/false时availability=false；合法ranking只取main reward，非法ranking仍-.2。混合别名保留作者行为。
- query normalize：与作者相同的去冠词、标点变空格、空白合并；token-F1和首代表greedy分组未跨初始问题。默认不按availability分组是论文优先的既有适配，报告明确区别作者recipe。
- reward：Hit1/3/5、availability、连续mainF1、阈值≤.5各分支对应论文式(6)。top3使Hit5等于Hit3，是用户要求固定Search-R1搜索预算的明确适配。非法主轨迹在ranker分组前剔除；gold只用于训练标签，不进入rank prompt。
- 无信号组：按整个group reward集合筛除同奖组，避免AdamW动量/衰减导致无信号漂移。informative group内零advantage成员仍计入batch序列分母，未错误只挑非零成员。组优势在完整group上预计算，之后可拆optimizer batch；每个batch每sequence等权，token数差异不再改变sequence权重。
- 序列目标：`sequence_ppo_loss`的min方向、clip、temperature、old/current score模式一致；`mean()`是每response内token均值，外层除实际batch response数。尾batch按自身样本均值，是常规SGD批处理语义，不宣称与一个总batch单步AdamW完全等价。
- 泛型HF输入：排除token_type_ids；实采样和old probability重新前向对齐；greedy validation不积累training event且score成本独立。

## 实际测试

```
.venv-cpu/bin/python -m pytest -q tests/baselines/test_cosearch_fixes.py tests/baselines/test_models_and_driver.py
10 passed, 4 warnings, 3.23s
```

含六项CoSearch反例与实际HFgenerate及完整fit driver更新路径。warnings为已有LibreSSL/Ray弃用以及greedy设置top_k=0提示；不是断言失败。未以tiny CPU测试替代目标ranker/50候选规模证据。

## 小项与尚未关闭事项

- `sequence_ppo_loss` docstring标“Eq.5”不准确：该论文式(5)是semantic grouping，GRPO目标来自式(3)。建议仅改注释，非梯度问题。
- 额外ranker参数 `batch_size/epochs/max_new_tokens/clip_ratio` 尚未完整正数校验；默认配置有效，但不合理override应在模型加载前fail-fast。已通知root。
- CPU FP32/eager ranker、真实top50服务/合法ranking生成、长度/显存/延迟、完整双agent checkpoint再恢复仍未验收；jointresume明确不支持。修复记录对此边界表述诚实。
- 用户明确保留Search-R1上下文不一致；不是本次遗漏修复。共享主actor数值修复由root整合，本审查没有重新宣称所有算法生产链条通过。
