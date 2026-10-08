# CoSearch 审计修复记录（2026-10-02）

依据 `audit-20261002/cosearch.md`，作者源码固定为 `snap-research/CoSearch@763bf8c4d7caa363ad845d39ddfd53b81ae377bd`。本次只修改 `search_r1/baselines/cosearch.py`、专属回归测试和本记录。主 actor、共享配置/Controller 由主任务整合，未访问或改变远端任务。

## 已修复

- CS-01：全 yes/no/true/false 答案统一禁止 document availability/relevance。合法 ranker 只接主奖励；非法 ranking 仍为 -.2。
- CS-02：文档匹配按作者 NFD Unicode tokenizer 连续精确 token 串，保留标点和冠词；query 分组另用作者去冠词、标点改空格规则。`The Hague`、`foo-bar`、Unicode 重音等反例已覆盖。
- CS-05：先过滤**整个**同奖励组，再做 optimizer；组内有奖励差异时保留零 advantage 的成员。报告同奖跳过组数、真实训练事件数和非零 advantage 事件数。即使已有 Adam momentum，同奖组也不执行更新。AdamW 权重衰减显式读取 `weight_decay`，缺省仍为 .01。
- 数学聚合：ranker loss 先取各 ranking response 的 token mean，再对一个 optimizer batch 的 response 等权平均，避免原总 token mean 给长序列更大权重。温度采样/old score/current score 相同；非有限 score/ratio 直接报错，不对有效值任意裁剪。
- 泛型 HF tokenizer 兼容：显式不返回 `token_type_ids`，实际未 patch HF generate 已通过。
- CS-06 部分：训练/验证 scoring token 成本独立累计；checkpoint 额外保存 `ranker_state.json` 会计快照并明示 `resume_supported=False`，未将该快照冒充联合恢复功能。
- CS-07 主 F1 空归一化边界及 BAPO 布尔特判隔离：由 BAPO 修复任务提供 `ordinary_answer_f1`、Controller 接入，CoSearch 不施加 BAPO 分类特判。

## 验证

`tests/baselines/test_cosearch_fixes.py` 六项：作者标签反例、trivial 规则、query 分组、长短 response 梯度一致、已有 Adam momentum 下无信号组参数不变、实际 HF generate/old score/greedy validation 成本隔离。

命令：`.venv-cpu/bin/python -m pytest -q tests/baselines/test_cosearch_fixes.py tests/baselines/test_models_and_driver.py -k 'not real_fit_driver'`，**9 passed，1 deselected**。原完整 driver 测试第一次遇到 BAPO 新 stage-validation 夹具无 val_dataloader，已通知 BAPO 修复任务；该项不能记成 CoSearch 自身通过。

直接导入固定作者 `answer_match_reward.py`（仅 stub 不参与 matcher 的 pytrec_eval），对 12 个答案匹配反例逐项比对，全部相等。记录：`evidence/cosearch-author-matcher.json`。

## 明确保留 / 尚未验收

- 用户决定所有算法统一继承 Search-R1 的生成 rolling window / 评分上下文不一致，CS-03 的该部分保留。共享空 mask 数值问题由主任务修复。
- 论文优先 continuous F1、threshold=.5、默认不按 availability 拆 semantic group、min group=3 保留；这是有意不同于作者默认二值 .8/split recipe。Search-R1 原 top3/4-action/500-observation 预算保留，Hit5=Hit3。
- CS-04 真实 top50 服务、目标规模 ranker 的长度/峰值/延迟与完整双 agent 验收仍待实测。HF CPU FP32 eager 是既有 backend 适配，六项 tiny CPU 测试不能证明 3B/7B 生产可行。
- CS-06 联合 trainer/ranker RNG、optimizer、scheduler、loader 恢复仍不支持；保存会计快照不能关闭该能力缺口。
