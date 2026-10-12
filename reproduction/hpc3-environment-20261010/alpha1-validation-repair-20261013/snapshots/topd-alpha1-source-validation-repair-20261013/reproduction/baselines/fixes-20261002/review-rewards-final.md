# BAPO / CoSearch 最终整合增量复审

2026-10-02；独立复审当前工作树，不修改实现，仅新增针对性回归测试。依据前轮 audit、`review-bapo.md`、`review-cosearch.md` 与本轮实际读取的实现。结论只覆盖以下修复整合；不是完整3B / vLLM生产验收。

## 结论

**已关闭前两个复审报告指出的 UID 缺列绕过、未更新 actor 仍推进 BAPO stage、ranker override 校验及目标式注释问题。未发现本轮奖励、分组、stage hook、ranker sequence normalization 新的 P1 / P2 缺陷。** 保留前轮报告明确的算法适配与未验收事项。

## 增量核证

1. `holdout.split_frames` 明确要求训练数据有 `extra_info.index`，缺列、缺 index、None 全部在划分阶段报错；不再继承 RLHFDataset 的默认 index=0 静默合并。已有不同问题同UID碰撞也拒绝；重复相同问题按 prompt 分组到同侧。原训练文件不被覆盖。
2. Ray 实际 `fit()` 的 stage hook 仅在当次 `actor/optimizer_steps > 0` 后触发；metrics 每 iteration 重建。新增真实 fit 回归分别模拟 critic warmup、actor 全跳过、正常更新，把 stage 设置为距离 plateau 转换只差一次观察，前两者保持 best / stalled / early，正常更新才完成转换。最终评估关闭不阻止独立 stage 验证，且独立 loader / reward 临时替换后恢复。
3. BAPO 双侧 yes / no / noanswer 类别精确匹配、空重叠 F1=0 与作者 scorer 修复保持一致；CoSearch `main_f1_reward` 继续用 ordinary F1，不误套类别规则。bonus / 全错重采样整组行为未因整合改变。
4. CoSearch 精确 NFD 文档 token matcher、trivial-answer exclusion、query 分组范围保持前轮已核对版本。ranker 过滤非法主轨迹，再在完整组计算 advantage；同奖组整体不进入 optimizer，含已有 AdamW momentum 的测试仍证明参数不漂移。informative group 的零 advantage 成员保留序列分母，避免改变期望测度。
5. 每个 ranker response 内 token 均值，外层每 minibatch 除实际 sequence 数；长短 response 不改变每序列权重。不同 minibatch 多次 AdamW 不声称等价于单次总batch更新。目标注释已改为 Eq.3，非误称 semantic grouping 的 Eq.5。
6. ranker batch_size / epochs / max_new_tokens 正数、clip_ratio 范围、grad_clip 正数及 similarity 范围在模型加载前检查；新增六项无效override回归全通过。泛型 tokenizer 排除 token_type_ids，真实 HF generate / old-logprob 重算及 validation 成本独立测试保持通过。

## 实际验证

新增 `tests/baselines/test_rewards_final_review.py` 共12个参数化场景：3个UID缺失、3个真实fit阶段hook、6个ranker非法override。

```
.venv-cpu/bin/python -m pytest -q \
  tests/baselines/test_rewards_final_review.py \
  tests/baselines/test_bapo_fixes.py \
  tests/baselines/test_cosearch_fixes.py \
  tests/baselines/test_models_and_driver.py
27 passed, 4 warnings in 3.39s
```

warnings 为现有 LibreSSL / Ray 弃用及 greedy top_k 提示，没有断言失败。本轮没有启用远端GPU、改动原agent checkout、占用现有训练或提交git。

## 仍保留的边界

用户明确保留原 Search-R1 滚动上下文 / 评分差异，全算法采用同一规则；不是遗漏修复。2%共用holdout按独立prompt划分，重复问题可能使行数比例略异；只能保证精确序列化prompt不重叠，不声称排除语义近重复。CoSearch top3、连续主F1、threshold=.5 / availability 不额外分组仍是文档披露的原论文优先及固定搜索预算适配。

真实169615题数据划分产物、目标3B / 教师 / ranker、50候选检索服务、显存/延迟及完整双agent checkpoint恢复尚无本轮验收证据；joint resume 明确不支持。共享actor数值 / 分布式归一化由另一复审覆盖，本报告不越权宣称生产链条全部通过。
