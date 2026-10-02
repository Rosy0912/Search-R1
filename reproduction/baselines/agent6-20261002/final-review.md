# Agent 6 提交后独立复审

对象：`work/cosearch@9c3d7ccd37e67b71ed8b225eba451065f33ab480`，追加复审 `c5d9c83` probe修复。共同部署source `297e1c1` / cherry-pick `cd86c9d`采用owner已独立冻结的审查，不在此重复作全面门禁审查。没有修改被审查repo或启动训练。

**最终结论：ranker核心诊断/strict50补丁通过代码与CPU复审；probe prompt问题在c5d9c83已修复，可以继续真实能力探测。真实模型/检索/联合GPU/效果仍未放行。** 全部来源和方法偏离沿用初审 `work/review-cosearch.md`，不把CPU条件通过写成效果通过。

## 关闭或改善的初审问题

1. `execute`现在逐行严格要求50个candidate且document.contents为str，缩成top3、49条、非法内容会在rank前拒绝。成功取50但内容重复是retriever品质问题，未要求额外去重改变任务。
2. rank成本分别记录训练/验证合法ranking、fallback、prompt最大token、生成/score token、CPU与wall time；update记录CPU/wall、informative group、distinct rankings、preclip gradient norm和每参数最多8点delta。未改变数学/奖励/温度/optimizer默认weight_decay=.01。
3. optimizer回归改为不同ranking、weight_decay=0、gradient norm>1e-5，修复初审“相同序列仅靠参数变化不足以证明RL信号”的证据问题。该测试仍patch generate；它证明真实tiny HF forward/backward/AdamW的有效信号路径，不证明base能自然生成合法ranking。
4. validation decorator在函数返回后添加cost_metrics并打印JSON，最终评估能捕获全部累计ranker成本，关闭“最后一次验证成本仅留在内存”的缺口。保存于final evaluation之前的checkpoint cost snapshot仍不含最终validation，最终JSON是补充会计来源。

## 独立实际验证

运行 `test_cosearch_fixes.py` 与 `test_models_and_driver.py -k 'not real_fit_driver'`：**10 passed,1 deselected**。覆盖strict50/schema fail-fast、同奖组Adam状态保持、真实未patch tiny generate、验证训练计数隔离、修改后的有效ranker梯度/参数probe等。没有重新运行完整140项，不把owner结果冒称独立执行。

本地按README脚本路径 `python reproduction/baselines/cosearch_probe.py --help` 在未editable安装该checkout的旧CPU环境报 `ModuleNotFoundError: search_r1`；按module运行成功。检查bootstrap第62行已 `-e repo`安装，故已按计划生产环境此项有明确解决路径，不列生产P1。

## 已修复：R-CS-F01 / 原P1 probe与生产初始问题上下文不一致

`cosearch_probe.py:44`无条件调用tokenizer.apply_chat_template；实际 `RLHFDataset.__getitem__:128–131`只在tokenizer.chat_template存在时使用，否则直接chat[0]['content']。缺template的base tokenizer会在probe失败，而训练dataset可运行；即使template存在，probe没有复现dataset tokenization/truncation和SearchRollout max_start_length2048→decode的初始ranker question context。长问题可导致probe测得与训练不同的50文档prompt长度和溢出结论。

建议用与dataset一致的template fallback和tokenize/postprocess参数，再取原max_start2048窗口、skip_special_tokens decode，最后传ranker；最好复用一个不含gold的公共helper以防prompt字符串漂移。输入query=原问题是允许的standalone检索能力探测，但不代表actor实际生成subquery的分布；报告必须保留该限制。不能为“修probe”截断50 passages。

追加实际核验c5d9c83：现在template存在才apply，否则chat[0].content；add_special_tokens=False与dataset一致；超过4096拒绝；取最后2048 token后skip_special_tokens decode，与SearchRollout.initial_states的unpadded prompt一致。未截断50候选。sys.path引导可靠，独立再次运行README脚本`--help`成功；`--questions 0`实际exit2并在读取模型/数据之前拒绝。Darwin RSS换算也已修正。这些关闭该P1及非正值/单位问题。

## 诊断实现边界与剩余项

- CPU参数抽样每个有梯度tensor最多8点，保存量随tensor数而非3B元素数增长；当前HF Qwen参数连续存储，reshape是view，不复制全权重。非连续参数reshape可能分配副本，泛型HF扩展须注意；当前固定Qwen没有已证实RAM风险。delta=0只表示抽样点未变化，不能推出所有参数不更新；delta>0也可能含decay/momentum/roundoff。grad norm与distinct-ranking指标能改善证据但不自动证明生产学习质量。
- gradient_norm_max是整个update各optimizer batch的preclip最大值，不是累计norm或平均；parameter_probe_delta_l1是各step抽样绝对差之和，不是完整权重L1且非最终与起点net delta。报告字段解释应保持这些语义。
- rank time包括prompt tokenization、generate、score及parse，不包括HTTP retrieve或完整actor rollout；update time包含训练梯度/optimizer及diagnostics，CPU process_time累加本进程线程时间，不能和wall直接等同。峰值RSS为driver全生命周期，不是ranker独占或全机job峰值；retriever进程/Ray worker/GPU峰值须额外采集。
- `ranker.cost_metrics`与修复后的probe.save RSS都正确区分Darwin/Linux。
- probe --questions/--repeats/--threads已做正值校验，但空train文件仍可能complete却无生成，建议reject empty data。默认3题×3重复有效，但小样本只反映该样本，并不能放行500-update。
- probe retrieval/schema、model load或generate异常时没有保存明确failure status；部分已生成报告仍可能停在in_progress或根本没有文件。建议try/finally写失败阶段/exception（不含凭据），保留已有证据。contexts overflow路径已写状态并exit2，不截断候选，符合用户约束。
- 共同cd86c9d已提供batch_answer_metrics和aggregate_answer_metrics；实际检查_validate的两条生成路径都调用collect_qa，并在返回metrics及JSON添加F1/EM/answer_format/format_rate/token/search_calls。因此初审R-CS-03实现缺口已关闭（此前增量复审只聚焦RewardManager而漏看独立QA metrics，该判断已纠正）。指标真实七集运行仍待验证。checkpoint未提供joint optimizer/RNG/loader exact resume。

## 后续真实验收门槛

修正probe上下文后跑冻结3B base+真实50候选合法率/长度/CPU/RSS能力；按共同Slurm/UUID门禁通过8卡基础验收；实际actor→retriever→ranker→双更新的2 outer update，记录有效奖励差异、实质梯度及参数变化而非仅events；5步save/load；再500实际更新与新GRPO公平比较、最终51,713题F1/EM/format及累计成本。真实失败若需要换ranker/截断passage/改预算，仍按计划由用户选方案。
