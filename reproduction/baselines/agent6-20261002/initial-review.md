# Agent 6 CoSearch 独立审查

审查日期：2026-10-02。对象：`work/cosearch` 的 `reproduce/cosearch@1620c2edb13809d19cc3f5af4a298ccd75488104`，冻结代码起点 `0723dca`。该范围差分仅增加计划文档，算法源码未相对起点变化。本报告没有修改被审查仓库、没有访问或启动远端训练。

**结论：代码/数学/逻辑有条件通过；正式 infra 与效果验收不通过（尚缺真实证据）。** CPU 逻辑可继续隔离验证，不能宣称 Search agentic RL 效果已经验证。历史 audit 的 CS-01/02/05/07 已被 fixes 修复，不能将历史 finding 当当前未修复缺陷；共同 rolling/scoring 上下文差异由用户决定保留。

## 来源与独立工作

- 论文：[CoSearch v2](https://arxiv.org/html/2604.17555v2)，本轮实际浏览 §3–4；公式和流程依次核对。使用 paper-glance 的串行方法框架，用户已要求审查，未重复询问菜单。
- 固定作者源码：[763bf8c](https://github.com/snap-research/CoSearch/tree/763bf8c4d7caa363ad845d39ddfd53b81ae377bd)。本轮独立下载到 `work/cosearch-author-review`，实际 `git rev-parse HEAD` 一致。
- 已读当前 README、audit/cosearch、fixes REPORT/cosearch，直接检查 cosearch.py、controller.py、ray_trainer.py、config 和作者 reward/group/tool/worker。
- 定向现有测试：算法+模型文件选择 ranker/ranking/semantic/cosearch，**7 passed,16 deselected**；`test_cosearch_fixes.py` **6 passed**。其中真实未 patch HF generate 的tiny测试仅验证生成、old-logp及验证成本隔离，不能证明合法ranking或真实50文档能力。
- 独立差分 probe：`work/cosearch-review-evidence/probe.py` 与 `probe-result.txt`。直接导入作者 matcher（仅 stub 未用于匹配的 pytrec_eval），8个冠词/标点/NFD/空答案/词边界反例逐项一致。布尔 relevance 排除另由当前回归确认。

## 一轮方法还原与一致性

1. Actor 按 Search-R1 生成 search。ranker.execute 向固定retriever请求50候选；初始问题prompt、query和所有文档进入独立CPU FP32 base ranker，没有gold。
2. HF sampling生成ranking；同温度eval-mode前向保存response old-logp。合法且恰好3个唯一整数ID返回对应文档；非法则原top3 fallback，同时记录非法event，训练奖励-.2。这是允许的运行回退，不能把fallback次数计成成功重排。
3. Actor继续观察500-token文档窗口直到答案/四轮预算。整轨迹合法时主奖励ordinary token F1，否则-.2；主actor按受控Search-R1训练规则更新。
4. after_actor用trajectory UID映射主奖励/gold/合法性。非法主轨迹在ranker监督前剔除；gold仅用于离线NFD精确token连续匹配，布尔全trivial答案不产生文档伪监督。
5. 同一原题内query token F1≥.8代表式greedy分组，组最小3；同奖组完全跳过。有效组用奖励标准化优势、token PPO clipped ratio、序列token mean后序列等权，独立AdamW更新。
6. Validation decorator切换ranker greedy且不记录训练events；Adapter继续用当前trained ranker。保存actor另存ranker HF/tokenizer/optimizer/cost snapshot，明确无joint exact resume。

论文先描述相似query分组、连续F1与γ=.5；作者recipe实际按availability再分query组，main F1阈值.8二值化、condition .8，worker保留至少4条并有duplicate reward重分配。本地采用论文优先的min3/continuous/.5/no availability split，是应披露的有意差异。论文Eq.3显示逐序列token求和，本地改成序列token mean，属于共同目标归一化适配，不能称逐字同公式。作者recipe两agent有reference KL；本地主actor保留共同KL，ranker不含KL，同样不等价作者完整recipe。

主3B base/辅助同3B base、50→3、5轨迹/4轮、500 observation、256 rank tokens与论文Instruct/7B ranker/50→5/8轨迹/6轮不同。K3导致Hit5=Hit3；奖励可取0,1/3,2/3,1。计划已授权这些受控适配，但不能用论文绝对分数直接验收本地结果。

## 需要关闭的严重验收缺口

### R-CS-01 / P1：warm-start未证明可产生学习信号

3B base生成合法ranking的概率没有真实证据。若多数非法，fallback维持检索表面可运行，但-.2同奖组被全部跳过，可能成为只训练actor的固定top3系统。需要真实模型+50候选逐query记录合法/非法率、fallback、group size、奖励差异、有效组数和非零adv事件；至少2个真实outer update证明ranker有实际有效RL更新。不能手构events或patch generate代替。

当前 `test_real_ranker_parameters_update_and_invalid_main_filter` 三条prompt/docs/response完全相同，只main reward不同，因此组优势理论上求和0、policy梯度抵消。默认AdamW衰减及浮点残差也会造成参数变化，旧测试“参数变了”不足以证明RL信号。独立probe最终记录在weight_decay=0的相同response仍测到grad L2约1.64e-7、parameter delta L1=.0001673，说明浮点残差经Adam也可被放大。三条不同ranking对照得到grad L2=.3949、parameter delta L1=.0080897，验证受控有效信号路径；仍为tiny+patched生成，不能升级成生产证明。正式应记录非零grad norm与参数delta，并用weight_decay=0诊断或明确剥离衰减/旧动量效果，避免仅凭optimizer_steps放行。

### R-CS-02 / P1：50候选输入长度和服务契约未实测

`cosearch.py:140–142`只检查输入token+256是否超过max_position_embeddings，明确raise且未截断；符合“不能偷偷缩减50候选”。但 `execute:183–185`仅核对返回batch行数，不核对每行确有50个文档；rank:133可静默keep=min(3,len docs)。错误服务返top3甚至0候选会改变任务而不报错。应验证真实服务每行50候选、ID/schema/排序，与固定retriever对照一致，缺候选明确拒绝或在报告注明异常。

实际question+query+50完整文档token分布、最大值、超过模型上限比例均未测。若溢出，应按计划请用户选择passage截断或作者ranker方案；不能擅自去掉候选、固定排序或加入gold。仅“GPU有80GB/机器2TB RAM”不能解决上下文限制。

### R-CS-03 / P1：效果指标未实现完整输出

当前 `main_ppo.py:25–28` validation RewardManager选择generated EM；`ray_trainer._validate`聚合该分数，当前final JSON也只有该路径指标。CoSearch训练F1不等于validation F1。计划要求按七集合报告F1/EM/格式率及成本，需要保存逐题可复算输出或新增一致的评估指标，且训练/holdout/最终测试严格分离。没有新GRPO公平对照与51,713题最终评估，不能回答CoSearch是否提升效果。

### R-CS-04 / P1：CPU ranker/RAM/吞吐仍未知

默认FP32 eager HF ranker在driver逐query串行生成，随后全序列score前向，训练再全序列forward/backward。3B参数+梯度+Adam双moment的纯权重状态下界约48GB（十进制估算），不含模型加载峰值/临时权重/激活/KV/logits/retriever/Ray/actor offload。prompt L、vocab V时完整logits约4LV bytes；eager attention还有L平方项，反向活跃激活会更大。真正获配cgroup内存而非整机RAM必须核查。

需测CPU时间/RSS峰值、首步Adam分配、50文档prefill/生成/score/backward分时，8CPU预算下吞吐；记录retriever RAM/线程和HTTP60s超时。256 rank token是固定配置，不应因慢悄悄减少。所有训练卡资源双阈值、Slurm allocation、UUID租约和GPU通信必须由共同门禁验证；审查没有替代预约授权。

## 其他重要边界

- P2 validation generation与score token成本已经分开，但验证的非法rank/长度/latency没有events诊断；cost累计只在下一次ranker.update发metrics，最后一次final validation后没有自动输出最终ranker成本快照。必须捕获final totals，否则51,713题ranker成本会遗漏。
- P2 checkpoint分别保存，不支持joint exact resume，且state缺完整RNG/dataloader/scheduler。要求5步save后加载actor+ranker做相同输入推理/权重hash检查；它是权重恢复验收，不能叫精确续训。final checkpoint在最终验证前保存，故ranker_state.json不包含最终验证成本，需要单独最终会计产物。
- 保留上下文错位及decode/reencode属于用户共同决定，不新增修复要求；数学报告必须承认不严格on-policy。共享空mask/数值安全修复已存在，本审查不重开历史CS-03。

## 放行的真实证据要求

按计划先公共环境/检索与资产hash、split身份通过，然后目标3B真实50候选长度/格式探测；合法ranking+真实奖励差异+非零梯度/参数变化的2-update联合smoke；5-update validation/save/reload验收及CPU/RAM/GPU峰值；再500实际outer update并与新GRPO按相同协议比较，最终七集F1/EM/格式/成本完整记录。无法取得资源时保留交接和具体阻塞；不能把CPU测试通过写成真实效果通过。
