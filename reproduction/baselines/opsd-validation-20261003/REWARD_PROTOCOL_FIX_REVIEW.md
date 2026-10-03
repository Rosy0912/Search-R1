# OPSD 最终回答协议修复独立审查

审查对象：本地 reproduce/opsd 的 8e60976d364ab7d7ae51714415813c1359dafe9a → 5f516208a0b0daed88c27c4800c7bcfcf908f043。实际 formal3007 的训练源码仍固定 8e；本审查未访问 GPU、修改远端源码、取消或提交任何作业。

## 结论

纯 OPSD 不需要因为这三项评分修复重新训练。旧源码产生的 EM/F1、answer_format、format_rate 不能直接作为修复后协议的效果结论；必须保留原始结果，并另外对保存的实际逐题轨迹重新评分，或用同一个训练 checkpoint 在固定修复源码下重新评估。这个结论针对当前纯 OPSD 路径，不适用于 reward 驱动的 GRPO、BAPO 等训练。

这并不证明漏洞曾在实际3007中被利用，也不证明实际效果变化的大小。仅靠聚合日志无法确定这些事实。

## 补丁与训练依赖

三个本地等价提交为 a59dbe4c62559416e95c1b54d1f87771e9810aeb（核心协议/EM）、63abd74e768370eafd60aca51b175173bee16d6b（共用 F1/动作合法性）、5f516208a0b0daed88c27c4800c7bcfcf908f043（可选格式奖励模型来源掩码）。

git diff 的生产文件只有 qa_protocol.py、bapo.py、qa_em.py、qa_em_format.py、main_ppo_format.py。actor.py、distillation.py、controller.py、rollout.py、llm_agent/generation.py 与固定训练8e逐字节相同。

main_ppo.py 的 RewardManager 调用 compute_score_generated_em，所以一般训练 reward 数值会变化；ray_trainer.py 将其记录到 token_level_scores/rewards。然而 controller.compute_advantages 对 method=opsd 明确将 advantages 和 returns 置零。DistillationActor 不调用 PPO reward loss：它直接使用 loss_mask、模型输出、私有训练 GT 的 teacher_inputs、全词表散度、固定 token denominator 和优化器。它不消费 EM、F1、advantages 或 returns。Controller.after_actor 对 OPSD 不进行根据分数的 ranker/stage 决策；stage_validation 对 OPSD 立即返回空字典；保存频率由固定步数决定。训练采样器/执行器也不调用本次修改的答案抽取函数。因此补丁没有改变当前 OPSD 的训练轨迹选择、训练目标或 checkpoint 选择。

## 旧指标与新指标的差异

对三条实际 CPU 反例分别比较 8e 和修复源码：
- 最终模型片段为 search 内嵌 answer Paris：旧 EM=1，新 EM=0。
- 较早片段 answer Paris，最后片段 search next：旧 EM=1，新 EM=0。
- think 内嵌 answer Paris：旧 EM=1，新 EM=0。

前两条的共用旧 F1=1，新 F1=0。修复保留合法独立终态 answer 的正常分数，拒绝嵌套、重复、多动作、未闭合、观察两侧拼接、早先回答沿用和尾部非 EOS 内容。模型来源 mask 继续区分 prompt、observation 与 policy token。

evaluation.batch_answer_metrics 使用 bapo.answer_from_spans 和 legal_action_records；因此周期与最终 qa EM/F1、answer_format、format_rate 受本次补丁影响。val/test_score 使用修复前后不同的 EM extractor，同样受影响。生成 token 与 search_calls 计数代码没有被补丁修改。

## 最低成本、可审计的纠正方案

1. 不热改正在运行的3007、不重复提交500轮训练、不把旧2999的九轮拼接。保留8e训练来源、原始64题阶段日志、实际终态、500次更新和 HF500 内容校验。基础设施/更新门禁与修复后效果门禁分别记录。
2. 优先检查是否存在完整逐题响应 token 或准确分离的 model_spans、info_mask、目标答案、source/question identity。若全部齐全，可 CPU 重计分，保留每条旧/新结果与聚合变化、文件 SHA，避免再次生成；不得从少数展示题推算全部。
3. 静态审查所见 _validate 仅收集内存 qa_rows、聚合并打印 JSON，RewardManager num_examine=1 仅展示每 batch 每 source 少数 model_spans。默认路径没有将全量逐题轨迹写入文件；runtime_audit 也不是 validation transcript 保存器。因此若没有外部完整轨迹文件，聚合日志不可离线还原修复后全量结果。
4. 若需重生成，等待训练/旧最终验证正常退出后，在新固定提交的独立副本执行 eval-only：先核验 HF500 文件全量 SHA、load info、真实 source/config/retrieval identity；只加载保存的同一个500轮模型，不做优化器更新；保持同节点 GPU 编码/FAISS、无 A800/RPC、同7集51713分母、确定性生成和原先工具预算。评估代码 provenance 应写为修复提交，训练 provenance 仍写8e，不可把两者合为一个来源或覆盖旧 receipt。
5. 若报告声称训练提升，应对比较对象（至少 base 与 HF500）使用同一修复协议/数据/预算；旧 base 分数也受漏洞影响，不能与新 HF500 分数直接相减。即便生成设置相同，重新评估仍是新实际采样证据，不能假称原先 trajectory 的精确重评分。
6. 原来每50轮的64题评估可按旧协议保留为历史。只有存在原始逐题轨迹，才能重新评分所有历史阶段；未保存50、150等 checkpoint 时，不能从100等 checkpoint 重造相同历史阶段。不得把另一次生成的64题分数冒充原始阶段分数。
7. HF500 blocked 预览原有门禁只绑定旧 source/旧 final 聚合。新增实际修复后评估证据需要先形成明确的独立补充门禁/receipt，再做 finalization；不能填假未来 job ID，不能仅依据旧 exit0 或目录存在宣称效果已验收。

## 本次 CPU 验证及限制

python3 -m pytest -q tests/reproduction/test_terminal_answer_protocol.py tests/reproduction/test_reward_and_validation.py：12 passed in 0.56s。

注意：本机缺少完整导入依赖，test_reward_and_validation 实际走测试自带的 AST RewardManager + 最小 DataProto fallback，已读取 RewardManager module=builtins、DataProto module=review_reward_tests 确认。它检验真实评分方法源码与来源掩码，但不等于远端真实 verl.DataProto/完整 Ray 环境验证。根任务应在原生 CPU 环境追加 SEARCH_R1_REQUIRE_REAL_DATAPROTO=1 的测试，不占训练 GPU。

另外独立动态提取 controller.compute_advantages：分别注入全0、全1、全-123 reward，OPSD advantages/returns 均逐元素全0；直接执行8e/修复 EM 三反例与 F1 两反例均符合上述旧1→新0；五个训练/执行源码文件逐字节相同。未进行额外 GPU 实验。

最终结论：保留当前纯 OPSD 训练，单独补足修复后的评分验证；拒绝把旧协议聚合分数包装成修复后效果。
