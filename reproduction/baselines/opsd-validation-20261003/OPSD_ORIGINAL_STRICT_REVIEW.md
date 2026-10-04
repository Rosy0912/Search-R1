# OPSD 原文与固定作者代码严格审查

审查日期：2026-10-04。生产对象只读固定 `8e60976d364ab7d7ae51714415813c1359dafe9a`（git show），作者对象固定 `ae7d2519e94920c4eb6206c0c26de46d9c50abae`；论文版本 v3。3007 已由 root 按用户要求停训；本审查不启动 GPU、不改生产、不提交 Git。

**结论：冻结初始教师和 .05 词表分量裁剪均有原文实验/作者入口直接依据，不能判为未复现原文的错误。真正明确的适配差异是 teacher 提示的角色/转场、仅答案而非参考解题轨迹、Qwen2.5 搜索环境、全参数更新、采样后连续10个mini更新以及温度/优化超参数。已有执行与等价验收不证明这些适配具有行为改善效果。裁剪存在确定的反教师梯度输入，但实际退化的主因仍未证明。**

## 1. 原论文到底要求动态、EMA还是固定教师？

[论文 §3.2 与 §4.1](https://arxiv.org/html/2601.18734v3)：方法叙述是同参数模型不同条件；实验明确固定 initial policy 并采用 LoRA。不能只读共享参数定义就要求在线教师。教师 stop-gradient 和整个实验冻结教师也不是同一概念。EMA不是本文主实验必需项。

固定作者 [opsd_trainer.py:128–216](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/opsd_trainer.py#L128) 三种路径：默认动态；fixed_teacher禁用LoRA adapter；use_ema_teacher暂换EMA，并在optimizer回调更新，二者互斥。compute_loss:678–695在no_grad下执行选定教师。作者主脚本 [run_opsd_4b.sh:27–36](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/scripts/run_opsd_4b.sh#L27) 明确use_peft+fixed_teacher。

本地 FrozenTeacher 是独立初始权重，不进optimizer且无EMA，**对应实验固定教师策略**。区别在作者共享base+LoRA，而本地独立FP32/eager base teacher+全参数BF16学生；初始target相同不代表学生更新约束相同。教师移到GPU/CPU储存的性能修复不改变这个策略。没有证据支持将当前退化归咎于“忘记在线/EMA更新”。

## 2. teacher hint：确实存在结构差异

作者 [data_collator.py:95–109](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/data_collator.py#L95)：把题目、solution、独立重新推理指令作为一个user message，再apply_chat_template、add_generation_prompt及teacher_thinking。transition定义在37–43。reason_first默认False（opsd_train.py:57–61）；原主路径不用额外教师生成，不能要求必须先生成rationale。

本地 distillation.py:41–50 去除prompt padding后，直接追加raw accepted-answers列表注释，再接相同学生response。没有重新建立user角色，也没有作者的独立重新解题transition。若原prompt已结束于assistant生成开头，注释便占据assistant续写前缀，语义不等同于user提供参考信息。**这是已批准answer-only适配之外的明确模板结构差异**：answer-only可以保留，user角色/转场仍可单独忠实构造；但如何在Search-R1工具协议保持动作历史边界必须设计和实际测验，不能直接照搬数学boxed模板。

只提供答案是PLAN授权，不是擅改；没有gold进student的证据，也没有next-token普通错位证据。缺少真实teacher格式/动作/答案分布诊断，不能从这项差异直接断言其造成退化。

## 3. clip默认：必须修正历史表述

作者 [opsd_train.py:71–76](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/opsd_train.py#L71) **实际CLI默认.05**；Trainer函数参数None只表示底层可选接口默认，不是训练入口默认。主1.7B/4B脚本.05，8B脚本.06；4B非思考消融1e-6。此前行为审查只强调函数None、不说明CLI.05是不完整表述，本报告明确纠正。

论文§3.2明确词表逐分量上界min后求和；作者generalized_jsd_loss:442–464同样clamp再求和。本地:18/27/28匹配该FKL路径，PLAN35也指定.05。不是把原作者“总KL裁剪”错误实现成分量裁剪。

[原始CPU数学证据](OPSD_BEHAVIOR_AUDIT_EVIDENCE.json)：teacher(.9,.1)、student(.1,.9)时，clipped loss=-.16972245；logits梯度(+.01,-.01)使teacher偏好token概率下降，真实KL增加。此反例是**原机制本身的数学风险**，不是本地独有bug。非负f-divergence性质在这种有负分量的上界截断后不成立。论文经验收益不是任意分布的优化方向保证；同样，toy也不是实际崩溃的因果证据。真实clip激活率、unclipped KL、梯度方向比较都未记录。

## 4. 采样与更新分布

固定作者training_step:1288–1441在每个训练microbatch生成学生轨迹后调用SFTTrainer backward；主4B脚本8×4/accum1，一次optimizer对应当前batch32新轨迹。8B/1.7B有梯度累积；不能泛称作者每micro立即optimizer。vLLM还存在权重同步调度，严格同步时点须按实际版本审计，不能只凭方法on-policy名称假定零滞后。

本地actor:42–62对一次outer已有2560轨迹依次10个local mini32更新，micro2只是累积；不是同一mini重复10遍。后9个mini的条件历史来自outer开始策略，学生已改变，生成/scoring state distribution更陈旧。其没有importance ratio不是普通PPO缺失bug：full-vocab条件蒸馏直接求和，作者主路径也没有PPO surrogate。加ratio会变目标，不能作为无证明修复。

两者均只对学生分支反传，teacher给定相同学生历史；本地环境observation参与context但不进loss，是工具环境必要适配。原数学completion没有工具观察，故不能说已经复现原任务分布。非法action tokens仍受纯蒸馏监督，无显式format/outcome reward目标。

## 5. 原目标与本地适配边界

[论文实验与附录B](https://arxiv.org/html/2601.18734v3)：Qwen3 instruct/math参考推理、LoRA64/alpha128、batch32、单轨迹、1024长度、100步、LR5e-6、sampling T1.1。其能力假设及局限不保证answer-only搜索迁移有效。

本地Qwen2.5-3B、answer-only、512×5搜索轨迹、全参数、mini256、500outer且10Adam/outer、T1/LR1e-6/gradclip1/warmup142，均不能称原论文训练配置复现。作者同一temperature参与generation和loss；本地T1与其1.1不同，会改变软目标而非只影响采样。teacher FP32/eager与作者BF16模型目标也不是相同数值；之前devicegate保留BF16严格迁移失败并采用precision-reference，只证明设备迁移可解释，不能借此宣称作者BF16目标字节等价。

全参数更强漂移、裸hint/缺转场、仅答案教师能力、10更新state滞后与裁剪均是合理候选解释；目前缺少单变量对照和教师分布诊断，**没有资格排序为确定退化主因**。更换为EMA、取消clip、加入reward也不能名为“修原文bug”，应为新实验设计。

## 6. 当前证据支持什么

3007末快照实际77outer，finish/valid-action约.05，grad_norm非零。评分旧协议缺陷限制EM/reward解释；解析完成率退化本身是真实行为。历史5outer/all8rank/独立HF及优化等价通过证明运行与保存，不证明搜索目标有效，更不构成500步成功。loss微批均值日志有约16倍量纲缩放（行为审查§2），负clipped loss也不是“真正KL下降”。

历史独审针对工程实现和作者算子一致性覆盖较强，针对迁移后的教师教学语义、目标的优化性质和任务行为改善覆盖不足。本次补齐原文依据并纠正“必须在线EMA”“.05偏离作者默认”两类可能误判；不能用通过作者probe免除这些目标风险。

后续应先固定Search-R1协议、以明确user角色/重新作答transition构建answer-only教师，并只读采集teacher合法动作偏好、真实clipping比例/unclipped KL；再事前定义单变量对照。任何新训练需新冻结源/数值容量与fresh5/HF及一致评分验收，不能追认已停3007为原文成功复现。本报告只给审核结论，不实施上述改变。

## 可复核来源身份

本轮从固定GitHub commit读取trainer/data_collator/opsd_train及四类脚本；保存的author_opsd_trainer.py与远端逐字节相同，SHA256 aa11fc2a3f3cd814da37db38c6fb2351cab7f98a5a572d808a9b75de16097c9d。data_collator SHA256 5b96f3b3ae2f04e2b9dbf03509d3cd723c1b637d9362ab4697135d9f885b8381；opsd_train SHA256 524edd3f9a5ee9ad1496410dd9c51019301e8e1ab72289f5982b34adaf81f26e。原论文与源码均以primary source当场核对；所有差异判断使用固定生产commit，不混入e3f评估分支。
