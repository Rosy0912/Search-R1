# OPSD 无裁剪候选修复与实际证据

用户2026-10-04最新授权：取消自动发送，先修问题，找出无裁剪情况。opsd heartbeat已通过原生automation工具设PAUSED并读取本地配置确认，不恢复。此授权执行软件修复及已有真实数据CPU复核；没有新训练或GPU作业。

## 有界修复

将generalized_jsd默认component_clip由0.05改为None，基线Hydra配置对应null。当前OPSD beta=0，得到完整teacher→student forward KL；保留显式component_clip=.05的历史配对开关。该改变属于本次Search适配整改，不能冒充作者原实验配方。teacher user-context候选、student历史/观测loss mask/全局token分母/微批累积/参数冻结/分块checkpoint路径没有改动。actor.grad_clip=1.0与PPO clip_ratio=.2保留，删除的是词表正分量上限，不是梯度范数保护。

配置是共享distillation区块；TOP-D使用其自己的逐token teacher logprob奖励/优势路径，并未使用OPSD generalized_jsd作为更新目标。显式旧启动快照及冻结8e不改、不部署；未来使用旧快照或显式.05会继续旧目标，不应宣称已使用此候选。

## 实际无裁剪情况

使用3146已保存的64有序训练初始状态×4条件全部256真实分布，不新增采样/模型/GPU。原始manifest3618cc9e2ccefe5be052b2b2ee13adaac25734f1895d0879011518c64ba11002保持不变，历史source4ebe/probebe保持不变；本修复函数显式None路径与该真实重放目标一致。

| 学生/教师提示 | 无裁剪KL下降/零/上升 | 平均真KL变化 |
|---|---:|---:|
| base/legacy |64/0/0|−0.0044715498|
| base/user_context |64/0/0|−0.0015920406|
| HF5/legacy |64/0/0|−0.0044628568|
| HF5/user_context |64/0/0|−0.0015899903|

上述是0.01归一化logit空间小步。独立原生Torch2.4 CPU解析公式q.sum()*p−q（不调用候选loss/autograd/模型）与原生autograd符号一致，最大KL变化浮点差1.43e−6。新user提示原clip=.05在base/HF5分别30/64、34/64上升；无裁剪在这批状态消除该反向现象。

这不是实际Adam参数更新、完整搜索轨迹或HF80；不能据此宣称reward改善或解决所有教师监督语义风险。完整直接首答案候选55/64弱化不等于思考/搜索后准确率下降，仍须后续真实轨迹验收，不能用KL下降替代。

## CPU验收

根任务61项相关测试通过：新增标准teacher KL/解析梯度/T=1.1/teacher无梯度、真实Hydra继承None配置+grad保护、历史显式.05保留和反例；既有提示对齐、模型mask、micro日志累积及checkpoint等回归。首次配置测试直接load子YAML未compose父配置而失败，已修为真实Hydra compose，不放宽保护值；最终stdout opsd-unclipped-fix-root-cpu.txt保留。独立actual无裁剪报告OPSD_UNCLIPPED_ACTUAL_REVIEW.md、逐行job3146-independent-unclipped-cpu.json；配置/入口独立审核结果见OPSD_UNCLIPPED_FIX_REVIEW.md（实际完成后记录）。

本候选尚未部署/训练；3007保持CANCELLED80轮，仅HF5，无HF80/HF500。没有新GPU成本。后续获明确授权才受控fresh5、阶段64题及行为退化停止门禁；所有GPU仍须同一H100，无A800或跨集群RPC。
