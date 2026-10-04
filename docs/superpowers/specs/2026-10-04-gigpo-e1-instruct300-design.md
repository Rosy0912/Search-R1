# GiGPO Search-E1 Instruct300：配置与兼容性检查

实验名称：gigpo-e1-instruct300-20261004。目标是在h100-8由官方Instruct初始化，完成300次真实outer更新，首5计入；第6取数前同进程独立验收，最终51713题。该实验不是旧2992/3130续训。旧源码、环境、run、HF、状态、日志及最终报告保持原位，新目录拟为/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct300-20261004。

## 已授权配置与实际差异

| 项目 | 新实验配置 | 与3130的差异／落实检查 |
|---|---|---|
| actor/reference | Qwen/Qwen2.5-3B-Instruct，aa8e72537993ba99e69dfaafa59ed015b17504d1，同一官方权重；reference冻结 | 3130 actor原2992 HF200；固定reference旧Base，不能复用 |
| 数据 | 共同manifest SHA 0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24 | 实际读取原件及其产物，不重新划分；169028训练、512验证（NQ/Hotpot各256）、64阶段、51713最终 |
| outer | 512问题×5响应=2560轨迹 | 保持真实响应group，不将vLLM n设5；n=1、TP1 |
| optimizer预算 | global mini128，epochs1，20step/outer | 旧mini256→10step；实测20，不把128当micro |
| actor micro | global16／rank2 | 旧global64／rank8；全局mini128每rank16，8micro/mini |
| old/ref micro | global32／rank4 | 均明确rank切分，真实路径验收 |
| LR/scheduler | 1e-6，57outer线性warmup后constant，每outer推进一次 | 不沿用142，也不每个optimizer step推进 |
| 正则 | entropy0，KL.001/low_var，clip.2，gradclip1 | 明确PG/KL相同sequence_mean，梯度范数使用已验证FP64/FSDP路径 |
| 长度／交互 | prompt4096/start2048/action500/obs500，maxturn4 | 原instruction与官方chat template，不增加提示 |
| rollout | temperature1/top_p1/top_k关闭，TP1/n1 | 5响应组显式audit |
| 检索 | 同训练节点GPU E5和完整8GPU FAISS，同Wiki2018/top3/127.0.0.1:18081 | 新共同原件比较；不访问A800，不跨节点在线RPC |
| 初始化与周期验证 | 同512题，初始化及50/100/150/200/250/300 | stage64产物保留独立身份，不能冒充512验证集 |
| 保存 | 5、25、50…300：完整HF+8rank model/Adam/scheduler/RNG+数据进度+GiGPO状态 | 5先重载验收，再同作业继续；无301 |
| 最终评估 | checkpoint300，51713题 | 独立结果逐集合计数；不以512或64替代 |

## GiGPO损失归约兼容性：必须显式区分

已验收60a472b的search_r1/baselines/objectives.py:attach_policy_weights(gigpo)是轨迹内每个动作等权：动作a有L_a个有效token、轨迹K动作时，每token权重1/(K L_a)。weighted_policy_loss使用每个动作所有有效token log-ratio之和形成严格joint ratio，并要求同动作优势常数。

用户明确要求的sequence_mean则是：轨迹i有T_i个有效动作token，权重mask/T_i；PG每个有效token的GiGPO joint-ratio PPO项与广播优势乘积先轨迹平均，然后在完整global mini128轨迹上平均。KL也按同mask/T_i归约。两者在动作长度不等时不同；2动作长度1与3、PPO项2与4，原action_mean为3，sequence_mean为3.5。不能声称损失和旧目标完全等价，不能用缩放优势的隐式补偿把变化藏起来。

兼容方案候选：A按明确共同要求采用token sequence_mean，同时保留GiGPO状态组、轨迹/步骤优势混合参数、joint action ratio、动作mask和优势广播；将通用归约差异明确记录。B保持旧action_mean，仅改其他共同配置——不满足用户sequence_mean。C通过反向缩放优势补偿长度——会改优势含义，拒绝。正在独立核对A是否属于已明确授权的通用归约更改；若证据证明与必须保留的算法定义冲突，保留具体反例，不擅改数学定义。

实现时必须先在完整轨迹上计算有效token数，权重随行/动作metadata一起经过重排和DP切分。rank mini16，global128，FSDP梯度平均8卡：每个micro贡献的加权sum除以128/8=16，再累积8个micro；不能各micro单独mean后直接求和，也不能再除gradient_accumulation。这样DP平均得到sum_global/128，PG和KL一致。padding/观察/检索token权重0；零有效token轨迹贡献0但仍在全局128轨迹分母内，不能按非空行再归一化。

## 验证与门禁设计

先冻结配置、数据/模型完整SHA、源导入路径与实际resolved config。CPU相关测试后，在同一个Slurm独占8GPU作业中先进行实际DP/micro/gradient与完整状态保存/重载GPU检查；不计训练iteration。通过后同机加载3B、实际初始化512评测，执行真实1–5。第5完整HF与可恢复状态保存/重载，暂停于第6数据读取之前，由独立审查实际8rank变化/20step与微批目标/finite/局部检索/数据游标/完整状态后继续同一进程1–300轨迹。无独审批准不得取6；失败保留现场，不从头启动替代连续性。

本次源需继承已验收最终回答评分、FP64 joint ratio/FSDP梯度范数、完整状态实现；新增共同数据消费者、官方tokenizer/chat模板、57warmup和fresh5同进程门禁仅用于本实验。旧60a/2992/3130产物不变。阶段/周期/最终评测分别记dataset身份与实际qa_count。

## 当前事实与待核实事项

2026-10-04 17:24实际h100-8无Slurm作业、GPU compute为空；本机目前没有共同Instruct模型和0b002新数据路径。h100-9资产审查记录指向Tree已验收共同产物和官方13文件模型。下一步在同H100集群内离线复制到h100-8独立目录，完整SHA与数据语义/来源核对；只复制输入，不建立在线跨节点检索依赖。实际source manifest、instruction、tokenizer golden、Wiki/E5身份、GPU实现及训练全部尚未宣称通过或启动。


独立兼容性结论：明确的共同sequence_mean授权覆盖旧action平均测度，采用候选A，不额外求确认。GiGPO状态/优势/joint ratio不改，明确通用损失归约是新协议差异。必须新增独立权重模式、绝对57warmup读取、20opt/160micro审计和fresh5门禁，不能抄Tree的2初始链、6叶子或h1009。独审文档COMPATIBILITY-INDEPENDENT-REVIEW.md SHA b9b513b20117925877bdb29f35c275dc97313dcac69bea226cb671efe79e0aef。


调度器具体时序：沿用真实LambdaLR的初始LR=0，outer1的20次Adam步骤LR为0，结束后scheduler推进到1/57；outer57结束后达到1e-6，之后常数。因此首轮参数可无变化，但首5必须证明从初始到5的八rank真实非零变化、每轮20个实际Adam步骤及有限有效梯度，不能把合法warmup首轮当成失败或改warmup定义。每outer开始LR=1e-6*min(1,(step-1)/57)，结束LR=1e-6*min(1,step/57)。
