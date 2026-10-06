# Tree-GRPO 官方 Instruct fresh200

本实验遵循2026-10-06新的人工goal，替代healthy110恢复和300轮目标。旧作业、checkpoint、Adam、scheduler、数据cursor和诊断保留在历史目录，均不继承为本实验训练状态。本文件是设计与源码说明，不是实际200轮或GPU验收证明。

actor和固定reference均从本机官方Qwen/Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`加载。运行包装实际计算完整资产manifest并与冻结身份比较。共同数据manifest `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`实际读取，原train169028/dev512(NQ/Hotpot各256)/final51713不重划分，worker再次检验过滤后内容和有序question IDs。

## 真实五叶树结构

每题两个独立根，根0叶预算3、根1叶预算2。先各生成一条根轨迹，再从已有动作节点继承完整前缀并续写：根0两条、根1一条。每题实际总共五条完成轨迹，不生成第六条后丢弃。固定quota模式约束每根候选恰等于quota，选择只重排现有所有叶。保持四轮搜索、长度2048/4096/500/500、不强制最终回答、topk3。

原`advantages.tree_grpo`不改：五条叶之间的树间分组中心化，加各根内部3/2条叶的树内中心化。原逐叶token权重1/T及全局叶分母不改；五条有效叶等质量，单题各叶相对质量1/5，各根总质量3/5与2/5。根标签来自两次相同策略分布的采样，quota差异为本次显式预算选择；不把两根伪称等质量。audit逐题/逐叶记录root ordinal、quota、tree/trajectory/hash、实际前缀继承及质量。

## 二十次实际更新与warmup

每outer512题×5=2560叶；8rank分配各320。全局mini128变为每rank16，全局actor micro16变为每rank2，每mini8个micro，epochs1共20次实际Adam调用。old/ref全局micro32，每rank4。每rank直接记录实际20 calls/0 skip及scheduler前后边界，并全rank同步验证，不能用平均20代替八个rank各20。只有完整outer才推进一次scheduler及outer计数；部分更新触发刹车受控退出，保留已发生写入和原始轨迹，不自动回滚。

LR1e-6、57个完整outer线性warmup后常数，沿用现有原生LambdaLR初始化语义：scheduler.last_epoch=0时LR=0，首outer20次Adam会更新moment/step但参数可不变化；不得报告首outer有非零参数变化。之后完整outer k结束scheduler=k、下一outer LR=min(k/57,1)×1e-6。entropy0、reference KL coefficient0、low_var_kl、clip.2、全局grad clip1；固定reference仍计算保留身份，不用reference KL替代采样策略刹车。

## 采样分布校正与三项防护

沿用独立说明`BEHAVIOR_MU_CORRECTION.md`的明确µ中心协议：old PPO分母为实际native选中token的FP32采样log probability，目标D(pi/mu,A)。整批actor重算β_full单列保留，固定同分区β_monitor用于自重复门禁。真实18项采样设置、token/prefix/positions/masks/原物理调用身份随Tree继承绑定。原vLLM/actor概率一致性门禁保持未通过，不放宽容差，也不把µ校正冒称原门禁通过。

Dual-Clip负优势c=3，指数前有效位置有限性检查和有界损失；全rank固定32行的真实mu→pi采样分布加权监控KL、负优势尾部及停止决定保留。shared-prefix/selected-branch measure不是无偏全策略KL，不扣初始mu KL基线。数值边界30不是允许训练偏移量。生产KL/tail阈值必须来自本次实际零写入观测和短训验收；候选.01只是零写入观测起点，未接受不能用于自动发布生产训练。calibration显式train模式对齐正式actor更新，成功与失败都恢复原各子module模式，保存实际模式证据；监控保留RNG边界。

## 执行与交付

独立新目录与唯一调度，禁止任何TREE_RECOVERY_环境。官方初始化直接检查8rank Adam state为空/scheduler0及driver/sampler0。先真实8卡零写入观测；短训前5轮属于同一个新训练实例，保存完整HF/tokenizer/状态和固定512评分，经独立核验后原实例继续6..200。保存5/50/100/150/200，初始和每5轮dev512，最终完整七集51713；逐轮报告reward/EM/F1/实际outer/optimizer/scheduler及成本。

所有训练、数据、权重、检索、存储和Ray在h100-9本机；8GPU Slurm预约、UUID租约、双15%外来资源门禁、原8背景PID及start ticks保持。只清理本任务。代码冻结后，执行前按用户要求启动不继承父会话上下文的subagent，单独提供目标、源码和冻结材料进行严格审查。source/CPU通过不等于GPU或效果通过，全部200outer、最终评测/checkpoint验证及资源收尾完成后才能关闭goal。
