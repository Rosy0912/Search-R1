# OPSD Search-E1 官方 Instruct 新实验：实际配置对照与兼容性

2026-10-04。用户授权启动新实验，但明确：若共同设置与OPSD存在必须改变算法含义的冲突，记录具体证据，不自行改变算法定义。本文件先记真实配置对照；没有提交GPU或训练，不借历史轮数。

## 用户指定共同条件

官方actor/fixed reference Qwen2.5-3B-Instruct revision aa8e72537993ba99e69dfaafa59ed015b17504d1；共同数据manifest SHA256 0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24；实际共同产物train169028、dev512（NQ/Hotpot各256）、stage64、final51713，禁止重划。

每outer512问题×5响应=2560轨迹；rollout n=1/n_agent5/TP1/temp1/top_p1/top_koff。globalmini128/epoch1 =>20 Adam步骤/outer（每rank16行mini）；actormicroglobal16/perrank2；old/refmicroglobal32/perrank4。lr1e-6，固定57个outer warmup后常数，scheduler每outer一次；entropy0/reference KLcoef.001/low_var KL/PPOclip.2/gradclip1。PG及KL sequence_mean：每行有效token均值再完整globalmini行均值，不能逐micro均值直接加。

prompt/start/action/observation=4096/2048/500/500，maxturn4；共同原始instruction+官方chat template，无自行额外提示。训练节点本机GPU E5/GPU FAISS/Wiki2018/top3/loopback，同共同实验资产。

总实际outer300，first5包含其中；初始化与每50 dev512；stage64；final51713。save HF+完整state在5/25/50/.../300。第5后在第6取数前阻塞独审，通过由同训练进程原位继续（模型/Adam/scheduler/RNG/游标/OPSD状态连续）；不得重新启动替代继续，不执行301。完整各rank状态保存/真实重载，8GPU实际参数变化/有限lossgrad/20steps/micro归约/检索/游标需要真实验收。

## 当前修复版与共同配置的确定性冲突（待独审）

当前固定修复HEAD66210885；旧3007/8e取消80仅HF5，旧数据/Base不能用于此新实验。

1. controller.validate_config对OPSD/TOPD明确拒绝actor.use_kl_loss、entropy非零或algorithm.kl_ctrl.kl_coef非零，错误文本 Pure distillation requires reference KL and entropy coefficients zero；run.sh OPSD也覆盖use_kl_loss=false/algorithm.kl_ctrl.kl_coef=0。
2. controller.compute_advantages OPSD分支令advantages=returns=0，注释 outcome scores are reporting only。
3. DistillationActor.update_policy调用opsd_loss而非PPO policy objective。当前无PG项、无reference low_var KL项、无PPOratio clipping；梯度范数clip1正常。公式是固定特权教师fullvocab KL(teacher||student)，不是sampled old/ref logprob KL。
4. 当前纯蒸馏按global supervised token count归一。把teacher KL改为trajectory sequence_mean会改变变长trajectory权重，不是日志/基础设施等价修改。用户共同条件只明指PG/referenceKL，不明确要求蒸馏目标改权重。
5. 教师user_context包含训练答案是OPSD算法特权监督；“不增加额外提示”可保持共同student instruction/template，但若也禁止teacher特权hint就会删除OPSD逻辑。不能默认删除。

这些不是从名称推测，需给具体源码行/作者资料独审。不能同时声称纯OPSD不变且PG+referenceKL.001生效。可同步核对/准备无冲突共同模型数据/预算/保存门禁；训练目标必须在冲突明确后按用户选择，不先启动含糊目标。

## 实际共同证据定位（准备读取，不代表本节点核验通过）

BAPO现canonical outputs/STATUS.json e1_instruct300报告同manifest实际train169028/dev512/stage64/final51713；源资产位于同H100集群h1005及Tree h1009。旧审核model manifest7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44，model13文件已报告核验。当前root仍需实际读取共同manifest及各产物，不能只继承对方状态true。

本任务新source部署、officialmodel在h10010实际SHA、共同数据在h10010实际SHA、resolved配置、8rank门禁、完整恢复state、first5暂停、fresh300/final均尚未完成。

## 本轮实际检查结果

共同manifest原始bytes已读取且SHA精确0b002b…；同一H100集群Tree h100-9的四份共同parquet已实际读取全部文件并计算SHA，各自匹配manifest，保留common-peer-verified-route-sha-check.txt。文件实际大小train17823593/dev62981/stage14032/final6445844字节。这是源产物通过，不是已部署h100-10。首次从h100-10直SSH peer因Host key verification failed失败已保留，不关闭主机校验；随后采用Tree已有已核验管理路由成功。未调用peer GPU/检索计算。

official模型完整manifest原字节SHA7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44已读，不代表本节点模型全13文件通过。

真正canonical run.sh --cfg job组合user要求referenceKL.001，然后执行真实validate_config，实际ValueError为Pure distillation requires reference KL and entropy coefficients zero；原始resolved与CPU输出保存attempt-common-kl-resolved.yaml / common-kl-compatibility-cpu.txt。不以报错冒充GPU验收。

独立算法审查ALGORITHM_COMPATIBILITY_REVIEW.md确认PG/refKL/蒸馏sequence_mean不同含义。固定warmup57需新增显式整数路径，不能沿用ratio.285；mini128正常非空可20，全球全空mini旧OPSD会skip，异常需报告不能静默改变Adam时钟。

独立状态审查STATE_GATE_COMPATIBILITY_REVIEW.md确认当前只有HF导出，无完整Adam/scheduler/RNG/数据游标恢复，acceptance_step5只有save没有阻塞；不得立即沿旧入口启动。后续须适配已审核SHARDED/DeviceMesh状态机制，OPSD专属固定教师状态、源/data资产身份；原位第5门禁及真实8rank恢复仍待实现和GPU验收。相同300预算需当前源码counter301 sentinel并严格终止actual300，日志最终不得用counter301冒充额外训练。

目前actual新iteration0，GPU/训练未提交。用户损失目标选择待答；automatic仍PAUSED，不自行恢复。上述与算法无冲突的实际资产核查和独立设计已完成，具体生产实现/硬件与训练未完成。

## 最新有效选择

用户明确“保留纯opsd”。目标固定无裁剪特权教师full-vocab KL、已有global supervised token归一；PG/referenceKL/PPOclip不生效，gradclip1。其它共同设置全对齐。新opt-in E1路径采用actual completed300结束，total_training_steps=300；旧301 sentinel推导不用于新路径。新src work/opsd-e1-300，branchcodex/opsd-e1-300。配置/状态/gate正在实施交叉审查，首5计入300并阻塞第6取数。暂无GPU或训练。官方manifest实际11原始文件（其中2权重），另元数据manifest，不沿用旧“13”叙述。

## Target-node actual asset check

ACTUAL h100-10 native direct LAN copy completed with all27 controlled files byte-SHA matched: official11 plus model manifest, four prebuilt parquet, raw9, exclusion audit, common manifest. asset-copy-verified.json preserves the receipt. Native Py3.10 CPU consumer checked actual identity/content/source allocations/original instruction and all six cross-split ID/canonical-question overlaps zero: train169028/dev512/stage64/final51713. Official revision and all11 model-file SHA checks passed. actual-assets-native-cpu.log preserves original full stdout. Scope is actual assets/CPU consumer only, not Slurm/runtime binding/key/GPU/training acceptance.

Root integrated test found dynamic sys.modules recovery identity drift: original106 tests1FAIL105PASS preserved; fixed deterministic source inventory plus actual origin check passed106 author integrated tests. Fresh native8GPU, samePID first5 and300/final not yet executed.
