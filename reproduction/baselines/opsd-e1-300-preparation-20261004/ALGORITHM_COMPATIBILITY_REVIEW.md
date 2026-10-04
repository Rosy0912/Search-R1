# Search-E1 / official Instruct 300：OPSD算法兼容独立审查

**结论：共同模型/数据/采样/预算/globalmini128及20 Adam/outer可以用于纯OPSD；PG、reference KL .001 / low_var、PPO clip .2与当前纯OPSD目标不能同时真实生效而仍声称算法不变。必须用户选择目标，不能静默添加、删除或把无效配置写成生效。** 此审查只读当前 `6621088`（上游已修复fae核心），未修改源码或启动GPU/训练。参数以root保存的 `REQUESTED_CONFIG_AND_COMPATIBILITY.md` 为实际请求，不猜Search-E1其它默认。

## 原方法与当前实现

[OPSD原文§3.2](https://arxiv.org/html/2601.18734v3#S3.SS2)区分full-vocab蒸馏和sampled-token policy-gradient替代目标；不能因其PG解释，把普通reward PPO额外项视为实现细节。当前选用前者：沿学生轨迹，固定特权teacher提供分布监督，梯度仅经学生；作者固定[ae7d trainer](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/opsd_trainer.py) `compute_loss`主分支736–744调用full-vocab JSD/KL，另706–726才是不同的sampled-token策略梯度分支。作者generalized_jsd_loss 466–473屏蔽label -100后用有效token数归一。论文实验固定initial teacher可用；EMA不是加入共同PG所需条件。

当前 `controller.py:65–70`明确拒绝纯蒸馏use_kl_loss/entropy/kl_ctrl系数非零；257–260 OPSD分支advantages/returns全0，reward仅报告。`actor.py:23–68`调用opsd_loss/backward，没有调用普通PPO actor目标。`distillation.py:12–33,104–166`默认无分量裁剪full-vocab teacher KL，mini全局监督token denominator；teacher.detach/no_grad/冻结，与普通reference actor不同角色。

`run.sh:24–35`当前明确覆盖teacher CUDA/user_context/component_clip=null/local h100-10/use_kl_loss=false/entropy0/kl_ctrl.kl_coef0。CLI若再设use_kl_loss=true或kl_ctrl系数.001，会被controller拒绝；仅设置actor.kl_loss_coef=.001但use_kl_loss=false，则该数字不产生任何损失，不能声称共同referenceKL已生效。old/ref logprob可以计算用于审计，但DistillationActor不消费它们作为训练项。

## 可共同使用、不改变蒸馏目标的条件

| 条件 | 纯OPSD兼容结论 |
|---|---|
| 官方 Qwen2.5-3B-Instruct 固定revision aa8e725… | 用户新授权取代旧PLAN只base限制；actor与固定特权teacher应来自同官方Instruct初始目录，独立reference若仅审计可同初始权重。必须实际核对全文件SHA、完整加载、词表/特殊IDs及ChatMLsuffix。不能将旧base/HF5资产当Instruct。 |
| 固定共同train169028/dev512/stage64/final51713、manifest0b002b… | 可共用，不重划；模型/data兼容性需要实际本节点资产验收，别人状态true非证据替代。 |
| 512问题×5轨迹、n=1/n_agent5/TP1/T1/top_p1/top_koff、4turn | 可作为明确Search适配和共同工作负载；不改蒸馏公式。比作者数学单响应实验的采样/复用设置不同，应保留标签。 |
| globalmini128/epoch1/globalmicro16 | 兼容：2560轨迹/8rank=320/rank，mini128/8=16，micro16/8=2；每rank20mini，每mini8micro，共20次实际Adam/outer。不是8rank相加160次全局Adam。fsdp_workers.py:97–101除DP且乘rollout.n，此处n=1不会额外乘5；n_agent5的轨迹已由controller生成。 |
| old/refmicro32 | 单独logprob审计的资源参数兼容，每rank4；不因此激活PG/referenceKL。 |
| lr1e-6、57outer warmup、entropy0、gradclip1 | 与纯蒸馏兼容；lr scheduler每outer一次，而不是每20Adam一次。当前源码只支持ratio推整数，300实际outer在当前counter需budget301，旧.285会得到85不是57。需明确实现/配置整数57与边界证据，不能套旧ratio。 |
| 本机GPU E5/FAISS/Wiki2018/top3/loopback、官方学生instruction/template | 可共同，不触碰跨集群服务；teacher-only特权提示仍需保留OPSD特性。 |
| 300actualouter、first5原位审核后连续、保存/评估日程 | 可以是工程条件，不改目标；当前complete恢复state/第5前第6取数阻塞尚未实现/验收，不能借旧job/HF轮数。当前总counter301只是源码结束条件推导，不等于新实验已实现。 |

官方[Instruct tokenizer](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/tokenizer_config.json)当前公开内容为EOS im_end、pad endoftext，普通user suffix仍为ChatML user结束+assistant开始；base与Instruct EOS不可混。FrozenTeacher会检查teacher/student tokenizer vocab和BOS/EOS一致。实际固定revision本节点下载/sha/template仍须核对，本报告没有把网页当前main当固定revision资产验收。

## 必须用户明确选择的冲突

| 共同项 | 当前纯OPSD真实行为 | 启用后的含义 |
|---|---|---|
| reward policy-gradient/GRPO/PPO项 | advantages0，PG不存在 | 加非零reward advantage和PPO项形成蒸馏+RL混合目标；不能仍标未变纯OPSD。只计算reward/完成率用于报告可保留。 |
| fixed reference KLcoef .001 / low_var | 训练禁用；与teacher full-vocab KL不同 | 独立reference正则新增目标/权重。low_var是采样token logprob estimator，并非privileged teacher full-vocab KL。 |
| PPO clip .2 | 配置可存在但OPSD不使用old/current ratio | 只有新PG或其它ratio策略项启用后才有作用；不是component clip .05，也不是gradclip1。不得用.2重新裁剪full-vocab组件。 |
| PG/referenceKL sequence_mean | 当前没有两项，因此没有可宣称的reduction | 若选择混合目标，两项需按每行有效token均值→完整globalmini行均值累加，不能每micro局部均值直接相加。当前普通DP actor对非paper路径仍masked_mean，并非自动满足该定义。 |
| distillation sequence_mean（如用户也要求） | 目前是global supervised token_mean | 改为各trajectory等权改变长短trajectory权重，属于新算法/适配选择；请求明确仅PG/referenceKL sequence_mean不能自动扩展到蒸馏。 |
| “无额外提示”同时约束teacher | 当前teacher加入privileged训练答案与教学transition，student不加 | 去掉特权信息删除OPSD监督机制；应将共同instruction约束用于学生，teacher内容另列算法专属，否则必须用户选择。 |

两个归约不可视为同义：一行1个监督token loss1，另一行9个监督token全部loss0；token_mean=.1，sequence_mean=.5。mini/micro切分保持整体分母是数值正确性，选择哪一个整体分母是目标定义。

参考KL亦不能用teacherKL冒充：初始student/reference同权重、同普通上下文时referenceKL为0；privileged teacher即使同权重、不同答案上下文，分布可不同而teacherKL非0。普通代码 `core_algos.py:274–278` low_var=exp(ref_logp-current_logp)−(ref_logp-current_logp)−1并clamp，reference evaluator和teacher全词表FP32/eager目标不是同一机制。

用户可选择：(A)保持纯OPSD，共同PG/refKL/.2标明确算法例外/不生效，蒸馏token_mean；(B)明确OPSD+reward-PPO+referenceKL混合目标，指定各项系数和reduction，随后独立数学/状态/容量验收；(C)作者sampled-token蒸馏替代目标，另定义为变体，不等于普通共同rewardPG。此报告不代用户选择，也不自动采用B或C。

## 更新频率与验收边界

globalmini128带来的20更新可保持纯OPSD。当前同outer全部rollout采样于更新前，后续mini学生已变化，因此不是每次Adam重采的新on-policy分布。20次相对此前10次增加outer内偏移，属于明确共同实验工作负载/超参数变化；purefull-vocab蒸馏没有PPO ratio控制，不能靠clip.2声称限制该偏移。可记录而不偷偷添加importance修正。

无GPU/训练已执行；official Instruct实际EOS/template/teacher target内容、共同数据、当前null/user_context数值容量、真实20steps/57warmup/300outer/第5原位继续/完整state恢复均需后续实际验收。现阶段目标冲突应先明确给用户，不能提交目标含糊的新长跑。
