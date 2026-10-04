# Search-E1 common profile independent review

2026-10-04；仅本地只读 CPU / 静态审查，作者为 gpu_probe，审核者未编辑这7个候选文件。原固定manifest `common-profile-source-manifest.json` SHA `0a5d5b749511d43c0b2b825b8f39258d24c791a80fca9c380b0e0e5950ccb24c`。

## 已确认

共同manifest原字节SHA `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24` 与官方模型manifest原SHA `7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44` 为literal pin。实际官方库存11，不是历史13。逐文件revision/字节数/全SHA、modelroot约束与未manifest普通文件拒；VERIFIED标记不是验收。actor/teacher同一初始官方目录。

共同消费者不split、不subsample、不写parquet：四产品全SHA/行数/列/完整prompt+reward身份/顺序ID/source计数/canonical问题hash，四split两两问题和ID交集、stage/dev内部问题唯一；严格raw9文件两train七final与exclusions原hash、路径越界、重复JSON都校验。numpy结构借既有plain转换。实际共同原manifest为train169028/dev512/stage64/final51713；作者合成测试169038不能充当本节点实际169028部署证明。

纯OPSD固定配置不添加PG/refKL；beta0/FKL、component_clip null、user_context特权教师，原global supervised tokenmean算法由其他源码模块落实。mini128/8=每rank16；globalmicro16/8=2；2560轨迹/mini128=20非空mini；logprob32/8=4。300 actualouter horizon、整数warmup57/ratio0、epoch1、local CUDA/H100 host、检索loopback/top3、阶段64每50与first5 gate条件明确。该配置算术不是实际20 Adam计数证明，需actor/trainer原始证据。

本审核独立执行两测试文件：35 PASS / 32.17秒，CUDA隐藏/OMP1；raw `common-profile-independent-cpu.txt`。测试真实parquet/DataProto CPU，但GPU模型/原始完整部署/300/最终效果均未验收。

## 原固定候选发现与修复待冻结

1. 原固定e1_profile.validate_profile expected未包含actor.grad_clip=1，虽然yaml设置1且launcher拒任意CLIoverride，固定用户契约仍可在修改resolved配置时漂移。已告作者，当前活动文件增加grad_clip=1及clip_ratio=.2；独立CPU实际将grad_clip改.5后已ValueError拒。**活动修复字节未包含于原0a5d manifest，须等新固定manifest及增量测试后关闭本项**，不能替旧快照冒认通过。

2. 原launcher通过 `python -m ...`，无自身repo定位/cd/PYTHONPATH或实际imported module origin/hash门禁。仅外部binding import_closure文件SHA不能证明执行的Python源来自该文件声明目录；当前CheckpointGate的外部file binding只读文件pin，与实际主进程/Ray各worker执行source witness必须分开。新任务不允许加载旧安装verl/trainer或shadow模块而仍声称新E1原位gate/300预算成立。此处为静态可确认缺少启动进口核查，尚未运行shadow exploit或GPU。作者正在处理/明确外层producer职责；新source freeze必须明确该缺口由哪个实际入口关闭，并有CPU阴性证据。

## 验收边界

原7文件共同数据/配置消费者可作为CPU软件候选保存；**当前不批准GPU/训练启动或完整source closure验收**。待新manifest固定后需复核上述最小修复，实际节点完整raw9+四parquet+官方11文件部署、source/config/runtime进口、owned8GPU lease与检索共存、实际容量和真实Adam20、fullstate/nativeDTensor/同PIDfirst5签名暂停与恢复、actual300/final51713及corrected原始评分全部独立证明。回退到旧.05、legacy、CPUteacher或旧3007证据均不能作为本实验满足要求。


## 新bootstrap活动候选增量（未等新冻结）

当前新增 `reproduction/baselines/e1_import_bootstrap.py` repo_first/实际module路径+全SHA、compiled function code核查，可以补原python-m入口闭包问题；但 `check_live_code` 对未在allowed的co_filename直接跳过，有可执行CPU反例：同module中good函数保持源码，bad函数用foreign.py替换并保持__module__等于当前module，good计数1而bad返回999，当前核查仍无异常通过。原始反例JSON `import-bootstrap-independent-counterexample.json`。必须拒本项目原生函数/方法的陌生来源；dataclass/library生成函数只能明确限定例外，不可泛跳过。已报root/作者，活动修复需新freeze后复审。

Ray trainer当前并非仅父进程bootstrap证明：`ray_trainer.fit` 的E1分支实际构造CheckpointGate，gate.__init__实际verify_import_closure核当前模块__file__/源SHA和关键对象实际bytecode对应编译源，发生在初始validation前。当前required7模块的若干关键对象：gate.verify_receipt、sampler/rng、controller.compute_advantages、distillation.opsd_loss/teacher_inputs、actor._update_policy、trainer.fit、worker payload/load。这个Ray实际自核不能扩大称整个Rayworker所有方法/进口的全闭包源码证明，仍需launch+worker实际运行闭包证据。

## 原始count backend数学核查

root改动只将mask.sum().float()得到的单元素total按实际backend路由：NCCL→current CUDA device，Gloo→CPU。sum/all_reduce(SUM)/world_size保持；actor仍使用total/world作完整mini监督token分母，每micro累积不再额外除micro次数，空mini仍所有rank同一raw count决策。这不会添加PG/referenceKL或改变全局tokenmean数学。CUDA路由API是CPU mock，真实2rank Gloo unequal3+1 total4和empty0均实测通过；独立2 tests PASS/6.32秒，raw `count-backend-independent-cpu-tests.txt`。尚无真实NCCL/GPU或真实3B梯度一致性证明，不把Gloo通过扩称NCCL实际验收。


## 最终固定9文件复审与结论

作者已冻结新manifest SHA `67598a2c0036e168cf1364d49f2357b221d887185966c60412672ff28daf49f0`，checker SHA `04304d563a71b3123e0da6c5350398bf0b8510e0b477ce47742798214bd5b3f0`。本审核独立全核9源码成员实际SHA；数据消费者/独立yaml字节较原0a5d保持一致，变更集中grad_clip/clip_ratio门禁、launcher实际bootstrap与新增进口阴性测试。

独立增量46 tests PASS/2.79秒，raw `common-profile-final-independent-fast-cpu-tests.txt`；全6组121tests已独立集成PASS/57.27秒，raw `state-rank-live-source-integrated-cpu-tests.txt`，其中包含该9文件完整57 profile/bootstrap/prebuilt测试，不把作者57结果代本审核执行。重放原foreign函数替换反例：未改good/bad时count2，替换bad保留__module__后明确ValueError拒bad；新raw `import-bootstrap-independent-fixed-counterexample.json`。grad_clip=.5/clip_ratio=.3、未知source filename、改__module__试隐藏、properties accessor、循环wrap、假dataclassmarker、源码明确constructor冒认generated、imported真实dataclass和伪constructor均测试覆盖。两项原阻塞在本固定CPU软件快照关闭。

实际Ray driver在CheckpointGate内调用同外部pin的strictchecker源码bytes和7critical expected-object校验；actor配置传rank4字段，新state helper在该rank执行同checker原始bytes、核actual modules/livefunction并输出PID/UID/rank证据，独立实际CPU接口与阴性pin测试通过。该rank0仅CPU参数，不是实GPUrank。动态import集合不作跨进程状态identity；确定性数学源码库存和实际origin/严格checker核查分层。

**允许把该固定9文件作为CPU软件候选提交与进一步native验收；仍不批准GPU训练、实际first5/300或效果完成。** source code检查证明当前普通项目函数的源码/bytecodeprovenance及critical对象对应原实现；明确生成的dataclass/library函数有受限制例外，不能把function_count称全运行时语义不可变证明或恶意内存替换全面证明。完整同节点部署、owned8GPU/源import runtime/检索共存/满形状容量、实际20Adam+FP32教师与FSDP/NCCL、原位first5签名pause、新进程完整driver恢复授权、300outer和final51713真实效果均尚未验收。负面效果应如实保留，CPU门禁通过不等于效果提升。
