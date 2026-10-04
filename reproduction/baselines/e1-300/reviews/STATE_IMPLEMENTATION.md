# OPSD E1-300 rank完整状态实现交接

2026-10-04。隔离checkout `work/opsd-e1-300`、parent f9798d7f3690e21ab63f8fd0f4c963b044d6a219。此agent只编辑 full_state.py、fsdp_workers.py 与新增CPU test/native fixture；不改actor/controller/trainer/config/holdout，不提交Git/remote/GPU。数学仍pureOPSD tokenmean fullvocab KL，无PG/reference追加。

## 固定候选文件

`work/e1-300/state-implementation-files.json` SHA e4a55259316d9a2a1f09259f8d27ba6882bac9d1b0b83f97c5215abce2aed81a，列4个实际文件绝对path/SHA：

- search_r1/baselines/full_state.py
- verl/workers/fsdp_workers.py
- tests/baselines/test_opsd_e1_full_state.py
- reproduction/baselines/e1-300/native_opsd_state_fixture.py

参照的是邻任务实际R6 raw source中的SHARDED/DTensor方案，不是旧LOCAL实现；本任务状态payload升级v3并含真实固定教师状态，无BAPO payload。

## 接口与集成

`full_state`: rng_state(cuda_device=None)/restore_rng、state_digest、file_digest、atomic_save、StatefulSampler、resolve_warmup_steps、teacher_state/restore_teacher_state。driver不调用CUDA，worker只当前rank CUDA RNG。digest包括DTensor完整global shape/stride/dtype/placements/mesh设备/ranks/dimnames/coordinate/local bytes；无gather/full_tensor/redistribute/mesh私有修改。普通Tensor dtype/shape/stride/content、容器类型/长度也被绑定，bytes通过memoryview避免大词表参数额外tobytes复制。

atomic_save独占临时文件并fsync后用hardlink发布不可变最终成员；已有最终成员/他人tmp拒绝，绝不删他人tmp。不等于远端传输原子manifest发布门禁。

StatefulSampler支持dataset或正size，保留RandomSampler(generator=None) seed消费，并保留epoch-end空slice permutation消费。`cursor`是已emitted，`consumed_cursor`为trainer成功outer后显式`commit_consumed(n)`；默认strict state_dict拒未消费预取。load验证原size/seed/shuffle/permutation/epoch/两cursor/generator并保留同sampler对象。要求num_workers0；trainer应在完成合法20steps后commit原question batch行数、所有save在commit之后、gate在下一取数之前。

resolve_warmup_steps优先严格整数`actor.optim.lr_warmup_steps=57`，不依ratio×horizon估算；保留通用legacy num_warmup_steps/ratio兼容。workers scheduler仍每成功outer update一次，没有改变Adam预算或loss归约。新版E1 trainer actualcounter300由另一agent实现，本接口不预设301 sentinel。

worker新RPC：`save_training_state(local_path,step,roundtrip=False)`、`load_training_state(local_path,step)`。公共SHARDED_STATE_DICT/offloadCPU DTensors、原生Adam state/paramgroups、scheduler、rank RNG以及OPSD固定教师状态。保存并从磁盘实际读回；roundtriptrue/load都调用同model/Adam/scheduler/teacher对象的真实load/restore，随后完整payload digest必须一致。严格rank/world/step/PID/object IDs，拒旧version/算法/LOCAL或异进程对象；默认分支不提供restart；下述显式恢复分支不允许world-size迁移。

rank_N.pt/json report：version3、algorithm pure_opsd、status EXACT_FULL_STATE_RELOAD_OK或FULL_STATE_SERIALIZED_OK、rank/world_size/step/path/sha256/state_sha256/state_digest、SHARDED_STATE_DICT/model_dtensor_entries、scheduler_last_epoch、optimizer_state_entries、pid/object_ids、teacher摘要。sameprocessload RPC读取同目录精确report/file pin，不产生独立PASS receipt。生产release必须另要求8rank/DTensor>0/actual counters/source和有效原始审查。

Teacher：idle CPU/eval/FP32 execution validation、requires_grad/grad/optimizer membership0；完整原资产regular filesSHA（拒symlink）、实际namedparameter内容digest/config/count/bytes/execution/storage device。实际Python计数八字段全部保存恢复（cpu/wall/transfer/cuda秒、CUDAupdateentries、scored_tokens、loadcpu/loadwall），不伪造BAPO stage。persistent及nonpersistent实际buffers和rope max_seq_len缓存元数据入payload并在同model对象恢复；JSON report省略buffer Tensor本体但保留buffer digest/缓存、全部identity/counter。release verifier应核payload教师buffer原文件hash，不要求JSON含Tensor。

state RPC不改变teacher参数值；固定teacher权重身份改变则拒。保存asset+content全量hash成本真实存在，不把checkpoint时间藏入updatebenchmark。时钟计数在**保存/恢复点**精确连续；真实后续不同执行branch的elapsed时间不应被要求物理逐位一致。原生toy fixture没有teacher interval计时，离散counter与内容准确；不能将它当完整生产teacher计时效果验证。

Worker另仅修改actor/rollout/ref边界data/prompts/TensorDict.to为显式torch.device(cuda,current_device)，保留原manual offload、Critic/RM不改。Ray单可见logical0与torchrun多可见3/7都显式路由；root另负责NCCL/Gloo token scalar数学等价修复，本agent不碰distillation归约。

## 实际本地验证

CUDA隐藏、OMP1，CPU环境执行32tests PASS/7.45秒，原输出 `work/e1-300/state-implementation-cpu-tests.txt`。包括：真实CPU Linear/Adam/LambdaLR、Python/NumPy/Torch随机源原对象step5save/load/roundtrip与第6更新oracle；teacher真实参数变更/optimizer overlap拒、buffer/cache/counter恢复；sampler同活跃iterator下一行恢复、未commit/错cursor拒；57整数；已有文件/tmp保护；12个**实际worker函数AST**逻辑GPU0/3/7路由。

真实world-one CPU Gloo DTensor保存加载digest一致，并实证不同local内容/placements导致digest变化。workerCPU测试的FSDP context和CUDA RNG API明确mock，模型/Adam/loss/RNG/磁盘是真实CPU行为，不能改称native FSDP证明。3个作者源/tool compile PASS。

## 新native fixture（已写，未运行）

默认8rank单节点、FP32 baseline和nested两decoder/三FSDP wrappers、FULL_SHARD/DeviceMesh、BF16compute/FP32reduce/buffer/persistent参数/Adam、manual offload；nativeCPUOffload为可选额外profile。固定FP32 toy teacher计算无clip完整2维vocabKL，与真实随机输入/decoder dropout，actual生产worker接口完成SHARDEDsave/load/roundtrip。每profile synthetic5outer×4Adam=20initial，no-reload/reload两个第6各4=>28calls，scheduler outer一次/warmup57；这些是**小模型state oracle预算，不是用户真实每outer20Adam的RL验收**。

报告含all model/Adam/scheduler/RNG/teacher component exact equality、sameobjects、DTensor>0与实际source/helperSHA。仅编译检查，未提交GPU。运行必须新source clean/外部E1_EXPECTED_SOURCE、SEARCH_R1_REPO、新fixture output及自有8rank torchrun allocation；root需要另做source/import/资源/Slurm门禁。完整officialInstruct3B checkpoint/teacher/tokenizer/实际20Adam/rollout检索/first5samePID gate/300与final51713仍由真实后续验收完成。

## 请求独立审查

请另一实现非作者代理复核固定文件和32CPU证据，重点SHARDEDTorch2.4/device/offload restore、教师buffer/计数语义、scheduler、sampler取数与drivergate组合。该交接不自产生PASS release，不签nativeGPU或first5，更不恢复旧3007。


## 显式新进程恢复补充（新快照，原32测试记录保留）

用户要求完整可恢复备份；同PID roundtrip仅能证明原位续跑。因此新增显式 `load_training_state(local_path, step, recovery_mode="authorized_new_process", recovery_binding_path=..., recovery_binding_sha256=...)`。默认参数不变，默认首5真实load仍严格旧PID和四对象IDs；未指定mode传binding、未知mode、缺binding均拒。**首5签名receipt不可复用为重启授权**。

保存payload增加 `recovery_identity.worker_source_sha256`（实际worker文件全SHA）与 `actor_config_digest`（实际resolved actor配置）；固定教师资产/权重/计数/buffer和rank/world/step/DTensor完整mesh仍包含在core。显式恢复仅移除payload的 `pid/object_ids` 两个运行身份做恢复后digest对比，其他字段和所有模型/Adam/scheduler/RNG/teacher/layout字节不能变化。报告返回 `EXACT_NEW_PROCESS_FULL_STATE_RELOAD_OK` 与新 destination_pid，不能冒充原位的 `EXACT_FULL_STATE_RELOAD_OK`。

外部JSON exact schema：`{version:1,checkpoint_sha256:<rank_N.pt实际全SHA>,rank_binding:{version:1,authorization:"EXPLICIT_NEW_PROCESS_RECOVERY",rank,world_size,step,source_state_digest,source_pid,source_object_ids,destination_pid,destination_object_ids,recovery_identity,teacher_asset_files}}`。运行身份必须逐项精确匹配源payload与本次新对象；binding文件SHA来自外部调用，不由load自产生PASS。该层只是**内容/身份绑定 primitives**，不能把任意SHA绑定当人类授权/签名证明。新controller完整source/config/data/assets/mesh/环境与独立签名授权、driver/sampler状态恢复仍需单独实现和验收；现有E1 controller只负责samePID pause5，没有自动restart。当前无重启作业。

新37 CPU tests PASS/8.05秒，raw `state-recovery-cpu-tests.txt`。包括真实新模型/Adam/scheduler/teacher对象恢复、独立 Python 子进程新PID（父进程外部生成精确绑定）实际读磁盘state和随机源，真实第6 Adam更新core oracle一致；默认同PID接口拒新对象与实际新PID。覆盖未知mode、binding缺失/默认模式旁路、配置漂移拒。FSDP和CUDA API仍mock；子进程Torch/Adam/RNG/磁盘均真实CPU。真实多rank新进程FSDP、完整3B状态/资源/签名restart尚未运行。

初次fork测试实际遇到Torch autograd拒fork后线程状态，原失败raw保留 `state-recovery-cpu-fork-failed.txt`；改独立Python subprocess，不将fork失败隐去或作为软件成功。


## 实际源闭包补充（新40测试快照）

`recovery_identity.source_closure` 增加实际 Git HEAD，以及当前已导入 search_r1/verl 模块、包initializer 的实际__file__路径和全SHA。先显式导入worker恢复依赖的full_state/distillation/controller/rollout/advantages/bapo/sharedholdout/actor/core_algos/fsdp_utils/torch_functional/protocol，确保恢复解析binding需要的模块在首次保存前已有稳定闭包；不导入未执行GPU模块、不运行模型。每个module必须实际来自当前repo对应.py或__init__.py，执行payload方法co_filename也必须为当前worker源。source/code/actor配置差异在任何model.load前拒，默认同PID和显式恢复都核。

完整40 CPU tests PASS/9.40秒，raw `state-source-closure-cpu-tests.txt`，包括实际source import shadow位置/source内容SHA/Git HEAD漂移拒，新PID第6真实CPU Adam oracle仍通过。一次新增closure因sharedholdout只在恢复时lazy import导致1fail39pass的raw保留 `state-source-closure-lazy-import-failed.txt`；将确需恢复解析模块纳入首次capture闭包后通过，没有放宽源集合等值要求。

恢复必须初始化与源state相同的实际模块闭包；额外/缺失依赖也拒，不自动import任意saved路径或改sys.path冒认当前源。此处校验的是实际import路径/文件内容，不等同于全进程每个函数动态bytecode或driver全配置签名验收；launch/runtime门禁仍须证明bytecode来源、完整数据与environment/restart授权。教师/数学源码已包含源闭包，不再仅worker单文件pin。


## 最终确定性源码身份与各rank live检查（覆盖上文动态集合设计）

根任务真实集成106例暴露旧9b快照使用sys.modules集合做身份：driver先导入gate/工具，与新进程模块集合不同，导致新PID恢复拒绝。该真实失败未跳过新PID测试，也未弱化源码字节比较。最新identity改为当前整个search_r1/verl目录全部.py文件的确定性相对路径→全SHA库存 +Git HEAD；实际已导入模块仍逐项检查来源，但其不同集合不作为身份需相等。所有恢复仍要求完整源码库存/actor config严格一致，并在model.load之前拒漂移；SHA库存包括OPSD actor/teacher/controller/math/loader，先前“额外模块集合也拒”的描述仅属于已被集成反例否定的旧设计，不能用于最新验收。

生产E1 trainer在init_workers前传播 `actor.e1_import_checker={checker_path,checker_sha256,closure_path,closure_sha256}`，每rank实际source capture调用 `verify_rank_import_checker`：checker只能来自当前repo固定script物理路径，closure是owned物理外部file、两全SHA与uniqueJSON，closure自pin同checker；以同一已hash源码bytes编译执行而非pyc，并调用strict verify_imports(required=())核该rank实际loaded modules/function code。输出 `E1_RANK_ACTUAL_IMPORT_CLOSURE_JSON` 含实际PID/UID/rank、checker/closure SHA、actual modules和verified_function_count。不能把driver导入证明代rank。CPUtoy/native小模型不传此field的state oracle仍无生产bytecode证明；生产E1实际字段必由固定trainer传入。

最终6文件集成121 PASS / 57.27秒（gate/bootstrap/prebuilt/profile/fullstate/backend），raw `state-rank-live-source-integrated-cpu-tests.txt`，保留先前活动checker117PASS+1 imported-dataclass误拒 raw `state-rank-live-source-imported-dataclass-failed.txt`。classalias修复后，实际source checker执行、wrong checkerSHA/closureSHA、自pin缺失拒、实际新PID第6Adam oracle均通过。源码库同等但不同import集合可恢复，不伪造GPUrank：接口测试rank0只是CPU调用参数。原106修后PASS50.02s raw保留 `state-source-inventory-integrated-cpu-tests.txt`。

最终checker依赖已由其作者独立冻结：profile manifest `67598a2c0036e168cf1364d49f2357b221d887185966c60412672ff28daf49f0`，scriptSHA `04304d563a71b3123e0da6c5350398bf0b8510e0b477ce47742798214bd5b3f0`。当前仅本地CPU与编译；整个3B actual500/HF/NCCL/native8rank/newrestart授权/first5实际paused签名/300与final51713依旧没有验收，default原位pause5与新restart授权严格分开。
