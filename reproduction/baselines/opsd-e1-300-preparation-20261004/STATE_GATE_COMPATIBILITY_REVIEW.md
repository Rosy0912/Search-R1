# OPSD official-Instruct300 完整状态／原进程第5轮门禁兼容性独审

2026-10-04。读取本任务REQUESTED_CONFIG_AND_COMPATIBILITY、现有OPSD actor/controller/ray_trainer/fsdp_workers以及邻BAPO outputs/e1-300最新 FULL-STATE、SHARDED、COUNTFIX、DEVICEFIX、DEVICEFIX-R2 CPU launcher独审及实际source模块。本轮仅本地只读，不改source、GPU、训练、上传或Git。**当前OPSD不具备用户所要求的完整状态保存/真实恢复/同进程pause5门禁；不能凭旧HF5证明兼容。** 算法目标冲突由根任务询问用户，状态准备不预设其选择，也不擅加PG/reference KL。

## 当前代码的具体差距

1. `verl/workers/fsdp_workers.py:514-547 save_checkpoint`只收集FULL_STATE_DICT rank0并save_pretrained/tokenizer，随后barrier和offload；TODO明确未支持sharded checkpoint。没有Adam param_groups/moments、scheduler、rank RNG保存或load_state_dict恢复。OPSD_CHECKPOINT_RANK_AUDIT仅设备/teacher审计，不是state恢复证明。
2. `ray_trainer.py:696-710 _save_checkpoint`调用HF导出和controller.save。`controller.py:307-321` JSON含method/global_step/BAPO stage/teacher路径，明确 `resume_supported=False`。路径不是teacher原始模型SHA；保存一个默认BAPO BoundaryStage也不是完整OPSD算法状态。
3. `ray_trainer.py:737-752`fit从global0初始化验证再增1；:775-776遍历普通loader。没有可序列化sampler顺序、cursor、generator和epoch，也没有恢复driver RNG。不能从HF目录推算已取数位置，更不能重新开进程跳过若干batch冒充连续。
4. :941-948只有save_freq/acceptance_step/final触发保存；:955-957指标后直接递增outer并迭代下一batch。**acceptance_step=5只额外保存，不暂停**。smoke_only/totalsteps6会return退出，不满足同训练进程保留所有对象并在receipt后原位继续。
5. :959终止条件使用次步计数>=total_training_steps。现框架若配置300则实际只有299 outer；用户要求300需清楚定义completed1..300/next301，并在最终验证日志与保存模型仍标actual300。不能以display counter301宣称又训练一次，也不能把first5单独另起一次后再跑300。
6. `actor.py`OPS Dmini逐token denominator和空mini skip是算法行为。新globalmini128产生每rank16行mini、320行/rank=>20予定Adam调用，但只有全部20非空才实际20。不能为了验收计数去伪造空mini optimizer step；需要实际global supervised token及skip日志证明，目标冲突未定时不能改归约。

## 邻任务可复用的是源模块／机制，不是其训练状态或验收

邻最新source链里曾真实发现 LOCAL_STATE_DICT/DeviceMesh不兼容3147、CPU scalar/NCCL失败3152、multi-visible CUDA路由失败3156。后续review显示这些已在软件级修正。不能从早期FULL-STATE报告复制LOCAL实现，也不能把旧component GPU成功签到新source。

* `search_r1/baselines/full_state.py`：通用rng_state/restore_rng（driver CPU、不枚举CUDA；worker只当前rank device）、DTensor内容/shape/stride/placement/mesh exact digest、atomic save、StatefulSampler、明确warmup steps解析。可在新source固定SHA后移植并独立测试，不改变OPSD损失。StatefulSampler要保持原RandomSampler generator=None的seed消费和顺序；num_workers0避免未记预取，driver RNG和sampler generator分别保存。
* `fsdp_workers.py:save_training_state` 与实际native fixture：采用公共SHARDED_STATE_DICT+CPU offloaded DTensors，保持FULL_SHARD/DeviceMesh；每rank原生model、Adam、scheduler、current-CUDA/Python/NumPy/CPU RNG，原对象load并重新digest。不是用HF模型再建Adam。需适配本OPSD actor/teacher offload和函数路径，不以BAPO payload替代。
* `checkpoint_gate.py`：可复用外部PASS consumer、缺receipt阻塞、错receipt硬失败、source/config/nonce、distinct rank_N.pt+报告、driver.pt真实digest、当前driver PID强绑定、live log前缀不可变等机制。**它目前含 `bapo/actual_outer_updates`、BAPO_RUN_ID及BAPO阶段driver状态**；必须在OPSD新工具里明确算法schema/实际counter，不能只改目录或method label通过。receipt不能由等待进程自己产生PASS。
* COUNTFIX的shared global_sum_count：NCCL时只把count scalar迁当前rank CUDA，Gloo迁CPU；保持SUM/world/empty语义。现OPSD `global_token_count`会在CPU incoming DataProto上sum后直接all_reduce，存在同样CPU/NCCL场景风险，需审查与同源修正，不先将整batch搬GPU掩盖。
* DEVICEFIX显式torch.cuda.current_device路由：actor/update/old/refworker和denominator应核实际rank logical device；Ray一可见卡logical0与torchrun多可见logical3/7不能混。移植仅路由修正，不能偷改OPSD token权重/PG/reference math。
* R2 CPU launcher import隔离：repo优先、前后module路径+关键function semantic code digest，能防旧installed package同filename冒充新实现。新OPSD fullstate/gate/actor/distillation/utils/source都需同样实际进口闭包证明；BAPO源hash、jobIDs、model/data SHA和review receipts一律不复用成OPSD证据。

邻报告只批准其固定source/CPU实现，不替OPSD目标Linux/Torch2.4、8rank、Instruct模型、完整state、first5或300效果作证。保存不同实验的BAPO patience/alpha/stage/ranker状态到OPSD文件没有恢复意义，应使用明确OPSD schema。

## 新OPSD必须实现的状态内容与顺序

### Rank state

每rank独占新路径保存：SHARDED/DTensor model、完整Adam moments/step/param_groups、scheduler计数与LR、Python/NumPy/TorchCPU/currentrankCUDA RNG、rank/UID/PID/start/source/config/world/mesh身份。payload版本必须拒旧LOCAL格式、错rank、重复文件与zero DTensor。保存前后按既有manual offload装载/回迁完整恢复设备与optimizer状态，参数/Adam/scheduler对象identity保持。

第5真实从磁盘加载到同对象再比较全部dtype/shape/shard metadata/content/RNG精确digest；不是仅文件sha或state_dict长度。保存顺序不得让roundtrip本身额外走scheduler或Adam。HF导出另保留官方Instruct tokenizer/config/全shard完整SHA和loaded finite/tied/参数变化验证，不代替rank state。

### Driver与OPSD state

当前actual outer、next6、epoch、sampler顺序/cursor/generator、driver CPU RNG、实际算法方法/配置、 teacher官方资产与固定内容SHA/冻结/精度/执行device/storage policy、OPSD累计rank审计计数按是否影响训练明确保存。teacher是固定且eval的资产，可通过严格原内容身份重建/保持，不需要假装有EMA/ranker/BAPO动态权重；若有动态状态新增，则必须真实序列化。老师不能包含optimizer、梯度或可训练参数。

待定的ref/PG目标若用户选择新增，需明确reference是独立fixed官方Instruct且完整资产/不可训练；不能先创建另一个可更新teacher或用OPSD KL混称ref .001。vLLM/rollout涉及的真实随机来源也需核清：Torch currentrank RNG/engine显式generator/seed及driver sampling/continuation。如果有不受上述RNG覆盖的活跃采样器状态，必须保持其同对象并证明pause/restore不消耗或改变；现源码没有完整engine恢复保证，不能扩大为任意restart。

### Gate

第5完成update20（实际）、dev/stage按预算、HF+完整state、全rankroundtrip、完整metrics打印与immutable step5 log快照后，在**第6次loader取数前**阻塞。保存driver已消费5batch的cursor并在samePID观察不变；外部独审绑定当前PID/source/config/run/nonce/5/300/57、全部native文件/报告、HF全量/资源/检索/finitegrad/模型变化以及原log前缀。外部receipt正确后samePID继续且只取第6batch；保存/pause不能额外改变RNG/游标。

总horizon明确actual1..300，outer scheduler一次且57warmup，不以20Adam误走20次scheduler。保存点5、25、50、…、300；结束不能301。dev初始化/per50是512，stage是固定64，final51713，不能把旧OPSDvalidation64当新dev512。共同dataset manifest禁止重划/重采样。

## 最小独立验收序列

1. 新本地实现CPU：真实小model/Adam/scheduler/RNG序列化原对象roundtrip与第6更新oracle；sampler save/restore下一行一致；实际trainer尾部AST300轮oracle含保存点/pause顺序/最终300；有效receipt基线和coherently-rehashed source/config/rank别名/driverPID/counter/path篡改负例。
2. 目标Linux导入隔离全回归+纯CPU官方资产/data actualSHA/schema/count以及Instruct tokenizer/template边界，源闭包全部冻结，不热改旧8e/3007。
3. 独立新自有8H100 hardware：生产匹配nested FULL_SHARD/BF16compute/FP32 persistent+moments/manualoffload，actualcurrentdevice,countcollectives与sameobjectSHARDED save/load，第6 no-reload oracle与reload更新模型/Adam/scheduler/RNG完全一致；本任务OPSD fullvocab loss/head/masks/teacher CUDA/FP32/冻结参与，不仅BAPO PG小模型。真实容量/finite/padding等价/检索和资源在本节点验证。
4. 用户算法冲突明确后冻结resolved新Instruct300目标与配置，再启动一条真实300。first5暂停独立HF+native全rank真实审查、samePID release和第6游标一致；真实300/periodic/final51713与终态成本另验。旧实际3007八十、HF5和3146分布诊断均不计新300轮数或保存验收。

当前可继续无冲突的资产/数据/状态机制准备；**未实现的fullstate/gate不允许直接训练long300再靠checkpoint5补救**。此报告只列兼容性和实现要求，不批准启动或代用户解决算法定义冲突。
