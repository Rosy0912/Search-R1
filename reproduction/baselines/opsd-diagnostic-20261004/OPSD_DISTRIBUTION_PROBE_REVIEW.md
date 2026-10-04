# Native binding门禁已补齐

已独立读取`work/opsd-distribution-be413032-native-binding-cpu.txt`，固定be/source/base-pins/binding实际SHA匹配，真实nativeCPU完整binding PASS ae51e1648e990afb9da3dd24d7889e61ef3da40a145f6d27a1687aa0ece07815。旧96统一435门禁FAIL保持；新be native通过，不追认旧版。单GPU外围独审另见[OPSD_DISTRIBUTION_WRAPPER_REVIEW.md](OPSD_DISTRIBUTION_WRAPPER_REVIEW.md)。实际GPU/完整replay尚待执行。

---

# Role-specific tied-serialization 修复独立确认

**新probe `be41303243221b8616d485bcfed5e7797ec6fecc775e91d682a97cb545533737` 静态/CPU通过；实际native全资产binding重验由root进行中，须真实通过才进入GPU。旧96失败保持，不覆写、不追认。** candidate manifest `7c17ab430ee1e2cf30d550f6b23dbcedbc781442fdaac1aeba1c02ee01a698de` 所有文件实际SHA相符；tests `8335b89af398adfacb2627e162793c9e5534d596cadccd9621be94005f63ccd8`。独立 **49 passed /3.67s**。

独立解析实际native header证据`work/opsd-distribution-actual-model-headers.json`（SHA `0007ddc31ef12394b536da13c5b07f493a515e722bd5de14e2b8c90467cd2359`）：base index/header均434，serialized numel3,085,938,688；HF5均435，serialized numel3,397,103,616。两配置tie_word_embeddings True；实际keyset差集恰lm_head.weight，未少其它key；差311,164,928=151,936×2,048，是重复embedding序列化别名，不是额外独立模型参数。

新validate_role_index严格列举36layer/12项每层+embedding/norm，base434，HF435只增加lm_head。独立把两份实际header keyset送该函数通过；CPU错key置换保持count不变、缺HF alias、base额外alias等负例拒绝。固定原asset/fullshard SHA/indexSHA约束保留，未通过宽松keycount放行任意缺项。collect要求真实config tieTrue且head.weight **is** embed.weight，记录并由replay验证tied_embeddings；exact unique parameters3085938688与finite/完整loading继续强校验。

此前把HF序列化key数硬套base是独审静态遗漏；mock replay不检真实资产，因此49CPU也只能验证修复逻辑，不能替代正在进行的native binding。新实际binding/加载/runtime/GPU收集/同版本CPU replay/Slurm终态未到，不能认定分布诊断已完成，更不能认定搜索效果改善。

---

# 真实native binding门禁失败后的状态修正

**96cfe版本尚不能进入GPU收集。** 已收到并独立读取`work/opsd-distribution-native-binding-cpu.txt`真实traceback：verify_binding错误地统一要求base/HF5 index435。前述CPU/static允许结论不能替代真实固定asset binding，此失败保留；等待role-specific fix与native CPU重验，不追认96实际通过。

index serialization条目与unique model parameters不同。历史真实base2933全参数gradient诊断实际count=3,085,938,688；真实HF reload2996 index435且tied_embeddings=True、all_loaded_parameters_finite=True、loading四字段全空。base若省略tied lm_head alias则434，而完整HF导出可含435；两者loaded模型仍共享embedding/lm_head同Parameter，其`.parameters()`只计一次。**因此精确parameter_count=3,085,938,688应保留**，不能通过放宽count解决index门禁。

修复需实际native读base/HF config与index差集，确认tie_word_embeddings、base仅省略lm_head.weight、HF相应alias与embedding相等及固定fullshardSHA不变，规范化keyset后严格相同。不能笼统接受任意434/435。CPU synthetic replay明确mock了asset校验，因此未覆盖这个真实资产序列化差异；本次native失败指出这一覆盖边界。

---

# 最后强化版本独立确认

**固定最终probe `96cfe2326cfea1e06c2793a9b4924382d938a511b1f33d86a8d539cfd527fc08` 静态/CPU通过，无剩余脚本阻断项。** 外部candidate manifest `9adc9834abaa4f08e819f2b510eb606da812a4efeff6c7e3c1ecd74ac796ee2a` 的全部4文件实际hash逐项相符；tests SHA `eccb1d2c964c80ff935b1ffe94972f01da273a0666c14eb0bb25ee9170fa1dfb`。独立CUDA不可见重跑 **47 passed /3.95s**。

加载检查已补齐：真实loadinginfo必须恰四个预期字段且全空；逐个loaded parameter检查finite，记录parameters_finite；精确参数numel=3,085,938,688，collect及replay同约束。模型加载后的dtype/device/freeze和actual attention backend检查保留。因此下面旧版本提到宽range/finite观测缺口已关闭。

新增完整CPU合成64×4 raw NPZ→manifest→replay执行，输出256 objective comparisons和256 conditional candidate rows；明确mock资产绑定/tokenizer、GPU前向与Slurm资源，不是GPU证明。缺一case、截断词表、置换prefix后重新计算SHA仍被semantic检查拒绝，说明不会只信文件hash。该测试与实际source数学counterexample、阶段/PCI/输出失败保护负例互补；并未声称真实模型logits可由mock核验。

审核结论仅准许继续准备用户授权的自有单H100只读诊断。bootstrap/launcher实际绑定、真实job/UUID、complete asset/train/source hashes、47tests原生环境和GPU收集/同版本CPU重放及终态仍需实际证据。审查者未启动GPU或修改远端/脚本/生产，也未提交Git。

---

# 最终增量结论（覆盖下面初版待修复状态）

**当前脚本静态/CPU审查通过，可在用户已授权范围提交新的自有单H100只读诊断；实际job/UUID/外部binding、收集完成及重放结果尚未验收，不是训练或效果PASS。** 最终审核probe SHA256 `b09b3138afc948b38b4bc69cc5e089ff1c43515609dfb6de794f777b0a19f791`；tests SHA256 `72e8fcfe42082782a62aafc0991d76204041de4432ecec5f53c11bddd6f450fd`。独立CPU重跑 **40 passed /1.25s**。

初版三项实际缺口已关闭：用已加载libcudart的cudaDeviceGetPCIBusId(0)与nvidia-smi PCI→UUID绑定，兼容Torch2.4且不靠可见序号猜测；validate_runtime核三阶段role/cases/dtype/backend、empty loadinginfo、冻结/设备、peak、初始/末尾与各阶段job/UUID/UID/cgroup/startticks；preserve_failure只写本次独占创建且当前job/uid匹配目录，已有output拒绝后不污染。CPU新增PCI正规化/ malformed、role/backend/dtype/freeze/UUID/jobprocess/startticks/peak/optimizer/缺loading权重等负例和旧目录保持不变测试。

模型加载使用真实output_loading_info并要求missing/unexpected/mismatched/error空；资产全部外部hash、原base/HFstep5固定shard SHA/index约束和既有独立HF全参数finite验收支撑无随机缺省权重。当前runtime参数数量guard仍是2B至4B而非精确3,085,938,688，且未独立保存all-parameter-finite布尔字段；建议强化，但不能仅因重复观测不足无限阻断已被固定全内容/加载证据约束的只读运行。最终实际阶段报告应核count确实3085938688，不能把宽range当完整性证明。若新增强化delta须重新pin脚本和CPU tests。

完整256case重放来源/训练行/64×4 count/词表/位置/候选ID/数组keyset校验已源码审查；CPU模拟前向验证每conditional position，纯stage负例通过。当前没有覆盖真实GPU收集后完整256case replay的实际执行，只有运行后才能完成此门禁。必须用同固定环境CPU重放（JSON数值exact），不得遇正常跨版本差异就静默放宽。

这只是64个**按原顺序选取的训练初始状态**，不能排除其它位置/晚期history裁剪或解释HF80（不存在）。只对首个accepted-answer别名候选打分，不是全部别名成功率。直接HF BF16模型推理也不能宣称与FSDP FP32 master+autocast训练前向相同。tag/候选完整条件概率可比较固定候选，不能当生成格式率或EM/F1。CPU归一化logit方向实验不是模型Adam更新。用户未授权重新训练，本通过不增加该授权。

执行前root仍需核自己的bootstrap wrapper持锁/传fd、新真实Slurm单H100/UUID、完整外部asset/train/source/probe SHA，绑定7ac候选与原8e历史训练身份，保留当前job failure/terminal/原始输出。没有提供bootstrap/binding最终bytes供本报告审，因此不能将本报告当该外围文件的SHA审查；脚本会要求实际新allocation。审查者未改script、提交Git、远端或GPU。

---

以下保留初版问题记录（已按上面状态关闭）：

# OPSD 初始状态分布诊断独立审查（候选待修复）

结论：**当前版本不批准真实GPU提交。** 现有17个CPU测试通过，不消除实际启动兼容性及重放身份缺口。只读诊断方向可行，必须修复以下项后增量复核。脚本当前SHA256 `34f8376faffbaf19d785ad3b1864384e61dc30378ee1ee2f44b053de9a4076d7`；未改脚本、未提交、未启动GPU/远端操作。

## 阻断项

1. collect:349直接访问torch.cuda.get_device_properties(0).uuid。实际固定环境PyTorch2.4；[PyTorch v2.4 primary CUDA binding](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/csrc/cuda/Module.cpp)不绑定uuid（当场核查）。这将在模型加载前AttributeError。应使用实际CUDA runtime logical-device PCI identity与NVML/nvidia-smi PCI→UUID核对，或同等真实device身份接口；不能仅凭CUDA_VISIBLE_DEVICES顺序推断。新增无uuid属性负例验证兼容路径。
2. replay:294–296只核initial jobid及final jobid/UUID，不核initial GPU UUID；更没有核m.stages及runtime.model_optimizer_steps。须将3阶段role/cases/dtype与实际loading/backend/full参数count/freeze/finite/device证据完整保存并验证；阶段ownjob/UUID/process/cgroup和85%峰值界限均要闭环。当前manifest里stages即使缺失/错误也不影响replay返回，不能称完整身份验证。
3. 目前17CPUcase没有完整collect-output/replay合成fixture及上述缺失证据负例。应至少补纯阶段validator测试以及case/prefix/vocab/身份篡改拒绝。完整原始数组重放不必真实GPU，但必须证明拒绝最小坏证据。

## 已审核正确方向

模型/训练输入/源码/probe外部SHA绑定，完整原始asset清单、固定base与HFstep5 shards及435key index，禁止HF80/500。输入是固定train前64，检查split=train/nq-hotpotqa及去重，不读验证gold。extract仅选定teacher_inputs/gjsd，已审padded替代依赖。以占位response定位teacher-prefix最后预测位置且排除占位，保存原prefixIDs。

所有GPU模型前向inference_mode，加载后eval/freeze，无model optimizer/backward/generate/RPC；CPU梯度仅saved logit vector。base/HF5 BF16Flash学生和base FP32eager teacher串行驻留，teacher legacy/user_context只改变其条件；不得称FSDP训练数值等价。

256case覆盖64×4，原始full-vocab初始分布、3tag和首个答案别名候选的每位置完整logits；可从固定token序列重算conditional logprob，不等于生成完成率或EM。候选seq始终在给定prefix下teacher-forcing，不采样。重放已检查cases去重/完整、prefix/source/训练输入重建、候选IDs、vocab shape/float32/finite、数组精确keyset，原始np.load禁pickle，数学反例方向明确。

概率/梯度scope仅初始state；CPU归一化.01 logit-space step不是实际SGD/模型更新。当前HF仅step5，不能解释step80模型分布或证明3007退化根因。tag比较必须按分词完整条件概率，不能把不同长度候选概率当统一准确率。

## 本轮执行

独立本地CUDA不可见CPU执行work/test_opsd_prompt_distribution_probe.py：17 passed /1.26s。该测试覆盖fixed源数学/NaNInf/shape/零梯度、两种teacherprefix、JSON/path/hash不可替换、train选择、禁optimizer/generate等，但不覆盖上述阻断项。

online candidate logprob和replay目前要求exact JSON数字；若在不同Torch/CPU版本重放可能正常末位不一致，应使用同固定CPU环境并记录版本，而非静默宽松忽略。GPU运行还需实际新Slurm job/UUID外部绑定、lease/lock继承、终态原始证据；不猜未来jobID。本报告待修复版本复核后更新，不提前批准。

## 追加输出边界缺口

main失败处理在collect拒绝已存在output目录后仍可能往旧目录写failure.json。必须仅在本次独占创建、当前job绑定成立的目录记录本次失败；已有结果拒绝后不得新增/改写文件。补已有output不变负例。该项也是提交前待修复项。
