# OPSD runtime观测与外部验收脚本独立复核

日期2026-10-03。基准HEAD `98c9661b030d8c2dcded1a298089ff3c83ab9292`，本次审查其尚未提交的teacher audit/CUDA peak观测差异。外部只读脚本为当前线程 `work/runtime-20261003/retrieval_compare.py` 与 `resource_sample.py`。仅新增本审查目录文档/证据，未改实现/脚本、未启动远端/GPU。

**结论：冻结状态检查合法路径不改变loss/gradient，非法状态拒绝测试通过。当前新增逐rank峰值还存在真实metadata收集缺口，不能称全部rank已可观测。检索比较确为真实E5/FAISS，但自动阈值不等同top3/top50内容与顺序完全一致；资源脚本还需明确作业cgroup和CPU服务归因。本报告未收到真实五iter日志，不认定GPU、检索或五步接受性通过。**

## 1. 数学等效和teacher audit

重复独立新旧loss比较：同一tiny Qwen参数/答案/mask，loss bitwise相同、15个student gradient tensor bitwise相同，teacher仍无gradient。当前loss函数没有新增audit计算；actor每outer结束才查询审计/峰值。观测 API 仅读状态，不加入图或optimizer，合法冻结状态不会修改参数。定向真实tiny更新和fake-teacher空batch回归 **2 passed/2 warnings（2.06s）**。

独立实际teacher audit反例：trainable参数元素数、存在grad的tensor数、teacher identity出现在student_optimizer group中的数正常均0；分别人为requires_grad=True、手动非空grad、加入optimizer时均RuntimeError拒绝。审计使用Python parameter object identity，适合独立teacher实例，不证明权重未被`.data`等其他路径更改；初末asset/hash仍需记录。teacher membership检查在outer后，不应宣称启动前已阻断所有非法状态。

证据：`evidence/runtime_delta_equivalence.json/.stderr`、`runtime_delta_pytest.txt`、`runtime_audit_probe.json`。这些是CPU/tiny证据，无GPU函数真实执行。

## 2. [R1] P1观测完整性：DP metadata仅保留首rank

`update_actor`注册DP_COMPUTE_PROTO；`collect_dp_compute_data_proto`最终调用`DataProto.concat`；`protocol.py:515–537`明确仅使用`data[0].meta_info`，SP1 sharding_manager.postprocess_data不聚合metric。因此actor在每worker分别添加`.../rank_0`到`.../rank_7`，driver只能获得rank0键，其余rank峰值被丢弃。独立CPU用两个真实DataProto重放concat，确实只有rank0峰值与rank0teacher cost。

**纠正此前报告：现有reduce_metrics对收集到的rank0局部list求均值，而不是8rank均值。** teacher per_rank后缀描述本rank值，当前driver实际首rank；不能乘8当真实各rank合计，也不能称RSS为rank均值。此前独立报告§7/§8和主REPORT里“rank均值”的推断缺少RPC收集核对，现明确撤回。

建议在OPSD worker同步收集观测（all_gather_object后每worker返回相同metrics dict），或写各rank独立可审计记录，不改全局DataProto.concat行为。必须保存各rank audit/cost/peak并检查8rank齐全，aggregate可另算max/sum/mean并命名准确。teacher违规rank会抛异常，但rank0审计0不足以代替全rank审计通过。已告知执行负责人，本项待最终修复复核。

## 3. CUDA allocator峰值与NVML采样口径

`cuda.max_memory_allocated/reserved`默认current device、无reset，表示该worker进程PyTorch allocator从开始到观测点的lifetime高水位，合法且不修改loss；不是每outer独立峰值、整GPU显存峰值、所有进程GPU资源或包含CPUteacher RAM。reserved包含缓存空间，allocated表示tensor分配，两者不可相加。

观测在update_policy末返回，step5保存发生其后；五outer短跑没有第6个update回传，因此最后一次allocator观测**未覆盖checkpoint保存峰值**。如验收要求覆盖保存/重载，必须在其后补实际各rankallocator观测或明确此范围缺口；NVML外部采样可补显存但不是PyTorch严格峰值替代。

NVML/nvidia-smi各时点memory.used包含分配卡上全部GPU上下文/进程（含vLLM、已有低占用任务），按UUID/PID/owner/job归因后才可标本任务。名义2秒采样最大值是**采样高水位**，可能遗漏瞬时峰值；不能声称精确allocator峰值。脚本每轮nvidia-smi两次查询各timeout8秒，再sleep2，所以实际间隔可能远大于2秒；sampling_seconds=2应解释为名义sleep值，以真实timestamp计算观测间隔。

## 4. retrieval_compare.py：真实执行范围与阈值限制

脚本使用同本地E5、mean pooling/max256、FP16 CPU/GPU Encoder；同本地FAISS index，CPU检索对照8可见GPU分片FP16 FAISS；固定NQ/Hotpot训练首/中/尾共6题和2个canonical queries，非最终集调参；记录完整top50 IDs/scores，top3/50内容SHA和有序内容相等行数、overlap、embedding差。序列符合PLAN真实CPU检索与原GPU参考比较要求的一部分，没有mock/BM25。

1. **阈值不能认证顺序/内容完全一致。** 硬assert仅平均ID overlap≥.95及排序位置score delta<.01；top3在8题×3=24个位置中允许1个不同（23/24=.958），top50允许20个不同。ordered_content_equal_rows和content_overlap只是输出，没有拒绝。所有query的top3顺序互换、score近似相同仍可能通过。必须人工/独立逐query查看ordered_content_equal_rows和差异，发现不一致按PLAN记录并处理，不能把OK标志写成原检索完全等价。
2. .95/.01是脚本新定的工程容差，不是PLAN已规定的阈值；应明确为数值检查初筛，不能用其静默接受任务改变。比较不同ID的同rank score差不等同同文档score误差，需对齐ID后解释。
3. 比较的是in-process Encoder/index，不调用127.0.0.1:18084真实HTTP服务。仍须验证运行server flags（device/FP16、资产hash、CPU index、topk）及同query的HTTP返回内容/排序/scores，不能拿脚本对比代替实际服务验收。
4. faiss.get_num_gpus()==8和index_cpu_to_all_gpus会访问全部8可见GPU，必须在本任务授权allocation且**训练尚未启动**时运行；检索参考process退出释放GPU、启动前重查occupancy再训练，不能与学生/vLLM重叠占卡。CPUteacher/训练过程中只用CPU检索。硬8卡不支持4卡备选，真切备选时须明确适配并记录，不绕assert。
5. timings混入encoder加载，GPU搜索未另显式synchronize（返回numpy通常已使结果ready，但应基于实际实现核对）；单8query非服务持续吞吐/并发/RAM压力验收。CPUFP16是否在固定Torch2.4真实Linux正常、内存是否足够必须实跑，当前CPU tiny不证明此脚本可跑。

## 5. resource_sample.py：采集范围及归因

- GPU query和process query成功时保存UUID/index/占用及每GPU PID的UID/RSS/VmHWM/start_ticks（可识别PID复用），失败会记录query_error，未伪造零值；有时间戳，可作为外部采样原始记录。
- cgroup_paths来自`/proc/self/cgroup`，即采样器，不是指定parent。只有采样器在同一授权srun/job memory cgroup中启动时才可作为本训练作业RSS/peak/limit/events；若外部SSH/tmux启动会读错误cgroup而不报错。应根据目标parent或实际已记录job cgroup交叉核对，记录有效memory.max/limit值；不存在文件被静默略过时不能假称已采cgroup峰值。
- per-PID usage只调用于nvidia-smi GPU apps。**CPU-only检索服务不会出现**，无法产出retrieval CPU/RAM成本；actor CPUteacher只因同actor持GPU才被记录，teacher全部加载阶段可能尚不在GPU app清单。必须显式记录本任务CPU service及actor PIDs的/proc和CPU时间/吞吐，或另外提供对应服务观测。
- `in_slurm_job`用substring `job_<id>`，可能job12误匹配job123；GPU可见进程PID宿主/namespace映射也须实际核对。需与完整cgroup组件/jobID和UID/UUID对齐，不仅依赖该boolean归因。整个GPU used不应全部归属于训练。
- os.kill(parent,0)只判断PID存活，不核对parent start_ticks；父进程退出后PID复用可造成采样继续。当前采样未捕获PermissionError（正常同用户一般无此问题）。父/服务shutdown后没有最终强制采样，终轮/重载短峰可能遗漏。这些是采集可靠性限制，不是训练loss缺陷。

## 6. 放行边界

合法teacher audit/CUDA allocator读取delta数学等效通过。R1多rank收集必须补齐才能声称逐rank观测达标；保存后峰值范围、真实HTTP检索对照及采样归因需在验收产物明确证明。检索差异未看结果前不称容差已可接受。主任务仍需至少五个真实3B/8rank search outer iter、step5可重载checkpoint、各rankfinite/非零后续更新/teacher冻结及真实资源记录，再交独立复审；本报告不是该真实验收完成证明。

## 7. 后续修正增量：完整rank观测与保存后日志

负责人已在actor outer末对teacher/cost/audit/CUDA峰值标量执行all_gather_object，所有worker将完整`.../rank_N`字典带回，原loss/gradient本地列表保持原样。独立真实两进程CPU Gloo运行实际actor的全空监督分支、收集不同rank的teacher计数与CPU区间值，再经真实DataProto.concat：rank0 legacy cost=1、rank1完整key=2，两rank键存活，参数不变，optimizer仍跳过。**R1关闭**，证据 `runtime_rank_gather_probe.py/.json/.stderr` 和 `runtime_rank_0.json/runtime_rank_1.json`。该测试未执行CUDA/NCCL、正常非空GPU backward；实际8rankcollective与合法teacher审计仍等五iter日志。

旧未加后缀的teacher成本在driver仍是rank0，新增完整rank键才允许实际sum/mean/max。actor/loss等旧指标仍保持已有rank0本地列表的统计口径，本次没有改变训练或全局DataProto行为。

保存checkpoint后，worker在全rankbarrier及offload完成后打印`OPSD_CHECKPOINT_RANK_AUDIT`（rank、实际path、lifetime CUDA allocator peaks、teacher审计）。静态位置覆盖保存阶段，截至打印点的峰值范围准确，未改保存数学。最终必须读取Ray原始worker stdout/stderr确认rank0–7齐全，不能仅依赖可能dedup的driver console；该日志不覆盖随后独立HF重载进程峰值，重载采样另列。

## 8. 资源脚本增量及独立race反例

已改为读取parent cgroup，添加job_processes遍历/proc记录CPU-only服务RSS，nominal_sleep_seconds明确只是名义2秒；job判断改成路径组件精确regex，job12不再匹配job123；加入utime/stime ticks与SC_CLK_TCK，可结合PID/start_ticks计算CPU秒。主路径stat字段索引11/12/19分别对应utime/stime/starttime，独立fixture核对正确。实际必须确认parent就是作业内目标、job cgroup路径格式与regex一致（脚本只识别`job_N`组件，不自动证明任何systemd命名格式）。

**[R2] P2采样race，初次新增CPU ticks版本已复现**：usage捕获/proc访问失败后又在except里重新读取/proc/pid/stat；消失PID导致第二次FileNotFoundError逃出，job_processes遍历可中断整个采样。独立AST执行已复现此失败并通知负责人。except直接返回pid/unavailable、不再读/proc即可；此项应在最终版本关闭并复测后才称CPU进程采样可靠。

检索threshold/content-order/HTTP范围和非严格NVML采样峰值的限制继续有效，尚无真实结果可审。新增逐rank观测与CPUfixture通过仍不等于真实3B五iter验收通过。

## 9. 最终脚本/启动协议复核：此前发现已修正，真实GPU验收待执行

重新读取负责人最终 `job.sh/http_retrieval_check.py/prepare_assets.py/model_preflight.py/resource_sample.py`，Python AST解析及bash -n通过，未执行GPU脚本。

- **R2关闭**：usage except现在直接返回unavailable，包含IndexError，不再二次访问/proc。独立原消失PID反例重跑通过，正常tick/RSS/starttime及精确job12/job123归因亦通过，证据 `resource_sampler_probe.py/.json`；保留初次失败记录，没有隐去失败历史。这只是模拟/proc fixture，不是Linux采样实跑。
- materialize先verify官方压缩/分片资产，再按固定verified源流计算expected SHA/size，已有dest或新staged必须逐字节hash/size一致才接受；新staged验证后replace，修复“已有错误dest也新记录hash通过”的问题。JSONL/FAISS真实文件大小/hash、读取/内存可用性仍等远端产物。
- HTTP服务除top3/top50有序内容hash一致外，新增逐位置score finite及CPU参考score差<1e-5检查；仍须取得真实输出，CPU-vs-GPU参考的content/order明细审核仍必需，.95初筛不改成完全等价认证。
- job要求Slurm JobID及CPU_PER_TASK≥96，Ray80CPU给teacher/student的线程预算、检索8及采样保留余量；先检查18084/16384端口空闲，保存自己新启动retriever PID，HTTP通过后确认PID存活、属于本job、ss listener精确绑定该PID。不会借用或结束现有他人服务。
- 真8rank NCCL/BF16、Flash forward/backward、3B vLLM推理都在授权job上下文执行；model预检后再查GPU双阈值；GPU FAISS overlay只给参考比较python进程的PYTHONPATH，训练仍CPU FAISS。GPU参考比较结束后launcher还会再次检查occupancy/租约，不能绕过残留占用。
- job训练使用全问题batch512、n_agent5、mini256、globalmicro8，total6/smoke_only=true；val_before_train=true与test_freq5提供原validation64的训练前/后测量；它不是七集51,713题正式效果。resolved Hydra、固定commit、uv实际env freeze、日志和退出状态写独立run路径，pipefail保持训练错误状态；不要求env必须有pip模块。
- step5保存后每rank worker audit/allocator日志已静态核对，需原始worker日志确认8rank齐全。HF重载仍明确在worker退出/Slurm清理后新进程或另一个授权allocation执行；job training-finished标记只说明训练进程退出，**不说明五iter/重载/最终独立验收通过**。初后student weight hash差、teacher初末immutable hash及实际reload推理仍是最终验收证据。

负责人报告此次完整CPU gate139passed/1skipped；此处只标为负责人报告，独立本轮实跑为2项focused、真实2rank CPU Gloo观测collective、tiny数学等效与资源函数fixture。**最终观测delta与所读启动脚本在所审范围内通过，无剩余已证实的新算法或脚本启动阻断。** 检索一致性实际差异、Linux8rank NCCL/Flash/FSDP/offload/3B吞吐、端口/cgroup真实格式、峰值、连续五outer和HF重载仍PENDING，必须以实际日志和产物复审，不能用脚本静态通过当GPU完成。
