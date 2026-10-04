# 2026-10-04 HKUST执行记录

代码来源：用户提供20261003-233328压缩包，301项manifest SHA256已核验；原样保存于archive，导入提交7ec86bd。训练适配另在control/build_runtime.py生成，runtime全文件hash及overlay.patch保存到每台任务receipts目录，原archive不改。

A节点hkust4 (xielab5, 10.123.4.11)，候选48GB4090 GPU 4/5/6/7。任务目录/工作盘：/weka_3/shaoshitong/search-e1-config-study-20261004。B节点hkust5 (xie-lab-4, 10.123.4.12)，候选48GB4090 GPU 0/1/2/3。任务目录：/home/shaoshitong/search-e1-config-study-20261004。卡位仅为实时空闲候选，每个CUDA阶段前重新验收；hkust5 GPU4异常不使用。其他进程不停止、不修改。

固定公开源模型Qwen/Qwen2.5-3B-Instruct revision aa8e72537993ba99e69dfaafa59ed015b17504d1；RUC-NLPIR/FlashRAG_datasets revision bcafb8dd07d453be3cbeeeb3f78be1841bddf92c；PeterJinGo/wiki-18-e5-index revision a4d31160a035f30764604f4827cd8f1d0315eb86。其余encoder/corpus修订及每项文件hash见远端assets/manifest.json。官方/镜像资产先下载到HKUST4，同HKUST局域网复制所需环境/模型/数据到HKUST5，逐文件SHA256校验；不接入其他集群资产。共享检索只绑定10.123.4.11:18000；仅公开资产中转只绑定10.123.4.11:18281。

每台使用自己的Python3.10 venv，不修改现有EditVerse环境。torch2.4.0/vLLM0.6.3/transformers4.46.3/flash-attn2.6.3/tensordict0.5.0等核心版本固定，同一wheel集合复制给B，两台pip freeze落盘。4090禁用NCCL P2P/IB，保留原CPU-offload和gradient checkpointing。所有Ray cluster强制local，临时目录任务私有，训练函数finally ray.shutdown，不执行全局ray stop。

启动命令：设置E1_BASE、E1_PROFILE=archive/r1、CUDA_VISIBLE_DEVICES对应上述卡，bash source/experiments/search_e1/control/pipeline.sh。flock防重复管线。自动顺序：等待环境与资产→冻结runtime→准备同一数据→单元反例测试→检索编码精度对照→真实两更新及显存调优→从初始权重正式GRPO1→独立64题测评→K5 fresh全训练集轨迹→OPSD1（保留基线adapter，按验证选优）→GRPO2→轨迹/OPSD2→七测试集完整评估。任一阶段非零退出停止，不覆盖已有runtime/rollout，不自动OOM重试。

GRPO容量调优每档0.4/0.5/0.6/0.7分别fresh开始、完整512×5轨迹真实两更新，保留完整201/501学习率schedule。下一档需按当前各卡峰值+缓存增量仍留6GiB。记录每2秒nvidia-smi样本，该峰值不是瞬时allocator最大值。独立vLLM推理采用0.85，因无FSDP actor共存；是否可用仍由真实阶段日志验收，不以加载成功宣称容量已验证。OPSD对loss和裁剪前gradient做finite检查，checkpoint完整key/shape/finite检查后才进入下一阶段。

公平性限制：两组共同采用严格模型来源最终答案评分，并记录历史reward旁路；共同CPU FP32 E5+FP32完整索引，启动时16题对照GPU FP16 top3并落盘，不能证明所有query等价。与旧曲线不做逐点等同。独立64题为新的固定分层样本，两组题ID一致，但不冒称旧R1原64题。单seed和配置组合差异只能诊断，不足以确证单一参数或稳定算法收益。

截至本记录写入：已推送原始代码；两台安装/资产准备正在执行，尚无真实GPU更新与训练评估结果。管线等待或安装完成不等于复现完成。

管线已于01:23:49 +08:00在两台tmux e1-profile-a-20261004/e1-profile-b-20261004启动，当前处于WAITING_FOR_ENVIRONMENT_AND_ASSETS，不是GPU训练已开始。安装发现当前PyPI pyairports0.0.1只有sample模块，与旧outlines所需pyairports.airports接口不符，因此改从原项目NICTA/pyairports提交9699eae8c6b03c97239018dea92e01daa8594936构建2.1.1 wheel，固定来源并保存源码SHA256。参见https://github.com/NICTA/pyairports 与 https://github.com/dottxt-ai/outlines/issues/1093 。此为兼容依赖恢复，不改变RL数学逻辑。

数据集Arrow、HF及编译缓存和临时文件均定位任务工作盘，避免hkust4根盘剩余约67GB被Wiki语料转换耗尽。训练管线只读下载好的本地资产，HF与datasets切换offline。


02:45更新：HKUST5所选GPU0–3的实际CUDA kernel探针在torch2.4.0环境中30秒强制超时（打印CUDA_COUNT 4但未完成分配/同步）；未进入训练。保留receipts/cuda-kernel-probe2.log及moved-to-hkust1.txt，停止仅本任务B管线，不重置驱动、不干预外部进程。B迁到HKUST1 (xielab1) GPU0/1/2/3，24GB4090，任务目录/data/shaoshitong/search-e1-config-study-20261004。现有只读torch环境的CUDA kernel测试正常退出；新的精确torch2.4环境仍需独立验收。使用ella Python3.10.19创建新venv，通过HKUST4私有公开资产HTTP复制同一wheels和模型/数据，逐项hash核验。

B仍为原Search-R1的四个rank、全局actor micro64及score128。24GB执行层设置E1_PHYSICAL_MICRO=4、E1_PHYSICAL_SCORE=8：逻辑micro16内每个物理子块的损失乘有效token数/逻辑micro有效token数，再除原gradient_accumulation；PG、entropy及reference KL共享mask，因此保留原token归约和optimizer边界。完整零mask逻辑micro明确报错；零mask物理子块仍执行forward/backward，临时使用非零归约mask后乘零权重，以保持所有FSDP rank的collective次数一致。tests/test_physical_micro.py直接提取归档与适配后的真实update_policy/compute_policy_loss/kl_penalty，使用不同mask长度、零mask子块、entropy及KL比较两次optimizer边界的梯度、权重和指标，CPU float64容差1e-12通过。真实CUDA显存及两更新尚待验收，不能用该测试代替GPU训练。

下载测速：HKUST1镜像128MiB索引range约11.4MB/s，HKUST1至HKUST4内网约117MB/s。剩余索引/语料range由HKUST1独立公开源下载，复用HKUST4已有32GB前缀及有效range；不跨集群，不重复下载完整资产。HKUST1目录/data/shaoshitong/search-e1-download-assist-20261004，仅公开range目录通过10.120.17.101:18282供HKUST4读取，逐块SHA256后组装，最后必须核验发布的整个LFS文件SHA256才允许assets.ready。个别慢range超时将保留partial后重试，不以分块大小宣称资产完整。

HKUST1精确torch2.4环境的GPU0–3与HKUST4 GPU4–7均完成CUDA/FlashAttention实际kernel测试；B环境及模型/数据完整就绪。四rank FSDP微型实际GPU检验通过（receipts/fsdp-physical-equivalence2.log）：直接调用原始与适配后的PPO update及原始FSDP optimizer step，各rank mask不同且各含零mask物理块，权重/指标float32容差1e-6一致，原始4次与物理8次forward均完成collective。此仍是执行适配验证，尚不是完整512×5 RL更新。首个测试使用torchrun默认TensorDict.cuda()映射到0导致rank设备不匹配；测试改为映射LOCAL_RANK以模拟Ray单GPU可见性后正常通过，训练代码未为此改动。

B首次冻结的runtime早于零mask同步补丁，尚在WAITING_FOR_RETRIEVER且未训练；已停止仅本任务进程组并将旧runtime及hash/patch保留为r1-unused-zero-mask-v1，重新冻结正确runtime。未热修改已训练模型或checkpoint。

公开range全部补齐并逐块SHA256就绪。HKUST4工作盘实际ST16000NM000J机械盘；合并改为直接流式生成e5_Flat.index，每个源分卷在写入过程中独立核验发布的完整SHA256，再原子发布索引。不另写64GB分卷副本，保留原下载前缀/分块供审计。manifest明确分别记录上游分卷SHA256和本地合并文件SHA256，不将后者冒充上游值。小文件真实字节拼接及源损坏禁止发布测试通过。算法/检索索引内容不变。


归档vendored vLLM0.6.3的KV容量计算实际为free_gpu_memory × utilization（不是整卡总显存比例）。容量候选扩展至0.8/0.9，仍需每档完整两更新、下一档保守按整卡显存增量预测并留下6GiB；实测不满足条件即停止增加。不会仅因0.7是预设上限而放弃实际剩余容量。

03:10三个上游完整文件SHA256全部通过（两个索引分卷与压缩语料）。最终索引直接生成成功。检索精度对照改为独立短生命周期GPU encoder子进程；CPU服务主进程只编码CPU/执行CPU Faiss，避免删模型后CUDA context仍占所选GPU而阻止GPU admission。子进程输出同16题FP16 embeddings，退出释放CUDA后由同CPU索引比较top3，比较数学不变。

03:20 A实际启动前admission拒绝GPU4约500MiB残留上下文；未生成RL更新。检索进程随原管线退出消失，B无法连接18000。修正为CPU服务CUDA_VISIBLE_DEVICES为空、只给短生命周期GPU对照子进程恢复所选卡；服务独立tmux e1-retriever-20261004，不随训练管线结束。A显式E1_REUSE_RUNTIME=1重启，仅在全部冻结runtime hash一致后复用，保持原runtime字节。原失败日志保留pipeline.log，新日志pipeline-retry1.log。真实阶段输出目录已存在时拒绝覆盖，需要先明确保留旧结果，防止重跑混入checkpoint。检索启动探测timeout提高至120秒以覆盖完整Flat索引冷查询。

03:32两组进入真实训练启动后，Ray的AF_UNIX socket路径因任务BASE+32位run_id+session路径超过107字节而失败，尚无更新。修复为/tmp/e1-<uid>-<run_id前8位>短链接指向任务盘BASE/ray/<完整run_id>，保留独立任务目录及碰撞检查。真实生成的allocated_init与实际Unix socket bind测试通过（路径<104字节）。仅执行路径变更，旧冻结runtime和smoke-040失败日志/receipt保留unused-ray-path-v1及smoke-040-ray-path-v1，再从已验证原archive生成新runtime；未原地覆盖旧runtime。新管线日志A pipeline-retry2.log、B pipeline-retry-ray-path.log，独立检索服务继续运行。

### Full-index retrieval bottleneck and common GPU execution

- Both fresh smoke-040 runs reached real search calls but waited at step 1;
  no optimizer update had occurred. CPU full-Flat batches took ~277 seconds.
- CPU FP32 encoder-only measurement: short128 queries 0.218s; long128 queries
  (256-token truncation) 6.64s. Keep that encoder unchanged.
- HKUST1 GPUs 4–7 were free (4 MiB, 0%); training B stays exclusively on 0–3.
- Reuse verified range shards on HKUST1, transfer only 34,015,805,440 bytes of
  original HKUST4 prefixes via LAN, validate each prefix SHA256, and assemble
  the full index with the published source-part hashes. No Mac asset transfer.
- Added full FP32 Torch Flat-IP sharding, hard query microbatch64, TF32 disabled,
  serialized shared HTTP service, top4 score/ID gate and concurrent512 peak gate.
  CPU tests on the exact remote environment passed2. Internal concrete-code
  review approved. Actual acceptance receipt is required before replacing CPU.

### Shared GPU oracle hardware acceptance (04:29–04:31 CST)

- All three complete assets passed their published hashes on HKUST1; the full
  corpus has exactly 21,015,324 rows, matching the index and original vector IDs.
- CPU Faiss vs GPU exact Flat on the unchanged CPU FP32 embeddings: all16 top4
  ID sequences identical; maximum absolute score difference 2.980232238769531e-7.
  CPU search43.3058495s; GPU search0.2361444s (same16-query input).
- Two concurrent512-query complete requests returned identical documents/scores.
  Torch peak reserved17,523,802,112 /17,548,967,936 /17,523,802,112 /
  17,532,190,720 bytes, against25,386,352,640 bytes/card. GPU4–7 only.
- Acceptance SHA256 b987f2d82e8499c0f562d0dc2eda9c941dcc80c2feb2e8f749cd27db900f24d6;
  asset-manifest SHA256 b1fe4576cd1d3314fe35fde3443cf63105110cf6dc9a71daa9e807f2cd2c4508.
- Both training arms recorded this exact common backend identity and began new
  smoke-040 real-update gates from the initial checkpoint. Old waiting runs have
  no actor/global_step checkpoints and were archived as cpu-retrieval-v1; their
  process groups were stopped only after verifying PID/session environment.
- The old read-only CPU service was replaced only after the GPU gate passed.
  Training B CUDA0–3 and retrieval CUDA4–7 are disjoint on HKUST1. Training A
  CUDA4–7 remains on HKUST4, whose CUDA0–3 foreign processes were untouched.
- Running acceptance still requires both smoke gates to produce changed finite
  step2 checkpoints, followed by full staged training. Initialization alone does
  not satisfy that requirement.

### HKUST1 physical cap refinement after real backward profiling

B's cap4 / score8 run reached actual backward passes but sampled peak24,183 MiB
on a24,564-MiB card. It was stopped and preserved as
smoke-040-cap4-score8-low-headroom before formal training. This was a proactive
headroom fix, not an OOM recovery and not a change to logical64/128 batching.
B now uses physical2 / score4, preserving exact logical token fractions and
optimizer boundaries. CPU actual-source gradient/weight checks passed2; all4
CUDA FSDP ranks matched original weights/metrics within1e-6 with16 physical
calls instead of4 original calls. A's ongoing run was untouched. Fresh B gates
use the same source/runtime/seed/initial weights/common oracle with new cap receipts.

### Full backward peaks supersede model-load estimates

A completed a real update/checkpoint1: strict EM0.195, finite grad norm2.316,
actor update481.792s, total step981.758s. LR0 is expected at the warmup start;
step2 actual parameter changes are still required. Its real peak reached48,126
MiB (49,140 total), so the run including checkpoint1 was preserved as
smoke-040-low-headroom-step1-saved, rather than promoted to a formal job.
B physical2/score4 reached23,017 MiB (24,564 total), preserved separately.
Both fresh probes now start0.25 and enforce6GiB headroom on accepted gates,
not just on prospective utilization increases. B uses physical1/score2, whose
four-rank FSDP proof matched the original actual update within1e-6 with32
physical calls versus4 originals. No foreign processes were stopped.

05:31 更新：下载和 SHA 校验已完成，A/B 共用已验收的精确 GPU Flat 检索后端。A utilization .25 的反向传播峰值仍为48126MiB，与.4相同；vLLM生成退出后释放cache，因此单降utilization不能解决actor峰值。保留失败运行与原669项冻结runtime/manifest后，两组改用physical actor1、score2，逻辑micro A2/B16及原归约不变。A trajectory-mean 四rank FSDP梯度/权重/指标等价测试通过，原4次forward变32次，容差1e-6；B token-mean cap1同样已通过。A/B实际训练于05:31进入epoch0 step1，尚未完成两更新验收。

B原smoke025在Ray worker初始化时SIGSEGV，GPU峰值仅4MiB，远端可用RAM934GiB，不属于GPU OOM。native栈为getenv→Abseil时区→gRPC StatusToString→Ray MetricsAgentClientImpl重试线程；原因仍不能断言是env竞争。任务私有Ray `_system_config.enable_metrics_collection=False`、启动前RAY_enable_metrics_collection=0/TZ=UTC；私有Ray+remote42小探针成功，真实四GPU worker已正常进入step1。不改变训练console指标、奖励或依赖版本，不操作其他Ray集群。

新增执行控制：两组按512条NQ/Hotpot训练留出集选择GRPO→OPSD权重，比较baseline与保存且已评估的50步checkpoint，EM并列选择更早权重；最终201/501日志映射到已保存200/500，测试64/51713绝不参与选择。四项计数/坍塌/最终步骤测试通过。OPSD另建冻结runtime，保持pair_loss逐字相同，分rank并行留出集评估并SUM整数correct/count；损失/梯度无穷统一all-rank拒绝，避免单rank提前退出造成其他rank挂起。七项分布式归约/不等尾batch/非有限值/原pair_loss测试通过。新增独立真实8192上下文OPSD更新与原生生成内存gate，synthetic输入仅检验显存和真实AdamW权重更新，不能作为科学EM结果；目前尚未运行通过。完整实验和因果结论仍未验收。

后续检查修正：`enable_metrics_collection=False`并不跳过Ray native MetricsAgentClient初始化；A新日志仍出现连接重试告警。实际dashboard_agent.log记录grpc/aiohttp_cors缺失、Loaded0modules，不能称为已消除该故障路径。补齐20个仅新增的Ray reporter依赖，使用原167项freeze为约束、dry-run确认无既有包被更换；所有22,639,229字节wheel按pip报告中的发布SHA256验证，再HKUST1→HKUST4局域网复制。两台ReporterAgent及optional_deps实际import均成功；既有Torch/vLLM/FlashAttention/Transformers/Ray等版本逐项未变。新增固定requirements-hkust-reporter.txt，后续bootstrap使用ray[default]。

B physical1/score2仍达22,941MiB，仅剩1,623MiB，故在提交正式训练前保留并停止自己的stage。定位fsdp_workers.update_actor在整个外层update开始即把Adam状态搬到GPU，直到全部10个mini optimizer步骤完才卸载；原optimizer_offload并不约束这期间的驻留。新增可选E1_JIT_OPTIMIZER=1：不提前加载状态；_optimizer_step完成原梯度裁剪后立即加载既有状态，执行同一个GPU AdamW，再用原offload函数卸载并同步CPU传输。没有改变LR、AdamW参数、步数、梯度、loss或逻辑micro。实际四rank FSDP连续4次更新的参数及每个Adam状态tensor逐位一致（atol0/rtol0），各rank CPU状态卸载验证通过，日志见receipts/fsdp-jit-optimizer-check.log。B原669项runtime/manifest/输出另存preserved/runtime-before-jit-optimizer、runs-before-jit-optimizer，再冻结新runtime并fresh启动smoke025；实际两更新与显存余量仍待验收。A原physical1测试继续，反向峰值已达35,932MiB，仍有超过12GiB余量。

A physical1 完成真实step1：strictEM0.195、legacyEM0.275、score_disagreement0.081、grad_norm2.208，finish_ratio0.984；gen416.779s、ref114.178s、actor912.930s、save21.418s、step1585.407s。global_step_1完整三分片已保存，首步LR0不能作为参数已学习的证据；当前进入step2。B新JIT配置fd4b10fb22ad4d228e0acad6f74c2a87已进入step1，dashboard_agent加载ReporterAgent及共7个模块，新的privateworker日志未发现原metrics连接失败。两组source控制程序继续更新，但各自运行中的669项runtime未被修改。

continue_after_gate.py等待明确tmux gate session退出，仅在run_id完全匹配、exit0、两更新checkpoint报告finite且changed、同run_id显存gate已accepted时接续完整管线。A等待913d684eeb7649a8ad8dbfa668ce7e38；B等待fd4b10fb22ad4d228e0acad6f74c2a87。失败不自动重跑、不覆盖原输出。当前等待并非完整200/500或OPSD已运行，更不是最终收益结论。各自独立tmux e1-accepted-continuation-archive-20261004 / e1-accepted-continuation-r1-20261004。

At 06:58 CST A completed two real updates at physical1/score2, vLLM .25. Step2 strict EM .175, legacy EM .268, grad norm2.005. All435 saved tensors finite; four q_proj tensors changed3.24–3.46e-7. The initial HF file omitted tied lm_head; the FSDP file materializes it. Validator allows only this extra alias when both configs tie embeddings AND saved head equals saved embeddings elementwise; all other key discrepancies fail. Four regression tests passed on the actual remote environment. Unchanged frozen runtime and exact run_id revalidated;6GiB reserve gate accepted. Continuation now tests .30 with two real updates. Full200/500 scientific comparison remains pending.

HKUST2 GPUs1/2/3/6 passed four-rank real OPSD8192-context/4096-response LoRA AdamW update plus500-token generation gate: finite gradients, changed LoRA B, reserved peak10.324GiB per rank. Synthetic memory gate provides no scientific EM. B JIT .25 completed step1 and runs step2;22941/24564MiB peak fails6GiB reserve. No formal B training promoted. Experimental entropy/logprob row checkpointing remains isolated from frozen training runtimes pending compiled four-rank equivalence and memory proof.

Download: HF mirror range transfers11.4MB/s, HKUST LAN117MB/s. Model/encoder/index/corpus original hashes verified. HKUST2 task-only direct LAN requests bypass its unavailable proxy variables; no global network changes. Portable Python headers were copied from the same version within HKUST and SHA checked after Python.h absence prevented TorchInductor compilation.

07:10 CST: Experimental row checkpointing passed all four actual FSDP ranks with compiled entropy and FlashAttention CE over full151936 vocabulary. Chunks7/16/128 match outputs, gradients and AdamW weights within1e-6. Additional CUDA allocation for257 rows falls468574720→322111488 bytes (31.3%) with chunk16. Nonreentrant checkpoint triggered Torch2.4 dynamic-compile FakeTensor errors; reentrant checkpoint passed actual compiled forward/backward and both shapes37/257. Optional E1_LOGIT_CHUNK defaults0 and retains original expressions; only row axis is chunked, full vocab and reductions unchanged. New runtime builder includes it and Ray forwards it; execution receipts and resume gates compare the flag. Existing A/B frozen runtimes remain untouched. B refinement supervisor waits exact old gate and continuation to exit, requires two completed finite changed updates AND measured headroom failure, preserves old run/runtime/manifest/patch, then builds fresh chunk16 runtime and runs new two-update memory gates before formal training. Failure stops the supervisor; no automatic retry.

Deployed refinement supervisor on HKUST1 in tmux e1-memory-refinement-r1-20261004, log logs/refine-after-jit.log. It is waiting for exact fd4b10fb22ad4d228e0acad6f74c2a87 and old continuation to finish, not modifying live runtime. A .30 gate remains running. Independent HKUST2 newly built row-statistics runtime verified304 file hashes and passed all7 actual reward/provenance/archive overlay tests. Latest source and four-rank proof pushed to search-e1. No notifications to the seven algorithm threads were sent, per user instruction. Full scientific comparison remains pending.

2026-10-04 09:55 CST status audit: both pipelines had stopped. A .30 completed two real updates, finite changed step2 weights, GPU peak35990/49140MiB; accepted. Next .35 admission saw1446MiB transient residual onGPU4 and rejected before a new stage existed. Selected GPUs4–7 now12MiB/no compute processes; explicitly resumed pipeline with unchanged runtime and accepted history in e1-resumed-pipeline-archive-20261004. B chunk16/score2 completed two real updates and finite changed weights, but actual peak23787/24564MiB failed6GiB reserve. The independent small tensor memory improvement does not establish improvement for whole RL training. Preserved its complete run and receipt in preserved/r1-row-statistics-score2-completed; started fresh score1 memory-gate pipeline in e1-score1-pipeline-r1-20261004, unchanged logical batches/objective/LR/optimizer and runtime. No formal200/500 result yet. Admission now retries only selected-busy/process rejection for at most180seconds in5second intervals; all UUID/process checks retained, other errors abort immediately, no processes terminated. This addresses stage-teardown residuals without weakening idle admission.

User acceptance tightened: do not yield until BOTH formal experiment runs actually finish at least5 iterations; smoke2/init/step1 is not acceptance. Prioritize formal training over additional utilization candidates after the exact live gate. Own tuner controllers1394861 (A) and2739918 (B) validated by /proc cmdline/E1_BASE/E1_PROFILE and SIGSTOPed only; child real-update gates continue uninterrupted. Dedicated supervisors await exact UUID442bccde5f484eee8f2fd620d9c4e68b /1132d114f04b4256bae134ad5866aed1, verify two-update finite changed checkpoint plus6GiB reserve and frozen hashes, append accepted history, terminate only paused own tuner after updates finish, and resume full pipeline with the single accepted utilization candidate. Validation failure resumes old controller and stops transition, preserving diagnostics. Live model workers/runtime are not altered. Logs formal-priority.log; tmux e1-priority-5iter-archive-20261004 / e1-priority-5iter-r1-20261004. Main agent remains active checking until both formal5-iteration acceptance.

Foreground monitoring found B score1/row16 still peaked22941/24564MiB during real backward. HKUST5 still exposes GPU4 N/A/unknown process, so it is not a usable replacement. Stopped only own B process group2739927 after matching /proc E1_RUN_ID1132d114f04b4256bae134ad5866aed1/E1_BASE and PGID; task supervisors exited, training GPUs0–3 now4MiB. Preserved complete partial run, runtime, hashes, overlay and failed receipt under preserved/r1-score1-before-chunked-adam. A gate442bccde5f484eee8f2fd620d9c4e68b continues undisturbed: step1 finite norm2.209, total1599.491s, step2 active.

New optional E1_OPTIMIZER_CHUNK=1 partitions only existing AdamW parameter groups during the SAME optimizer.step, preserving per-parameter state objects/hyperparameters and restoring original group identities before scheduler/zero_grad; CUDA foreach arithmetic remains identical, without simultaneous full-group temporary allocation. Empty only unused caching blocks before moment-state GPU load. Four ranks,8 nested FSDP parameter tensors,4 consecutive updates: parameters and every Adam state tensor BITWISE equal to original; actual additional CUDA optimizer-step allocation9438208→1049600 bytes. Separate row128 full-vocab compiled entropy/Flash CE equivalence passed all4 ranks;1025-row additional peak1868823040→1323685888 bytes. Using row128 reduces many tiny row16 dispatches while preserving reduction and tested tolerance1e-6. These isolated proofs do not themselves prove RL memory acceptance. Fresh frozen B runtime built and e1-formal-safe-r1-20261004 started with physical1/score1/JIT1/row128/optimizerChunk1/util.25. Single candidate .25 advances to local OPSD gate and formal training only if two real finite/changed updates AND6GiB reserve pass. Five formal iterations for both arms remain required before yielding.

A passed .35 exact gate:435 finite saved tensors and q_proj delta3.46–3.61e-7; both updates finite, step2 1636.848s. Local four-GPU OPSD8192-context gate passed; formal round-1-grpo started with unchanged200-step config at10:52 CST. Baseline heldout strict NQ.176/Hotpot.203 at step0; now formal step1 active. Initialization/validation still does not satisfy5 iterations.

B whole-state JIT/row128/optimizerChunk1 actual peak eventually23001/24564MiB, not a headroom pass despite isolated foreach proof. Stopped only own PGID2794374 after /proc runUUID d0cbf9ec3f7e4d9a9f5e003a8dee2054 check; retained run, runtime/hash/patch/receipt/pipeline.log in preserved/r1-whole-state-before-staged-chunks. Optional E1_STAGE_CHUNK_STATES=1 loads/updates/offloads states for only each parameter piece inside one outer logical optimizer boundary; original parameter-group identities restored before single outer scheduler step. Empty unused activation caches at physical-piece boundaries. Four actual FSDP ranks ×8 parameter tensors ×4 updates: parameters, allAdam states AND LambdaLR scheduler states BITWISE identical. Actual-source overlay AST passed. New B frozen runtime and e1-staged-state-pipeline-r1-20261004 now active; logs/formal-priority.log, physical1/score1/JIT1/row128/optimizerChunk1/stagedStates1/util.25. Record actual Ray-worker execution env and allocated/reserved memory before/after each outer optimizer call; never infer actual memory from isolated tests. Must pass two real updates plus6GiB reserve before formal training. Both formal runs reaching5 remains pending and main agent continues foreground work.

12:02 CST foreground check: brief HKUST4 SSH/SOCKS and same-cluster HTTP routing failure recovered without training restart. Both owned training tmux sessions survived. Relay health again reports the identical full index/source/acceptance hashes. A formal round1 completed steps1 and2 with finite gradients (2.139/1.904), strictEM.172/.155, now step3; observed peak39434/49140MiB. B staged-state gate remains in first update with four GPUs at100% utilization and sampled peak15533/24564MiB, not yet a formal iteration. Foreground monitor now retries transient SSH/read failures six times, without modifying or stopping training. Both formal5 acceptance still pending.

13:09 CST: B staged-state gate completed both real updates. Step1 strictEM.197/grad1.054/3186.790s; step2 strictEM.180/grad1.129/3211.295s. All435 checkpoint tensors finite; four q_proj deltas7.45058e-8. Actual PyTorch peak reserved16344MiB and sampled15683MiB leave over6GiB reserve. Local OPSD gate also passed and pipeline attempted formal startup. Meanwhile HKUST4/5 network became unreachable again: HKUST1 ping to10.123.4.11 returns TTL exceeded from campus10.120.42.212; local SOCKS and campus HTTP/SSH fail. Actual full GPU oracle onHKUST1 remains healthy with identical source and acceptance hashes. B initial formal run ab76f92c2fdf403ea34cea53f32d99fd failed during step0 validation due old HKUST4 relay URL, before any formal iteration. Preserved complete failure folder/log/receipt under preserved/r1-relay-unreachable-formal-init, changed only future control endpoint to direct SAME HKUST1 oracle, SHA-verified train_profile.sh/pipeline.sh, explicitly launched e1-direct-oracle-pipeline-r1-20261004 with E1_RETRIEVER_URL and resumed accepted gates; frozen runtime unchanged. A had completed4 formal steps before route failure; latest live state unknown until access returns. Read-only monitor retains CONNECTION_UNKNOWN and keeps checking B rather than exiting on an unreachable peer; unknown never counts as acceptance. Both formal5 criterion remains unmet; main agent continues work.

13:19 CST: new B formal UUID e2e33c84f289410384586e04b45e7497 is running; actual Hydra log contains direct10.120.17.101:18001 URL and contains no old relay URL. Initial heldout rollout is progressing. A remains unreachable from local SOCKS and HKUST1 campus route. Prepared relay retry for deployment after access returns: up to30minutes, identical payload bytes, identical backend, only transport errors/502/503/504; invalid payload/backend computation errors still propagate. Real HKUST1 oracle probe with an isolated first connection drop passed: two attempts forward exactly the original request bytes and return exactly the original top3/results/scores. No live A runtime/service was modified while inaccessible. Formal5 acceptance remains pending.

13:22 CST campus route recovered without a local VPN change. A was not stopped by the network incident: formal UUID f04fe593830c4831918ede872812c016 completed steps1–5 and is running step6. All numeric metrics for1–5 verified finite, process exists, receipt RUNNING with fresh heartbeat;669 frozen runtime files reverified. Actual step gradients2.139/1.904/2.058/1.930/2.102, strictEM.172/.155/.155/.130/.171. Saved full live proof receipts/archive-formal-five-live.json. B formal UUID e2e33c84f289410384586e04b45e7497 continues initial heldout validation against the direct same backend; formal5 still pending. Continue foreground monitoring until BOTH pass. Relay retry deployment will wait for no active retrieval connections and an actor/reference phase, preserving the ongoing A training.

13:33 CST: B baseline heldout EM matches A exactly (NQ.176, Hotpot.203); formal step1 now active. Safely activated A relay retries only after no established port18000 connections AND all four training GPUs had>20GB resident in actor/reference phase. Own relay tmux pane cmdline matched this task before replacement; before source/PID preserved. New relay SHA2fd4f0a7… verified against committed source; real health source/acceptance hashes unchanged, formal training PID/runUUID remained alive. Full deployment receipt saved archive-relay-network-retries.json. A continues step6 without a training restart; B still requires five completed formal steps.

13:51 CST heartbeat/live inspection: both pipelines RUNNING. A completed6 formal steps; step6 strictEM.168, grad2.049,1553.630s. Now step7; retry relay has actual POST/retrieve200 responses. B UUID e2e33c84… completed baseline validation with matching NQ.176/Hotpot.203, generated2560 trajectories in formal step1, scoring finished and first actor setup underway (no optimizer END yet). Sampled peaks A40130/49140MiB and B14391/24564MiB; no new OOM or failure. First iteration is still incomplete for B; both formal5 acceptance remains pending and foreground monitoring continues.
