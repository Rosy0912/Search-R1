# 七实验会话 CUDA 非法访问与实际规避证据调查

调查主体完成时间：2026-10-08 14:29 CST；定稿纳入 root 随后的最新人工执行授权。执行者：GiGPO 主线的只读后台子 agent。

## 结论与适用边界

在七个实验会话已经保存、可返回的记录中，**Search-E1 有直接相似的 vLLM 生成非法地址访问 + NVIDIA Xid31/MMU fault 现场，并有切换 XFORMERS 后完成全量生成的真实运行证据**。这是可以借鉴的规避方案。其余六会话未找到同类现场及同类修复证据；它们的概率精度、非有限梯度、OOM、序列化、元数据、监督器和迁移故障不能作为 GiGPO 非法访问的共同根因。

Search-E1 的 XFORMERS 长运行证明是 **169,103 题 × 5 = 845,515 条全新 vLLM 轨迹**，不是启动成功或几条短检。后续同一管线八 rank 完成 **3,156 次 OPSD Adam 更新/一 epoch**，但这部分训练是 HF/SDPA：不能把 3,156 次更新写成 vLLM 与 FSDP 反复权重切换的验收。两种注意力后端会改变 BF16 舍入、logits 和抽样轨迹；科学参数相同不等于位级等价。旧故障算子仍未确认。

GiGPO 的对照现场来自主线已有原始证据及 root 本次通报：3337 在 H100-8、官方 Qwen2.5-3B-Instruct、Torch2.4/vLLM0.6.3 下实际完成 2 outer/40 Adam；第 3 轮生成 rank1 在 `random_samples.cpu()` 报非法访问，伴随 Xid31 `FAULT_PDE`。异步同步点不能定位真正失败 kernel。正式 3338/runtime19 把 `CUDA_LAUNCH_BLOCKING=1` 传给八 rank；root 于 2026-10-08 06:29:24 UTC 实读第一轮八 rank 各19次真实Adam写入、仍RUNNING、首outer尚未计完整、未见CUDA错误。**CUDA blocking 是诊断设置，当前不能称为根因修复。它与 Search-E1 已有全量运行证据的 XFORMERS 规避不是同一级别证据。**

定稿时 root 转述的更新人工指令已明确要求：**立即主动停止3338、切换XFORMERS正式运行，并在指定Mac故障表新增一行**，由root执行。这一较新授权取代此前“健康3338继续、真实复发再换后端”的安排。主动停止不等于CUDA复发，也不能称3338必然会失败。本子 agent 未修改GiGPO源码、参数、后端、任务状态或故障表；仅交付本文，不声称新的GiGPO XFORMERS作业已经通过运行。

## 调查范围与证据强度

并行只读调用 `read_thread`，对七会话逐页读取，直到各自 `hasMore=false`；最新页和必要历史包含工具输出。部分较早分页只读会话文本/消息，未请求所有工具原始输出。再只读核查会话提到的 Mac 本地故障报告、实际验收文档、配置、控制源码和 receipts。搜索覆盖 illegal memory access、device-side assert、Xid31、FAULT_PDE、sampler、CUDA blocking、FlashAttention/XFORMERS、概率 NaN、FSDP、KV/offload 与相关真实后续运行。

本调查没有 SSH、GPU/大模型执行、运行其他会话脚本、发消息/唤醒其他会话、提交推送、创建/暂停 automation、取消作业或修改其他会话文件。唯一新写入是本文。向 root 传递后台发现不等于联系其他实验会话。

“未找到”仅指本次可返回会话记录及已核查本地材料，**不证明未保存或未返回的原始日志中绝对不存在故障**。旧摘要、计划、CPU PASS、合成 GPU fixture、首次初始化均不升级为真实持续训练修复证明。

| 调查代号 | 工具返回的会话标题与 ID | 本次可返回记录时间范围（UTC） | 同类非法访问证据 |
|---|---|---|---|
| Search-E1 | Search-E1研究；`01a1028b-b626-7f91-93db-0099ff1623c5` | 2026-10-03 13:12:05 → 2026-10-08 04:53:13 | 有：prep2 GPU3 illegal access/Xid31；有全量 XFORMERS 后续生成 |
| Search-G1 | Search-G1；`01a0fb1b-d3ff-74b3-aa13-49ca57871f92` | 2026-10-01 10:51:40 → 2026-10-08 06:09:29 | 未找到；有 OOM、概率重算精度及资源配置问题 |
| Tree-GRPO | 验证 Tree-GRPO 效果；`01a0fcbc-0b46-7502-8b66-d8b21a0e032d` | 2026-10-02 13:10:41 → 2026-10-08 06:04:17 | 未找到；有非有限梯度 norm、概率路径和部署/状态问题 |
| BAPO | 验证 BAPO 在 Search Agentic RL 中的效果；`01a0fcbd-41d8-7910-a410-372ce5c8870c` | 2026-10-02 13:11:24 → 2026-10-08 06:22:34 | 未找到；有不同类别的数值稳定化真实运行 |
| OPSD | 验证 search agentic RL 中 0PSD 效果；`01a0fcbd-c50e-7cf1-8073-256a13ac9f7e` | 2026-10-02 13:11:44 → 2026-10-08 03:21:29 | 未找到；有 teacher 词表缓冲显存/数值、源码守卫和验收问题 |
| TOP-D | 验证 TOP-D 在 Search Agentic RL 中的效果；`01a0fcbe-15f8-7600-ae84-9487d924560a` | 2026-10-02 13:12:05 → 2026-10-08 06:18:14 | 未找到；32B 教师共存/监控/迁移问题，学生训练未开始 |
| CoSearch | 验证 CoSearch 在 Search Agentic RL 的效果；`01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef` | 2026-10-02 13:12:20 → 2026-10-08 06:18:44 | 未找到；空 ranker、ContextVar 序列化、可选元数据与迁移 I/O 问题 |

范围是读取开始时的快照，各会话之后可能继续产生新记录。TOP-D 有读取其他会话后转述的 Search-E1 Xid31 内容，已按事件身份去重，不能算 TOP-D 自己发生一次非法访问。

## Search-E1：可直接借鉴的规避与真实运行

### 故障现场

2026-10-07 约 16:20 CST，HKUST1 preparation-2 的固定权重 vLLM rollout worker3/GPU3 报非法显存访问，kernel 同时记录 Xid31/MMU fault。run `d588a1450f8343edbed9c1e1901ac836`、controller PID2308665，故障前保存 80,320 条完整轨迹，八分片为 9920/10240/10240/10240/9920/9920/9920/9920；当时 OPSD 更新数为 0。采样相关异步栈最终在 `SamplingTensors.from_lists` 的 `temperatures_t=torch.tensor(...)` 报错；失败输入 pickle 也被同一 CUDA 错误打断。无 OOM 消息，不能因此认定 sampler、温度张量构造或硬件损坏是真根因。

故障日志可追溯会话 turn `01a114b0-d961-7c71-8a89-621142a05eb7`。本地 [故障封存 receipt](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-publish/experiments/search_e1/docs/receipts/preparation-2-failure-preserved-20261007.json) 的完整 SHA 为 `b3ab7529d001280155f7b08e95c889c48e59c80e06b1e5ba08849e7aeb409738`；原节点目录为 `/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/preserved/preparation-2-cuda-failure-d588a1450f8343edbed9c1e1901ac836`。本调查未重新连接节点读取 kernel 日志。

### 相对直接故障运行，究竟改变了什么

环境版本是 vLLM0.6.3、Torch2.4.0+cu121、flash-attn2.6.3、xformers0.0.27.post2。共享依赖未升级。独立 prep3 仅在新子进程环境新增 `VLLM_ATTENTION_BACKEND=XFORMERS`，以及新 run/输出/路径身份。启动审计将原 prep2 标为 FLASH_ATTN；本调查未额外远端重读旧 selector 全日志。

[准备控制源码](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-publish/experiments/search_e1/control/opsd_from_archive100/prepare_preparation3_xformers.py:214) 的 `verify_prepared_config` 强制 prep3 的 datasets、source100、retrievers、execution_change、科学函数证明、科学设置分别等于 SHA 固定的 prep2 配置。配置继承 `dict(old)`，冻结科学 runtime305 文件逐字节复制，`changed_files=[]`，manifest 同为 `3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc`。

| 设置 | prep2 → prep3 |
|---|---|
| 模型 | 同一 archive GRPO100 权重，12 文件验收身份保持 |
| 数据/规模 | train169103/validation512/test51713，同 SHA；169103×5 完整新生成 |
| batch/随机采样 | 8 worker、每批64题、worker seed42–49、T1/top-p1保持 |
| 长度/搜索 | context8192、每turn500 token、5turn/4search、top3、obs480保持 |
| KV/执行 | GPU22671 blocks、CPU7281 blocks、utilization0.85、eager=True保持 |
| 通信/检索 | NCCL24h、原私有18004 endpoint及同检索身份保持 |
| OPSD | epoch1、lr1e-5、LoRA16/alpha32、clip10/gradclip1、chunk32、accumulation4、global trajectory batch32保持 |
| 新变化 | 新子进程 attention backend XFORMERS；完整新生成，旧80320不拼接 |

`execution_change` 中的 KV override 和 NCCL3h→24h是**此前 prep1→prep2** 的既有调整，直接故障 prep2 已有；不能误写成 illegal fault 后同时缩小缓存、batch 或增加通信超时。失败前后 T、权重、数据、科学损失均未换。XFORMERS 的 kernel 算术变化仍可能改变采样结果，旧运行未保存可恢复 vLLM RNG/cursor，不能声称精确续跑。

本地 [prep3配置](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output/opsd-debug-plan-review-20261008/evidence/old/receipts/preparation-3-config.json) 的完整 SHA 为 `3ac56bfa4455ecc93175172db6353bbc4a7cae95aee24781797428e50abeeb5b`。准备/管线源码实读 SHA 分别为 `bbb2a6bdc7c65e3c070a1ba1dcb1c7e370848b8d118188ad5ad08e1fa09618ea` / `85c09c3bd5a58a59509b63e7fcc70aeb0cc1b6c139cc11e4ab6a039f4fe48bb4`，与配置 pins 相符。

### 真正后续运行了多少

新 controller2459676/run `914bfff786c84ede8d82414dff65f60e` 于 2026-10-07 16:31 CST 启动。八 worker实际日志均为 `Using XFormers backend`，同时确认 GPU/CPU blocks22671/7281和原 seeds；不是只核启动器环境。完整 [实际启动 V2 receipt](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-publish/experiments/search_e1/docs/receipts/preparation-3-independent-controller-startup-readonly-audit-v2-1791362186.json) SHA `140474ea4c7c8bc8b07c321ae0e2e924182328542274905afe28cd5a5138e95f`。controller环境为null不构成失败，它在 spawn时给 rollout parent/worker设置后端；实际 worker环境/selector已确认。

2026-10-08 09:23–09:26 CST 的 [独立真实运行与五更新审计](/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output/preparation-3-actual-five-opsd-independent-audit-1791422595.md)（本文实读 SHA `3635e170747f572468568b870f6d3ed45e5ada31b64e8849bf4a2c3632148272`）证明：

- 全169103题/845515轨迹完整生成、原题位置和五sibling顺序有效；八原生成 worker正常退出，GPU进程释放。
- 原检索 PID2239529/身份未变，峰654MiB；生成期连续记录最低 actual free3099MiB，未见新可读 dmesg Xid/NVRM/CUDA。
- 同一管线筛出100974 pairs；native更新序列1…3156完整无重复，末批14条，八rank真实Adam均3156、权重有限且变化。
- updates1/5的16个保存rank bundle被真实CPU重新读取：504 optimizer states/rank、Adam step1/5与有限moments、RNG完整，rank状态相同。八rank update5时实际显存free至少15077MiB。
- 3156的最后live allgather不是3156 optimizer checkpoint冷重载证明；native512/完整七基准效益也不是本次CUDA规避验收的内容。

已保存发布记录：规避实现推送 `407ad92`，真实五更新/完整epoch验收提交 [980989e341eea9407b38821d773dee2f2c754f31](https://github.com/shaoshitong/Search-R1/commit/980989e341eea9407b38821d773dee2f2c754f31)。本调查不重新执行Git网络核验；commit来源是会话当时的推送与核验记录。该实际审计明确写着旧 CUDA/Xid31 根因尚未确定。

[官方 verl v0.4 FAQ](https://verl.readthedocs.io/en/v0.4.0/faq/faq.html#illegal-memory-access)也给出 vLLM≤0.6.3 rollout illegal access采用XFORMERS的建议，要求Ray启动前设置并在各rank检查。本调查已打开该官方页面核实。文档建议与Search-E1全量实跑相互支持，但都不能单独证明GiGPO在第3轮权重同步、KV缓存、offload或H100 kernel的具体错误机制。

## 其余六会话：已验证不同类别的方案，不能混为同根因

### Search-G1

未找到 illegal access/device-side assert/Xid31 同类记录。保存故障包括BF16 vLLM采样与HF重算最大logprob差1.550987、独立FP32诊断差0.0005569458，以及真实基线CUDA OOM和同卡另一约59GB进程。FP32诊断 native3260的8rank/64 ordered questions/一次DTensor同步是**0optimizer writes**，不能当长期训练或非法访问解决。最新人工要求重新对齐Search-E1 BF16，并取消额外全量HF独立前向；会话正在改性能配置，不能将FP32作为应迁移给GiGPO的“已证明必须”方案。

证据：[COMPUTE_PRECISION.md](/Users/shaoshitong/Documents/Search-G1/repo/reproduction/common8/COMPUTE_PRECISION.md)；会话 OOM turn `01a119c5-62b6-7642-a61a-46cd764b045e`、诊断与最新BF16 steering turn `01a11a1b-ee2d-70f1-a855-40829abf71d3` / `01a11a21-8e1c-7230-8d5c-1d925e167452`。这是概率数值/资源问题，未证明FlashAttention非法地址故障。

### Tree-GRPO

历史2971完成307 outer后，第308在 `_optimizer_step` 拒绝非有限梯度norm。失败microbatch/rawgradient未捕获，KL overflow或FP32 norm overflow仍是候选，早期308微批可能已写参数，不能写0更新。公共数值方案分别为饱和KL先限制exp输入、FP64全局L2梯度范数及正确shard/replica计数；这些处理浮点溢出/NaN，不修复非法指针访问。

本地 [故障与候选审查](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/FAILURE_AND_CANDIDATE_REVIEW.md)明确未定位原始数值根因；[真实8rank3063摘要](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/TREE_NATIVE_NUMERIC_3063_SUMMARY.json)是20合成FSDP案例/rank、17秒作业、非3B训练峰或2/5/500验收。最新3335实际完成5 outer/100Adam每rank及same-world完整state保存，后续因人工指出重复检查拖慢而取消，非CUDA非法访问。5轮状态与新性能实现也不构成2971数值根因确定或XFORMERS效果证明。相关当前证据：[报告](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/REPORT.md)、turn `01a11a1c-cceb-7dc3-afa0-76e970e8d612`。

### BAPO

未找到同类非法访问记录。数值源 `1cceda2f7e201de3f68c876bb9a3b895563c20ce` 引入与Tree相同的FP64 norm与有限饱和KL稳定化，目标硬件3086包含真正8rank FSDP/offload与KL反例；随后3094真实两轮COMPLETED，20optimizer调用/全部数值有限及HF2检查通过。这是已实跑的**不同数值故障**修复，原始smoke仅两轮，不能改称长期500验收。

证据：[3094独立实际报告](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-3/outputs/bapo-numericfix-smoke3094-independent-review.md)。当前3315另一已冻结实现已确认138/200且数值有限（14:18 CST会话记录）；它包含后续概率/预算等调整，不是1cc数值改动的单因素对照，更不是非法显存规避。证据：[当前报告](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-3/outputs/REPORT.md)、turn `01a11a2a-1828-77b3-ba97-083c832f2288`。

### OPSD

未找到同类非法访问记录。历史3303/旧候选的问题涉及teacher full vocabulary缓冲/数值与显存门禁；后续 `fcecbea5d939e41dcfb3b118a26ef049a9bda3d6` 保持teacher原四行整batch前向，将head projection/目标mask/fullKL放入nonreentrant checkpoint重算，减少保存target词表缓冲。CPU全参数loss/梯度等值是软件证明；真正3304运行已通过首5/第10连续实测，最新会话另确认200/200、2000Adam每rank与final51713汇总，但outer验收 wrapper/资源归属退出1及HF200/fullstate终验仍待补齐。不能把训练child0误写为整体验收0。

证据：[候选数学与bytes审查](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-unclipped-multitraj200-teacherrecompute/independent-review/TEACHER_RECOMPUTE_CANDIDATE_BYTES_INDEPENDENT_REVIEW.md)、[真实10轮显存/连续性](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-4/work/e1-unclipped-multitraj200-teacherrecompute/ACTUAL3304_COMPLETED10_CONTINUITY_AND_MEMORY_INDEPENDENT_REVIEW.md)、200实读turn `01a11970-275c-7762-9001-5fc18776f4a3`。第10 NVML最高77.91GiB仍高，不能把reserved/used高直接诊断为泄漏、非法访问或持续15%余量。teacher-recompute不是vLLM注意力后端修复。

### TOP-D

未找到同类非法访问记录。32B teacher scorer/vLLM共存有实际显存压力与CPU/GPU概率超差；一些门禁/清理失败、watcher“采样时间过期”是监督器问题，记录明确无模型OOM/异常时不能改称CUDA crash。教师3228至少真实到55轮，每rank20optimizer/80micro，数值有限且OOM0；当前迁移选teacher HF85，TOP-D student实际更新0。不能把教师GRPO训练或仅共存GPU测量写成TOP-D学生算法与CUDA修复验收。

证据：[TOP-D报告](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/outputs/REPORT.md)、55轮实读turn `01a11157-b9d2-7c30-8c96-752b02f0cf2a`、监控失败turn `01a10958-b10b-70b3-895d-195f65b7543a`。当前迁移I/O断流/部分分片不能解释GiGPO的Xid31。

### CoSearch

未找到同类非法访问记录。U3201实际联合12后，在13前因空ranker `_postprocess([])`/`torch.cat([])`失败；V3205在checkpoint load前因Ray cloudpickle捕获ContextVar失败，vLLM shutdown信息是退出后果，不能当新GPU kernel故障。W改为worker运行时导入模块保留process-localContextVar；X进一步适配四个可选metadata原None语义。CPU实际完整Ray类序列化/新解释器回归只是CPU证明。后续X3211有五真正joint稳定窗口且最新源X203完整24文件SHA核验，证明这些不同路径恢复后确有长运行；同时main-only emptyranker skip不能计joint，strict600/save冷恢复等不能借此宣布通过。

证据：[V/W原故障及序列化修复](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/review-native-v-serialization-fixed-w.md)、[X可选元数据修复](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/work/review-native-w-optional-metadata-fixed-x.md)、[CoSearch现有记录](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6/outputs/REPORT.md)与当前heartbeat绑定源203。迁移EIO/BrokenPipe当前重试不是GPU非法访问，不借用为GiGPO调整算法、ranker或预算的理由。

## 给 GiGPO 主线的可执行判断

1. 现有直接证据支持最新人工已选择的XFORMERS规避：采用已兼容Torch2.4/vLLM0.6.3的过程级后端、在CUDA/Ray启动前设置并核查全部实际rank/selector。GiGPO旧版本强制FLASH_ATTN，新独立版本必须避免该旧硬编码覆盖XFORMERS；由root实现，本调查未改代码。
2. 不把E1的0OPSD准备失败当作GiGPO已完成40Adam后的精确训练恢复模板。保留最新完整训练状态/失败现场，依据主线自身真实checkpoint能力处理，不拼接或虚构旧RNG状态。
3. 不将NaN处理、FP32概率诊断、FP64norm、teacher重算或emptyranker等不同类别方案一并加入GiGPO。没有本次证据支持它们是3337真根因。
4. 已存在的CUDA blocking真实生效证据只支持同步定位与运行观察。3338的19写入和无CUDA错误现场必须保留；最新人工授权的主动停止另记为主动切换，不能覆盖为故障退出。新正式XFORMERS作业的启动/outer/Adam结果须由root真实证据确认。

本报告是一份只读交叉会话调查，未增加测试、审查门禁或训练实施。后端规避的可借鉴性已向root尽早报告，主线训练没有等待本文。
