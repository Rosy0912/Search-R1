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
