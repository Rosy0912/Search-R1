# Search E1 两套训练配置对照

用户已授权推送 search-e1 分支、查找24/48GB空闲GPU并运行复现，不需要再次审批这些操作。只在HKUST集群内运行，资产从官方源下载或同集群已有可信缓存获取，禁止其他集群资产/检索服务接入。

archive/内容和301项SHA256保持不变。实际runtime从archive复制，适配改动另存patch及source manifest。两组都为Search-E1的两轮GRPO与OPSD，统一Qwen2.5-3B-Instruct权重、seed42、NQ/Hotpot训练留出512、共同独立评测题ID、E5/Wiki2018/top3、严格最终答案评分及同一推理预算。原始reward作为旁路诊断，不用于主收益结论。

A archive配置：每轮200更新、512×5轨迹、mini128、逐轨迹mean、entropy0、lr1e-6、warmup0.285约57步、KL0.001、clip0.2、四次搜索。B R1训练配置固定取自 search-r1 提交84e7d1000331c4965b907ac804019f4b25362b67的reproduction/a800/train_continuation.sh及verl/trainer/config/ppo_trainer.yaml，但不复刻历史fresh-Adam迁移：每轮500更新、全局mini256、全局actor micro64（四卡每卡16，全局梯度累积mini/micro=4；每卡内16条轨迹有效token统一归约）、ref/rollout概率micro128、micro内token mean、entropy0.001、warmup0.285约142步；其他不变，OPSD沿用archive配置。共同50/100/150/200节点可比较，B额外至500；报告总optimizer步数及token/GPU时间，不能把组合配置差异归因于单一超参数或初始权重。

优先各使用4张48GB卡，A hkust4:4-7；B hkust5:0-3。上述是实时检查候选，不等于永久预留，启动前及阶段切换必须再查UUID、显存及foreign compute PID。hkust5:4异常，永不使用。显存方案允许降低仅执行层microbatch并保持mini及loss归约，但若B无法保留原micro内token加权关系，停止并明确记录，不能冒充原配置。CPU-offload和gradient checkpoint保持原设置。vLLM utilization不是越大越好：从0.4开始，以真实rollout→logprob→update峰值及OPSD/检索共存测试选最高可通过值，保留临时张量余量；不得只靠模型加载成功验收。

检索优先CPU FP32 FAISS索引、GPU E5编码，共享训练卡时必须计入峰值；如果改成CPU E5或其他索引精度，须记录并核验逐query top3，不默认等价。同集群可在另一HKUST节点部署检索但需要独立空闲GPU及服务身份/资产哈希记录，避免跨实验串用错服务。

训练先完成完整512×5批次真实更新的小测，保留权重变化、mask/评分反例、NCCL及显存峰值证据，然后fresh seed正式两组。小测的总schedule保留各自201/501，用独立stop_after控制停止，不用短总步数改变warmup。正式训练保存50/100等checkpoint并检查非有限loss/gradient/tensor及独立EM；发现NaN停止本任务，保留证据，不热修改或重启其他训练。OPSD只允许从有效checkpoint进入；禁止默认选最后checkpoint而无质量验证。

结束时报告每阶段实际完成状态、旧/严格reward分歧、相同题集EM、运行身份和Git提交。环境安装、数据下载、烟测通过都不等于复现效果成功。资产不齐或GPU验收失败时保留完整日志并说明限制，仍推进其他可完成部分。

实际执行适配：两组共享CPU FP32 E5编码与CPU FP32完整FAISS索引，避免编码器侵占训练显存。服务启动时用固定16个留出问题比较历史GPU FP16编码器在同一索引上的top3并保存差异；该样本不能证明全体query等价，因此新结果不宣称与历史GPU检索曲线逐点可比。GRPO KV比例从0.4逐级0.5/0.6/0.7，每级先做两次真实更新，根据各卡实测峰值加下一档缓存成本、至少保留6GiB再决定是否继续。任何档位失败停止并保留证据，不自动重试OOM。独立推理无FSDP共存，KV比例0.85，保留15%显存；未经过实测前不宣称打满。
