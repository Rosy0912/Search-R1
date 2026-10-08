# 共享 infra 与 Search-R1 基线一致性审查

固定审查 SHA：`7b93fbd01742715dc14290b0a6e2d3b42e6c19f7`；基础版本：
`b251ba6b503217a59cd3395c1326d7b66b602989`。主 agent 补充审查，六名算法 agent 各自独立出具报告。
本轮只读实现，不启动远端任务，不更改另一个 agent 的 checkout/训练，不修复本轮发现。
总体结论：**有条件通过代码接入检查，不能放行为完整生产训练验收**。

## 1. 数据流及兼容边界

Hydra compose → `validate_config` → Ray `main_task` → worker 初始化 → trainer 创建 Controller。
每批原始问题按 n_agent 重复，新的 SearchRollout 生成真实动作/工具交互或树叶，sources映射回问题行。
原 trainer 重算 old logprob、负载均衡同时重排行级 metadata、算 reward/advantage，再调用实际 actor worker。
worker 的参数/optimizer offload、FSDP和vLLM同步仍使用原入口；OPSD替换内部actor类，TOP-D替换token loss归一化。
纯蒸馏不创建RefPolicy，aux teacher独立冻结；CoSearch ranker在driver独立更新和保存。
旧 `python -m verl.trainer.main_ppo` 不传baselines时仍走原generation/reward/advantage。
actor的ppo_epochs从原来配置未实际迭代改为真实迭代；基础默认E=1保留原行为，E>1是有意改动。

## 2. 已确认问题

### INF-01 / P1：全零 action mask 的 PPO microbatch 产生 NaN（继承缺陷）

位置：`verl/utils/torch_functional.py:107–109`，`verl/trainer/ppo/core_algos.py:187–193`，
`verl/workers/actor/dp_actor.py:264–290`。当一个microbatch没有任何有效actor token时，masked_mean除以0。
实际调用原compute_policy_loss，old/current=0、advantage=1、mask=0，pg_loss/clipfrac/kl及反向梯度均为NaN。
[反例输出](evidence/shared-infra-probes.json)确认，且该core文件与b251逐字一致。
影响 GiGPO、Tree-GRPO、BAPO、CoSearch 的主actor，以及原GRPO；TOP-D的独立token-mean和OPSD避免这个特定分母问题。
零动作轨迹能进入新rollout，现未做microbatch级拒绝或安全loss处理。普通非空tiny测试不能排除它。
建议在共同actor路径安全处理空micro，同时保持DDP/FSDP collective同步和零梯度backward，原GRPO也用相同修复。
这里是条件性代码缺陷，不声称已在正式训练日志里发生。

### INF-02 / P1：长轨迹 generation 与 old-policy scoring 使用不同前缀（继承缺陷）

位置：`search_r1/baselines/rollout.py:87,96–100,149–152`、原generation rolling/right-side caps、
`verl/trainer/ppo/ray_trainer.py:775–779`。
生成rolling保留尾部，训练response保留头部；初始prompt超过max_start也可立即不一致。
重构prompt+response重算old logprob不等于该动作实际采样条件。
新增report默认仅记录context_mismatches，error可在不一致动作生成前拒绝，未修复分布。
影响六种算法和原Search-R1的可比性，尤其TOP-D teacher/old/current同状态及PPO on-policy解释。
建议在正式实验前共享审计/修复策略，GRPO对照一并处理；不能仅修改一种算法以获得不公平优势。
各算法报告有不同上下文/长度反例。不能把“与原代码一致”当作缺陷不存在的证明。

### INF-03 / P2：资源启动器仍依赖外部正确分配，未形成完整部署流程

`reproduction/baselines/run.sh:6` 只要求CUDA_VISIBLE_DEVICES非空，Hydra默认仍是nnodes=1、n_gpus_per_node=8。
仅暴露2卡不自动让Ray配置变成2卡；需要明确override并检查global mini/micro→per-rank除法结果和batch可整分。
启动器不自动检查他人进程/集群调度、不提供全局独占锁，不启动/验证检索服务，不创建隔离的RAY_TMPDIR/TMPDIR/HF缓存。
基础H20 pipeline含独立环境/资产/GPU/模型/检索/checkpoint检查；新run.sh没有调用这些全套门禁。
这是当前部署契约缺口：应先独立预约、GPU实时双阈值/进程检查、隔离缓存、验证依赖/资产/retriever，再运行。
本轮没有为此请求GPU或自动启动生产训练；之前A800 gpu_smoke有自己的三次空闲检查，不能代替run.sh门禁。
OPSD独立反例进一步确认：global micro=1/world=2可通过早期校验，worker floor后local micro=0，
加载模型后才报错；默认8/8正常。该限定配方应提前检查global micro等于DP rank数与mini整除。

### INF-04 / P2：无效 token 的 exp 溢出仍能污染 masked PPO loss

位置：`search_r1/baselines/advantages.py:92–99`，以及原`core_algos.py:185–193`的相同先exp后mask模式。
TOP-D agent发现后root独立调用真实token_mean_ppo复现：old=[0,-100]、current=[0,0]、
advantage=[1,0]、mask=[1,0]，loss=NaN、gradient=[-1,NaN]，见[反例输出](evidence/shared-infra-probes.json)。
第二个token无效却在exp(100)溢出后以0×inf污染loss。单纯mask最后相乘和分母clamp不能隔离非有限值。
这不是正常小policy drift已被证实失败，也不是TOP-D核心proximal reward数值不稳定；是条件性PPO实现风险。
建议exp之前先选择有效token或清零无效delta，对有效位置非有限值明确拒绝，并处理entropy同类风险。
原PPO先exp后mask的模式也需要一起复核；本轮确证反例是在新token_mean_ppo函数。
原raw生成token经过decode/标签截断/reencode的ID一致性另见[TOP-D T05](topd.md)，尚未在真实Qwen量化。

## 3. 相较原基础版本保留了什么

[逐字校验清单](evidence/base-unchanged.json)：原generation.py、core_algos.py、qa_em.py、RLHFDataset、
H20 train_paper_grpo/run_pipeline/prepare_paper_data 均与固定base一致。
新配置对照base train_paper_grpo脚本保留Qwen2.5-3B base、paper-grpo train/test路径、NQ+HotpotQA、
512问题/256 mini/64 global micro、LR1e-6/warmup.285、四动作、topk3、温度/top_p1、
start2048/prompt4096/action500/obs500、参数/梯度/optimizer offload、TP1、KL.001、E=1共同默认。
TOP-D/OPSD纯损失关闭KL/entropy，Tree轨迹数变化、BAPO reward/holdout、CoSearch额外ranker/candidates是算法必要偏离。
OPSD micro1/每rank以及TOP-D E2通过launcher显式更改。详细作者recipe差异见各算法报告。

paper-grpo数据脚本合并后以len(output)赋index并assert唯一，避免旧脚本跨来源local index碰撞。
RLHFDataset仍信任extra_info.index，任意其他parquet缺失/重复index不会自动拒绝；不要把这种一般风险说成当前固定数据已碰撞。

## 4. 实际复核证据

- `.venv-cpu/bin/python -m pytest -q tests/baselines`：24 passed，[输出](evidence/cpu-baselines.txt)。
- `.venv-cpu/bin/python -m pytest -q tests/reproduction`：39 passed、1 Linux procfs检查在Mac skipped，[输出](evidence/cpu-reproduction.txt)。
- 六种Hydra配方compose+validate_config通过；辅助路径为占位，没有读模型或验证数据/网络。[输出](evidence/shared-infra-probes.json)。
- `bash -n reproduction/baselines/run.sh`：通过。
- 既有[A800两卡tiny GPU证据](../gpu-validation.md)：真实NCCL/FSDP actor和ranker更新/保存，非生产验收。

测试通过与上面两个P1反例同时成立；现有测试没有覆盖所有异常训练路径。
目前没有正式3B+Torch2.4/vLLM0.6.3+remove-padding/offload+真实搜索的整链验收。
OPSD每rank FP32 CPU教师、TOP-D driver教师、CoSearch CPU AdamW ranker成本未获生产吞吐/峰值预算。
checkpoint支持权重和部分算法/ranker状态保存，没有完整actor optimizer/dataloader/RNG resume，manifest明确resume_supported=false。
BAPO还缺独立阶段holdout，TOP-D缺明确训练过兼容teacher，CoSearch缺明确ranker及50候选检索服务验证。
这些属于未完成验收/输入条件，不能写成“所有infra已经没问题”。
