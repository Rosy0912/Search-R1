# Search-E1 reproduction

论文：[Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning](https://arxiv.org/pdf/2605.22511)。独立复现，非作者官方代码。GRPO 使用固定版本的 [Search-R1](https://github.com/PeterGriffinJin/Search-R1)，OPSD 在本项目实现。

## 实现

- `search_e1/core.py`：标准 EM、最少检索/最短正确参考轨迹、优先选择错误且字符差异最大的学生轨迹、检索内容 token 掩码、Eq. 7 前向 KL。
- `search_e1/opsd.py`：冻结 GRPO 基座，关闭 LoRA 得到教师分布；开启 LoRA 得到学生分布。两端使用相同响应 token ID，并按各自前缀长度对齐下一个 token 的 logits。KL 在词表项层面仅做上界裁剪；分块 checkpoint 降低显存使用。多 GPU 使用每轮轨迹 batch 的梯度求和，保持全局平均目标；每轮根据独立验证集 EM 选择 adapter，然后合并。
- `search_e1/rollout.py`：训练后的新 rollout 池、固定 E5/Wikipedia top-3 检索、四次搜索预算、最终 EM 评测。
- `search_e1/prepare.py`：NQ+HotpotQA 训练数据，训练集留出的验证集和七个独立评测集。`data/manifest.json` 记录实际样本数。
- `scripts/grpo.sh`：标准 GRPO，200 更新、lr=1e-6、KL=0.001、clip=0.2、G=5、temperature=1、检索内容掩码。
- `scripts/cycle.sh`：两轮 GRPO → fresh rollouts → pair mining → OPSD → merge，最后七集评测。

## 与论文的差异/未公开细节

公开 PDF 未完整给出 LoRA rank/alpha、OPSD 学习率、epoch、batch、字符距离、rollout 池 K、验证集构建及 prompt 精确文本。当前明确采用 rank=16、alpha=32、lr=1e-5、1 epoch、梯度累积8、K=5、字符级归一化 Levenshtein 距离（[RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html)），并从训练数据留出512样本验证；这些是可配置复现假设，不能声称与作者训练完全一致。完整清单在 `configs/paper.json`。

论文配置目标8卡，当前调度脚本使用 Lumia 的4张L40S 48GB，论文为8张H800。使用 FSDP CPU offload、每卡 microbatch=1（全局 microbatch=GPU 数）、512GB 主机内存。检索保持 E5-base-v2/Wiki2018/top-3；原始 float32 FlatIP 索引分片到四张已分配 GPU，每卡约15GiB；禁用 TF32，按分片取 top-3 再合并，E5 编码器使用第一张已分配GPU。启动时与 CPU FAISS 核对8条真实训练问题的 top-3 和分数，失败则退出；可用 `RETRIEVER_BACKEND=cpu` 回退。全量 fresh rollout 使用 vLLM 批量采样，Transformers 提供逐条参考实现；全量约85万条/轮生成，吞吐尚未验证。GRPO 沿用 veRL/vLLM。

论文文字使用共享参数视角，而实现段规定关闭 LoRA 的冻结 GRPO 教师；这里按实现段执行。`<information>` 及其边界全部不参与损失；检索预算通知也属于外部信息。rollout 保存策略实际生成的字符范围，保证检索插入的换行也不参与损失；跨越生成/外部边界的 BPE token 保守排除。单个响应超过教师或学生上下文预算时整对跳过，并记录数量，避免错位截断。

## Lumia

目录：`/home/jmzhang/Workspace/Search-E1-20261001`。独立 `.venv` 基于既有 `searchr1` 依赖，只在项目环境安装 peft 与真实 FlashAttention 2.6.3，未改共享环境。

已有资源：

- `/home/jmzhang/self-evolution/Search-R1/data/retriever_data/e5_Flat.index`
- `/home/jmzhang/self-evolution/Search-R1/data/retriever_data/wiki-18.jsonl`
- `/home/jmzhang/models/e5-base-v2`

```bash
ssh lumia
cd /home/jmzhang/Workspace/Search-E1-20261001
bash scripts/prepare_lumia.sh
sbatch scripts/lumia_smoke.sbatch
sbatch --parsable scripts/lumia_pilot.sbatch
# 全量作业必须指定真实且通过端到端验证的 pilot 目录
sbatch --export=ALL,PILOT_DIR=/absolute/path/to/pilot scripts/lumia_full.sbatch
```

`normal` QoS 已验证可申请空卡；账号此前4张运行卡使用 `high` QoS，达到该 QoS 的4卡上限。无需停止已有实验或指定未分配的物理 GPU。

日志位于 `runs/`。作业查询：`squeue -u jmzhang`；已结束作业：`sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed`。

## 本地/其他服务器

```bash
bash scripts/bootstrap.sh
python -m pip install -e .
# 另按 upstream/Search-R1/README.md 安装匹配的 veRL/vLLM/flash-attn 和检索依赖
python -m unittest discover -s tests -v
python -m search_e1.prepare --output data
```

Upstream 固定 commit：`598e61bd1d36895726d28a8d06b3a15bed19f5d3`。原项目许可在 `upstream/Search-R1/LICENSE`。训练脚本保留该版本的保存结构 `actor/global_step_N`，将配置的 total_training_steps 设为更新数+1，以补偿其在更新后递增再判断退出的边界行为。

## 验证与恢复

核心测试检查 EM、配对排序/丢弃规则、信息掩码、KL 方向和教师 stop-gradient。GPU smoke 用 tiny Qwen2 检查真实因果对齐和 LoRA 参数更新；这只能证明代码路径，不代表论文指标复现。

各轮 `.done` 仅在对应阶段成功退出后写入。GRPO 原版不能恢复优化器状态；失败时不可将旧阶段视为完成。rollout 用独占创建保护已有输出；若中途失败，先保留原文件，再明确重新运行该阶段。全量结果以实际输出的 `.metrics.json` 为准，不预填论文的0.440。当前没有完整复现结果。

## 本次运行记录

端到端试验 `133148` 已 COMPLETED（退出码0，12分46秒，node11，4张L40S）：2次 GRPO 更新 → 40条新轨迹 → 2对蒸馏样本 → 四卡 OPSD → adapter 选择/合并 → 合并模型评测。4条 NQ 验证样本 EM=0.25，训练前后持平，因此选择基线 adapter。这是流程验证，不能当作论文结果。

原全量作业 `133149` 在首个更新的 CPU 检索阶段停止，尚未产生训练 checkpoint。CPU 检索每128查询约一分钟，是主要瓶颈。精确 GPU 分片检索基准32查询：CPU18.87秒、GPU0.35秒，top-3 索引完全一致，最大分数误差1.2e-7。此为检索加速比，不代表端到端训练加速比；记录在 `reports/gpu_index_benchmark.json`。GPU 检索端到端试验 `133179` 已 COMPLETED（退出码0，9分10秒）：2次GRPO更新、40条新轨迹、3对OPSD样本、adapter选择/合并及评测。蒸馏前后4条验证样本EM都为0，因此选择基线adapter；小样本没有证明质量提升。全量作业 `133180` 已在node10自动启动，目录 `/home/jmzhang/Workspace/Search-E1-20261001/runs/paper-3b-133180`。查看：`ssh lumia 'squeue -j 133179,133180'`。论文硬件与当前4张L40S不同，完整运行时间尚未确认。

修复记录：旧版 Search-R1 将全局 microbatch 除以 GPU 数；设为1会在四卡下变成0，导致 TensorDict split 死循环。当前三个 microbatch 配置都设为 GPU 数，启动时检查整除关系；每卡实际 microbatch=1。

## 分阶段完整实验（当前运行）

批处理验证 `133223` 已 COMPLETED（退出码0，5分17秒），完成两个GRPO更新、保存与验证。6项单元测试包含不等长轨迹上批处理和逐轨迹累积的损失/梯度等价性。`search_e1/grpo_batching.py` 在启用批处理时将策略、参考KL和熵归约为逐轨迹均值，保持原每卡microbatch=1的目标；未修改group advantage、PPO clipping或优化器更新。普通 `grpo.sh` 默认仍为每卡microbatch=1；阶段作业采用训练2、概率计算4，并显式将配置传给Ray worker。小验证集EM=0.25，只验证运行路径，不能证明完整训练效果。

原作业133180完成首个更新：训练采样平均奖励0.264，单步2213.6秒；它在第二个更新期间被本项目停止，日志保留，尚无完整训练checkpoint。新实验从初始模型完整执行第一轮200更新，不声称恢复了旧优化器。由于L40S分区单作业上限7天，完整流程拆成五个afterok依赖作业，每个GRPO阶段包含完整200更新，不在更新中途切断：

| 阶段 | 作业 |
| --- | --- |
| 第一轮 GRPO | 133235 |
| 第一轮新轨迹池 + OPSD + 合并 | 133236 |
| 第二轮 GRPO | 133237 |
| 第二轮新轨迹池 + OPSD + 合并 | 133238 |
| 七数据集完整评测 | 133239 |

统一实验目录 `/home/jmzhang/Workspace/Search-E1-20261001/runs/staged-3b-20261001-170407`。实际提交回执在服务器 `runs/latest_submission.json` 和本地 `reports/lumia_submission.json`，包含 `jobs` 数组；30分钟定时检查已更新为跟踪该作业链。训练数据、每轮200更新、K=5的新轨迹池、OPSD和七个完整评测集保持不变。完整批次的稳定性和实际提速幅度仍待实测。

新实验提交方式：`python scripts/submit_phases.py`。只有启动全新实验时才调用；不要重复提交运行中的实验。失败阶段的恢复应核对已有完成标记、产物和依赖状态后处理。

新配置首个全量更新实测：作业133235单步1017.9秒（17.0分钟），原133180为2213.6秒（36.9分钟），约2.17倍提速；策略更新591.6秒、参考概率65.8秒、采样291.8秒。两次轨迹长度略有不同，因此这是单步实测对比，并非严格固定轨迹基准。梯度正常，训练采样奖励0.272，不代表独立评测EM或质量提升。记录在 `reports/staged_first_update.json`。当前已进入第2个更新，完整论文复现仍未完成。

并行采样验证133283已COMPLETED（退出码0，40秒）：2卡、8问题、K=5，40条轨迹的question/sibling覆盖和唯一性检查全部通过，产生2对样本。EM=0.15仅为流程验证。后续阶段脚本已设置按分配GPU数启动独立vLLM采样进程（首轮4卡，后续阶段已申请8卡），问题互斥分片，每问题K保持5，随机种子为42+worker编号；四卡全量运行尚待验证。生成期间进度位于`rollouts.jsonl.parts/worker-*.log`及各分片输出，完成后才合并最终文件。失败保留分片，不自动当作可恢复结果。最终七数据集评测同样并行，仍为贪心解码。记录在`reports/parallel_rollout_benchmark.json`。

第2个全量更新已完成：1001.4秒，梯度范数2.251，训练采样平均奖励0.261，完成率0.984。前两步时长约17分钟且接近，未依据这两批随机采样奖励的波动改变训练配置。记录在`reports/staged_second_update.json`；完整200步及独立验证仍待完成。

训练前完整512条验证基线作业 `133489` 已在node11启动（4卡），采用与第50步相同的上游验证路径，开启 `val_before_train` 和 `val_only`，不执行训练。结果用于第50步效果比较及扩卡决策。首轮已完成26次更新，第27次运行中。

完整512条训练前基线验证 `133489` 已成功完成（16分25秒，退出码0，完成标记已核查）：NQ EM=0.21875，HotpotQA EM=0.2734375，宏平均0.24609375。证据见 `reports/validation_baseline_result.log` 和 `.json`。第50步使用同路径验证后比较；该基线不是论文最终测试结果。

## 第50步独立验证与后续扩卡（2026-10-02 16:30）

首轮GRPO第50步使用与训练前基线相同的512条留出验证集及验证路径，两个数据源均提升：

| 验证集 | 初始模型 EM | 第50步 EM |
| --- | --- | --- |
| NQ | 21.875% | 约37.9% |
| HotpotQA | 27.34375% | 约45.3% |
| 宏平均 | 24.609375% | 约41.6% |

宏平均提高约17.0个百分点。第50步控制台指标仅保留三位小数，表中训练后数字是近似值；这属于留出验证，七个完整测试集尚未评测。第50步checkpoint已保存，检查时52/200更新完成，第53步运行中，无需重启首轮训练。原始证据及比较见 `reports/step50_validation_and_queue.log`、`reports/step50_validation_result.json`。

根据用户此前授权，已通过 `scontrol update Gres=gpu:8` 将尚未开始的133236、133237、133238、133239改为每阶段申请8张L40S。作业编号和afterok依赖不变；133235仍使用4卡持续训练。Slurm已核查四个待运行作业的ReqTRES均为8卡，实际分配需等待依赖和调度。检查时node10空闲4卡、node11空闲6卡，8卡单节点作业仍须等待足够资源。未停止其他项目作业。

待运行作业保留提交时的sbatch脚本文本，因此 `run_phase.sh` 也从 `SLURM_GPUS_ON_NODE` 获取实际卡数，避免旧脚本的GPUS=4覆盖8卡分配。OPSD使用8卡时accumulation=4，保持全局轨迹batch=32；GRPO batch=512、mini-batch=128、actor/score每卡microbatch=2/4保持不变。两轮各200次更新、每轮169103问题K=5、OPSD及七个完整测试集范围保持不变。8卡全量运行、采样覆盖及吞吐仍待实际启动验证。

近期普通更新约18分钟，首轮剩余148次更新估计还需44–46小时，预计10月4日中午前后完成；后续8卡阶段和最终评测的完成时间待实际吞吐及调度确认。扩卡回执保存在 `reports/scaling_step50_submission.json`，服务器latest_submission.json亦已更新。继续每30分钟核查，正常推进不重复通知。

<!-- latest-monitor-start -->
最新核查（2026-10-03 07:54）：首轮133235完成100/200步，第101步进行中。第100步同配置512条验证NQ EM约0.402、HotpotQA约0.461，宏平均约0.4315；较第50步约0.416提升1.55个百分点，较初始0.24609375提升约18.54个百分点。两数据源均提升，控制台指标三位小数，尚非七个完整测试集结论。global_step_100及global_step_50保留，无非有限指标或近期异常；第100步包含1101.1秒验证和48.0秒保存，总2403.5秒。近期训练完成率下降保留观察，第100步独立EM未下降，继续既有配置。后续四阶段仍申请8卡、等待依赖；完整范围保持两轮各200更新、全量K=5轨迹池及OPSD和七集评测。继续30分钟检查；证据 `reports/monitor_20261003_0753.json`、`reports/step100_validation_result.json`。
<!-- latest-monitor-end -->

第100步验证（2026-10-03 07:54）：NQ约40.2%、HotpotQA约46.1%、宏EM约43.15%，第50步约41.60%、初始24.61%。第100步checkpoint已保存，首轮训练继续，后续8卡请求保持。结果记录 `reports/step100_validation_result.json`；宏EM变化来自同配置留出验证，完整测试结论仍待全流程完成。

第100步后的时间估计：最近20个普通更新中位耗时约19.0分钟；首轮剩余100更新，连同后续验证预计32–34小时，即10月4日下午。后续8卡阶段和最终评测总耗时仍待实测。
