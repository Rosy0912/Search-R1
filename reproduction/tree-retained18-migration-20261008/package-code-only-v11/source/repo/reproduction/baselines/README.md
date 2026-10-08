# Search-R1 上的六种算法移植

基线固定为私有仓库 `shaoshitong/Search-R1` 的
`9ab694499ad3913b67499f7a4ea13da6b544d459`（分支 `search-r1`）。算法实现迁移自旧独立分支 `6818496`，原始审计的b251基点保留在历史报告。
[迁移报告](migration-search-r1-20261002/REPORT.md)说明兼容处理和最新验收。本地独立目录：
`/Users/shaoshitong/Documents/Search-R1-RL-Baselines`；分支
`search-rl-algorithm`。没有修改原复现 agent 的 checkout、H20 环境、数据或正在运行的任务。

当前状态：六种算法的入口、rollout/reward/advantage/optimizer 数据流已实现，
CPU 集成和独立论文/源码审查已完成；A800 两卡 tiny CUDA/FSDP 更新和辅助 ranker
短测试已通过。**正式 3B/vLLM 训练验收、资源预算和全量七集评估仍待完成**。
GPU 测试范围和原始记录见 [A800 测试报告](gpu-validation.md)。

2026-10-02 的[六算法原始审计](audit-20261002/SUMMARY.md)及后续
[修复与独立复审报告](fixes-20261002/REPORT.md)均已保存。已修复审计确认的代码问题；
按用户决定保留 Search-R1 生成/评分上下文差异。最新目标归一化和门禁通过本地测试，
本轮 A800 SSH 连接失败，未取得这些最新改动的 GPU 验收，不能复用历史测试替代。

## 1. 控制变量

继承 reviewed base 的 Qwen2.5-3B **base**、原始 Search-R1 prompt、NQ+HotpotQA 混合
169,615 条原训练数据中按问题固定划分 98%训练/2%阶段验证（六算法与新增GRPO共用），
周期验证按新基座使用validation-64，训练终端切到全部七集51,713题；两者均不参与阶段切换，
Wiki18/E5-base-v2 检索器保持。
四个动作、无额外最终强制生成、每动作最多 500 token、observation 500 token、
起始前缀 2048、rolling/right-side cap 4096、检索 top-k=3、采样温度/top-p=1。
学习率 1e-6，warmup ratio=.285，500 个外层更新（旧 driver 门限 501）。
主 actor 仍全参数训练。模型和数据输入默认共享只读；输出放到另一个目录。

算法固有的 reward、rollout 数量、loss、辅助模型变化如下。它们必须单独报告，
不能把这些移植称为作者原实验配方的直接复现。

| 方法 | 相较 Search-R1 的改动 | 保留/明确偏离 |
|---|---|---|
| GiGPO | 动作前 observation/context anchor；episode + 同题同 anchor 的 discounted step advantage | 保留 EM、5 条轨迹、reference KL；按论文 Eq9 整段文本 action 比率裁剪、episode/action 等权，区别于作者 token PPO；默认 mean-only、γ=.95、ω=1，exact observation grouping；不照搬作者 invalid penalty/近似 anchor |
| Tree-GRPO | 2 个 root，各进行 1 轮×2 个真实 continuation，选 3 个叶子；同题和同树两次 z-score 相加 | 6 叶训练而非原5轨迹；EM、原 parser/tool/四动作；跨root续写合批；共享前缀在叶 loss 中重复，按论文逐叶 token mean 再叶均值，弃叶和 padding 成本计入 |
| BAPO | token-F1 + 格式惩罚 -1、组全错的合法 IDK bonus .5；阶段和多样性 gate；后期整组重采样最多2次 | 不修改任务 prompt；缺 IDK 提示可能让拒答稀疏；paper gate/连续5次停滞，不复制官方 G=8、计数器和阈值硬编码 |
| OPSD | 学生搜索 rollout；冻结同初始化教师只看到训练 accepted answers 提示；full-vocab KL/JSD | answer-only 特权上下文适配，原数据没有 gold rationale；主 actor 全参数，不复制原脚本 LoRA。默认 β=0、温度1、vocab 分量 clip=.05；无旧 reference KL/entropy |
| TOP-D | 独立冻结已训练 teacher；proximal dense reward、未来有效动作 token 均值、同题 pooled token normalization、逐题目标权重及 PPO 内循环 | 默认**无特权答案**；教师与旧学生同一重构前缀；E=2 launcher，纯损失无旧 reference KL/entropy；不是 OPSD，也不是 TrOPD |
| CoSearch | 固定 dense retriever 召回50候选；独立生成式 trainable ranker 选3；主 agent F1/格式-.2，ranker 复合 reward 和自己的 GRPO/AdamW | 用户授权的联合系统扩展；主模型/数据不变，观察经过排序；原论文K=5→保持原Search-R1 K=3；CPU HF ranker backend 替代官方新verl dual-Ray，单独报告成本 |

保留原 `python -m verl.trainer.main_ppo` 入口。没有 `baselines` 配置时仍使用原流程。
原 actor 已存在的 `ppo_epochs` 现在实际生效，默认 E=1 不改变 reviewed recipe。

## 2. 文件和实际接口

| 文件 | 作用 |
|---|---|
| `search_r1/baselines/main.py` | 新 Hydra/Ray 入口，启动前验证必需参数 |
| `verl/trainer/config/ppo_baselines.yaml` | 继承原配置和固定 paper-grpo 输入；六种方法参数 |
| `rollout.py` | 完整 trajectory state、动作前 anchor、step/token provenance、真实 tree continuation；旧 DataProto 专用 row selection |
| `advantages.py` | GiGPO、tree、TOP-D advantage 内核；共享数值安全 PPO helper |
| `objectives.py` | 完整rollout预计算action/trajectory/question权重，DP行分母和GiGPO联合action比率 |
| `holdout.py` | 固定2%问题组划分、SHA校验、UID检查、共同GRPO对照 |
| `preflight.py` | GPU双15%阈值/计算进程检查、持锁、最终资源数和单节点检查、离线资产/检索预检查 |
| `bapo.py` | generated-span 答案解析、合法 action record、F1/IDK/group gate、阶段状态 |
| `distillation.py` | 冻结教师、显式 target-index−1 映射、full-vocab OPSD、全 DP token denominator |
| `actor.py` | OPSD 独立 actor loss/update，多内循环、教师不进 optimizer/vLLM |
| `cosearch.py` | 实际 HF ranker generation/scoring/独立 optimizer、semantic grouping、reward、checkpoint |
| `controller.py` | 接入原 trainer generation、reward、advantage、ranker 更新、独立阶段验证、算法状态保存 |
| `verl/trainer/ppo/ray_trainer.py` | opt-in hooks；验证时 CoSearch 使用当前 ranker；其余验证仍原 EM 流程 |

`step_ids` 与 response token 一一对应，observation/pad 为 -1；`info_mask`/`loss_mask`
阻止外部文档参与 actor loss。anchors、tree/trajectory UID、action records 存在
逐行 `non_tensor_batch`，随负载平衡一起重排；分组按稳定 uid，不依赖相邻5/8行。
答案标签不可跨 observation 拼接，也不读 prompt 示例的答案。

树状态深拷贝保存真实 rolling prefix、原 prompt、全部保留 response、mask、已用预算
和 observation。分叉不重新搜索共享前缀、不重置四动作预算；稳定 node ID 去重候选。
预算耗尽不标成环境已完成。零 token 动作仍进入 GiGPO discount 时间轴和 anchor baseline。

TOP-D：`r=logaddexp(log(α)+teacher_logp-old_logp, log(1-α))`，默认 α=.1；
每个 token 的 return 为其 reward 加**后续**有效 actor token reward 均值。
reward/advantage detach 后在 E 个内循环中固定，current logprob 每次重算。
TOP-D 完整rollout预计算每题 token 数和 B/(Q×Nq) 权重，minibatch用全DP行数分母；
完整rollout目标对题等权，固定排列mini是这个加权目标的mini优化，不能宣称每mini完整题等权。
Tree/CoSearch先每轨迹token平均再行均值；GiGPO先整action概率比裁剪再动作/episode平均。
OPSD维持全DP有效token平均，BAPO/GRPO保留作者/基座token PPO。

OPSD：β=0 forward KL(T||S)、β=1 reverse KL(S||T)，中间为 weighted JSD。
clip 在 vocab 分量求和**之前**，与官方实现一致；不是总 token KL clip、也不是 top-k
归一化近似。teacher 和 student token-to-ID vocabulary 必须完全一致；特权答案不会
进入 student、tool、ranker 或验证输入。只支持 SP=1、固定 microbatch，其他组合明确拒绝。

CoSearch：ranker prompt 只含原问题/子查询/候选，不含 gold；训练 reward 才读 gold。
合法排序必须恰好K个不同、在范围内的整数ID。非法排序回退 dense 前K保证工具可用，
但 ranker 格式 reward=-.2。初始问题内按子查询 token-F1≥.8 贪心聚组、过滤组大小<3及全同奖无信号组；
默认按论文不额外按 answer availability 拆组，可显式启用官方源码的拆组变体。
相关性为 Hit@1/3/5 均值；候选含答案且 relevance≤.5 时只用 relevance，否则含答案
则 relevance+main reward，无答案则 main reward。非法主轨迹不训练 ranker；文档匹配采用作者NFD精确token串，yes/no/true/false答案不作相关性伪监督。
ranker响应序列等权，informative组的零adv成员仍保留。
生成使用干净 GenerationConfig，关闭 checkpoint 附带的 repetition/no-repeat processors，
使 rollout 与 old/new 温度 logprob 分布一致。

## 3. 启动（目前没有自动启动生产 GPU 任务）

先分配独立 GPU 作业，使用 reviewed 环境的 Python，进入本分支 checkout。
例如已有资源仍在 `/data/search-r1`，代码部署到独立目录：

```bash
cd /data/search-r1-rl-baselines/repo
export PATH=/data/search-r1/env/bin:$PATH
export SEARCH_R1_ASSETS=/data/search-r1
export SEARCH_R1_BASELINE_OUTPUT=/data/search-r1-rl-baselines
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # 仅填写已分配给这个独立作业的GPU
bash reproduction/baselines/run.sh gigpo
bash reproduction/baselines/run.sh tree_grpo
```

上述两条是备选任务，不能在同一组 GPU 上同时启动。脚本不杀其他进程，
不接管原 pipeline，也不覆盖其 checkpoint。默认 offline；所需模型必须先准备在目标节点。
下载按用户 memory：目标远端直下→失败时本地中转→全部校验后删除仅本次中转副本。

```bash
bash reproduction/baselines/run.sh bapo

# 同初始化的冻结教师 + answer-only 特权context，保持学生全参数训练。
export OPSD_TEACHER=/data/search-r1/assets/qwen2.5-3b
export OPSD_GLOBAL_MICROBATCH=8             # 8 ranks时每rank=1；仍需显存实测
bash reproduction/baselines/run.sh opsd

# 独立已训练且 tokenizer 兼容的教师，不能使用同一未训练学生初始化。
export TOPD_TEACHER=/path/to/frozen-trained-qwen2.5-3b
bash reproduction/baselines/run.sh topd

export COSEARCH_RANKER=/path/to/auxiliary-generative-ranker
bash reproduction/baselines/run.sh cosearch
```

所有方法的新入口自动从原训练集按完整prompt组固定选择2%，seed42；重复题同侧，
阶段集排除最终测试的相同prompt。产物写入独立common-data目录、带源/产物SHA，原数据不改。
BAPO每次实际actor update后独立阶段验证（默认频率1），连续5次无改善转后期；
最终七集测试仅报告，不控制阶段。2%是独立问题比例，有重复题时行数比例可略异。
每步约数千题验证会增加成本，可显式调整stage_validation_freq并报告新阶段间隔。

启动器自动匹配CUDA_VISIBLE_DEVICES与Ray本机GPU数，持跨checkout的本用户GPU锁，
双阈值≤15%且拒绝已有compute process；实际运行只支持单节点。
直接绕过启动器、尾部override超配GPU或nnodes>1会在模型/Ray加载前拒绝。
独立TMPDIR/RAY_TMPDIR/HF缓存/output，不启动或停止他人的服务；检索HTTP和模型/数据预检查失败即退出。
这些锁不能取代外部调度器的独占分配；其它用户不会自动遵守本用户锁。

辅助模型路径由运行者显式提供；不会自动下载大模型或复制账号凭证。
全量训练前可用原环境执行 CPU gate，再单独安排2个外层更新的 GPU smoke，
检查非零参数更新、loss/logprob finite、梯度 mask、各模型实际 GPU/RAM、checkpoint
和验证完整性，随后才进行正式训练。CPU gate 命令如下：

```bash
python -m unittest discover -s tests/baselines -v
python -m unittest discover -s tests/reproduction -v
```

## 4. 资源、成本和已知边界

- rollout 每个外层更新记录实际 `generated_tokens`、`prefill_tokens`、`retrieval_calls`、
  `trajectories`、`padded_rows`，包含被弃树叶、BAPO重采样、真实GPU padding结果。
  轨迹计数包括 continuation，不等于仅最终训练叶数。
- actor 记录真实 `optimizer_steps`；正常总数为 E×每rank minibatch数；全局空监督mini共同跳过，另列skipped_empty_minibatches。
  局部空rank仍参与backward；无实际optimizer step不推进scheduler/BAPO阶段。
- TOP-D teacher 默认一个 driver CPU 模型，OPSD 一个冻结CPU模型/**每actor rank**。
  记录参数数和累计 scoring token。CPU方式解决无网/独立资源接入，不保证训练吞吐；
  3B FP32 每份权重约12GB，8份会明显占RAM，另有模型激活和加载峰值。
- OPSD 分块 loss **仍保留完整 student logits 和各 chunk autograd 图**；chunk16不意味着
  memory bounded。默认起点每rank micro1，full-logits显式估计预算4GiB；这只是提前拒绝
  过大分配，不能替代真实峰值显存验收。tiny FSDP 路径已通过；正式 3B、remove-padding 和 offload 路径仍待 GPU 验收。
- ranker 默认 CPU unsharded FP32，独立 AdamW；可指定额外专用设备，但不得挤占原任务。
  大 ranker CPU 训练可能很慢；50候选超过model context会明确报错，不偷偷丢文档。
  记录生成/prefill、scoring/training forward tokens、optimizer steps和参数数；validation
  的生成/prefill另列，不把辅助模型成本算成零。
- 继承 reviewed base 的滚动上下文裁剪和最终重构轨迹重算 old logprob 行为。
  长轨迹的实际 generation conditioning 与最终 full-trajectory scoring 可能不同；
  这是原流程的限制，尤其不能据此宣称 TOP-D 理论条件或 PPO on-policy 性质已在
  长上下文严格满足。每次动作记录上下文是否一致，并报告 `rollout/context_mismatches`。
  设置 `baselines.context_alignment=error` 会在不一致的动作生成前明确拒绝；默认
  `report` 保留共同基线行为。用户明确选择不修复此项，六算法和GRPO共用原规则，默认只报告。
- checkpoint 包含原 actor 权重、ranker权重/tokenizer/optimizer以及算法 state.json。
  state manifest 在辅助保存后原子写入；BAPO best/stalled/early/patience一同保存。
  原 driver 没有完整 optimizer/dataloader/RNG恢复入口，本分支明确
  `resume_supported=false`，不承诺中断后精确恢复。冻结教师不复制到actor checkpoint。

## 5. 验证证据与审查

本地 Python3.9/Torch2.8 的 `.venv-cpu` 是独立测试环境，不替换H20的Torch2.4/vLLM0.6.3。
最新整合测试及独立反例证据见[修复报告](fixes-20261002/REPORT.md)。覆盖真实离线tiny HF、
六方法actual fit（GPU transport mock）、两进程Gloo/DDP actual actor不同微批/跨题rank、
空监督AdamW动量保持、作者scorer/matcher差分、真实fcntl锁、stage/UID边界等。
Linux procfs检查在Mac跳过。此轮没有新的GPU成功运行，旧tiny证据仅代表旧代码。

独立审查记录：[GiGPO/Tree](review-gigpo-tree.md)、[TOP-D/OPSD](review-topd-opsd.md)、
[BAPO/CoSearch](review-bapo-cosearch.md)。首轮、二轮发现并修复的具体问题保留在报告里。
算法审查 PASS 范围是 CPU 测试和源码审查；新增真实 GPU 测试的范围单列于
[A800 测试报告](gpu-validation.md)，不能据此宣称正式训练或论文指标已复现。

## 6. 论文和固定源码

| 方法 | 论文 | 审查的官方源码commit |
|---|---|---|
| GiGPO | [2505.10978v2](https://arxiv.org/html/2505.10978v2) | [verl-agent/20bd331](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8) |
| Tree-GRPO | [2509.21240v1](https://arxiv.org/html/2509.21240v1) | [Tree-GRPO/19bf3fa](https://github.com/AMAP-ML/Tree-GRPO/tree/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59) |
| BAPO | [2601.11037v2](https://arxiv.org/html/2601.11037v2) | [BAPO/8ddbb4f](https://github.com/Liushiyu-0709/BAPO-Reliable-Search/tree/8ddbb4fa23f105d52d3621d232f839212e895c3f) |
| CoSearch | [2604.17555v2](https://arxiv.org/html/2604.17555v2) | [CoSearch/763bf8c](https://github.com/snap-research/CoSearch/tree/763bf8c4d7caa363ad845d39ddfd53b81ae377bd) |
| OPSD | [2601.18734v3](https://arxiv.org/html/2601.18734v3) | [OPSD/ae7d251](https://github.com/siyan-zhao/OPSD/tree/ae7d2519e94920c4eb6206c0c26de46d9c50abae) |
| TOP-D | [2607.04751v1](https://arxiv.org/html/2607.04751v1) | 未核实官方源码，按论文审查；未核实NeurIPS录用信息 |

这里只实现这六项；TOP-D没有被替换成名称相近的另一算法。
