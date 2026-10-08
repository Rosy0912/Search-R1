# Search-R1 六算法复现交接计划（2026-10-02）

本计划供六个独立 agent 执行；本轮只准备文档、分支和资源候选，不启动训练、不持有资源预留。现有六算法已实现，CPU 验收 134 passed / 1 skipped；最新代码尚未完成真实 3B 搜索训练的 GPU 验收，不能把“实现完成”当作“复现成功”。

代码冻结起点：`search-rl-algorithm@0723dca0c1458482ac3e78f8ed43af04893955d2`，祖先为 `search-r1@9ab694499ad3913b67499f7a4ea13da6b544d459`。仓库 https://github.com/shaoshitong/Search-R1 。Search-G1 仅作为另一路参考，不把其 probe/reward shaping 混入本计划。

## 1. 六个任务、分支和 GPU

| Agent | Algorithm / launcher | Branch | H100 node | Physical GPU IDs | Count | HTTP / Ray port |
|---|---|---|---|---|---|---|
| 1 | GiGPO / gigpo | reproduce/gigpo | h100-8 | 0-7 | 8 | 18081 / 16381 |
| 2 | Tree-GRPO / tree_grpo | reproduce/tree-grpo | h100-9 | 0-7 | 8 | 18082 / 16382 |
| 3 | BAPO / bapo | reproduce/bapo | h100-5 | 0-7 | 8 | 18083 / 16383 |
| 4 | OPSD / opsd | reproduce/opsd | h100-10 | 0-7 | 8 | 18084 / 16384 |
| 5 | TOP-D / topd | reproduce/topd | h100-11 | 0-7 | 8 | 18085 / 16385 |
| 6 | CoSearch / cosearch | reproduce/cosearch | h100-16 | 0-7 | 8 | 18086 / 16386 |

最新用户决定：优先 H100 > H20 > A800 > 4090，能8卡就8卡；A800仅4卡。**有进程不再直接排除，只要利用率≤15%且显存占用≤15%即可分配。** 当前快照2026-10-02 20:55:52北京时间，六台H100整机共48卡满足双阈值，因此本方案六个算法分别独立8卡H100，可并行准备及训练（TOP-D等待教师）。

h100-5、9每卡约1131MiB，h100-11、16每卡约1135MiB，利用率0%，均有一个小显存进程；原进程保持不动。h100-8/10没有compute process；h100-4/7每卡约25–30GB显存超15%，不分配。H20已有任务、A800 GPU7约84%显存占用，仍排除。监控表目前“低占用候选”标签额外要求无进程，本计划根据原始util/used/total重新计算，不沿用标签。

**先决适配：当前preflight拒绝任何compute process，不符合用户最新规则。** 公共setup负责人必须将其改成仅以双15%阈值决定GPU是否合格，仍采集并记录进程，未知/查询失败拒绝；保留UUID租约、可见GPU验证和启动前复查。必须增加有低占用进程可通过、任一阈值超标仍拒绝的有意义测试。此改动尚未实现，本轮仅写计划；不能通过绕过preflight或kill既有进程来启动。节点需按Slurm实际调度规则预约；不借用别人的活跃作业allocation，训练过程中资源变化按计划暂停自身任务再报告，不结束原进程。

GPU是候选分配，并非已预留；每agent启动前复查双阈值及Slurm分配。TOP-D教师准备完成前可做CPU准备，不先加载学生占满卡。用户现在授权H100进入本次六算法训练计划，不扩展Codex凭据同步或软件全局升级范围。

候补顺序同用户优先级：释放且合格的H20整机8卡；其次A800 GPU0–3四卡（真实入口重试已成功，GPU7排除）；最后4090。切换候补要更新具体UUID/卡型/实验配置/成本，不能无记录切卡型。主方案已满足六个算法全部H1008卡，无需使用这些候补。

## 2. 各算法原理与必须保留的行为

| 算法 | 原理 | 本实现关键规则 / 验收重点 |
|---|---|---|
| GiGPO | 组合整条轨迹回报优势和同一问题、相同观测锚点下的步骤优势，改善长搜索过程信用分配 | gamma=.95、omega=1、观测精确哈希、默认只中心化。严格 joint textual-action ratio=`exp(sum(token logratio))`，按动作裁剪、动作等权；与作者源码 token-PPO 有意不同，报告须说明。空动作不参与 ratio，但保留时间线。 |
| Tree-GRPO | 对搜索轨迹作共享前缀的树状续写，用问题组和树内组的相对优势训练 | 2 roots、1轮、2 expansions、每root3 leaves，共6叶；真实共享前缀，问题与树 zscore 相加；每叶 token 均值后叶等权。不要改回独立完整采样或添加作者的强制最终回答。 |
| BAPO | 根据模型阶段切换边界/不确定性奖励和重采样，使错误组产生有效学习信号 | yes/no/noanswer F1、合法IDK奖励、early ratio<.05、late diversity<G/2；每次实际 actor outer update 独立阶段验证，连续5次不改善切阶段，后期最多2次重采样。沿用作者 token-GRPO surrogate。阶段验证很昂贵，不擅自减少频率。 |
| OPSD | 让冻结教师看到训练答案形成特权分布，学生只看原搜索输入；用分布蒸馏更新学生 | 教师同一 Qwen2.5-3B base 初始权重，answer-only 特权，无额外 rationale；全词表 FKL/JSD（beta=0 默认FKL）、温度1、component clip=.05。禁用额外KL/熵；每rank微批1、SP1。teacher每actor rank在CPU，8卡有8份教师。 |
| TOP-D | 把教师信号转成信赖域的平滑 proximal reward，再对学生策略进行更新 | alpha=.1、logaddexp稳定实现、当前及排除自身的未来有效token均值、同题白化；全rollout按问题权重 B/(Q*Nq)，2 PPO epochs。教师必须独立已训练且tokenizer兼容，不输入答案；固定minibatch不保证每批都等同完整问题目标，须记录。 |
| CoSearch | 主搜索策略与可训练生成式文档ranker共同优化；ranker从50候选中选3个 | 相同问题query F1≥.8分组、最少3条，奖励无差异组跳过；答案NFD精确token匹配，排除boolean伪监督；主连续F1/格式奖励与ranker Hit@1/3/5及主reward门控。ranker默认CPU FP32独立优化器；prompt不含gold。 |

数学和来源对应见 `reproduction/baselines/README.md`、`audit-20261002/`、`fixes-20261002/REPORT.md`、`migration-search-r1-20261002/REPORT.md`。这些审查报告是历史记录，不应把其旧 SHA 当作本轮代码 SHA。每个 agent 必须在自己的分支提交后再做论文/原始代码对照审查，并解释有意偏离。

## 3. 统一实验协议（六算法与新 GRPO 对照）

- 主模型固定 Qwen2.5-3B **base**，不能换主模型为Instruct；保持 Search-R1 数据、prompt、检索语料/index/E5模型、四轮动作和搜索任务。
- 从原训练集按完整问题划分2% holdout、98%训练，seed42；重复题同侧；holdout排除周期/最终测试题。用当前入口自动生成并核对所有任务路径无关的split内容身份及产物SHA一致（第8节）；原 train.parquet 不覆盖。所有方法包括新GRPO用同一规则。
- 起始长度2048、rolling4096、动作/检索观测各500、最多4轮、不强制final。按用户决定保留原 Search-R1 的生成窗口与训练历史差异，所有算法一致，不另行修复。
- 全局 train batch512、mini256、lr1e-6、warmup .285、TP1、SP1、rollout温度/top_p1；一般每题5轨迹，Tree为6叶，属于算法区别。
- 主方案8卡统一 `ppo_micro_batch_size=8`（每rank1）；备用4卡统一为4，包括GRPO；比旧默认64更保守，属共同显存适配，需要验收并报告，不能把旧100%训练/不同micro的GRPO直接当公平对照。若显存不够，先查真实峰值，不静默改长度、batch、数据或算法。
- 正式500个实际outer update，默认total_training_steps501；test_freq50/save100/acceptance_step5。完整七集合最终评估51,713题；周期validation-64用于周期日志，BAPO独立2%阶段集用于阶段切换。最终测试绝不用于阶段/教师/调参选择。
- 各方法同时报告F1/EM/格式率、搜索次数、生成token、samples、outer updates、ranker/teacher CPU时间和RAM、GPU小时、检索CPU成本、墙钟、显存峰值。Tree多样本、BAPO重采样/阶段验证、TOP-D多epoch的额外成本单列，不能仅用步数比较计算成本。

GiGPO agent兼任一次**新 GRPO 对照**：先在自己的8张H100上串行运行 `grpo`，在2% holdout上选择教师checkpoint，然后将教师导出为独立不可变HF目录，核对tokenizer和权重/hash，交TOP-D，再运行GiGPO。GRPO代码/配置提交 `reproduce/grpo-control`，不混入GiGPO算法变更；teacher训练只能用98% train。旧H20活跃训练和其 checkpoint 默认不作为教师，避免holdout泄漏、并避免干扰另一个agent。

TOP-D 的教师选择是明确依赖，不是假定已经有可用教师。可使用新GRPO训练所得兼容检查点，但这不保证复现论文的最佳教师质量；报告teacher的holdout表现、训练来源和成本。若效果要求更强教师或该选择与论文设置冲突，先向用户说明。不要直接使用Qwen2.5-3B-Instruct：其EOS可能不同，当前FrozenTeacher要求词表、BOS/EOS一致。

CoSearch 的首次可运行ranker warm-start采用相同本地base模型目录（独立实例、独立优化器、不改资产）；这保持主模型/数据，辅助模型初始化与作者配置差异必须在报告中说明。先测50文档输入长度/合法ID格式；不能因失败就改成固定排序。若要换作者ranker或增加确定性passage截断，先说明差异并取得用户方案选择。

## 4. 分支与目录隔离

六任务分支从同一冻结代码起点建立；同名已有分支不得强推覆盖。新公共部署适配由GiGPO agent协调，在 `reproduce/common-setup` 完成，独立审查后冻结一个公共提交，六分支各cherry-pick同一提交；先冻结公共环境，再进行算法特有更改。TOP-D不要自行实现另一套数据/检索准备。共享只读资产可以复用，每个算法的env、checkout、output、cache、Ray、tmux独立。

H100 persistent roots (verified local data mounts; check permissions/quota before creating):

| Node / algorithm | Agent root | Data disk available |
|---|---|---|
| h100-8 / GiGPO | /mnt/data/shaoshitong/search-r1-rl/gigpo | 12TB |
| h100-9 / Tree-GRPO | /dataset/shaoshitong/search-r1-rl/tree-grpo | 7.7TB |
| h100-5 / BAPO | /dataset/shaoshitong/search-r1-rl/bapo | 3.3TB |
| h100-10 / OPSD | /mnt/data/shaoshitong/search-r1-rl/opsd | 4.6TB |
| h100-11 / TOP-D | /dataset/shaoshitong/search-r1-rl/topd | 9.3TB |
| h100-16 / CoSearch | /dataset/shaoshitong/search-r1-rl/cosearch | 7.4TB |

All six have192 CPUs,about2TB machine RAM,and/usr/bin/python3.10. Actual job RAM/CPU depends on cgroup/Slurm allocation; do not assume all machine RAM is allocated. uv availability must be checked and installed only into the project tool prefix if absent. Separate repo/env/assets-root/runs/logs per algorithm; shared assets only immutable with owner/hash manifest. Never use existing docker/data project directories. Node16 also relays SSH, so keep own CPU/RAM quotas and services isolated.

备用方案目录：GiGPO `/root/search-r1-rl/gigpo`（A800 overlay剩余约13TB，检查平台持久性）；Tree `/data/shaoshitong/search-r1-rl/tree-grpo`（hkust1）；BAPO `/weka_2/shaoshitong/search-r1-rl/bapo`；CoSearch `/mnt/95a003e2-1274-441f-88bf-ca7f5b636032/shaoshitong/search-r1-rl/cosearch`。HKUST默认SSH PATH没找到Python3.10/uv，不代表没安装，先查隔离环境。hkust2根盘剩12GB，不作为主分配节点；使用它时所有缓存/环境/输出放/data。

H100路由是 Mac→mgmt-host→h100-16→worker，使用现有内部SSH配置，不猜IP、不复制密钥。例：

```bash
ssh mgmt-host 'env -u LD_LIBRARY_PATH ssh h100-16 "env -u LD_LIBRARY_PATH ssh h100-8 hostname"'
```

`control.py run`当前只接受原8个外部SSH别名，不支持H100训练；不要用`control.py run h100-8`。登录h100-16后先检查`sinfo/scontrol`、可用partition/account/QOS，按集群规则为目标节点申请单节点8GPU独占作业；模板为`salloc -p gpu -w h100-8 -N1 --gres=gpu:h100:8 --ntasks=1 --cpus-per-task=32 --mem=256G`，参数须按实际partition、GRES格式、内存/QOS限制调整；此处不是已经获批的预约。训练在`srun`分配的上下文内启动，尊重调度器给出的CUDA_VISIBLE_DEVICES，核对实际UUID。Slurm显示IDLE仍需查GPU进程，以免系统外已有任务。若无法取得独占预约，等待并向用户报告，不能root直接绕过调度。首次部署可先在各自目录准备CPU环境，GPU验收也必须预约。

Mac拉取分支后将代码bundle/归档传到远端独立repo也可以；不要给远端复制Mac私钥或把账号凭证写进仓库。Mac上的本地Codex+SSH能执行整个训练任务；远端Codex API不可用时直接SSH，不把安装/认证作为训练先决条件。

HKUST连接使用 `ssh -o ProxyCommand=none hkustN`（需要EasyConnect校园路由）。A800普通域名目前走BoostNet fake-IP，重试成功用：

```bash
ssh -o HostName=110.43.160.159 \
  -o 'HostKeyAlias=[ssh.platform-multi.baai.ac.cn]:2222' \
  -o ProxyCommand=none baai-a800 'hostname'
```

真实IP来自本轮独立DNS与mgmt-host解析一致，仅作为临时会话覆盖，未修改~/.ssh/config。入口IP可能变化，重试前核对DNS；主机身份仍匹配原known_hosts，禁止删除已有key来绕过验证。登录存在既往RSA hostkeys扩展签名警告，但认证及采集成功。

## 5. 公共新集群准备：先完成这些，再执行训练命令

**这些是待执行的公共部署工作，并非当前已有的一键新集群安装能力。** 当前训练入口存在，但原H20辅助脚本不能原样套到每个新目录。

1. `common-setup`负责人清点模型/语料/E5/index/原parquet的固定revision、大小和SHA；只读复用既有manifest。优先目标机器直接下载；卡死/不联网时Mac下载Linux x86_64适用wheel和资产，依次传目标，全部校验后才删除本次中转副本。规则见 `/Users/shaoshitong/Documents/remote_codex_control/memories/download-transfer.md`。原H20部署可只读核对来源，不能改其目录或抢用其检索进程。
2. 将 `download_assets.py`、`prepare_paper_data.py`、`verify_assets.py`、`retriever_local.py` 中硬编码 `/data/search-r1` 改成明确资产/代码根路径。保留原数据变换、prompt、row/schema及asset hash；不要运行原 `run_pipeline.sh` 把新任务落进原H20项目。
3. `bootstrap.sh`已经支持SEARCH_R1_BASE/SEARCH_R1_PYTHON。工具齐全、完整git历史（不能缺其固定上游commit）且repo就位后使用下述命令。每任务独立env，核心固定Linux x86_64/Python3.10、torch2.4.0、vLLM.6.3、transformers4.46.3、Ray2.40.0、tensordict.5.0、numpy1.26.4、xformers.0.27.post2、flash-attn2.7.4.post1。以requirements-h20/constraints-h20为权威锁文件，不随意升级最新版。

```bash
export SEARCH_R1_BASE="$AGENT_ROOT"
export SEARCH_R1_PYTHON=/absolute/path/to/python3.10
bash "$AGENT_ROOT/repo/reproduction/h20/bootstrap.sh"
source "$AGENT_ROOT/env/bin/activate"
```

Torch正常依赖先安装，cuBLAS12.4.5.8是既有H20验收的有意覆盖，bootstrap最后--no-deps安装；不能让resolver重新乱配。新H100/4090/A800仍需验证兼容性。Flash wheel SHA256 `a8719a86d0cd12a105739151f9e45ac8c2ecb840a83ed7a080a88d1645a98440`。离线模式先完整准备wheelhouse/flash wheel再BOOTSTRAP_OFFLINE=1，不照搬Mac arm64环境。

4. **检索资源必须与训练卡隔离。** 现有retrieval_server.py无论是否--faiss_gpu都无条件将E5和输入.cuda()，HTTP端口8000也硬编码。因此省略--faiss_gpu并不等于CPU检索，直接在训练卡起服务会被preflight拒绝。公共部署适配增加显式CPU device、host/port参数，把encoder和输入均移CPU，CPU FAISS仍用原index；绑定127.0.0.1及表中各端口。当前不存在这些CPU CLI参数，必须先实现、审查再使用。对固定真实query验证top3/top50与原检索的内容/排序/数值误差，记录并处理不一致，不能换mock/BM25或改变任务。H100整机RAM提供条件，实际作业获配RAM是否可容纳须以加载实测为准，限制每检索服务线程预算例如8 CPU并测吞吐。CPU检索过慢时，另行申请未分配的检索GPU并单列成本，不能偷偷使用训练卡或A800 GPU7。
5. `cuda_preflight.py`原硬编码8卡sum ranks==28；公共适配为实际卡数 `n*(n-1)/2`（8卡28、4卡6），验证BF16/NCCL/flash/vLLM基础推理。不能以原八卡检查失败判定四卡不支持，更不能忽略真正通信失败。
6. 运行完整CPU gate（README所列测试）、真实检索schema和topk验证、实际8卡基础通信。然后各算法2 outer update smoke，检查loss/logprob/梯度finite、非零参数更新、mask、CPU/GPU峰值、split hash、实际n_agent/teacher/ranker更新。
7. 验收通过后跑5步保存/验证接受性，确认导出/加载检查点，再启动正式预算。当前完整actor+optimizer+RNG+dataloader精确续跑尚不支持，不宣称中断后等价恢复；用tmux保证一次活跃实例，断连重attach，不重复resume第二个训练进程。

## 6. 统一启动模板与独立 Ray

公共部署和依赖验收完成后，在对应agent远端会话中执行。`AGENT_ROOT`、`METHOD`、`GPUS`、`HTTP_PORT`、`RAY_PORT`按表填写；`RUN_ID`每次新建。启动前用调度器获得独占资源，检查GPU双阈值和记录现有进程；下面示例不能代替资源预约。

```bash
export AGENT_ROOT=/mnt/data/shaoshitong/search-r1-rl/gigpo
export METHOD=gigpo GPUS=0,1,2,3,4,5,6,7 HTTP_PORT=18081 RAY_PORT=16381
export RUN_ID="${METHOD}-$(date +%Y%m%d-%H%M%S)"
source "$AGENT_ROOT/env/bin/activate"
cd "$AGENT_ROOT/repo"
export CUDA_VISIBLE_DEVICES="$GPUS"
export SEARCH_R1_ASSETS="$AGENT_ROOT/assets-root"
export SEARCH_R1_BASELINE_OUTPUT="$AGENT_ROOT/runs/$RUN_ID/output"
export SEARCH_R1_BASELINE_RUN_ROOT="$AGENT_ROOT/runs/$RUN_ID/runtime"
mkdir -p "$SEARCH_R1_BASELINE_OUTPUT" "$SEARCH_R1_BASELINE_RUN_ROOT/ray"
export TMPDIR="$SEARCH_R1_BASELINE_RUN_ROOT/tmp"
mkdir -p "$TMPDIR"
# 预先启动已完成公共适配的独立CPU检索服务；不使用当前尚不存在的CLI参数。
# 以该服务真实健康检查通过为后续先决条件。
ray start --head --port="$RAY_PORT" --num-gpus=8 --num-cpus=24 \
  --object-store-memory=8589934592 --include-dashboard=false \
  --temp-dir="$SEARCH_R1_BASELINE_RUN_ROOT/ray"
export RAY_ADDRESS="127.0.0.1:$RAY_PORT"
bash reproduction/baselines/run.sh "$METHOD" \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  retriever.url="http://127.0.0.1:$HTTP_PORT/retrieve"
```

Ray ports must be free. Each algorithm owns one separate H100 node and one Ray instance. Preserve existing low-memory processes; never use global ray stop or process-name kill. Slurm GPU visibility must match all eight allocated UUIDs. The 4-GPU A800 fallback changes GPU count and global microbatch to4; keep the shared data/model protocol and record the hardware difference.

算法特有环境和启动调用见 `agents/` 六个文件；每个文件可连同本PLAN交给对应agent。OPSD/CoSearch的CPU辅助模型不额外占训练GPU；改变到GPU必须重新申请单独辅助卡并报告成本。OPSD当前full student logits仍整体分配，chunk16不等于总显存受控，8卡80GB也需真实峰值验收。

## 7. 执行顺序、停止条件与交付

顺序及TOP-D教师依赖以第8节的精确启动顺序为准。六分支合入同一公共部署提交；其他方法可在自己的独立节点验收后并行训练。TOP-D只能在教师交付后进行算法smoke和正式训练。

遇到算法核心与Search-R1冲突、需要截断CoSearch 50候选prompt、改变BAPO阶段验证协议、教师质量/兼容性不足或8卡OOM需要改变实验预算时，说明具体冲突和选项，请用户决定。一般依赖/路径/端口问题按公共计划修，不擅自动数学。已有用户授权保留上下文差异，不再反复询问。

每agent产物：自己的分支提交（不推main/search-r1/search-g1）、固定最终commit和source SHA、公共commit、resolved Hydra配置、模型/数据/teacher/ranker hash、GPU UUID和占用快照、环境freeze、smoke日志、完整训练/评估日志、指标表与计算成本、`REPORT.md`、严格独立subagent审查（代码/数学/逻辑/infra，论文+原代码逐项对照）。报告区分验证通过、未完成、偏离作者协议。不要将auth.json、私钥、checkpoint大文件或敏感日志提交Git。

本地最终报告建议 `/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/<method>/REPORT.md`；远端原始大产物留各agent runs目录。采用本地codex账号路由时遵守account-switching memory，不能靠memory热切当前App账号，不能覆盖auth.json或将本地账号切换当远端凭据同步。

## 8. 审查补充：跨路径数据身份、教师选择与启动依赖

**数据身份**：当前holdout manifest的spec包含绝对输入路径，六台机器/不同目录的manifest ID和整个manifest字节可能不同。因此本计划的“一致”指源文件内容SHA、split fraction/seed/算法版本、问题集合/行集合摘要以及产出parquet SHA一致，不能要求本地路径不同的manifest原文或ID一致。公共setup增加路径无关的content_identity（规范化排序输入SHA、参数、问题/行哈希）并保留各机绝对路径为provenance；不改分组/切分数学规则。每agent对比该身份及产物SHA，记录本地manifest。

**教师选择**：当前GRPO只有validation-64与最终测试评估，没有从2%holdout选择checkpoint的现成流程。公共setup必须添加并审查只用common holdout的检查点评估器，再执行教师选择。候选为完成实际update 5、100、200、300、400、500所保存的完整actor HF目录；以2%holdout的平均答案F1最高选择，F1相同时选最早update。使用原Search-R1搜索/生成设置及固定评估seed42；这个选择器不得读取最终七集，也不得用最终测试选教师。标准正式评估仍独立报告。若保存频率/实际步数导致缺失候选，先修公共验收保证上述六个产物，不静默换选择集合。save_checkpoint已经输出HF权重/tokenizer，选择复制的是这个完整HF目录，不是raw FSDP分片。比较词表/BOS/EOS、student与teacher权重hash差异，并交付teacher manifest/holdout指标/选择理由。现成selector尚未实现，需先完成；不是凭假设选择教师。

**精确启动顺序**：公共部署冻结→全部任务通用CPU/硬件通信/检索检查→GRPO及非TOP-D方法的算法smoke→新GRPO完整控制训练及holdout-only教师选择→TOP-D算法smoke/5步验收→TOP-D正式训练。其他方法验收后可在各自独立节点并行正式训练；TOP-D交付教师前只做CPU准备和通用硬件检查，不能使用未训练base教师替代其算法smoke。
