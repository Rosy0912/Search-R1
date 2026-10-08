# Tree-GRPO 独立审查（2026-10-02）

**结论：CONDITIONAL。** 固定实现确实执行树分叉，独立 intra/inter-tree advantage 与论文及作者默认 `tree` 路径一致；不是把独立完整轨迹贴 tree UID。尚不满足正式 Qwen2.5-3B/vLLM 训练验收：长上下文条件分布错配和共享 infra 的空 loss 微批问题应先处理，成本与作者配方差异需要实测并披露。未发现 Tree-GRPO 独有 P0。本审查未修改实现、未启动远端作业、未 kill GPU、未 commit/push。

## 固定证据与范围

- 本地审查 SHA：`7b93fbd01742715dc14290b0a6e2d3b42e6c19f7`（实际 `git rev-parse HEAD` 核实）；Search-R1 reviewed base：`b251ba6b503217a59cd3395c1326d7b66b602989`。
- 主文：[Tree Search for LLM Agent Reinforcement Learning v1](https://arxiv.org/html/2509.21240v1)，§3.1、§3.2 Eq.6–8、§4.1；独立重新读取，未沿用已有 PASS。
- 作者源码：[固定 commit 19bf3fa](https://github.com/AMAP-ML/Tree-GRPO/tree/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59)。缓存 `/tmp/rl-gigpo-tree-audit/Tree-GRPO` 不是 Git checkout，故对以下文件逐个下载固定 SHA raw 内容并与缓存逐字比较，五项均 MATCH：

| 文件 | SHA256 |
|---|---|
| search_r1/llm_agent/generation_ts.py | 8189df2c33d00300a75ccce044e3017cf3197c3f43943bc4e3cf204644e70d38 |
| search_r1/llm_agent/tree_node.py | 643d2ee589fce25b09e973287f8aca93a136a31d8d739f6dd928c69c2afb6b96 |
| verl/trainer/ppo/ray_trainer_ts.py | be85b049d3d05fef367c8525d4a3a3342eb0b9efb570b843d16cd9625e818e39 |
| verl/trainer/ppo/core_algos.py | 6d62c3a6c20e3621c9cddef1a5ab7252867489aa44028274304124bc37813363 |
| train_multihopqa_tree_search.sh | d1dd6b8dda6ffbe7c6c97c72132c56083ea3d2c3ccc213d3cdeb36789807cba0 |

## 串行机制与公式映射

1. `ray_trainer.py:733` 把每题重复 `n_agent=2` 次。`SearchRollout.initial_states`（rollout.py:83–88）建立两个独立 root/trajectory/node UID；prompt、rolling、响应、mask、剩余 turn 状态均逐行保存。
2. `complete`（162–172）在每次真实动作前保存深拷贝 snapshot，然后生成 action、原 parser 执行 search、追加 observation。生成所得 observation 被缓存于 state。初始链包含最多四动作；回答 done 后停止；耗尽 turn 只是 pending=false，不伪装为环境 done。
3. `run`（201–225）从非终止动作前状态随机有放回选 N=2 个 parent，深拷贝 continuation。`branch` 只换 trajectory UID，不换 tree/node/source；因此共享前缀确实复用，turns 不重置，不重复执行其工具查询。root 状态也在候选中，与作者 tree_node.py:132–142 相符。
4. 新 continuation 的动作前节点按稳定 node UID 去重（216–221），保留以后轮次可用的新节点，而不把多次选同 parent 的重复 snapshot 加权。L=1 后得到每 root `1+LN=3` 终端/预算耗尽轨迹，随机不放回选 K=3；默认没有弃叶，K 更小配置才有弃叶。作者 tree_node.py:145–175 同样随机选 K 个叶并 prune。
5. `compose`（174–199）保留完整叶路径，prefix 在每个叶的训练样本重复出现；这是作者叶路径目标的行为，不应改成 prefix 只训练一次。observation mask=0，step=-1。sources 用原输入 row index 映射 QA metadata（ray_trainer.py:766–769）；uid 来自固定数据 index（781），不是叶位置。tree_uid、trajectory_uid 是 non_tensor_batch，随 DataProto reorder；无按相邻行分组依赖。
6. `controller.py:175–193` 保留 outcome sum，并把 scalar 落在最后保留的 action token；`advantages.py:57–61` 计算 `A_i=z_question(R_i)+z_(question,tree)(R_i)`，sample std correction=1，epsilon=1e-6，再乘有效 action mask。这是 Eq.6–7；两组 z-score 均基于原 reward，**不是**先全题 z-score 再在树内二次归一化。作者 ray_trainer_ts.py:168–189 默认 `tree` 相同；190–204 的 `tree_2norm` 是另一个源码模式，不应拿来要求当前实现。
7. old logprob 在生成后完整叶重算（ray_trainer.py:775–777），reference logprob 随后计算；PPO ratio token-wise、clipped surrogate、独立 reference KL，遵循作者/原 Search-R1 actor。实际 loss reduction 对论文 Eq.8 有差异，见 F2。

## 四维判断

| 维度 | 结论 | 证据与边界 |
|---|---|---|
| 代码 | 核心通过、有条件 | 真 continuation、节点去重、mask、叶 source/UID 数据流已核；8 个相关 CPU 测试通过；未执行生产 CUDA actor |
| 数学 | advantage 通过；目标权重有偏离 | 独立两组 sample-zscore 相加正确；共享 prefix 不应盲目去重；PPO microbatch token 均值继承作者，但不等同论文逐轨迹均值 |
| 逻辑 | 有条件 | 预算继承、叶选择正确；长上下文 generation/scoring 前缀错配继承且可触发；root 串行扩展造成额外 padding |
| infra | 未验收 | Hydra/validate_config 通过，真实默认接 vLLM、FSDP/offload；已有两卡 tinyFSDP 只证明小模型参数更新，不证明 3B、多轮检索或 vLLM/offload 集成 |

## Findings

### F1 — P1，继承缺陷：超 cap 后真实 generation/scoring 条件不同

位置：`search_r1/baselines/rollout.py:86–87,96–100,149–152,177–187`；默认 `ppo_baselines.yaml:69` 为 `context_alignment: report`。rolling 保留尾部 4096，response 保留头部 4096，scoring 拼接截短 prompt 与保留 response；initial prompt 超 max_start 也会使首步不一致。原 Search-R1 `generation.py:94–119` 同样尾截 rolling，故不是新 Tree-GRPO 独有错误，但新实现仅计数继续训练，不能证明数据固定就不会触发。

反例：四轮 search-only、cap=80，真实 rolling 长80、scoring context长98，记录两次 mismatch，equal=false（探针结果附下）。生产四轮每 action/obs 各500 token加 prompt，可超过4096；是否在具体数据触发尚未统计。后果是 PPO old/current logprob 用另一上下文，训练样本可能截掉后期答案，跨分支节点虽正确复制仍继承错配。建议采用每动作真实 scoring prefix/窗口一致的 logprob 与 loss，或在统一协议下确保完整路径不超 cap并在超限前停止；正式 smoke 开启 error gate只能暴露问题，不能作为修复。任何修复应同时保持对照组条件一致。

### F2 — P2，有意保留/继承偏离：PPO 按微批有效 token 平均，而非论文逐轨迹平均

位置：`verl/trainer/ppo/core_algos.py:185–193`；`verl/workers/actor/dp_actor.py:260–269,289`。原 base 同一 `masked_mean`，作者固定 core_algos.py:572–603 也是 token masked_mean。论文 Eq.8 是先每条轨迹除长度再组均值；实现是在每个 microbatch 按有效 action token 总量平均，再按 gradient_accumulation 平均 microbatch/rank。

反例：一个1-token叶 A=+1，一个3-token叶 A=-1，共享当前 log-ratio参数0；实际 loss/grad=0.5，论文逐轨迹 grad=0。共享 prefix cancellation 也只有在一致权重、ratio/clip条件满足时才成立，不应宣称在实际 PPO 中总是完全抵消。保持作者/原 Search-R1优化器是合理适配，但应披露实际目标；若严格重现 Eq.8，应实现逐叶平均并验证微批/rank独立性，且统一对照组策略。

### F3 — P2，新 infra 偏离：逐 root 扩展使八卡 padding 成本显著放大

位置：`rollout.py:208–214` 每个 root 单独 complete(children)，以及108–118按总GPU补行。作者 generation_ts.py:594–620 把所有 root 的候选合并后一次并行扩展。默认 production 512题×2root，每轮每root两个 children仍独立发起生成：八卡时每次2有效行补到8，浪费6行；不能用“共享prefix”直接推导论文节省率或对原5链的等成本比较。

反例：两root、两children、8GPU、强制选 shared search 后节点，真实工具查询仅2（前缀未重跑），但4次worker call均batch8；24 discarded padding rows，720 generated tokens。全部padding generation已经正确计入成本，不存在漏计。建议跨root/round聚合 continuation、按有效token/工具调用/实际GPU生成与walltime分别记录成本，并按实际预算比较。计数轨迹6不意味着等成本5-chain。

### F4 — P1，共享 infra 继承缺陷：全零 action loss 微批

Tree-GRPO 使用原 actor 的 `masked_mean`，缺零分母保护（core_algos.py:187,192–193 与 dp_actor.py:269,273,284）。全部有效 action mask为0时可能NaN；零token/截断样本可进入叶选择。跨算法具体反例与证据由本次 shared-infra 审查统一记录；本文件没有独立运行此反例，因此不冒称独立已测。正式训练应消除/跳过空loss微批并处理DP同步，不能由树advantage零值认为安全。

### F5 — P2，已披露适配和未验证风险

当前 `max_turns=4,force_final_answer=false`（ppo_baselines.yaml:51–52），作者脚本 `max_turns=3` 且 generation_ts.py:451–539 会对仍active路径额外 final rollout。当前将预算耗尽路径作为可训练终端候选，与原Search-R1控制变量一致，但不同于作者“强制最后一轮”的动作约束。当前EM沿用原协议，未照搬作者额外格式/retrieval score；lr等训练时长也固定原对照。属于授权的任务适配，必须明确，不能称作者原配方逐参数复现。

## 一致性矩阵

| 项目 | 原Search-R1 base | 作者19bf3fa | 当前审查SHA |
|---|---|---|---|
| 主模型/任务 | Qwen2.5-3B，paper-grpo QA/search | Qwen2.5-3B/search | 保留原固定数据/任务 |
| rollout | 5完整链 | M2,N2,L1,K3真实树 | M2,N2,L1,K3真实树，6叶 |
| 分叉prefix/预算 | 不分叉 | node保存context、depth | snapshot深复制，继承turns，复用工具观察 |
| 候选采样 | 无 | root+nonleaf、有放回 | 同；去重node UID |
| 叶采样 | 无 | 随机K不放回、prune | 同样随机K不放回，sources显式QA重排 |
| advantage | 题内zscore | tree模式inter+intra独立zscore | 同（singleton安全零值；作者singleton返回score，合法默认组均≥3） |
| observation loss | state mask | state mask | 同，mask=0 |
| prefix loss重复 | 链独立 | 每叶重复prefix | 同 |
| PPO reduction | microbatch token均值 | 相同源码均值 | 保留；论文Eq.8偏离 |
| cap/scoring错配 | 有尾/头截断机制 | 继承相似窗口机制 | report记录继承风险，非修复 |
| 扩展批处理 | 无 | 跨root批处理 | 逐root串行，padding高成本 |
| final动作 | 四动作无强制final | 三轮+active最后生成 | 保留原控制变量 |
| rollout backend | vLLM | vLLM ts实现 | 旧vLLM生成+独立state树，不需移植ts backend |
| FSDP/offload | actor参数/梯度/optimizer，ref参数 | 脚本启用 | 开启，调用load/offload链存在；未生产实测 |

`run.sh:14` 实际设置n_agent=2；main.py:7–13选ppo_baselines Hydra配置并validate，再进入原main_task；controller.py:59–66检查root/budget；fsdp_workers.py:84–93,364–400,410–428与vLLM rollout:89–103有实际offload调用。默认vLLM在ppo_trainer.yaml:68继承，TP=1/温度1/top_p1/offline和模型/数据路径均实际进入配置，而不是仅README宣称。

## 测试与关键输出

本轮独立执行（本地CPU，无远端GPU）：

```bash
.venv-cpu/bin/python -m unittest tests.baselines.test_algorithms.Estimators.test_tree_independent_inter_and_intra_normalization tests.baselines.test_algorithms.Rollouts -v
.venv-cpu/bin/python /tmp/tree-grpo-independent-probe.py
CUDA_VISIBLE_DEVICES=0,1 .venv-cpu/bin/python -m search_r1.baselines.main baselines.method=tree_grpo actor_rollout_ref.rollout.n_agent=2 --cfg job
```

- 相关8 tests全部OK，覆盖prefix不重搜、node去重、来源映射、padding、耗尽预算、response cap、metadata、独立normalization；这些使用字符tokenizer/mock生成，不代表真实3B/vLLM。
- 独立探针 `/tmp/tree-grpo-independent-probe.py`、原输出 `/tmp/tree-grpo-independent-probe.json`；重要输出持久保留于本报告：

```json
{"two_roots_8GPU":{"sources":[0,0,0,1,1,1],"tree_ids":2,"generated_tokens":720,"prefill_tokens":2016,"retrieval_calls":2,"trajectories":6,"padded_rows":24,"worker_batch_sizes":[8,8,8,8]},"cap80":{"context_mismatches":2,"retained_response_len":80,"rollout_context_len":80,"scoring_context_len":98,"equal":false},"zscore":{"values":[2.2071027755737305,-1.2071046829223633,-0.4999989867210388,-0.4999989867210388],"permutation_equivariant":true,"singleton":0},"PPO_weight":{"loss":0.5,"grad":0.5,"paper_equal_trajectory_grad":0}}
```

- Hydra打印配置成功，另调用compose+validate_config成功：`/data/search-r1/assets/qwen2.5-3b vllm roots=2 rounds=1 expansions=2 leaves=3`。此步骤没有初始化Ray/模型，因为使用--cfg；不能算运行训练通过。
- 已有证据 gpu-a800-20261002-run2/result.json 的 Tree-GRPO tinyFSDP两rank各2步、delta L1约60.02/95.80，是既有日志而非本轮重测；gpu_smoke.py使用toy模型和合成advantage，并不执行vLLM树rollout。辅助ranker更新与Tree-GRPO没有直接关系。
- 未覆盖：真实Qwen tokenizer、4096 cap实际发生率、真实长树/多轮reward、8GPU rollout开销、vLLM缓存/权重同步、全参数3B FSDP/offload显存、梯度归约、实际checkpoint恢复、七集全量EM。没有据此声称production PASS。

正式训练阻碍：处理/确认不存在F1并统一对照，修复共享空loss微批问题，完成独立资源上的真实3B/vLLM至少两外层更新、成本与checkpoint验证。F2/F3/F5可作为明确适配接受，但其统计意义与成本不可隐去。
