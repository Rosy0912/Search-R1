# GiGPO 独立复现审查（2026-10-02）

总体 **CONDITIONAL**。在固定 Search-R1 输入、mean-only / exact-observation 适配模式下，GiGPO 核心优势公式有独立数值证据，anchor/uid 重排和 observation masking 的代码链闭合；但 context 截断导致采样条件与全轨迹重算条件不同、全零 action 微批导致 NaN 的继承缺陷仍阻碍无条件正式训练放行。没有本次 production Qwen2.5-3B /真实检索 /vLLM /offload 长跑证据。既有 CPU 与两卡 tinyFSDP 证据只能说明各自覆盖路径，不等同正式训练通过。

## 固定对象与 primary evidence

- 被审 HEAD：`7b93fbd01742715dc14290b0a6e2d3b42e6c19f7`；比较 base：`b251ba6b503217a59cd3395c1326d7b66b602989`。本次没有修改实现，没有远端任务、GPU 启停、commit 或 push。
- 论文：[arXiv 2505.10978v2](https://arxiv.org/html/2505.10978v2)，重点 §4.1–4.3 / Eq.3、5、7、8、9。本次直接打开论文，未以此前 PASS 或研究 notes 为证据。
- 作者固定源码：`langfengQ/verl-agent@20bd331bdbc9026a5668e11362178e10ab7400c8`。[core_gigpo.py](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/gigpo/core_gigpo.py) 与 [run_search.sh](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/examples/gigpo_trainer/run_search.sh)。缓存 `/tmp/rl-gigpo-tree-audit/verl-agent` 没有 `.git`，不能声称 git HEAD 已核验；本次从固定 raw URL 重新下载并逐文件核 SHA256：core `fa709b91baf2331c4857fee32a249c4ca8d8d69e864d1e71f08730cc8c5cc456`，run `6ff12a5ce26aea27076b99df85fcbbc73a62e1ab5a81c12d5cc46b58c0fb33e6`，均与缓存完全一致。作者其他缓存文件只作辅助代码阅读，不借此声称全树固定哈希通过。
- Search-R1 对照以 base `git show` 的 `search_r1/llm_agent/generation.py` 及现有 `reproduction/h20/train_paper_grpo.sh` 为本仓库原行为证据；作者框架版本与 Search-R1 的旧 verl 不能直接混同。

## 一轮串行流程与数学映射

1. `baselines/main.py:7–13` Hydra 合成 `ppo_baselines`，先 `validate_config` 后进入 Ray `main_task`；后者创建同一 Qwen2.5-3B tokenizer /actor /ref，`adv_estimator=grpo` 只选择 critic-free 外壳，实际优势由 Controller 替换。`ray_trainer.py:733` 每题重复 `n_agent=5` 次。
2. `rollout.py:83–88` 为每条轨迹保存 prompt、rolling、初始 observation。逐轮只调度 pending states (`162–171`)，不足 GPU 数时重复首行补齐、生成后丢弃 padding (`108–119`)，搜索仍调用原 manager 的 action parser /检索 /obs tokenizer (`120–136`)。
3. 在执行下一轮前，状态使用当前 observation token 的 SHA256 做 anchor (`140–143`)，初始 anchor 是题目，随后 anchor 是上次工具反馈。action token 的 step id 为本轮，observation 为 `-1`，模型 mask 分别为 1/0 (`146–148`)。这是 retroactive 分组，无额外 per-state rollout。
4. 完整 trajectory 合并后，`ray_trainer.py:775–785` 全轨迹重算 old_logp、设 `uid=index`，接着 `_balance_batch` 重排。`DataProto.reorder:539–545` 同步重排 tensor 与所有 non_tensor 元数据，因此 anchor、step_ids、uid 对齐。组运算在 driver，发生在 DP shard 前。
5. `controller.py:169–186` 取 attention × info_mask，将 terminal EM 标量重新放到最后保留的 action token。`advantages.py:29` 算 episode return `R_i=Σr`，每题每轨迹一次，`A^E_i=(R_i-mean_uid R)/F`。默认 `F=1`；可选 `F=sample_std+1e-6`。
6. `advantages.py:35–47` 按 anchor timeline 从末轮递推 `Q_it=r_it+γ Q_i,t+1`（Eq.5，γ=.95）。包括零 token action 的时钟步和 baseline 贡献；以 `(uid,anchor)` 分组对 `Q` center /可选 std（Eq.7）。不存在跨题 anchor 污染。
7. `48–53` 把 `A^E_i+ωA^S_it`（Eq.8，ω=1）广播到该 action token，最后乘模型 mask。singleton step group 得到 0。`ray_trainer.py:905–911` 创建 actor loss_mask，`dp_actor.py:249–268` 对这些 token 做 clipped PPO，KL 同样使用该 mask。工具反馈参与 attention，直接 PPO/KL/entropy loss 为零。

## 四维结论

| 维度 | 结论 | 依据与边界 |
|---|---|---|
| 代码 | CONDITIONAL | estimator、metadata、source selection、组重排闭合；缺陷 G1/G2 属于继承路径，不能因继承而放行 |
| 数学 | 核心 PASS；目标实现 CONDITIONAL | 默认 mean_norm、均长 case 与固定作者函数数值一致；episode 按轨迹一次符合论文；token/microbatch 平均并非逐 action 等权，见 G4 |
| 逻辑 | CONDITIONAL | observation anchor 不泄漏未来 answer；零 token action baseline 已保留；超 cap 的 scorer 分布不匹配，且扩展 response cap 时奖励时钟有风险 |
| infra | 未通过正式训练 gate | Hydra 与 launcher 真入口可合成；FSDP load/offload、vLLM weight-sync 路径存在但本次只静态检查；2 张可见卡不会自动修改默认 8 卡资源池 |

## Findings

### G1 — P1，继承缺陷：采样 prefix 与训练/old-policy prefix 不同

位置：`search_r1/baselines/rollout.py:83–100,149–152,174–188`；`verl/trainer/ppo/ray_trainer.py:775–777`；配置 `ppo_baselines.yaml:69` 的 `context_alignment: report`。原 Search-R1 base 的 generation `_update_rolling_state` 同样保留窗口尾部，而原 left/right 拼接评分保留初始 prompt /response 首部。

触发：初始有效 prompt > max_start_length=2048，或 prompt + 前几轮 action/obs 超过 rolling cap=4096。默认四轮至多 4000 response token 并不保证 rolling prefix 不溢出。默认 report 仅统计 mismatch 后继续更新。

独立 cap=45 的四轮 search 反例：mismatch_count=3；第 2–4 轮实际 prefix 为 `</search>\n\n<information>Paris</information>\n\n`，评分却用 `Question: capital?` 加首部保留 response。对不同上下文重算 old_logp 无法恢复真实采样分布；PPO 的行为策略分母与数据来源不相符。

建议：正式 gate 至少开启 `baselines.context_alignment=error` 并在真实 dataset/4-turn rollout 验证 0 mismatch；根治用逐 action 保存真实 prefix 做 logp/update，或在任何 cap 前终止并一致评分。单纯 report 不能构成修复。此风险 cross-reference root 的 shared-infra 审查。

### G2 — P1，继承缺陷：全零 action 微批产生 NaN 梯度

位置：`verl/workers/actor/dp_actor.py:264–289`、`verl/trainer/ppo/core_algos.py:187–193`、`verl/utils/torch_functional.py:107–109`。

触发：某 DP rank/微批所有输出经 skip-special-token /postprocess 后零 action token（仅 EOS 等），或 metadata 指示的 action 全被截断；虽然 `advantages.py` 安全保留其 baseline，actor 接口仍将全零 loss_mask 给未防 0 的 masked_mean。独立直接调用得到 `loss=nan, clipfrac=nan, kl=nan, gradient=nan`。KL 与 entropy 也调用同类分母。

后果：FSDP 梯度同步可污染整组；已有 estimator 空 action 测试并不覆盖 actor 空 mask。建议：保持分布式 collective 一致、空本地块给 graph-connected 零 numerator，并对所有 loss/metric 分母 clamp 或显式安全处理；别只跳过某个 rank 的 backward。正式训练前修复并验证一空 rank 与整空 microbatch。cross-reference shared-infra。

### G3 — P2，有意偏离：不是作者 search launcher 的逐项复刻

位置：`advantages.py:24–29`、`rollout.py:140–141`、`ppo_baselines.yaml` 的 `gigpo.normalize=false, anchor_mode=observation`；作者固定 `run_search.sh` 配置 `mean_std_norm`、similarity=True/.9、invalid_action_penalty=.01、entropy=0。

本适配默认 mean-only + token exact equality（论文允许固定 F=1/exact state）。每 episode 一次，相比作者 core 的 cross-step episode stats 消除了长度加权；均长 case normalize=False 完全匹配。normalized override 与作者默认仍不等价：两条两步 reward=[1,0]，本实现每 token ±1.4142115，作者 ±1.5731297，原因是作者 duplicated episode rows 的 sample std；不齐长 episode `[success×2 steps, fail×1]` 作者 episode center `[1/3,1/3,-2/3]，本适配 `[.5,-.5]`。

此外继承 Search-R1 entropy=.001、无作者 invalid-action penalty，且保持主模型/数据/EM/caps/training schedule。建议：实验和表格明确命名“GiGPO on fixed Search-R1, mean-only/exact/per-episode adaptation”；不要表述为作者 search.sh 参数等价。若对齐作者设置，需要独立消融，而非悄改固定 Search-R1 对照。

### G4 — P2，继承目标差异：微批/rank 局部 token 均值决定更新权重

位置：`dp_actor.py:264–268,289`、`core_algos.py:192`、`fsdp_workers.py:97–109`。非 TOP-D 用每微批 masked mean，再等权梯度累计与 DP 平均。独立两微批 token 数 1/3、advantages 1/3，梯度=-2；global token mean 应=-2.5。论文 Eq.9 按 action/episode 写目标；当前既不是跨 DP global token mean，也不是 action 等权。这是 Search-R1 基础目标的继承适配，不能拿 TOP-D 的 global-denominator 测试为 GiGPO 证明。

建议：在固定 Search-R1 比较中保留并披露该权重；若声明 strict GiGPO objective，需要明确 action/token aggregation 并验证 DP gradient 与单进程参考一致。本次没有证实该差异会导致 3B 发散。

### G5 — P2，未验证部署边界与非默认 cap 风险

`run.sh:6` 只要求 CUDA_VISIBLE_DEVICES 已设；`main_ppo.py:175–178` 仍按 trainer.n_gpus_per_node 建资源池。用两卡 env 做 --cfg，仍是 n_gpus_per_node=8；使用 2 卡需显式 override，不能宣称自动匹配。`fsdp_workers.py:97–109` 整数除法规范 mini/microbatch，任意 DP/SP 配置须满足整除和非零；本次默认 8 卡/64 micro 合理，未做 GPU 启动。

`rollout.py:152` response 首部截断但 anchor timeline 不截断；若用户增加 max_turns /force_final_answer 使 response 超过 cap，`controller.py:176–181` 会把终局奖励搬到最后“保留”action，改变真实 terminal reward 时刻（默认 4×(500+500)=4000<4096 不触发 response cap）。建议对 override 的 cap/time horizon 做 fail-fast 或保留真实 step reward，不能用默认4轮的安全界限推广到任意配置。

## 一致性矩阵

| 项目 | 固定 Search-R1 | 作者固定代码 | 本审适配 |
|---|---|---|---|
| Qwen2.5-3B /paper-grpo data /EM | 保持 | 作者示例 7B-Instruct、另一数据路径/环境 | 固定任务允许的必要适配；非作者性能复现 |
| n_agent=5 /4 turns /检索 topk3 | 保持 | group5 /maxsteps4；独立环境实现 | 原 manager/parser/retriever 复用 |
| initial observation /后续工具反馈 anchor | 新 metadata | anchor_obs 同题分组 | exact token hash + uid；POMDP 历史未放入默认 anchor，与 observation 方法一致 |
| episode mean /std | 原 GRPO std | 支持 mean /std，默认跨 step | 每轨迹一次、默认 mean；明确有意偏离 |
| step discounted return | 原 GRPO 无 | gamma=.95 | .95 + omega1，含空 action 时间步，数值核查 |
| observation loss mask | 保持 | action/step row response mask | attention 保留、loss排除；无发现默认mask泄漏 |
| DP reorder /优势组 | 原 index→uid | 单独 trajectory/episode uid | driver 用 uid/hash 而非邻接组；固定数据 prepare:23–24 生成全局唯一 index |
| invalid penalty /entropy | 无 penalty、entropy=.001 | penalty .01、entropy0 | 继承 Search-R1，披露参数差异 |
| offload /vLLM | 同框架配置 | 新版 verl runtime、作者示例 offload不同 | 复用旧 FSDP manager；不能移植作者runtime保证 |
| context cap | 原窗口尾/评分首不一致 | 此次未核全作者 rollout 树 | G1 已复现，未作作者缺陷断言 |

## 实际命令与结果

工作目录均 `/Users/shaoshitong/Documents/Search-R1-RL-Baselines`。

1. `git rev-parse HEAD` → 固定 SHA 如上；`git show b251ba6:search_r1/llm_agent/generation.py` 核继承截断。
2. `curl -fsSL https://raw.githubusercontent.com/langfengQ/verl-agent/20bd331bdbc9026a5668e11362178e10ab7400c8/gigpo/core_gigpo.py -o /tmp/gigpo-core-fixed-audit.py`；run script 同 URL/download 方法。Python hashlib 核两份缓存，SHA 见前文。初次 shasum 因本机 C.UTF-8 perl locale panic 失败，改 hashlib 完成核验，不是源码或网络失败。
3. `.venv-cpu/bin/python -m unittest discover -s tests/baselines -p test_algorithms.py -v` → 19 tests OK（含其他算法；GiGPO empty baseline/time、uid permutation、observation/source metadata、context guard）。
4. `.venv-cpu/bin/python -m unittest discover -s tests/baselines -p test_models_and_driver.py -v` → 4 tests OK；真实 driver 使用 stub rollout worker /tiny CPU model，不是 Ray cluster/vLLM。原始输出 `/tmp/gigpo-audit-driver.out`。
5. `.venv-cpu/bin/python /tmp/gigpo-audit-probe.py` → author differential /zero mask /cap /local mean 独立反例，exit0；关键输出保存如下：

```text
normalize False ours [[0.9750000238,1.0],[-0.9750000238,-1.0]]
author [[0.9750000238,1.0],[-0.9750000238,-1.0]]
normalize True ours +/-1.4142115116; author +/-1.5731296539
ZERO_ACTION loss nan clipfrac nan kl nan gradient nan
CONTEXT_CAP mismatch_count 3
AUTHOR_LENGTH_WEIGHTED episode [0.3333333135,0.3333333135,-0.6666666865]
OURS_PER_EPISODE episode [0.5,-0.5]
LOCAL_MEAN gradient -2.0 global_token_mean_expected -2.5
```

临时探针/输出 `/tmp/gigpo-audit-probe.py`、`/tmp/gigpo-audit-probe.out`；公式匹配使用本次固定 raw author core 动态 import，非自己重写“expected”。normalizeFalse 结果有 torch.assert_close。

6. `.venv-cpu/bin/python -m search_r1.baselines.main --cfg job > /tmp/gigpo-audit-hydra.out`；`CUDA_VISIBLE_DEVICES=0,1 PATH="$PWD/.venv-cpu/bin:$PATH" bash reproduction/baselines/run.sh gigpo --cfg job > /tmp/gigpo-audit-launcher.out`；`cmp` exit0，`bash -n reproduction/baselines/run.sh` exit0。证明 shell/Hydra 入口合成同配置，不执行模型加载，不证明 GPU资源池、retriever、Ray worker imports 或 vLLM 能运行。

未覆盖：真实 paper-grpo prompt/cap 分布（未访问远端资产）、完整作者 rollout 树固定下载、生产 vLLM采样与logp一致性、Qwen3B peak VRAM、8卡 DP/SP、offload restore/weight-sync、save/resume长跑、真实检索 timeout/服务吞吐。全局唯一 uid 由固定 prepare 代码保证，但未读取目标服务器实际 parquet；不能把其他自定义合并数据 index 冲突假设当本 paper-grpo 已发生事实。

## 正式训练前阻碍

必须完成 G1 的真实 rollout 条件一致性 gate（或修复逐 action scoring）、G2 的空微批/空 rank NaN 修复及分布式验证；固定适配说明 G3/G4 应进入最终实验方法。生产 3B、目标真实数据/检索、实际 vLLM和offload、正确 GPU 资源数至少跑受控端到端 step 和 checkpoint验证后，才可从 CONDITIONAL 升级。本次无 P0 新实现缺陷证据；不能据此推断不存在其他 P0。
