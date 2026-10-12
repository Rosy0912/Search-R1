# EasyPPO：论文与官方实现核对

核对日期：2026-10-12。只读论文/官网和小型官方代码；没有加载模型、安装环境、访问 GPU 节点或启动训练。

论文是 [EasyPPO: Stabilizing the Critic Is Key，arXiv:2609.36802v1](https://arxiv.org/html/2609.36802v1)，提交于2026-09-29；[官网](https://easyppo.github.io/) 实际链接到 [官方仓库](https://github.com/EasyPPO/EasyPPO)。代码固定为 `0a490a88ad7b84a8f7974fe168f0ba63600b1e31`（2026-10-04）。以下 GitHub 链接全部固定此 commit，不引用浮动 main。

## 结论先行

EasyPPO 是**保留 learned critic 和 token-level GAE 的 PPO 改进**。三个改动是：截断响应仅从 actor loss 中排除；按同题 return 的逆标准差加权 critic loss；一个 rollout batch 分成4个 critic mini-batch，每个重新前向/反向、clip、优化一步。它没有提出“把所有 actor 优势按置信度衰减”的新 actor loss，也没有把 critic 目标降为人为长度惩罚。[论文 §§2–4](https://arxiv.org/html/2609.36802v1#S4)、[官方说明](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/README.md#method)。

## 一轮实际怎样执行

1. 用当前 policy snapshot 生成同题多条响应，取得原 reward，保留全部响应行；标记达到 response budget 的行。发布的同步 v1 路径按 `response_len >= max_response_length` 判 overlong，保留 keys/cardinality；不是检测所有“答案错误”或所有工具失败。[标记实现，217–255行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/ray_trainer.py#L217-L255)。
2. **此时不掩掉 overlong**。计算 old/ref logprob、旧 critic values；用所有有效响应计算 group return STD、critic weights、GAE/returns。[v1执行顺序，461–518行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/v1/trainer_base.py#L461-L518)。
3. 对全部响应做4次 critic mini-batch 更新。actor advantages 已由更新前 values 计算，不随这4次 critic 更新重算。[critic分批，1551–1572行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/v1/trainer_base.py#L1551-L1572)。
4. critic warmup 后才更新 actor；在 actor update 前临时将 overlong 的 response/loss masks 置零，结束后恢复。官方发布配置 `critic_warmup=31`，配合从1计 step，对应前30个 rollout step 只训练 critic。warmup 不是优化器的20步 LR warmup。[actor-only mask时序，520–540行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/v1/trainer_base.py#L520-L540)、[配置，135–140行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/configs/easyppo_aime.yaml#L135-L140)。

actor-only filtering 本身仍有选择偏差，不能称无偏优化完整 rollout reward。论文的完整分布 critic 推导只说明：在理想精确 baseline、响应级、未加 PPO clipping 的分析下，它比只训练 completed responses 的 critic 多出一个有利于 completion 的项；不构成真实 token PPO 收敛保证。[论文 §3、附录9.2](https://arxiv.org/html/2609.36802v1#S3)。

## Critic 权重：是 1/STD，不是 1/variance

对题 (q\) 的 (n\) 个 sampled returns，代码使用**总体方差 correction=0**：

\[
\hat v_q=\frac1n\sum_{j=1}^n(R_{qj}-\bar R_q)^2,\qquad
u_q=\frac1{\max(\hat v_q^{\beta},\varepsilon)},\qquad
w_q=\frac{u_q}{|Q|^{-1}\sum_{q'\in Q}u_{q'}}.
\]

同题每条响应共享一个权重，unique prompt 均值为1；权重从本 rollout batch 计算一次并冻结。官方 EasyPPO 配置 `variance_weight_beta=0.5`，即 (1/\max(\hat\sigma_q,\varepsilon)\)。**通用函数签名默认 beta=1.0，是支持逆方差的通用接口，不能拿它冒充 EasyPPO 配方**。官方 CriticConfig 默认也是0.5；没有设置 normalized-weight upper cap。[实际计算，2521–2592行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/core_algos.py#L2521-L2592)、[CriticConfig，92–98行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/workers/config/critic.py#L92-L98)、[发布配置，105–113行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/configs/easyppo_aime.yaml#L105-L113)。

为何是逆 STD？独立展开线性 value head 的 MSE 梯度，若 (g=(V-R)h\)、(m=E[R|s]\)、(v=Var(R|s)\)，则

\[
E[\|g\|^2|s]=\|h\|^2[(V-m)^2+v].
\]

将**loss**乘 (1/\sqrt v\)，梯度第二矩乘 (1/v\)，噪声项变成1；若 loss 乘 (1/v\)，噪声项会变成 (1/v\)，不是同一归一化。实际仅用 prompt STD 近似各 token prefix 的 conditional STD，且由用于训练的同一组 sampled returns 估权重，有估计偏差；“保持最优 conditional mean”仅针对理想固定正权重的点态 population regression，不保证共享有限网络的最终参数或 actor EM。[论文式(5)–(7)、附录9.3](https://arxiv.org/html/2609.36802v1#S4)。

零方差组不是删除：floor 控制最大原始权重。离散 reward 范围 (\Delta\) 的提议 floor 为 (\Delta/(2\sqrt n)\)。(n=16\) 时 Search-R1 的[0,1]取0.125，AIME 的[-1,1]取0.25；FrontierCS 连续分数取0.075。[论文附录表4](https://arxiv.org/html/2609.36802v1#S8.S3)。

## Critic loss 和目标到底改了什么

代码保留 PPO clipped value loss。令旧 values 为 (V^0_{it}\)，新预测为 (V_{it}\)，`returns` 为 (G_{it}\)，(V^c=clip(V,V^0-\epsilon_V,V^0+\epsilon_V)\)，则该 critic mini-batch 的默认目标是

\[
L_V=\frac1{2N_{\rm token}}
\sum_{i,t}M_{it}w_{q(i)}
\max\{(V_{it}-G_{it})^2,(V^c_{it}-G_{it})^2\}.
\]

权重只乘分子；分母仍是**未加权的原 mini-batch 全DP有效 token 数**，不除以 (\sum wM\)，也不按每条轨迹先取均值。physical micro-batch 的分片贡献按同一 global denominator 累积。[value loss，2084–2155行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/core_algos.py#L2084-L2155)、[跨micro/DP归一化，158–206行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/workers/utils/losses.py#L158-L206)、[token denominator，1168–1173行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/core_algos.py#L1168-L1173)。

注意“prompt 权重均值=1”不意味着按不同响应长度加权后的 token 权重均值也=1，更不保证总梯度范数完全不变。当前源码将 return variance 从原 `token_level_scores.sum(-1)` 算出，然后正常生成 `returns`；没有把 overlong 的目标再乘一个衰减因子，也没有专门降低它的 critic target。[权重/returns路径，1454–1547行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/v1/trainer_base.py#L1454-L1547)。具体任务 scorer 可能本来就给未完成响应低分，那是原 reward，不是新的 target 修改。

## GAE 与 actor objective

标准反向 GAE：(\delta_t=r_t+\gamma V_{t+1}-V_t\)，(A_t=\delta_t+\gamma\lambda A_{t+1}\)。源码对 observation mask 为0的位置跳过 value/TD-error，先保存 (G_t=A_t+V_t\)，再对白化 actor advantages。官方 配置 (\gamma=\lambda=1\)，无 KL-in-reward；终末奖励设置下，有效 policy token 的 unwhitened return 为原终末 (R\)。因此不应将 critic return 说成“组标准化 reward”或“白化 advantage”。[GAE，216–264行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/core_algos.py#L216-L264)、[GAE直接沿原路径，143–171行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/v1/utils.py#L143-L171)。

actor仍调用 `vanilla` PPO：logratio先限制[-20,20]，PPO上下界和 negative-advantage DualClip 后 token mean，另加 reference KL。EasyPPO critic weights 没有传入 actor 优势或 PG loss。AIME上/下clip为0.28/0.2、DualClip3、KL系数0.001、entropy0；论文 Search-R1 EasyPPO 上/下clip均0.2。[actor loss，1311–1369行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/core_algos.py#L1311-L1369)、[actor调用和KL，76–145行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/workers/utils/losses.py#L76-L145)。它改变 actor 的 overlong loss mask及其 learned baseline来源，不能简单说 actor训练完全未变；但不是新的置信度 actor objective。

## Mini-batch 的具体默认与版本边界

| 设置 | 论文 FrontierCS | 论文 AIME / 实际发布AIME YAML | 论文 Search-R1 |
| --- | --- | --- | --- |
| 每轮 prompts × responses/prompt | 16×32 | 32×16 | 64×16 |
| 总响应数B | 512 | 512 | 1024 |
| critic mini响应数 | 128 | 128（YAML写8 prompts，再乘n=16） | 256 |
| critic optimizer机会/epoch | 4 / 1 | 4 / 1 | 4 / 1 |
| actor optimizer机会/epoch | 1 / 1 | 1 / 1（32 prompts×16） | 1 / 1 |
| critic STD floor | .075 | .25 | .125 |
| actor / critic LR | 1e-6 / 2e-6 | 1e-6 / 2e-6 | 1e-6 / 2e-6 |
| critic warmup / LR warmup | 30 rollout / 0 optimizer | 30 rollout / 20 optimizer | 30 rollout / 0 optimizer |

表中除发布 AIME YAML外的任务数字来自论文附录表1–4；不能声称公开仓库已提供所有三个任务的原始 recipe。当前 `configs` 仅有 `easyppo_aime.yaml`，`scripts/run_aime_ours.sh` 调这个配置；本次没有自行补写 Search-R1 配置。[AIME全配置](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/configs/easyppo_aime.yaml)、[官方入口](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/scripts/run_aime_ours.sh)。

每个 critic mini 独立重新 forward/backward，clip-norm1再AdamW，下一 mini 用新的 critic 参数；micro只是一个 mini的梯度累积，不是额外optimizer步。[mini循环，234–303行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/workers/engine_workers.py#L234-L303)、[每mini清梯度/更新，113–132行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/workers/engine/base.py#L113-L132)、[clip/非finite拒更新，689–728行](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/workers/engine/fsdp/transformer_impl.py#L689-L728)。4步为配置机会数，遇非finite时源码可跳步；本次没有运行训练来证明实际Adam计数。

## 对当前 G1/Tree-GRPO 的迁移边界

当前已核 G1/Tree-GRPO 是 critic-free 的组优势路线。EasyPPO 的主要分析对象是 (\nabla_\phi L_V\)、learned baseline 和 critic clipping；无 critic 时没有这个优化对象。把 (1/STD\) 再乘到已经按组STD归一的 GRPO actor优势，会成为另一种 actor重加权，可能再次强化近零方差组，**不是直接复现EasyPPO，也不受其critic理论支持**。

可迁移的是诊断方法：完整记录 truncated比例、原reward方差、每mini preclip norm和相同批次贡献；不能从 critic 论文的稳定性结果断言我们当前 R 造成退化，或断言替换 G1奖励/优化设置会改善 EM。若考虑 learned critic，是新增 actor–critic算法及模型/计算预算的独立baseline，而不是只改一个现有GRPO系数。

证据边界：论文主图各任务每方法一条run；不同critic初始化的三run研究只覆盖FrontierCS。其结果不能视作我们Qwen2.5-3B Search-G1/Tree-GRPO的已验证收益。[论文 §5](https://arxiv.org/html/2609.36802v1#S5)。

本地只保留必要官方源文件（小于1MiB）及此说明；源码身份/文件SHA见 `easyppo-source-audit.json`。本次未运行官方测试，源码中的CPU tests只作为可读执行意图证据，不报告其通过。
