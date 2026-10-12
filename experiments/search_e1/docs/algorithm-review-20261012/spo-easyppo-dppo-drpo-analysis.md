# SPO EasyPPO DPPO DRPO 的机制与 Search G1 迁移分析

核对日期：2026-10-12。结论基于指定论文版本和作者官方源码；数学推论另行标明。

这四种方法处理两个不同问题：Simple SPO、DPPO、DRPO控制给定优势信号如何推动策略；EasyPPO改善 learned critic，进而改善优势估计。它们没有直接证明 Search-G1 的探针奖励是否正确，也没有把轨迹 NLL 当作学习难度。

## 方法身份与选择依据

| 简称 | 本文主版本 | 原文 | 官方实现固定版本 |
| --- | --- | --- | --- |
| SPO | Simple Policy Optimization，ICML 2025 最终版本 | [arXiv 2401.16025v9](https://arxiv.org/html/2401.16025v9)、[ICML](https://proceedings.mlr.press/v267/xie25m.html) | MyRepositories-hub/Simple-Policy-Optimization `9fdeda315b00cee82d1dc4e10c01db200c97e909` |
| EasyPPO | EasyPPO Stabilizing the Critic Is Key | [arXiv 2609.36802v1](https://arxiv.org/html/2609.36802v1) | EasyPPO/EasyPPO `0a490a88ad7b84a8f7974fe168f0ba63600b1e31` |
| DPPO | Divergence Proximal Policy Optimization | [Rethinking the Trust Region in LLM Reinforcement Learning，2602.04879v3](https://arxiv.org/html/2602.04879v3) | sail-sg/Stable-RL `c6af1d119c0fe30d16dd9a837d35fdf695acd226`；verl `2035f58dcb06d8cad4caee107b30f4045d724367` |
| DRPO | Divergence Regularized Policy Optimization | [Rethinking the Divergence Regularization in LLM RL，2606.09821v1](https://arxiv.org/html/2606.09821v1) | Tencent-Hunyuan/UniRL `0fe4f99dc2384a022fe0c8472aec889c216d3c07` |

优先选择与 LLM policy optimization 有直接关系、作者代码可核对的版本。Simple SPO 被当前 DRPO 原文直接作为对照；DPPO 有 verl 官方实现。所读取 GitHub 页面上 Simple SPO 为123 stars、Segment SPO为56，但 stars不是引用量，也不能比较整个 verl-recipe 的 stars 与单个算法仓库。OpenAlex版本记录存在覆盖/去重问题，GitHub API额度受限，故不声称完成全球引用量或讨论量排名。

SPO还有 Single-stream、Segment、Soft 等同名方法；DPPO有机器人 diffusion policy版本；DRPO还有 Decoupled Reward 版本。主表不将这些方法混为一谈。特别 Simple SPO 早期v4和最终v9目标不同，此处只用v9及相符源码。

## 统一变量和一次更新

令 \(s_t\) 为问题及已生成历史，\(y_t\) 为原采样 token：

\[
p_t=\mu(y_t\mid s_t),\qquad q_t=\pi_\theta(y_t\mid s_t),\qquad r_t=q_t/p_t.
\]

\(\mu\) 严格表示真实采样行为策略；不是静态 SFT reference、特权 teacher 或每个 optimizer step 重新重置的 current policy。若训练框架重算一个冻结旧anchor \(\nu\)，实际ratio是 \(q/\nu\)，但数据仍由 \(\mu\) 生成；不能未经核对就把二者当作同一分布。\(A_t\) 在一次采样批的策略更新中视作固定值。这里只比较 policy-gradient主项；reference KL、entropy、有效token mask和loss归约另行保持。

实际流程为：自回归生成轨迹并保存行为概率 → 计算奖励及优势 → 在原历史上并行 teacher-forced 前向当前模型 → 取得所有有效 policy token 的 \(q_t\) → 计算 surrogate → 按指定分母归约 → backward和optimizer更新。并行前向不等于重新生成轨迹；多次mini-batch更新也不等于每条轨迹都训练多次epoch。

未加约束的token级局部surrogate梯度为

\[
\nabla J=\mathbb E[ r_t A_t\nabla_\theta\log q_t ].
\]

忽略边界点和辅助项，PPO的clip给这个梯度乘0/1门：正优势且 \(r>1+\epsilon\)，或负优势且 \(r<1-\epsilon\) 时置零。这个token没有继续推远的自身梯度，但共享参数仍可能被其他token更新而继续移动。

## Simple SPO 平滑地控制相对概率位移

最大化目标为

\[
J_{\rm SPO}=\mathbb E\left[rA-\frac{|A|}{2\epsilon}(r-1)^2\right].
\]

它以二次惩罚替换PPO硬clip。对固定非零 \(A\) 求导：

\[
\nabla J_{\rm SPO}=\mathbb E\left[
\underbrace{\left(1-\frac{\operatorname{sign}(A)(r-1)}\epsilon\right)}_{w_S}
rA\nabla\log q\right].
\]

正优势的相对概率接近 \(1+\epsilon\) 时推力逐渐降到0，超过后反向；负优势在默认 \(\epsilon=.2\) 下同理围绕 \(1-\epsilon\)。孤立驻点还须满足 \(0<r^*\le1/p\)：负优势只有 \(\epsilon<1\) 才有正内点，正优势则需 \(p(1+\epsilon)\le1\)。这是软惩罚及局部驻点，不是把真实模型参数或全部token概率硬投影回可行集合。共享softmax约束、参数耦合和有限步长仍然存在。

官方 `Trainer.compute_policy_loss` 直接求导上述二次目标；旧logprob来自生成时冻结的batch。训练还包括GAE、critic/value clipping、entropy、mini-batch优势标准化及grad norm clipping。主入口epsilon默认0.2。原作者实验为Atari/MuJoCo，不能把原文成绩当作LLM搜索任务复现收益。[实际loss代码](https://github.com/MyRepositories-hub/Simple-Policy-Optimization/blob/9fdeda315b00cee82d1dc4e10c01db200c97e909/atari/trainer.py#L70-L93)、[采样与GAE入口](https://github.com/MyRepositories-hub/Simple-Policy-Optimization/blob/9fdeda315b00cee82d1dc4e10c01db200c97e909/atari/main.py)。

机制边界：SPO仍以ratio衡量位移。当旧概率极小时，相对ratio及其惩罚可能很大；它不保证全部词表散度受限。最终论文将该二次函数写成convex，但最大化目标二阶导为 \(-|A|/\epsilon<0\)，应为concave；本文以公式、代码和独立求导为准。

## DPPO 以分布散度替换 ratio 门限

理想未加IS上限的surrogate形式为

\[
J_{\rm DPPO}=\mathbb E[M_t r_t A_t],\qquad
M_t=\begin{cases}
0,&A_t(r_t-1)>0\ \text{且}\ \widehat D_t>\delta,\\
1,&\text{其他情况}.
\end{cases}
\]

mask被冻结、不反向求导。只有已经越界、且该优势还会继续推远时才停掉该token梯度；不是所有散度大的token都删除。朝旧策略返回的更新仍保留。[作者实际mask](https://github.com/sail-sg/Stable-RL/blob/c6af1d119c0fe30d16dd9a837d35fdf695acd226/verl/trainer/ppo/core_algos.py#L1241-L1308)。

完整TV是 \(\tfrac12\sum_v|q_v-p_v|\)。Binary-TV将采样token与其余词表合并，得到

\[
\widehat D_{\rm TV}=|q_t-p_t|=p_t|r_t-1|.
\]

Binary-KL用同一Bernoulli聚合计算 \(p\log(p/q)+(1-p)\log[(1-p)/(1-q)]\)；TopK额外保留部分词表概率，代价比完整词表小。重要边界是：这些聚合散度为完整散度的下界，通过近似门限不代表完整散度一定通过。

例如 \(p=10^{-5},q=10^{-3}\)：ratio为100，绝对位移仅0.00099。正优势的PPO会clip；Binary-TV DPPO在 \(\delta=.15\) 下不会仅因ratio=100删除它。DPPO相当于给予低旧概率token更宽的相对更新区间，而不是根据NLL把低置信轨迹统一降权。[原文方法](https://arxiv.org/html/2602.04879v3#S4)、[另一作者官方框架中的Binary-TV门](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/dppo.py#L40-L89)。

实际作者及verl源码用等价的一阶梯度写法，并加IS上限c：

\[
L_{\rm code}=-\operatorname{sg}(M)\operatorname{sg}(\min(r,c))A\log q.
\]

不对mask、IS权重求导，最小化该loss时梯度下降方向是 \(M\min(r,c)A\nabla\log q\)。无cap生效时才与上面的无截断surrogate梯度一致。源码对ratio计算还有logratio数值clamp。trust mask只改变分子，归约分母仍用原response mask，不仅对保留token重新平均。[作者主loss](https://github.com/sail-sg/Stable-RL/blob/c6af1d119c0fe30d16dd9a837d35fdf695acd226/verl/trainer/ppo/core_algos.py#L1160-L1199)、[verl Binary-TV](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/verl/trainer/ppo/core_algos.py#L1380-L1457)。

代码移植必须检查真实anchor：作者Stable-RL主实现使用rollout logprob，verl DPPO函数接收old logprob，而常规trainer会重算old。仅打开名为bypass的开关也可能同时改变loss mode。因此不能凭 `dppo_tv` 函数名就认定行为锚点与论文一致。verl函数fallback cap为20、ActorConfig默认为3、示例覆盖为10000，不能混称同一默认；该示例也关闭group STD优势归一，并使用固定horizon的seq-mean-token-sum-norm。[真实配置](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/examples/dppo_trainer/run_qwen3_30b_a3b_megatron.sh#L56-L85)、[旧概率调用链](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/verl/trainer/ppo/ray_trainer.py#L1570-L1586)。

TopK也需分开论文和代码：论文定义TopK与采样token的并集；作者vLLM实现保持K槽，若采样token不在TopK则替换最后一个槽。仍是合法粗粒化分区，但不应声称逐字实现了K+1集合。[采集源码](https://github.com/sail-sg/Stable-RL/blob/c6af1d119c0fe30d16dd9a837d35fdf695acd226/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L543-L573)。

## DRPO 以连续修正控制绝对概率位移

它将SPO的ratio二次项乘上行为概率：

\[
J_{\rm DRPO}=\mathbb E\left[rA-\frac{|A|p}{2\delta}(r-1)^2\right],
\]

由公式自行求导得

\[
\nabla J_{\rm DRPO}=\mathbb E\left[
\underbrace{\left(1-\frac{\operatorname{sign}(A)(q-p)}\delta\right)}_{w_D}
rA\nabla\log q\right].
\]

SPO平滑控制 \(r-1\)，DRPO平滑控制 \(q-p\)；DPPO在该概率位移越界后硬置零，DRPO理论上可以连续降低推力并在可达越界时反向。局部形式驻点为 \(q=p+\operatorname{sign}(A)\delta\)，仍受[0,1]及共享模型可行域限制。[原文式8–10](https://arxiv.org/html/2606.09821v1#S3)。

**默认参数是关键**：原文、类默认及发布recipe的 \(\delta=12.5\)，不是0.125或它的倒数。有效概率给出 \(|q-p|\le1\)，故此配置的 \(w_D\in[.92,1.08]\)，不会触发负权重反向机制。这个范围只限制乘法修正，不限制 \(rA\nabla\log q\) 的总大小，也不能仅凭此断言最终算法收益来自哪一个因素。[默认及loss源码](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/drpo.py#L28-L64)。

源码直接求导二次loss。若移植成 \(-\operatorname{stopgrad}(w_D)rA\)，其一阶梯度才与原目标一致；若让w再参与autograd，会多出一项。实际代码也将logratio限制到[-20,20]，上述无截断公式需限定在该区间内。

完整recipe不按group STD归一优势；loss按每轨迹token sum/固定8192再轨迹mean，不能声称沿用了我们每轨迹有效tokenmean设置。README仍写一更新，但当前类支持multi-update且recipe采用4个不同mini-batch，不能称同一个token训练了4epochs。[实际recipe](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/examples/ar/qwen3_drpo_4b_base_dapo_sglang.yaml#L185-L231)、[优势配置](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/examples/ar/qwen3_drpo_4b_base_dapo_sglang.yaml#L35-L39)。

## EasyPPO 改善 critic 的统计目标与梯度噪声

它保留learned critic及token GAE，actor沿用clipped PPO。三个关键操作如下。

1. 达到长度预算的截断轨迹仅从actor loss排除；critic保留这些轨迹的实际returns。若两者均删，critic会学到“已经完成条件下”的平均回报，可能掩盖越来越高的截断率。actor-only filtering仍有选择效应，论文理想baseline推导不是真实PPO收敛保证。
2. 同题多轨迹估计return总体标准差 \(\hat\sigma_q\)，令 \(u_q=1/\max(\hat\sigma_q,\varepsilon)\)，按unique prompt均值归一 \(w_q=u_q/\operatorname{mean}_{q'}u_{q'}\)。用w加权critic regression loss；不把w再乘给GRPO actor优势，也不改变return目标。
3. 采用适度更小的critic mini-batch，各mini分别重新前向、反向、clip、Adam。异常样本影响的范围因而与整batch一次clip不同；GPU micro-batch仅是梯度累积，不等于额外optimizer更新。

批内GAE和actor优势先由更新前critic计算，再做critic更新，最后临时掩actor的overlong行；不是critic每个mini更新后重算一次本批优势。[论文方法](https://arxiv.org/html/2609.36802v1#S4)、[执行次序源码](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/v1/trainer_base.py#L461-L540)。

源码通用权重函数默认beta=1，但实际EasyPPO配置beta=.5，即逆STD，不是逆variance。对于简化value head，梯度噪声二阶矩正比于return variance；loss乘1/STD会使梯度二阶矩噪声项乘1/variance，这解释选择来源，实际prompt估计仍只是各prefix条件噪声的近似。[权重实现](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/verl/trainer/ppo/core_algos.py#L2521-L2592)、[真实配置](https://github.com/EasyPPO/EasyPPO/blob/0a490a88ad7b84a8f7974fe168f0ba63600b1e31/configs/easyppo_aime.yaml#L105-L113)。

官方可运行AIME recipe每轮32题×16响应，critic mini128响应，共4次critic机会；论文Search-R1是64×16、mini256，同样4次。不能把AIME配置冒充Search-R1原始配置。实际critic loss为加权clipped value MSE、全DP有效token分母，而不是我们每轨迹先平均。当前Search-G1/Tree-GRPO无learned critic，因此引入EasyPPO主体将增加一个critic及计算预算，不是单换actor loss。

## 数学比较和当前实验的含义

| 方法 | 改动位置 | 梯度控制对象 | 不直接解决的事 |
| --- | --- | --- | --- |
| Simple SPO | actor surrogate | 相对概率 \(q/p-1\) 的连续修正 | 探针奖励可靠性、全对组优势放大 |
| DPPO | actor门限 | 分布散度近似及方向；越界硬停 | 低NLL等于易学、完整词表严格约束 |
| DRPO | actor正则 | 绝对概率 \(q-p\) 的连续修正 | 默认一定触发反向、w有界等于总梯度有界 |
| EasyPPO | critic数据及loss、actor截断mask | critic条件目标与噪声 | critic-free GRPO的直接替换 |

以下为针对我们方案的独立推论，不是论文已验证的Search-G1结果。

**奖励、优势、更新几何应分开。** 若全对组 \(R_i=1+\lambda s_i\)，在稳定项很小时：

\[
A_i=\frac{\lambda(s_i-\bar s)}{|\lambda|\sigma_s+\varepsilon}
\approx\frac{s_i-\bar s}{\sigma_s}\quad(\lambda>0).
\]

小奖励差异依然可成为量级1的优势。SPO/DPPO/DRPO把A当作给定输入，并不判断这些偏好是不是正确。在理想 \(q=p\) 时，SPO和DRPO的权重为1，DPPO的mask为1；最初梯度均沿原A。故替换这些loss不能自动关闭全对组里的探针更新信号。

**轨迹熟悉度和本轮策略位移不同。** 用户提出的

\[
\ell_i=-\frac1{T_i}\sum_{t\in\mathrm{policy}}\log p_{it}
\]

可复用已经存在、mask正确对齐的行为logprob，通常不增加模型前向；其含义是该历史和响应在该行为策略下的平均NLL。若保存的是未经temperature/processors变换的raw model概率，应明确标作raw-policy NLL proxy，不能混称真实行为NLL。SPO/DPPO/DRPO的 \(r\)、\(q-p\) 则需本来训练前向的current概率。高NLL轨迹未必更新位移大，也未必不值得学；DPPO反而允许罕见token更宽的相对位移。NLL门控是另一个训练假设，不是DPPO的等价写法。

**对当前混合组采样计划的建议。** 保持已选定“512训练组、最多2048独立题、mixed优先、只用all-correct保底”的方案不变，先分别记录mixed与fallback两类的严格EM \(C\)、原任务评分、shaped \(R\)、归一优势、有效policy NLL、ratio分位数、Binary-TV、越界方向和clip比例。优先检查是否存在与更新位移相符的异常，以决定PPO与新surrogate对照的优先级。若问题主要来自fallback组的A方向，优先解决奖励/优势层，不能预期更换trust-region形式自动修复。正的NLL权重若只乘A且不再归一，在同一 \(q,\mu\) 下不改变DPPO的方向门，只缩放许可PG；后续参数路径仍可能改变，组合收益需另验。

EasyPPO的选择偏差分析也提醒：过滤会改变被训练的分布。但它研究的是截断轨迹及learned critic，不能直接用来证明我们的mixed筛选有效或无效。

## 同名 Single stream SPO 的补充

另一种与我们group退化议题更直接相关的方法是 [Single-stream Policy Optimization](https://arxiv.org/html/2509.13232v2#S4)，作者提供 [verl-recipe/spo](https://github.com/verl-project/verl-recipe/tree/242c73fcadf331e7214c4b48f67423f45239cea8/spo) 官方源码。它不采用Simple SPO的二次目标，而是给每题保存历史baseline：

\[
a_i=R_i-\hat v_{q(i)}^{old},\qquad
\widetilde a_i=\frac{a_i-\bar a_{\mathcal B}}{s_{\mathcal B}+10^{-8}}.
\]

先用尚未纳入本轮reward的baseline计算优势，随后更新策略，再更新按题历史。二值reward版本使用带遗忘的Beta计数，等价于自适应EMA。归一作用于当前batch跨题，并非历史batch的全局累计均值/方差；它绕开本题当前group恒定reward导致的0优势，但仍可能有陈旧baseline或全batch标准化为0的问题。[实际优势代码](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/spo_ray_trainer.py#L529-L561)。

实际源码控制遗忘率的漂移量为旧轨迹token绝对logprob差的均值，不是精确全词表KL；更新actor后额外回读历史轨迹概率，不能宣称零额外forward。它同时改变每题rollout数、prompt采样分布和状态缓存，不能拿它的成绩支持“只换loss就有效”，也不能把当前连续shaped reward无定义地塞入二值Beta计数。[漂移实现](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/spo_ray_trainer.py#L240-L303)。完整边界另存 `single-stream-spo-notes.md`。

## 核验范围

源码均固定commit，保留必要小型官方源码及文件SHA。独立标量有限差分复核了SPO/DRPO导数、负优势符号、罕见token例子及DRPO默认权重界，结果在 `independent-math-checks.json`。这些不是官方训练、GPU实测或端到端收益复现；未启动、停止或修改任何训练，也未改已选定的mixed采样方案。

分项代码审计：`easyppo-notes.md`、`drpo-notes.md`、`dppo-notes.md`；它们记录类默认、发布recipe和论文设置的区别。文档推送和逐文件读回结果另存 `publication-verified.json`。
