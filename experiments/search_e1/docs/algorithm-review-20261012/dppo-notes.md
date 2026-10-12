# DPPO：论文与官方源码核验笔记

只读研究，2026-10-12；未访问训练节点、未启动训练、未改变运行源码。DPPO在此指 **Divergence Proximal Policy Optimization**，不是diffusion-policy PPO。

## 0. 固定证据身份

- 论文：[Rethinking the Trust Region in LLM Reinforcement Learning，arXiv:2602.04879v3，2026-06-12](https://arxiv.org/html/2602.04879v3)。实际读取§3、§4.1–4.4、§5.2–5.4及Appendix B/C/E。
- 作者主仓库 `sail-sg/Stable-RL` HEAD：`c6af1d119c0fe30d16dd9a837d35fdf695acd226`；Git远端实际读取后，按该SHA读取下列源码。
- verl移植仓库 `verl-project/verl` HEAD：`2035f58dcb06d8cad4caee107b30f4045d724367`。
- [verl固定版本文档](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/docs/algo/dppo.md#L75-L81)明确仅实现Binary-TV/Binary-KL，TopK指向作者仓库。文档SHA256 `293272e4588b437a8ef70c2ecf29c28c6fc9dd990c71e6e2a696f7aff8b75ad2`。

## 1. 它改什么

DPPO改PPO/GRPO的**token更新信赖域掩码**，不定义新答案奖励、难度分数、组选择或专用advantage估计器。论文的动机是ratio对小概率token过敏：旧概率很小，移动很少概率质量也可产生很大ratio；高概率token移动较多质量却可能仍在ratio区间。其修正依据是概率质量/分布散度，而非仅ratio大小。[论文§4.1–4.3](https://arxiv.org/html/2602.04879v3#S4)

## 2. 一轮的串行执行与方向条件

1. 行为策略 \(\mu\) 生成轨迹，保存采样时各token的概率；奖励/advantage沿选定基线计算。
2. 当前策略 \(\pi_\theta\) 在同一历史上计算采样token的概率，得到 \(r_t=\pi_\theta(a_t|s_t)/\mu(a_t|s_t)\)。
3. 估计当前分布相对行为分布的散度 \(\widehat D_t\)，用下述掩码判定是否停止该token的PG贡献。
4. 仅在已经沿advantage方向继续远离行为锚点、且散度超阈值时屏蔽；回向锚点的更新保留。
5. 按既定response mask/reduction聚合PG，再执行既有优化器；不是拒绝轨迹后重新采样。

论文Eq.12：

\[
M_t=0\iff \widehat D_t>\delta\ \land\
\big[(A_t>0\land r_t>1)\lor(A_t<0\land r_t<1)\big];\qquad M_t=1\text{ otherwise}.
\]

正优势却 \(r_t<1\)，或负优势却 \(r_t>1\)，即使散度超阈值，也保留能把ratio拉回1的更新。\(A=0\)时实现可能走负分支，但PG仍为0。这里的“mask”是更新许可，不是严格EM是否合法的评分mask。

## 3. Binary、TopK与下界边界

记 \(u=\mu(a_t|s_t),p=\pi_\theta(a_t|s_t)\)。Binary把词表分为采样token与其余token：

\[
D^{\rm Bin}_{\rm TV}=|u-p|,
\qquad
D^{\rm Bin}_{\rm KL}=u\log\frac up+(1-u)\log\frac{1-u}{1-p}.
\]

KL方向是**行为策略到当前策略**，不是当前策略到初始reference的KL正则。

论文TopK保留 \(S=\operatorname{TopK}(\mu,K)\cup\{a_t\}\)，将其余token聚为other，计算该粗粒化分布的TV或KL。保留集合加other是词表分区；triangle inequality/log-sum inequality给出

\[
D^{\rm partition}_{\rm TV}\le D^{\rm full}_{\rm TV},\qquad
D^{\rm partition}_{\rm KL}\le D^{\rm full}_{\rm KL}.
\]

因此**近似散度超阈值证明full散度也超阈值；近似未超阈值不保证full散度安全**。other内部概率重分配可能被漏掉。该掩码也没有逐步硬投影保证所有状态full散度均在阈值内；不能把论文信赖域理论直接等同于近似实现的单调收益保证。[公式13–16与Appendix B](https://arxiv.org/html/2602.04879v3#S4.SS4)

作者实际TopK输入也有论文/代码细节差异：[vLLM采集L543–573](https://github.com/sail-sg/Stable-RL/blob/c6af1d119c0fe30d16dd9a837d35fdf695acd226/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L543-L573)保持固定K个槽；采样token不在topK时，用它**替换最后一个topK token**，而非追加形成K+1显式槽。这仍是合法分区，保留下界性质，但不是论文集合的逐字实现。

## 4. 实验结论与配置边界

论文§5.2比较行为锚点与trainer重算锚点，支持用实际rollout \(\mu\) 定义信赖域；不能用同参数、不同推理后端重算概率默认为相同。Appendix C的稳定性实验不做advantage标准差归一化，DPPO的IS cap取无穷；Appendix E较大规模实验采用有限cap（默认5），阈值依Binary-TV/KL与实验配置变化。这些是具体实验设置，不是DPPO定义强制使用的奖励/标准化选择。[论文§5.2](https://arxiv.org/html/2602.04879v3#S5.SS2)、[Appendix C/E](https://arxiv.org/html/2602.04879v3#A3)

## 5. 源码：stop-gradient、IS cap、reduction、anchor

### 作者主实现

- [主损失L1160–1199](https://github.com/sail-sg/Stable-RL/blob/c6af1d119c0fe30d16dd9a837d35fdf695acd226/verl/trainer/ppo/core_algos.py#L1160-L1199)：明确使用 `rollout_log_prob` 构造ratio；log-ratio上界20；ratio再以 `clip_ratio_c` 截断并detach，mask亦detach。损失为 \(-A\,\mathrm{sg}(\min(r,c))\,\mathrm{sg}(M)\log\pi_\theta\)，其PG为 \(M\min(r,c)A\nabla\log\pi_\theta\)，没有对mask或IS权重反传。
- [mask L1241–1308](https://github.com/sail-sg/Stable-RL/blob/c6af1d119c0fe30d16dd9a837d35fdf695acd226/verl/trainer/ppo/core_algos.py#L1241-L1308)：Binary-TV、Binary-KL、TopK均有对应分支；`*_recompute`是明确的另一种锚点实验，不应与标准DPPO混写。
- `compute_policy_loss_with_mask`函数fallback cap是10，**但** [ActorConfig L143–144](https://github.com/sail-sg/Stable-RL/blob/c6af1d119c0fe30d16dd9a837d35fdf695acd226/verl/workers/config/actor.py#L137-L145)已定义cap=3、reduction=`token-mean`；真实配置的显式值优先，不能说“实际默认一定10”。
- [TopK核函数L3–51](https://github.com/sail-sg/Stable-RL/blob/c6af1d119c0fe30d16dd9a837d35fdf695acd226/verl/workers/actor/utils.py#L3-L51)：`no_grad`、转FP32、读取当前完整logits上的所选token概率，other以剩余质量计算；数值floor=1e-12；同时返回TV/KL。
- PG numerator用trust mask，`agg_loss`分母仍用原response mask；不会仅对“未被屏蔽token”重新归一化。[reduction L1026–1081](https://github.com/sail-sg/Stable-RL/blob/c6af1d119c0fe30d16dd9a837d35fdf695acd226/verl/trainer/ppo/core_algos.py#L1026-L1081)

### verl移植

- [Binary-TV L1380–1457](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/verl/trainer/ppo/core_algos.py#L1380-L1457)、[Binary-KL L1461–1542](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/verl/trainer/ppo/core_algos.py#L1461-L1542)：log-ratio clamp为[-20,20]；mask和截断ratio都detach；可另乘传入的rollout correction weights；函数fallback cap=20，但 [ActorConfig默认cap=3、token-mean](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/verl/workers/config/actor.py#L155-L165)。
- [官方示例L56–85](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/examples/dppo_trainer/run_qwen3_30b_a3b_megatron.sh#L56-L85)覆盖为cap=10000、`seq-mean-token-sum-norm`、GRPO std-normalization=False、KL奖励/正则关闭。它不是“每轨迹token均值再轨迹均值”： [agg_loss L1181–1202](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/verl/trainer/ppo/core_algos.py#L1181-L1202)实现为轨迹token-sum后按序列数及固定horizon归一化；若scale未设，horizon取response mask宽度。
- **锚点不能凭函数名认定**：移植函数只接 `old_log_prob`。 [ray trainer L1570–1586](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/verl/trainer/ppo/ray_trainer.py#L1570-L1586)在普通路径重算old，bypass路径才令其等于rollout。 [apply_bypass_mode L1137–1144](https://github.com/verl-project/verl/blob/2035f58dcb06d8cad4caee107b30f4045d724367/verl/trainer/ppo/rollout_corr_helper.py#L1137-L1144)同时把 `loss_mode` 改成 `bypass_mode`，因此**不能盲开bypass后声称仍执行dppo_tv/dppo_kl**。示例本身未显式设置这个锚点路径；实际移植必须核resolved config与传入old tensor来源，本笔记未运行示例或认定其已满足论文锚点。

源文件SHA256：作者core `4f3738abecebbe4f6e1202103fa57deeb2b9ea38b14653152b38f04fe8db7684`；作者TopK utils `834da2b21867d4ee98e500af4a0fb3807ea91bdd8c1792c30d889e73453d2c09`；作者vLLM采集 `9276fe844626aefe5d11a5752a183e2eddcbec00fe1015b1f6501225cdab7ad1`；verl core `369b614f239f237a04a38a40b1877cd6fd14aa7bd5f9ff4fe4d7758b682a6948`；verl actor config `d7631944b186f162b897022c9fac8b2f0ede7f8bb6cba4276d98e5d5033fd60b`；verl示例 `42a82250d6d664006f394cfb22ba000ca2e922bb539135846095b2271737a0e5`；verl ray trainer `eb41f99ccc66976eab14ebe35aa1aa5b7318e02469240ef1569ac4c884c94312`；bypass helper `763158342551bc4077805a8a009d959a7a2043c4d43b772fe6045f4a40932f3e`。

## 6. 与当前NLL难度权重、mixed/fallback方案的联系

这是数学联系，不是新训练结论：

- NLL方案按一条轨迹在冻结行为策略下的熟悉度连续缩放PG；DPPO按**当前与行为策略的分布变化及优势方向**给token二值许可。Gaussian-CDF难度D与DPPO divergence D不是同一量。
- 若用正权重 \(w_iA_i\) 且不再归一化，同一 \(\pi,\mu\) 下优势符号不变，DPPO mask不变，允许token的PG再乘 \(w_i\)；后续参数路径改变后，mask当然可能不同。不能说两个机制效果等价。
- DPPO倾向释放“旧概率很小但移动质量很小”的探索token；高NLL downweight可能同时削弱这些稀有正确轨迹。这是互补/冲突均可能的取舍，不证明应自动合并。
- mixed优先/all-correct候补改变采样分布与组信号；DPPO不筛题、不消除奖励误导，也不自动消除全对组小probe gap经std放大的优势。官方示例恰关闭std-normalization，不能将这一配置变化的收益归因于mask本身。
- 首个迁移比较必须固定奖励、组规则、advantage normalization、reduction、behavior锚点和IS cap；只替换PPO ratio clipping为DPPO mask，之后才单列NLL/候补机制消融。该建议没有授权或启动新训练。
