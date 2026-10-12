# DRPO 原文与官方代码核对（2026-10-12）

本笔记只读公开论文/官方源码，未访问训练节点，未下载模型、环境或轨迹，未运行训练。已按 paper-glance 的方法流程提取规则核对。目标为 **Divergence Regularized Policy Optimization**，不是同名 Decoupled Reward Policy Optimization。

## 身份与固定版本

- 原文：[Rethinking the Divergence Regularization in LLM RL，arXiv:2606.09821v1](https://arxiv.org/html/2606.09821v1)，2026-06-08，Yao / Zhou / Qi / Lee / Bo / Pang。
- 官方：[Tencent-Hunyuan/UniRL DRPO](https://github.com/Tencent-Hunyuan/UniRL/tree/0fe4f99dc2384a022fe0c8472aec889c216d3c07/DRPO)。实际 `git ls-remote` 获取 HEAD/main 均为 `0fe4f99dc2384a022fe0c8472aec889c216d3c07`；以下 raw 源码均按此 SHA 读取，而非浮动 main。
- 本次 GitHub API 请求返回403 rate limit，随后使用只读 Git ref 和 pinned raw 获取成功；不将 API 失败描述为代码不可用。

| 实读小源码 | 完整 SHA256 |
|---|---|
| `unirl/algorithms/drpo.py` | `3da9d8c4b11050cd58285ff6089bf684242714877ed015c95c764a9f6c0a79df` |
| `unirl/algorithms/dppo.py` | `990b5513cb9390735544995f85f3a5889cada744739f5e0f7832812f337dfa2b` |
| `unirl/algorithms/base.py` | `326291439342130064abb865a5ef69fb5a49475edb65f040d38e0db39edfae52` |
| `DRPO/config.yaml` | `75800127f2cd3b5751908000d297a852bce30aac4a7952f407378a4a48b1e0d8` |
| `examples/ar/qwen3_drpo_4b_base_dapo_sglang.yaml` | `39cec24501c0b4492d519fd13e4c6cd8713476cc5d5842080e20fa7c6d516024` |
| `DRPO/README.md` | `df8124341076c94d6dadae7b5d4801ea0550205a5feb5ae06b4e039b99ea7364` |

## 核心公式与一轮执行

固定采样行为策略 \(\mu_t\)，令当前策略 \(\pi_t\)、\(r_t=\pi_t/\mu_t\)、固定优势 \(A_t\)。原文 Eq8 的最大化目标为

\[
J=\mathbb E_\mu\sum_t\left[r_t A_t-\frac{|A_t|}{2\delta}\mu_t(r_t-1)^2\right].
\]

先采样并保留 rollout token/logprob，再算同题优势；冻结 anchor；对原 token/history 做当前模型 replay；展开轨迹优势到 token；计算比率及二次惩罚；应用有效 token mask、归约并反传。官方实际程序遵循这个顺序。[源码 L120–174](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/drpo.py#L120-L174)

由上式自行求导得到

\[
\nabla J=\mathbb E_\mu\sum_t w_t r_t A_t\nabla\log\pi_t,
\quad w_t=1-\operatorname{sgn}(A_t)\frac{\pi_t-\mu_t}{\delta}.
\]

这与原文 Eq9–10 的符号×Binary-TV写法等价。正优势且概率增大、或负优势且概率减小，均属于继续远离行为概率：\(w\) 随位移减小，越过 \(\delta\) 后为负，反向拉回；朝行为概率移动时 \(w>1\)。孤立 token 的形式驻点为 \(\pi_t=\mu_t+\operatorname{sgn}(A_t)\delta\)，但需注意概率可行域及 softmax/参数耦合，不能认为所有 token 都能同时达到该驻点。[原文 §3](https://arxiv.org/html/2606.09821v1#S3)

**自己的数学边界检查：** 有效概率下 \(|\pi-\mu|\le1\)，故 \(w\in[1-1/\delta,1+1/\delta]\)。权重有界不等于完整 \(rA\nabla\log\pi\) 或参数梯度有界。

## 可求导惩罚与 stop-gradient

官方直接实现负目标：\(-rA+|A|\mu(r-1)^2/(2\delta)\)。`A`、旧概率 detach，`ratio` 保留梯度，二次项正常求导，未另乘一个可求导的 \(w\)。[源码 L51–64](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/drpo.py#L51-L64)

移植若改写为加权 surrogate，应使用 `-stopgrad(w) * A * r` 才得到上述一阶梯度；若把 `-w(theta)*A*r` 整体求导，会额外求导权重，变成另一目标。反之，直接二次目标中不能 detach `ratio_delta`，也不能再重复加入同一修正。anchor logprob也必须是真正冻结量。

实现还将 \(\log r\) 截到 `[-20,20]`，超出该范围时不再严格等于无截断论文梯度；这是数值实现边界。[源码 L51–52](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/drpo.py#L51-L52)

## 默认阈值的关键边界

- DRPO class默认、官方recipe、原文§4均为 **12.5**（代码名 `drpo_epsilon`，就是公式 \(\delta\)，不是其倒数）。DPPO code/原文推荐为 **0.15**。[DRPO默认 L28–38](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/drpo.py#L28-L38)，[原文 §4](https://arxiv.org/html/2606.09821v1#S4)，[DPPO默认 L28–37](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/dppo.py#L28-L37)。
- **由公式推得**：12.5使 \(w\in[0.92,1.08]\)，正常概率下不会出现负权重；其 Binary-TV 区间覆盖整个概率可行域。理论上的越界 corrective 机制存在，但不能说官方默认运行实际触发了这种负修正，也不能说它使用与 DPPO .15相同强度的边界。
- `|A|`同时缩放惩罚，令形式边界不依赖优势幅度；但它不修改奖励/优势来源，不能据此解决 probe错误或全对组小差距经std标准化产生的信号。特别 \(\pi=\mu\) 时 \(w=1\)：初始梯度就是原 \(rA\) 梯度。

## 归约与 anchor：默认类和示例要分开

- class默认 `loss_agg_mode=token-mean`；完整recipe选择 **seq-mean-token-sum-norm**：每轨迹有效 token loss之和除固定 `horizon=8192`，再轨迹平均，**不是每条轨迹除自己的有效 token 数**。[recipe L185–204](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/examples/ar/qwen3_drpo_4b_base_dapo_sglang.yaml#L185-L204)，[归约 L94–129](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/base.py#L94-L129)。
- recipe `normalize_adv_by_std:false`，只组内均值中心化；不能套入我们旧 G1 的 group sample std设置。[recipe L35–39](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/examples/ar/qwen3_drpo_4b_base_dapo_sglang.yaml#L35-L39)
- 默认 `old_logp_source:rollout`：使用采样器logprob作冻结anchor，同batch后续更新沿用它，因此ratio兼含engine mismatch与策略位移。`replay`路径则预更新用训练端同温度replay冻结一次，后续固定，不每步重置。[anchor源码 L32–53](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/base.py#L32-L53)。recipe将 `sampling_temperature`绑定实际sampling.temperature；processors/温度的实际一致性仍需集成验收，不能仅凭同名参数宣称bitwise。
- **官方文档内部矛盾**：README L142–145仍写只1次更新/`supports_multi_update=False`，但同commit `drpo.py` L118实际True、recipe L231实际4次disjoint mini-batch更新。以代码与recipe为准，保留这个文档过时证据。[旧文档段](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/DRPO/README.md#L142-L145)，[实际类](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/drpo.py#L115-L118)，[实际recipe L220–231](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/examples/ar/qwen3_drpo_4b_base_dapo_sglang.yaml#L220-L231)。

## 同仓库 SPO / DPPO 入口及可用范围

- **SPO**：已核可用入口是同一 `_drpo_loss(mu_weighted=False)`分支，去掉 \(\mu\) 因子，固定ratio二次惩罚。recipe注明 `penalty_mu_weighted:false`即plain SPO。此处只确认官方实现入口，不代替阅读 SPO 原论文。[分支 L56–64](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/drpo.py#L56-L64)，[recipe说明 L188–192](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/examples/ar/qwen3_drpo_4b_base_dapo_sglang.yaml#L188-L192)。
- **DPPO**：`unirl/algorithms/dppo.py` `_dppo_mask`/`_dppo_loss`，keep=朝 \(\mu\) 移动 OR \(|\pi-\mu|\le\delta\)，keep在no_grad/detach下计算；越界且继续远离时硬置零，不提供负修正。[L40–89](https://github.com/Tencent-Hunyuan/UniRL/blob/0fe4f99dc2384a022fe0c8472aec889c216d3c07/unirl/algorithms/dppo.py#L40-L89)。不要与同repo `flowdppo.py`扩散模型版本混淆。
- 当前官方代码公开且可静态读取；它依赖 UniRL Stage replay、packed TextSegment、loss_mask及TrainStack，**不是直接替换我们VeRL函数即可的独立脚本**。本次没有安装依赖或实际跑fixture，故不宣称端到端可运行/与现有Search训练兼容。官方recipe自己的comments说移植了早期 `MaxwellJryao/SPO-DPPO`参考，但本笔记没有实读那个repo，不拿这条comment冒充已核对其源码。

## 本次结论

DRPO改变更新几何/正则化，不改搜索采样树或正确性奖励。核心是**绝对概率位移的连续修正**；默认12.5应解释为很弱的 bounded reweighting，而不能展示成实际会频繁出现负权重的硬边界方案。迁移比较时应固定奖励、优势归约、loss归约、行为anchor及采样温度，再单独比较更新规则；上述静态分析不构成Search任务收益或新训练授权。
