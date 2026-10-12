# Single-stream SPO：同名边界与实际 tracker

2026-10-12 只读补充。本文对象为 [Single-stream Policy Optimization，arXiv:2509.13232v2](https://arxiv.org/html/2509.13232v2)，不是 Simple Policy Optimization / SimplePO，也不是 Segment Policy Optimization。它提供一个与当前组优势退化直接相关、但具有不同采样和历史状态预算的替代设计；不能混用名字或结果。

实际读取 [verl-project/verl-recipe 的 spo recipe](https://github.com/verl-project/verl-recipe/tree/242c73fcadf331e7214c4b48f67423f45239cea8/spo)，固定 commit `242c73fcadf331e7214c4b48f67423f45239cea8`（2026-09-22）。[REQUIRED_VERL.txt](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/REQUIRED_VERL.txt) 指定依赖 `verl` commit `d7944c01e63e9eb639c8357648b7958550591158`；本次没有下载整个 verl、环境、模型或训练数据，没有访问GPU节点或执行训练。代码可读成功，下述不是仅据摘要猜测。

## 核心流程

1. 离线先为每题估初始成功率，建立按 prompt 字符串索引的 Beta参数表 (\alpha_q,\beta_q\)。默认 `offline_N=8`；实现额外加0.5平滑，避免纯0/1初始值。由 `rho.clip_lower=.875` 得初始有效计数8：[初始化356–376行](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/spo_ray_trainer.py#L356-L376)、[配置13–20行](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/config/spo_trainer.yaml#L13-L20)。
2. 每轮用历史 (p_q=\alpha_q/(\alpha_q+\beta_q)\) 设置采样权重 (\sqrt{p_q(1-p_q)}+.05\)，从 dataloader提供的候选池选择当前 prompt batch。代码是确定权重的加权随机抽样，**不是从Beta后验抽成功率的Thompson Sampling**，尽管README使用该称呼：[380–415行](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/spo_ray_trainer.py#L380-L415)。
3. 当前每题生成一条，取原scalar outcome reward；用**尚未纳入本次reward的历史 baseline**算 (a_i=r_i-p_{q(i)}^{old}\)。然后跨当前完整 prompt batch 计算

\[
\widetilde a_i=\frac{a_i-\bar a_{\mathcal B}}{s_{\mathcal B}+10^{-8}},
\]

源码是 `torch.std()` 的sample std；将同一个 (\widetilde a_i\) 贴到该响应有效token。此路径已给出 advantages，后续普通GRPO advantage函数不会覆盖它。不是 per-question 当前group中心化，也**不是跨历史batch累计均值/方差归一化**：[529–561行](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/spo_ray_trainer.py#L529-L561)、[630–639行](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/spo_ray_trainer.py#L630-L639)。
4. 正常更新actor，随后估策略漂移/遗忘率并更新历史表：

\[
\alpha_q' = \rho_q\alpha_q+r_q,\qquad
\beta_q' = \rho_q\beta_q+(1-r_q),\qquad
p_q'=p_q+\frac{r_q-p_q}{\rho_q(\alpha_q+\beta_q)+1}.
\]

这是按prompt历史的自适应EMA，不是每token的critic神经网络。上述等价式由代码更新直接代数展开；历史baseline并不等于当前policy的精确成功概率。[actor先、tracker后，648–672行](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/spo_ray_trainer.py#L648-L672)。

## KL-adaptive 的实际近似

论文用 policy之间KL控制遗忘；实际recipe取每题上次存下的已生成轨迹，在当前actor上再算logprob，使用

\[
D_q=\operatorname{maskedMean}_t|\log p_{old,t}-\log p_{new,t}|,
\qquad\rho_q=clip(2^{-D_q/.06},.875,.96).
\]

**这是样本token绝对logprob差均值，不是全词表KL，也不是有符号logratio的精确期望。** `_get_spo_rho` 在actor更新后另做历史轨迹和当前轨迹的HF logprob前向，再存最新轨迹/probability。因此不能将此方法说成与我们“复用行为NLL、无新增forward”的置信度权重方案相同。它group-free，却仍有历史轨迹缓存和额外评分成本。[240–303行](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/spo_ray_trainer.py#L240-L303)。

## 对组退化意味着什么

GRPO的全对/全错组会因本组reward常数而得到0；SPO先减历史baseline，所以本轮哪怕只抽到一个错误答案，若过去 (p_q>0\)，原 (a_i<0\)，仍可能有更新信号。本轮全对也可能保留正优势。因此它处理的是**从同批组baseline改为历史perprompt baseline**，不是用reward shaping给本组全错轨迹重新排序。[论文§§4.1–4.4](https://arxiv.org/html/2509.13232v2#S4)。

边界：若整个batch的 (r_i-p_i\) 恰好相同，batch标准化仍得到全0；若tracker陈旧/失准，信号仍可能不合适。跨题中心化后优势符号也可能变化。历史action-independent baseline支持raw REINFORCE的无偏性，**不自动证明同批中心化、随机STD、PPO clip和优先采样后的更新仍是原均匀数据目标的无偏梯度**。普通 (E[R|q]\) 也不是任意score-gradient范数下的严格最小方差baseline；这里应称低方差估计设计，不称普适最优保证。

## 发布配置和复现边界

`train.sh` 的SPO分支是1024 prompts×1、global actor mini128；GRPO分支是128 prompts×默认8，同为1024响应，但prompt覆盖、历史baseline、优先采样、初始化成本均不同。SPO关闭KL-in-reward和actor KL loss，clip下/上为.2/.28，actor LR1e-6，最大工具turn8。不能把它和当前512×5的Search-G1说成只替换一个优势公式：[train.sh 37–67行](https://github.com/verl-project/verl-recipe/blob/242c73fcadf331e7214c4b48f67423f45239cea8/spo/train.sh#L37-L67)。

源码tracker字典为 `fit()` 内局部变量，跨迭代持续更新；本次所读recipe未发现把该字典/历史轨迹缓存交给checkpoint恢复的路径。不能把“训练期间persistent”扩称“原入口resume会完整恢复tracker”。重复prompt字符串、离线值键的完整覆盖及reward范围[0,1]也需要明确；把当前G1的-0.1/0.05/连续shaping直接投入Beta更新可能使参数解释不成立，应另走经过定义的EMA扩展。

本次只能确认官方recipe实际存在。没有SPO子recipe独立采用量统计，**不使用整个 verl 或 verl-recipe 仓库星数证明 Single-stream SPO 社区热度**。论文和当前主报告所选SimplePO是不同版本谱系，收益数字不得互借。

必要源码7文件逐字节等于固定commit，身份与SHA见 `single-stream-spo-source-audit.json`；未运行其测试或启动新实验。本说明是供主报告补充边界，不改变正在进行的任何训练。
