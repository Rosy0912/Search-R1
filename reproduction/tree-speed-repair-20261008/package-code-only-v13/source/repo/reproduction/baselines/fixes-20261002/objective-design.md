# 损失归一化设计审查（2026-10-02）

用户明确保留 Search-R1 的生成/评分上下文不一致；本设计只修复算法目标聚合，不改变搜索窗口、模型、数据、检索协议。审查者只改本文与另行授权的 Tree 批处理，未编辑 actor/controller/advantages。

## 严格目标和原源码适配须区分

GiGPO [v2 §4.3 Eq9](https://arxiv.org/html/2505.10978v2#S4.SS3) 的比率是整个文本 action 的概率比，外层是 episode/action 平均。作者固定源码 `/tmp/gigpo-core-fixed-audit.py` 和 step-flatten trainer 使用 token PPO；只加 action 长度权重不能使 token PPO 等于 Eq9。root 选择 strict action-ratio；将变长 Search-R1 horizon 明确扩展为每 episode 的有效文本动作数 Ti。

Tree-GRPO [v1 §3.2 Eq8](https://arxiv.org/html/2509.21240v1#S3.SS2) 的比率是 token 比率，先在每条叶轨迹平均，再组均值。作者固定缓存 `/tmp/rl-gigpo-tree-audit/Tree-GRPO/verl/trainer/ppo/core_algos.py:572–603` 是旧 token mean，有意采用论文归一而非照搬源码的这个差异。

TOP-D [v1 §3.2 Eq7 / Algorithm1](https://arxiv.org/html/2607.04751v1#S3.SS2) 每题在 G 条回答全部 token 中平均，再对题求期望。没有已核实官方源码；审查报告 `audit-20261002/topd.md:T02` 已记录其权重反例。

OPSD 保持作者实际 full-vocab/token-mean reduction。源缓存 `/tmp/topd_opsd_source_audit/OPSD/opsd_trainer.py:441–473`、固定 SHA ae7d2519e94920c4eb6206c0c26de46d9c50abae；不能无说明改成 sequence mean。

## 通用跨微批/DP接口

在 driver 的完整 rollout 中、DP shard 前构造独立 TensorDict 字段 `policy_loss_weights[B,L]`；DataProto reorder、select、split 均需携带。权重不含 advantage。有效 mask = attention 后缀 × info_mask；工具观察、padding、无有效 token 动作均权重零。

Tree：每行 Li = 有效动作 token 数，权重 mask/Li（Li=0 时整行0）。全 mini 分母是各 DP rank 行数之和 / DP size。每微批 numerator 直接 backward，不再除 gradient_accumulation；FSDP 平均之后等于 mini 的逐轨迹均值。空轨迹保留在行分母中、贡献0，避免依内容丢弃改变采样分布。全局没有有效监督时所有 rank 一起跳过 optimizer/scheduler。

TOP-D：完整 rollout 中 B 行、Q 个问题，Nq 为每题全部响应有效 token 总数。权重 mask * B/(Q*Nq)，Nq=0 时该题零权重、Q 分母仍包括它。mini 用全局行数/DPsize分母；完整 rollout 时精确等于逐题目标。若 mini 随机采行，mini 是完整 rollout 逐题目标的无偏估计。若 minibatch 被固定按长度/问题排列，它只能叫该加权目标的 minibatch 优化，不能声称每 mini 严格均匀题均值。现 `_balance_batch` 可能拆开题，绝不能按 rank 或 micro 的局部题 token 数重算分母。要求 exact 每 mini Eq7 则必须完整题组分 mini 或跨 rank exchange mini 的 uid/token 数，此处不建议额外改采样计划。

GiGPO strict：对每行每个存在有效 token 的 step，先求 d_it=sum_k(new_logp-old_logp)，rho_it=exp(d_it)，取该动作的 A_it（验证每 token 相等），在 action 层执行 clipped surrogate。行 loss = sum_t surrogate_it / Ti；Ti 只数非空有效动作。零-token action 仍参与现有 advantage 的时钟与 anchor baseline，但不能作为 policy 动作概率项（否则空串ratio=1的常数进入 loss/clipmetrics）；空轨迹行 loss=0、仍保留行分母。每行序列和必须先完整汇总再 exp/clip，不能分别 micro 切 token，现 micro 是切行可行。可将同一 surrogate 广播到 action token，权重 mask/(Ti*len_action)，然后使用同一全局 mini 行分母，但直接 action sum 更清晰。对有效 d 的非有限或 exp溢出要 fail-fast；任意 clamp logratio 会改变原文 surrogate。

GiGPO KL/entropy 是保留 Search-R1 的额外项，建议沿用独立 action-token均值并明确不是 Eq9 的完整整段KL；如选择与 policy 一样的行/动作权重，需文档记录。不能将 action-ratio surrogate 的 KL指示器偷换为原 token ratio指标。

## 必须验收的独立反例

1. Tree 1token A=+1、3token A=-1，ratio参数0：逐轨迹梯度0，而旧tokenmean=.5。
2. GiGPO 同一个2token action，共享 token logratio z：actionratio=exp(2z)，与两个token PPO平均 exp(z)不相等；长短action分别加扰动、跨clip阈值测试。
3. GiGPO 同一轨迹1token/3token两个动作，动作均值应等权；再加空动作、空轨迹，明确Ti与行分母规则。
4. TOP-D 两题长度2/4，token梯度值1/3：逐题mean=2，而globaltokenmean=7/3。完整题分散到不同 DP rank、两种 micro 切法仍需匹配参考梯度。
5. 两 rank 一空一非空仍 collective正常、全空所有 rank无参数/AdamW动量/调度器移动。用真实Gloo/DDP梯度验证，而不是只核权重tensor。

## 其余必修/验收项

P1：所有 PPO/entropy/KL masked inputs 在 exp/聚合前安全处理；全空 mask finite graph-connected0；有效非有限值明确报错。GiGPO/Tree/OPSD资源配置整除/localmicro非零及可见GPU数量早期验证。TOP-D真实teacher身份、权重与tokenizer哈希及长轨迹资源验收。

P2：Tree所有root的child continuation合批（保持每root node去重、树/来源/turnbudget/共享prefix）；OPSD全空mini跳过更新；OPSD globalmicro=DP ranks；response cap截断丢失动作后终局reward时钟风险需要 fail-fast 配置门禁。GiGPO mean-only/exact/per-episode统计属于已有论文允许适配，应记录而非强制作者search.sh的跨step统计。

生产 gate：Qwen3B+真实检索+vLLM/offload/remove-padding短验收、真实teacher CPU/RSS/GPU峰值；checkpoint目前不是精确resume，不能仅因HF权重保存成功宣称可恢复。用户已接受的context问题不作为阻止代码修复的 gate，但依然要在论文适配边界披露。

## BAPO 目标补核

直接重读 [v2 §2.2 Eq2](https://arxiv.org/html/2601.11037v2#S2.SS2) 后，其形式确实写整轨迹 probability ratio、组内轨迹均值。仅把tokenPPO改成sequence-tokenmean也不是该公式的逐字实现。但 Appendix A.1 明确实现基于 verl、工具反馈不进loss、所有非BAPO参数匹配standard GRPO；独立audit已将作者tokenGRPO视为合理基座适配。因此建议 reward/modulator 修复沿用作者standardGRPO路径，并明确公式/实现层的差异；如做字面strict整轨迹ratio，应作为明确模式另行设计，不能把sequence-tokenmean标为Eq2精确实现。此补核纠正此前仅凭audit认为未有序列目标区别的粗略判断。
