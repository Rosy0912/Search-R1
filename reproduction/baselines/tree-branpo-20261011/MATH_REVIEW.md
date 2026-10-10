# Tree-GRPO + BranPO-inspired bounded tail sampling：独立数学审查

审查日期：2026-10-11。范围：本地静态源码及原文方法；未启动训练、未访问训练节点。结论：**可实现的最小版本是采样机制改动，保留现有 Tree 优势与 PPO；不能声称完整 BranPO 复现，不能保证五叶混奖，不能声称无偏策略梯度。**

## 1. 原文与本实现边界

唯一论文来源：[Training Multi-Turn Search Agent via Contrastive Dynamic Branch Sampling，v1，§4](https://arxiv.org/html/2602.03719v1#S4)。原文 §4.2 从尾部截断、采样后检查终局奖励，保留对比结果；§4.3 的式 (10–12) 将同前缀分支均奖用于 prefix，并分别标准化 prefix 和 suffix；§4.4 对冗余动作另加 mask。这里仅借用末段优先与向前回退的采样思路。原论文奖励筛选、独立 prefix/suffix 优势与冗余 mask 均不应悄悄加入本版本。

以下公式、反例、预算与统计为本次独立审查推导，不是论文复现声明。既有源码 `advantages.tree_grpo`、`objectives.attach_policy_weights` 与 `weighted_policy_loss` 是保留对象。

## 2. 既有估计器与论文估计器的区别

对题 q，根 r∈{0,1}，叶 i∈{1,…,m_r}，m=(3,2)。每题五叶，512 题恰为 2560 行；两根分别有 3/5、2/5 的叶质量，不能擅自改成根各 1/2。令 R_qri 为终局 token rewards 的和，Z_G(x)=(x−mean_G)/(sample_std_G+10⁻⁶)；单元素组返回零，零方差组返回零。现有估计器为

    A_qri = Z_全部五叶(R_qri) + Z_同根m_r叶(R_qri).

对该叶的每个有效模型 token，A 相同；观察 token 的 mask 为零。每叶长度 T_qri 是有效模型 token 数，权重 w_qrit=mask_qrit/max(T_qri,1)。全局归约是五叶等质量、每叶 token 平均，不是整个 batch 的 token 平均。

论文式 (10–12) 则先得到每根均奖 b_qr=mean_i R_qri，再以跨根标准化值赋给共同 prefix；suffix 以题内所有分支奖励标准化。这里不把论文 suffix 误解成仅同根标准化。若替换成这个分段估计器，就删去了当前 suffix 的 intra 项并改了 prefix 统计，已经是另一实验版本。仅将两种优势相加也会新增目标，不能当作无影响集成。

一个明确反例：两根奖励分别 [0,0,1] 与 [0,1]。当前根0 intra 不为零、根1 intra 不为零；论文 prefix 只比较 1/3 与 1/2，两根优势分别为负与正；当前同一共享 prefix 的叶副本还可能携带不同符号的优势。因此两者不相等。

## 3. 最小机制：三次新增完整 continuation，全部保留

1. 正常生成两个 root 完整轨迹，缓存每次模型动作之前的真实状态 snapshot，包括已有 token IDs、行为 logp、工具观察、turn/search 计数及剩余预算。奖励只在轨迹完成后由既有 reward reader 计算。
2. 为每根将可合法重放的 snapshot 按动作序号从晚到早排列。首个新增 continuation 从末次模型动作之前分叉。分叉不是末尾字符串截断，不把已生成答案留在 prefix，不倒退已消费的历史预算。
3. 根0有两个新增叶槽，根1只有一个。根0第一次新增叶与原轨迹奖励相同（或按明确的 tolerance 未形成奖励差异）时，第二槽回退至更早的合法 snapshot；已有对比时第二槽仍从末段采样。根1的一次槽用尽即结束，不能再额外生成回退候选。没有更早 snapshot 时仍从最晚合法点采样。若采用固定晚→早顺序，不用奖励控制分叉位置，可进一步避免奖励驱动分配；须记录实际选用规则。
4. 每个生成成功的完整 continuation 不论奖励都成为训练叶；不得丢弃同奖叶、不得只挑异奖叶、不得先生成十个再选五个。保留两个原叶及三个新叶，共五叶。奖励可控制根0剩余槽位置，但不能向模型 prompt 或工具请求注入标答、正确性标签或“改成错误答案”等控制指令。
5. 每条 continuation 仍受 max_turns=4、2048/4096/500/500 原限制、force_final_answer=False 约束。没有答案的合法终止照旧评分，不能为补对比强制答案。
6. 将五叶交给原优势、原 policy_weights、原 PPO clipping/dual clip/ratio 防护、原 KL/entropy 与分布式归约；只增加采样和审计字段。错误退出不属于“奖励相同”，必须显式失败；不能用复制既有轨迹伪装新增采样。

这仅给出**至多三次新增 continuation** 的严格样本预算；各 continuation 长度不同，因此不承诺相同 token/FLOP/wall-clock 成本。根0有限回退最多一次，根1无剩余回退槽。更深递归与多候选筛选需要新增预算，不能偷偷改变 2560 行契约。

## 4. 无偏性、有限预算与共享 prefix 的限制

### 五叶不能保证奖励不同

若给定所有可分叉状态 s 都有 P(R=0|s)=1，则任何合法五叶必全零；全一同理。即使五叶为独立 Bernoulli(p)，混奖概率也仅为 1−p⁵−(1−p)⁵，p=1/2 时仍有 1/16 全同奖概率。真实叶共享 prefix、并非独立，不能拿该数值当实际保证。需要报告实际混奖率；无法保证“每题产生有效梯度”。

### 奖励挑选有 selection bias

假设固定 prefix 下成功概率 p，原叶奖励0，然后持续挑选成功 continuation 进入训练，则被保留的新增叶条件成功概率为1，而非p。例如 p=.1，强制保留一正一负的均奖为.5，并不估计原策略条件回报 .1。PPO 的行为 token ratio 校正不了由筛选事件改变的样本分布；需要真实 inclusion probability 才可能讨论选择校正，当前版本不具备它。

最小版本保留所有预算内候选，避免这个奖励丢弃偏差。但奖励驱动回退改变状态分布，root 枝条相关，组标准化依赖同组回报，PPO 裁剪本身也改变估计。因此仍应描述为“既有 Tree clipped surrogate 下的适应性采样”，不能描述为无偏 on-policy PG。即使不用奖励分配，也不自动恢复无偏性。

### prefix 不添加第二份信用；保留原叶质量

既有五叶目标（省略符号无关正则）是

    L = (1/5) Σ_叶 i Σ_有效token t [ell_PPO(A_i,ρ_it)/T_i].

共享模型 token e 在后代叶 D(e) 出现，原目标的该 token 总项恰为

    L_e = (1/5) Σ_i∈D(e) ell_PPO(A_i,ρ_e)/T_i.

共享 prefix 副本是同一个采样事件在不同叶目标中的既有质量，不是新采样事件；审计必须区分 unique sampled tokens 与 leaf token occurrences。**不另加 prefix loss、不再加一份 mean reward 信用**即可避免新增重复计数。物理保留各叶副本实现原式是正确的。若计算优化要只处理唯一 prefix token，必须保留以上精确聚合项和 Σ_i∈D(e)1/(5T_i) 质量；不能简单除以后代数或改根质量。

PPO 是关于 A 的分段函数：同一 ρ 下正、负 A 可能走不同 clip 分支。因此平均 A 后只调用一次 PPO 不一定等于先算各叶 PPO 再求和；例如 ρ>1+ε、一个正 A 一个负 A 会出现该反例。唯一 token 合并应分别聚合正负优势的带权质量，或保持逐叶 PPO，再做后处理聚合。既有逐叶表示是最小且安全的实现。

## 5. 每 iteration 的可审计统计

必须从完整 2560 行、DP sharding 之前计算统计，避免 rank-local 分组和重复 prefix token 改变样本权重。

对每题每根，定义 intra B_qri=Z_同根(R_qri)，**未加 inter**，以及

    V_intra(q,r) = (1/m_r) Σ_i (B_qri−mean_i B_qri)².

记录根均值 mean_(q,r) V_intra，按 m_r/5 加权的题均值 mean_q Σ_r(m_r/5)V_intra，零方差根比例，以及样本方差版本（分母m_r−1）的明确名称。不能把 inter+intra 的方差称为 intra 方差。

样本标准化下非退化组的 population V_intra 约等于(m_r−1)/m_r，即3叶约2/3、2叶约1/2；epsilon 会轻微压低它。因此该日志主要显示有多少组退化，不能独自证明信用精度提高或梯度降噪。

同时记录原始奖励树内 population variance、各根 mixed_reward_rate（max R−min R>tolerance）、题级五叶 mixed_reward_rate、binary_correct_mixed_rate（若定义正确性阈值须命名区分）、新增 continuation 同奖/异奖数、回退深度分布、可回退状态数、实际新增叶/新采样 token/工具调用数及空 loss 行数。连续 F1 奖励不同并不等价于一正一负，阈值统计不能替换训练 reward。

没有跨独立 rollout 重复实验，intra advantage 的单次离散程度不等于策略梯度随机方差。共享 prefix 不同后代相关：若 M 个回报方差σ²、等相关ρ，则均奖方差为σ²[1+(M−1)ρ]/M；ρ=1 时没有1/M降方差。原估计器保留，不能引用论文 prefix averaging 的结论声称本版本已经获得同样收益。

## 6. 必要不变量与轻量反例检查

- 每题两根、根配额[3,2]、五个独立 trajectory ID、512题/2560训练行；所有生成预算内叶均保留，失败不得当成同奖结果。
- 同题只在对应 root 的真实 snapshot 分叉；共享 token IDs/logp/工具观察一致；分叉 suffix 用真实新采样行为 logp，观察永不入 loss。
- max_turns、输入/上下文/输出/观察限制及不强制答案均保持；没有新增奖励、DAPO/IGPO损失或自动 RSM。
- policy_weights 每个非空叶和为1、空叶为0，全局分母不变；根质量仍3/5与2/5。共享 prefix 不新增 loss 项。
- 全零/全一 reward：inter/intra全零、variance全零、finite，不“修造”对比结果。
- 根0全零、根1全一：intra全零但inter非零，日志不得误报树内有效优势。
- 奖励[0,0,1]、[0,1]：intra取样标准差标准化；核对population variance约2/3与1/2、题加权约3/5，无错用token权重。
- 对共享 prefix 用相反符号优势与超 clip ratio：逐叶 PPO 保留，若合并须与原目标数值相等。仅通过 prefix 均值测试不充分。
- 行重排/DP切分前后adv、tree_uid、policy_weights及统计不变；fallback无额外槽时停止，零合法snapshot显式处理而非静默复制叶。

上述检查足以核验这次机制边界，不需要启动训练证明数学恒等式。实际“提高混奖率”是待验证经验假设：后续训练必须与同预算随机分叉基线比较，并逐iteration记录以上统计；本审查不预先声称收益。
