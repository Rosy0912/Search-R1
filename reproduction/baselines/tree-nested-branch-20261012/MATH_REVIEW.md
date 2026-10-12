# Nested 五叶采样数学审查（只读独立审查）

审查日期：2026-10-12。仅读取本地源码，未访问远端、修改运行代码或启动训练。主实验 3360 必须完成后才可启动新实验。

## 结论与边界

可实施目标是**奖励条件选择的五叶 Tree-GRPO surrogate**：改变采样器，保留既有优势、每叶 token 权重、全局行归约和已配置 PPO/Dual-Clip 形式。它不是原始未筛选 rollout 分布下的无偏 policy gradient，也不能保证每题混奖、树间差异增大或训练效果改善。真实 nested genealogy 可完整记录；原损失不使用内部结点信用，因此不能称为新建的层次信用分配算法。

## 当前实现核对

- `rollout.py::_tail_branches` 目前配额为 `(3,2)`，有限三条 continuation 均保留，后续仍用原 root snapshots，不是本目标 `(2,3)` 和 B→B1→B2。
- `complete` 保存每次模型动作之前的状态，包含初始状态；这些 snapshots 未自动筛选“有效动作”。`valid_actions` 是累计数；记录中的逐动作 `valid_action` 目前只在 trajectory 输出开关开启时存在。因此新实现必须明确逐动作有效性并用它过滤候选，不能把所有 snapshots 宣称为有效动作。
- `State.branch` 保留 `tree` 和 `node`、更新 trajectory；新分叉事件需另有唯一事件 ID，不能只靠 clone 时的 node 判断父子。
- `advantages.py::tree_grpo` 的所谓 inter 是**全部五叶的题内标准化奖励**，不是先算两个 root 均值再相减；intra 是 `(题, root tree)` 内标准化奖励。
- `objectives.py` 的 Tree 权重为每叶有效 token 的 `1/T`，总归约为全局行数，不是全局 token 数，也不是两棵树各占一半。

## 精确定义建议

每题保留原始 A、B。A1 在 A 的有效动作**起点**集合上均匀随机选一个位置，采样一次 continuation 并保留，不按奖励筛选。无有效动作时单次初始 fresh fallback；不得通过复制 A 假装新增一次采样。

对 parent P（首次 P=B，第二次 P=B1），建立 P 实际完整路径的有效动作起点列表，由尾到头，每个位置至多一次新 continuation。接受第一条 `R(child) != R(P)` 的结果。失败候选只计采样成本，不进入训练、优势、leaf 权重或训练分母。若全部候选同奖，则从初始状态单独 fresh 采样一次，无论奖励如何都保留为 child，不再次搜索。B2 比较基准是 B1 的终奖，不是 B 的终奖。

初始位置去重建议：常规扫描排除初始状态，把它留给最后单次 fallback；因此每阶段 `(parent_trajectory_id, action_start_id)` 最多一次。如果常规扫描包含第一个有效动作且其起点就是初始状态，那么 fallback 是同一状态的第二次采样，必须公开写作允许重复的独立 fallback，不能同时声称“每动作最多一次”。两种规则不可混用，推荐前者。B 与 B1 两个阶段允许各自尝试初始状态，它们是不同 parent 阶段，不是同阶段重复。

五叶按 `A,A1,B,B1,B2` 保留，树 ID 为 `A,A,B,B,B`。B1 的路径需由继承 B 前缀加新生成 suffix 组成；B2 的起点必须取自这条 B1 实际路径，不能复用原 B 的完整 snapshots。B2 若在 B1 已继承的较早前缀处分叉，事件谱系仍是 B1→B2，但 B1 新 suffix 并不被 B2 继承；日志应区分“事件父亲”和“实际共享前缀”，不得画成所有 B1 token 都是 B2 的祖先。

有效动作列表、初始状态排除、奖励判等规则、剩余 turn/最终回答预算、无有效动作 fallback 必须成为协议配置。当前终奖若为精确二元 EM，可精确判等；非二元奖励须预定义容差，禁止结果出来后改阈值。

## 保留的优势和目标

设题 q 五叶 i 的终奖为 r_i，根组 g∈{A,B}、n_A=2、n_B=3。定义

`z_G(i)=(r_i-mean_G(r))/(sample_std_G(r)+1e-6)`，单叶组定义为 0。

训练优势仍为 `a_i=z_q(i)+z_(q,g(i))(i)`，所有有效模型 token 使用同一标量，检索 observation 不受损失。使用停止梯度的 r、优势、采样 μ 和选择结果。

完整批次 N=5Q，`w_it=m_it/max(T_i,1)`，`T_i=sum_t m_it`，保留目标

`L(θ)=1/N Σ_i Σ_t w_it ℓ_PPO/DualClip(ρ_it,a_i)`。

有效非空叶总质量为 1；空叶贡献 0 但继续占行分母（并显式记数）。A 树总质量为 2/5，B 树为 3/5。不可为平衡根组改成每树 1/2，也不可因试了较多失败候选改变分母。按 DP/microbatch 切分必须恢复同一全局目标，不能局部重算均值或标准差。

### reward 差异与“树间”差异

B1 与 B 异奖保证相应已接受 B 子树存在奖差，不能保证两个 root 均奖不同。二元例 `A=[0,1]`、`B=[0,1,0]` 已混奖但 root 均值只差 1/6；更一般实值例 `A=[0,1]`、`B=[0,1,0.5]` 均值完全相等。只要五叶混奖，代码 inter 项就可能非零，即使 root 均值一样。因此必须同时报告全五叶方差、root 内方差和 root 均值差，不把五叶 inter 非零解释为纯树间信用提升。

## Nested 共享前缀信用

重复前缀沿每个叶的完整 response 进入损失，这是原每叶目标的约定。某个共享 token e 在不裁剪且同上下文/概率时，局部梯度系数与 `Σ_(i containing e) a_i/T_i` 成比例。intra 虽满足树内 `Σ_i z_g(i)=0`，不同叶长度导致 `Σ_i z_g(i)/T_i` 通常不为 0；只覆盖 B1/B2 的子集也不自动抵消。共享 token 可因所选后缀奖差得到非零信用，不能误称已实现公共前缀抵消或对该 token 的无偏动作优势。

审计应记录共享发射 token 的叶重数及聚合 `a_i/T_i`，不改变损失。若要采用结点去重、前缀屏蔽、edge-return 或仅 suffix 训练，它们都改变原目标，需要另一明确实验，不能静默放进当前修复。

## 奖励筛选与 μ 的数学限制

令未筛选在位置 s 生成的 continuation 分布为 p_μ(y|s)，接受事件为 E_s={R(y)≠R(P)}。接受后分布为 `p_μ(y|s,E_s)=p_μ(y|s)1[E_s]/Z_s`；第 j 个首次成功候选还乘之前各候选失败概率，fallback 乘所有失败概率。初始 root、随机位置、继承前缀和 B1 依赖也共同影响完整输出五叶分布。

逐 token 的 `ρ_it=π_θ(token|h)/μ(token|h)` 只使用实际采样 emission 概率，未包含成功事件的归一化 Z、首次成功/停止概率、位置选择和 fallback 混合概率。它不能把此 selected dataset 自动还原为原未筛选轨迹分布；有限支持、相关五叶、数据相关 group baseline 与 PPO clipping 也不能据此宣称无偏。标准化优势本来就需要作为 surrogate 解释。

建议直接使用现有真实 μ 分母及原裁剪规则，但把目标明确命名为 `reward_selected_nested_tree_grpo_surrogate_v1`，解释为**固定奖励筛选数据上的 token PPO surrogate**，μ 作用是实际发射概率 provenance 和局部比率分母。固定一次 rollout 内全部选择、优势、μ；不要用当前 θ 重新筛选旧批次，不添加未经推导的 selection importance weight。若现有配置使用 recomputed β 而非 μ，应如实记录该实际分母，不能仅改名称声称已校正。

## 有限搜索不保证混奖

若全部候选及 fresh fallback 都与 parent 同奖，必须仍保留 fallback，五叶完整且允许同奖。即使不同奖励在策略支持内，每位置只试一次仍有全失败正概率；若策略/环境只产生一个奖励，则无任何保证。二元奖励且 B1 异奖、B2 也异于 B1 时有 `R(B2)=R(B)`，B 树组成只能是 0/1/0 或 1/0/1；它不创造第三种质量级别。禁止为追求混奖无限重试、只丢弃同奖题或谎报成功。

## 最小必要日志与测试

每阶段记录：题/根/parent/child trajectory ID、分叉事件 ID、action 起点和倒退次序、候选有效性、原始与 child 奖励、接受/拒绝/fallback 原因、所有 attempt 次数、初始状态试过次数、继承 token/observation/raw ledger hash、remaining budget。B2 另记起点属于 B1 继承部分还是新 suffix。保存小型数值审计，不保存原始全量 rollout 到 Mac。

分开计数：attempt/accepted/discarded/fallback、新生成 token、prefill、检索、wall time；失败候选的所有成本记入 total，仅已保留五叶计入训练样本。报告根配额、混奖比例、root 均奖差、树内及树间 leaf-mass 方差、原优势方差分解误差、共享前缀重数/信用、empty loss row、μ provenance 错误与分母类型。监控 KL 应标记固定所选叶的加权监控量，不能称全策略 KL 无偏估计。

最少测试（无需启动模型/训练）：

1. 固定假 reward 序列检验 tail→head 每位置一次、首次异奖即停止、失败不训练、cost 全记；全同奖也返回完整五叶。
2. 初始起点只在 fallback 一次；无有效动作和零新 suffix 正确处理。B2 使用 B1 实际 snapshots，比较 B1 奖励，哈希验证 prefix/observation/μ ledger。
3. 五叶 `tree_uid` 恰为 2/3，保留 root A/B，shuffle 后 genealogy 与 μ 同步；证明 B2 不回到已拒绝的 B1 候选或原 B 完整路径。
4. 用实值等 root 均奖例验证 inter 仍非零；标准差 `correction=1`、epsilon、零方差均与原 `tree_grpo` 一致。
5. 手算不同 token 长度的共享前缀系数与总 loss；统一批次和分片归约相等；失败候选不改变 N。空 mask 仍占原行分母并触发日志。
6. μ token 比率相同但 accept probability 不同的合成案例，确认协议/报告明确标记 selected surrogate、绝不输出“selection corrected/unbiased”的声明。

完成这些本地验证、提交推送新实现与配置后，还需确认主实验 3360 最终完成才能启动新实验；此文不授权提前启动或改动该实验。
