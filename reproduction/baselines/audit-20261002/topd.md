# TOP-D 独立单算法审查

审查日期：2026-10-02。结论：**CONDITIONAL**。核心 proximal reward、未来有效 actor token 均值、逐题 pooled whitening、固定 old snapshot 的 E 轮 PPO 和全 DP token mean 接入成立；不能视为作者实验复现完成，也不能直接放行正式训练。正式训练首先需要处置共享的生成/评分前缀不一致、落实兼容且训练过的真实 teacher，并完成真实 3B / vLLM / remove-padding / offload / 搜索整链资源与数值验收。

## 固定版本、来源和范围

- 实现 HEAD：`7b93fbd01742715dc14290b0a6e2d3b42e6c19f7`，本轮 `git rev-parse HEAD` 实际确认。
- Search-R1 基础版本：`b251ba6b503217a59cd3395c1326d7b66b602989`；实际读取其 `reproduction/h20/train_paper_grpo.sh`，并比对 actor diff。
- Primary：[Trust Region Policy Distillation，arXiv:2607.04751v1](https://arxiv.org/html/2607.04751v1)，方法 §3.1–3.3、Eq.(5)/(7)/(8)、Algorithm 1、理论 §4、Appendix B。此处 TOP-D 不等于 TrOPD、ERPD 或其他名称相似方法。
- 本次论文 HTML 无作者代码链接；检索精确论文题名、作者和 GitHub 未确认可归属于作者的源码仓库。检索出现论文笔记和其他 PD 仓库，均未拿来充当官方代码。**作者代码 SHA：未验证，不能填写，也不能声称源码级一致。** 下文作者证据仅限可核原论文。
- 独立阅读 `advantages.py`、`distillation.py`、`controller.py`、`dp_actor.py`、`fsdp_workers.py`、Ulysses sharding/gather、`main_ppo.py`、`ray_trainer.py`、两份配置及 `run.sh`。仅写本报告，没有修改实现、远端运行、GPU 使用、进程终止或 git commit/push。
- 主学生固定 Qwen2.5-3B，固定 Search-R1 paper-grpo 数据；辅助教师由用户授权，但模型身份/训练来源尚需输入与核实。遵循 paper-glance 的串行算法展开和不补造来源要求；用户已明确审查输出，无须再展示技能菜单。

## 一轮串行执行与公式映射

1. `run.sh:21–25` 指定 TOP-D、独立 teacher，关闭 privileged answers、actor KL、entropy 和 reward KL，选择 E=2。`main.py:9` 先调用 `validate_config`；`controller.py:36–58` 强制 FSDP 搜索、action masking、rollout.n=1、teacher 路径、零 KL/entropy 系数。默认 YAML 是共同基线，直接只改 method 而不经 launcher 不构成有效纯蒸馏配方。
2. `ray_trainer.py:733,765–777` 每题重复 `n_agent=5`，执行工具搜索 rollout，再于**尚未更新的学生**计算 old logprob。`controller.py:125` 在 driver 加载一个 FP32 CPU FrozenTeacher，`distillation.py:113–123` 核 vocab/bos/eos、固定 pad、eval 且 requires_grad=False。参考模型角色在 `main_ppo.py:184–187` 移除。
3. `controller.py:168–171,194–203` 取有效 mask = response attention × info mask，以 uid 分组。每条 trajectory 使用同一个 prompt + response（含 observation）的真实教师 forward，默认 targets=None；`teacher_inputs:36–55` 去 padding、压紧 response、建立每个 target 的前一个 hidden-state 位置，`FrozenTeacher.log_probs:129–142` 在全词表 softmax 后 gather 实际 token。教师与 old/current 都除以 rollout temperature。教师保留 observation 作为后续动作条件，observation 不获得 TOP-D reward/loss。
4. 令 d=log π_teacher−log π_old。`advantages.py:69–73` 算 r=logaddexp(d+log α, log(1−α))，即 log[α exp(d)+1−α]。α 必须处于 (0,1)。在 mask=0 的位置先令 d=0 再清零 reward，避免无效教师 infinity 污染该估计器；这是概率空间 proximal mixture，不是简单 αd。
5. `advantages.py:74–76` 用反向 cumsum 算 exclusive future reward sum 和 future **有效动作 token 数**，R_t=r_t+sum_{j>t,mask_j=1}r_j / max(1, future_count)。最后有效 token 的 future=0；observation gap 不进分子分母。这是论文单串 Eq.(8) 到搜索动作 token 的明确适配。
6. `advantages.py:78–86` 将**同题所有响应的所有有效 tokens 合并**，μ=mean(R)，σ=std(R, correction=1)，A=(R−μ)/(σ+1e−6)。空组或一个有效 token 返回零 advantage。不是逐响应标准化，也不是先算每条 response 平均。使用 sample std，论文只写 std，未明确 ddof；没有作者源码证据证明 population 或 sample 为唯一选项。
7. reward_fn 的 EM 分数仍为报告/验证服务；`ray_trainer.py:829–847` 不叠加 reference KL，`controller.py:203–204` 用教师 dense reward 替换 token_level_rewards。adv/returns/old_log_probs 整批一次生成后传入 actor，E 轮不重算教师或 old。
8. `dp_actor.py:225–239,260–290` 对每个 minibatch 求全局有效 token 分母，按 microbatch 累加 PPO clipped numerator，再一次 optimizer.step；不另除 gradient_accumulation。目标为负的 min(ratio A, clip(ratio) A)，clip 默认 0.2。E 轮循环复用 old/adv TensorDict；`actor/optimizer_steps`=minibatches×E。不等价于每轮更新 old policy。
9. worker 在 `fsdp_workers.py:361–400` 先载入 CUDA 参数/optimizer、SP allgather 后更新、再 offload；原 FSDP clipping、lr scheduler、vLLM 同步入口继续使用。权重与算法配置有保存流程，但共享报告已注明不支持完整 optimizer/RNG/dataloader resume，本审查不将 checkpoint 当作完整恢复训练保证。

## 四维审查

| 维度 | 独立结论 | 边界 |
|---|---|---|
| 代码 | 核心数学路径与真实 actor 接入通过；teacher 冻结且不加入 actor optimizer | masked token 指数溢出反例仍可使 PPO loss/gradient NaN，见 T03 |
| 数学 | Eq.(5) 稳定算子、exclusive future mean、逐题 pooled whitening、固定 old 的 PPO clip 正确 | ddof 无官方代码证据；跨题目标权重偏离 Eq.(7) 的逐题期望，见 T02 |
| 逻辑 | 默认非特权；teacher/old/current 在压紧后的 full-trajectory prefix 对齐；纯 teacher loss 无 EM/KL/entropy 混入 | full-trajectory scoring prefix 不一定等于真正 rollout prefix，见 T01 |
| infra | SP=1 的 DP/FSDP token 分母、微批累积、有限输入空 mask 安全；沿用 worker offload 入口 | SP>1 只有代数与源码核验；生产 remove-padding/offload/3B/真实搜索/真实教师吞吐未验收 |

### DP、SP 分母的独立核验

`global_token_denominator` (`distillation.py:94–101`) all_reduce 全 world 的 mask.sum，再除 world。SP=1 时每 rank 的 numerator / (N_global/W)，FSDP 平均梯度以后得到总 numerator / N_global；不同 microbatch 长度不会被等权平均。全零有效 mask 时分母 clamp_min(1)，有限 logp/entropy 的零 loss 与零 gradient 安全，区别于原 compute_policy_loss 空 mask 除零问题。

SP>1 不能只看全 world all_reduce 就判定重复计数 bug：`fsdp_ulysses.py:58–79` 在同 SP group 重复完整 mini 数据，SP=s、DP=d、W=sd，累计 token 数为 sN_unique，分母=sN_unique/W=N_unique/d。`dp_actor.py:109–128` gather 回完整 logp；`ulysses.py:226–229` backward 乘 s；随后 FSDP 的 W-rank 平均把每 SP rank 的序列分片贡献合并为按 d 个独立数据副本归一化的梯度。默认 SP=1；以上是静态代数证明，**并未执行真实 SP>1 FSDP 梯度对齐**。动态 microbatch 在前向/backward collective 次数上也需真实分布式验收，不将源码可选项等同于已测试能力。

## 分级问题与训练门槛

### T01 / P1 / 继承问题：实际生成 prefix 与 teacher/old/current 评分 prefix 可不同

位置：`rollout.py:87,96–100,149–152,177–187`、`ray_trainer.py:775–777`；诊断默认 `ppo_baselines.yaml:69` 的 report。`State` 字段顺序为 source、prompt、rolling（`rollout.py:32–35`），初始评分 prompt 只保留原输入尾部 max_start_length，而实际生成 rolling 保留完整原输入；之后 rolling 持续保留尾部，训练 response 保留头部。teacher 使用保留的评分 prompt+retained response，同 old/current 评分可以相互一致，却不一定对应实际采样状态。

触发反例：原输入 prompt token 长度 3000，max_start=2048，max_prompt=4096。首动作由完整 3000-token 输入生成，旧学生与教师却在原输入[-2048:] 的评分 prompt 上评分。另一个反例是后续上下文超过 max_prompt，rolling 丢头但评分 response 保头。`test_inherited_context_mismatch_can_fail_before_generation` 本轮实际通过，max_start=2 的题在 error 模式于调用生成前拒绝；report 模式记录 mismatch 但继续。

后果：r_t 中 π_old 不再是采样条件概率；严格 on-policy 及 PPO importance ratio 解释不成立。TOP-D 的 proximal bounded-reward 形式仍存在，但不能据此迁移论文全套理论结论。建议正式训练前让共享 GRPO 对照和 TOP-D 使用共同 context 对齐策略；短上下文预验收先开 `baselines.context_alignment=error`，保证 mismatch=0。error 是拒绝门禁，不是自动修复，长样本会中止。

### T02 / P2 / 有意适配未明确声明：跨题全 token mean 与论文逐题期望不同

位置：`dp_actor.py:233,261–262`、`distillation.py:94–101`；论文 Eq.(7) 外层是 x 的期望，每题内部除该题 G 条响应的 token 数。实现 denominator 合并一个 minibatch 内所有题的有效 tokens。

触发反例：两个题分别有 2、4 个有效 tokens，每个题内 clip numerator 的单 token 梯度分别为 1、3。逐题等权平均梯度=(1+3)/2=2；全 token mean=(2×1+4×3)/6=7/3。本轮独立算例输出 `2.0`、`2.3333333333333335`。

后果：长搜索题更重；这不是 microbatch 归一化 bug，也不是逐题 advantage pooling 错误，而是跨题 loss weighting 的 recipe 差异。建议在实验声明 global-token adaptation，或单独实现逐题权重且保持跨 DP microbatch 一致；不要将当前实现写成 Eq.(7) 全部逐字复刻。

### T03 / P2 / 条件性数值缺陷：mask 最后相乘不能隔离 exp 溢出

位置：`advantages.py:92–99`，`dp_actor.py:261–272`。TOP-D estimator 会隔离无效位置的 infinity，但 token_mean_ppo 先对所有 token 算 exp(delta)，再乘 mask；0×infinity/NaN 仍是 NaN。

触发反例：old=[[0,−100]]，current=[[0,0]]、A=[[1,0]]、mask=[[1,0]]、denominator=1，所有 logprob 非正。float32 exp(100) 溢出；本轮对这一组输入实际输出 loss=NaN，gradient=[[-1,NaN]]。第二个 token 完全不参与训练目标，仍污染其梯度。正常 finite 小 drift、包括全零 micro mask 的现有测试通过，故不声称该问题已在默认 lr=1e−6 的生产记录发生。

后果：在严重 policy drift 或无效 logits 异常时可污染全部 FSDP 参数；单纯 denominator clamp 并非任意输入空 mask 安全保证。建议在 exp 前用有效 mask 选择/清零 delta 与 advantage，valid tokens 的非有限值则明确拒绝并记录；actor entropy 也应在聚合前选择有效值，避免非有限 entropy ×0。不用此反例否定通常有限输入下已正确的 token mean。

### T04 / P1 / 未验证训练输入和资源门槛：真实 teacher 尚未验收

位置：`controller.py:125,197–201`、`distillation.py:121–142`、`run.sh:22`。driver 单个 FP32 CPU teacher，对每条 trajectory 串行做 eager-attention 全长 forward，chunk 仅切 lm_head 词表输出，**不切 transformer attention 的序列长度**。G=5、512题意味着约2560条 teacher forward/outer batch。

资源触发场景：3B教师仅 FP32 参数量级约12GB（十进制，估算不是实测）；CPU eager attention 按序列长度平方增长，长题+多轮 observation 保留会增加峰值。更大的辅助教师按其真实参数进一步增长。actor optimizer/参数 offload 同机也占 CPU RAM；24 CPU 已有 tiny 作业记录无法证明此配置吞吐或内存足够。

后果：可能 RAM OOM、长时间阻塞 driver 或严重 teacher bottleneck；尚无正式配置测量，**不宣称已经 OOM，也不是数学错误**。`validate_config` 只比较 canonical 路径，不验证模型确实训练过或更强；不同路径复制同权重依然初始信号为零。需要记录 teacher checkpoint/hash/tokenizer 身份、训练来源，测一次真实长样本 peak RSS 和 tokens/s，并估计一个 2560条批次时间；必要时单独部署 teacher scoring，但不能仅用人工 teacher_logp 放行。

没有本算法独立 P0 发现。T01 是正式训练的实际算法正确性门槛；T04 是资源和输入验收门槛。T02 是声明/复现精度问题；T03 是已证实的条件数值缺陷。

### T05 / P2 / 继承机制、频率未验证：raw sample tokens 与训练 label 未核等价

位置：原 `generation.py:55–75`，新 `rollout.py:120,138`。原 base 后处理先 decode(skip_special_tokens=True)，再在闭合 search/answer 标签后截断文本，然后重新 encode。重编码的 canonical action tokens 成为训练 label；`context_mismatches` 只比较 canonical rolling 与重建 prefix，没有比较 raw生成 token IDs 与重编码 label。special-token移除和非唯一分词均不能由普通 encode/decode roundtrip 自动证明等价。

触发边界：raw response含被skip的special token，或存在 decode后重编码改变分词的 token 序列。后果是即使context mismatch=0，也不能据此声称训练label完全等于采样动作序列；本轮未测真实Qwen频率，未声明实际训练发生。建议真实vLLM smoke记录raw retained IDs、canonical IDs与实际stop边界，先度量差异再确定与base共同修复策略。标签之后的未执行尾部截断本身可作为环境action定义，不能把所有文本截断一概判为bug。

## Search-R1 与作者证据一致性矩阵

| 项目 | Search-R1 固定 base | 当前 TOP-D | 可核作者证据 | 判定 |
|---|---|---|---|---|
| 学生与数据 | Qwen2.5-3B、paper-grpo train/test | 保持主学生和数据 | 原文 §5 为数学蒸馏设定 | 有意跨任务/模型适配，不称论文 benchmark 复现 |
| 工具观察 | 四轮、topk3、action state masking | 保持搜索环境，observation 只做条件 | 原文自回归单串 | actor-token future mean 属适配；搜索理论额外边界需声明 |
| teacher | 无 distillation teacher | 独立、冻结、默认同 prefix，无 gold hint | §3、Alg.1 固定 teacher | 方法一致；真实 teacher 身份与质量未验证 |
| proximal α | 无 | α=.1，logaddexp | Eq.(5)，Appendix B .1/.2 | 一致 |
| return | GRPO outcome advantage | 当前 r + exclusive future有效 token mean | Eq.(8) | 单串一致；动作 mask 适配 |
| whitening | 同题 outcome 级 | 所有同题有效 tokens pooled，sample std | §3.2 pooled token std，ddof 未述 | pooling 一致；ddof 不能声称官方核实 |
| actor objective | 原 PPO masked token mean/micro accumulation | DP 全 token mean，固定 old 的 PPO clip .2 | Eq.(7)、Alg.1 | clip/old 一致；跨题权重见 T02 |
| E/minibatch | base E=1 / mini256 | launcher E=2 / mini256 | Alg.1 支持 E；Appendix B off-policy epoch1 / mini32 | 有意不同 schedule，不称作者 recipe 一致 |
| G | n_agent5 | n_agent5 | Appendix B group8 | 为共同 Search-R1 budget 适配 |
| 训练温度/top-p | 1 / 1 | 保持1 / 1；teacher/old/current 同温度 | Appendix B 1 / 1 | 默认一致；用户改 top-p/top-k 需重审 sampling 分布 |
| KL/entropy | actor KL .001，base entropy 默认项 | 关闭 actor/reference/reward KL 和 entropy | Alg.1 pure proximal teacher clipped objective | 必要偏离共同基线，代码已落实 |
| remove-padding/offload | true，param/grad/optim CPU offload | 保持 worker 入口 | 无足够作者 infra 源码证据 | 静态接入成立；production路径未测 |

## 严格 on-policy 和理论适用界限

old/teacher/adv 的固定快照满足“先 rollout，再 E 轮有限 off-policy reuse”的实现结构；E=2 不叫严格每步重新 on-policy 采样。默认 top-p=1/top-k=-1、temperature=1 与论文训练抽样接近，但还要保证 T01 mismatch=0、vLLM与FSDP权重同步与采样 logprob 一致；本轮无真实 vLLM 验收。

论文 §4.1 的 bounded score-function 假设没有在神经网络训练中被验证；其 immediate-reward estimator 的界不能自动替换成 pooled standardized return PPO 梯度界。group std≈0、经验 whiten、token-only clipping、AdamW 和有限 minibatch 都不能据论文标题推出逐次单调改善。§4.3 的理论 surrogate 含状态分布/TV 误差控制，而代码只有经验 PPO clip，没有逐次计算并强制理论 lower-bound 上升；clip 并非完整 trust-region 约束。

搜索环境的 action→检索→observation 转移、预算终止、动作 mask 及 long-context 截断与论文纯 EOS absorbing autoregressive MDP 不同。因此理论可作设计动机，不能承诺本适配具备原证明的无条件 convergence/monotonic-improvement。另，论文“无额外开销”应理解为 proximal 算子相对已有 OPD teacher scoring 的成本，不能解释为新增真实 CPU teacher 相对 Search-R1 GRPO 免费。

## 实际执行、结果和未覆盖

本轮本地命令（未启动远端/GPU）及实际结果：

1. `git rev-parse HEAD` → 固定 SHA；`git diff b251... 7b93... -- verl/workers/actor/dp_actor.py` → 实际确认新增 E 循环、TOP-D 分母/损失分支。
2. `.venv-cpu/bin/python -m unittest discover -s tests/baselines -p test_algorithms.py -v` → **19 tests OK**（0.053s）；其中本算法两项覆盖 future actor mean/mask 和微批不等长+有限输入空 mask；共享 prefix mismatch 门禁反例同时通过。包含其他算法测试只代表一次测试模块运行，非本 agent 审查那些算法。
3. `.venv-cpu/bin/python -m pytest -q tests/baselines/test_models_and_driver.py -k real_teacher_logprob_alignment tests/baselines/test_distributed_normalization.py` → **1 passed、4 deselected**（1.93s）；`-k` 也排除了 distributed 项，故随后另跑下一条，不把被 deselect 当作已测。
4. `.venv-cpu/bin/python -m pytest -q tests/baselines/test_distributed_normalization.py` → **1 passed**（6.64s）；实际两个 CPU/Gloo DDP ranks，长度1/3不等、单 rank空 mask，核对全局梯度2.5/3。
5. 本地 Python 定向独立反例调用真实 `topd`/`token_mean_ppo`/`teacher_inputs`：masked teacher NaN/Inf 被隔离，adv finite；returns 第一条 `[1.4930888,0,2.3953886,1.5721736]`；pooled sample std≈0.9999990，population std≈0.9128700；有限输入全零 mask loss/clipfrac/kl=0、gradient零；masked delta100 → loss NaN、gradient `[-1,NaN]`；teacher输入 `[2,3]+[4,5,6]` 的 target prediction positions `[1,2,3]` 对齐。
6. 在线只将 arXiv primary 用为作者方法/recipe/理论来源；HTML GitHub 查找无作者实现链接，精确题名作者检索无已确认作者仓库。未以任何 third-party 笔记证明作者实现。
7. root复核指出初始 State 字段方向后，本 agent 重新读取 dataclass 和 `initial_states`，调用真实 `SearchRollout.initial_states`（测试 fixture 的18-token输入、max_start=2）实际输出 `scoring_prompt_len=2, generation_rolling_len=18`；T01已纠正为初始生成完整输入、评分尾片。原P1错位结论不变。

已有 A800两卡tiny FSDP证据只作补充读取：`gpu_smoke.py:102–105` teacher=old+人工 outcome/position signal；run2/result.json:35–42 记录 TOP-D 参数 L1改变量58.7566、optimizer_steps2、finite非零grad；其 config 在 `gpu_smoke.py:80–84` 为 remove_padding=False、SP=1、固定micro1，直接 actor入口，而非完整 worker offload/vLLM/retriever。CPU真实教师测试使用 hidden16、一层随机 tiny Qwen，对照直接 forward logprob，能核 target 对齐，不能证明训练过教师、更强教师或论文效果。

未覆盖：正式 Qwen2.5-3B、实际 trained teacher、完整 Ray/worker/vLLM权重同步、remove-padding=True、offload=True、SP>1、动态分布式 microbatch、检索服务/长轨迹、不等长多题跨DP paper-weighting等价性、实际吞吐/peak RAM、论文指标或训练收敛。共同资源 launcher 门禁和完整 resume 缺口参见 [shared-infra.md](shared-infra.md)，不以已通过 tiny evidence 覆盖这些条件。

最终放行意见：**核心代码接入 CONDITIONAL；完整正式训练未通过验收**。优先统一处理 T01，再固定真实 teacher 与 T04资源预算，明确 T02适配，修复/门禁 T03条件数值风险；完成一次真实3B完整搜索更新和保存、测量教师评分与worker路径后才可评估正式长跑。
