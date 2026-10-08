# GiGPO / Tree-GRPO / TOP-D 目标与共享数值安全独立复审

日期：2026-10-02。仅复审本轮工作树，不编辑算法；历史独立 audit 对象是 7b93fbd，本报告不是该旧提交的重复放行。用户已明确接受 Search-R1 生成/评分上下文不一致，故该继承限制不列为本轮必须修复项。

当前结论：**目标权重数学与 SP=1 跨 DP / micro 梯度通过；发现一个 actor 接入错误，root 已修正并通过独立非空 actor 探针；原始 logits 非有限的 autograd 反例已通过 entropy_from_logits 在 softmax 前整体 fail-fast 解决；这是一律拒绝非法 forward，而不是对非法分布继续零梯度训练。生产 3B/vLLM/remove-padding/offload gate 尚无本轮证据。**

## 论文与实现对应

- [GiGPO v2 §4.3 Eq9](https://arxiv.org/html/2505.10978v2#S4.SS3)：文本 action 的联合概率比，episode/action 外层均值。objectives.py 在单 action 内先加 token logratio，再 exp/clip；policy_weights=1/(有效动作数×该动作 token 数)，广播的同一 action surrogate 聚合后恰等于 action 均值。同 action advantage 必须一致且有合法 step_id；空 action 不产生 policy 项、空 trajectory 在行分母中保留。与作者 token-PPO 源码有意不同，不能称为源码逐字复制。mean-only / exact anchor / per-episode advantage 适配保留。
- [Tree-GRPO v1 §3.2 Eq8](https://arxiv.org/html/2509.21240v1#S3.SS2)：保持 token ratio，先每叶轨迹 token mean，再轨迹 mean。权重 mask/L_i、全 mini 行数分母实现该顺序，解决旧 token 总量加权反例。固定叶数时轨迹均值等于等权题/树组均值；不对任意变叶数外推。
- [TOP-D v1 §3.2 Eq7](https://arxiv.org/html/2607.04751v1#S3.SS2)：完整 rollout 提前计算 N_q，权重 mask×B/(Q×N_q)，全 rollout 用 B 行分母时严格为逐题 token mean 再题 mean。权重在 driver、DP shard 前计算且随行重排传递。任意 mini 使用固定 rollout 权重；随机均匀抽行是完整目标无偏估计，固定排列 mini 是该加权目标的 mini 优化，**不能写成每个 mini 都严格执行均匀题均值**。同题不同轨迹/跨 rank 不能局部重算 N_q。
- OPSD 保持 full-vocab/token mean；BAPO 保持作者 standard token GRPO。本文不将这两个方法宣称为新序列比率版本。

## 独立验证

1. 执行 tests/baselines/test_objective_measures.py、test_numeric_safety.py、test_distributed_normalization.py、test_paper_ddp.py：**12 passed**（evidence/review-objective-tests.txt）。覆盖 Tree 长短轨迹梯度抵消、TOP-D 两题长度反例、GiGPO action 联合比率跨 clip 与动作等权、空 mini 不移动 AdamW 已有动量/参数、masked logp 非有限与 exp 溢出、fractional mask。
2. 独立真实两 rank CPU/Gloo/DDP 探针：evidence/review-objective-ddp.py 和 .txt。Tree/TOP-D/GiGPO，各 micro=1/2、完整题跨 rank、一 rank 完全无有效 action 共 12 情形，梯度均等于单进程完整 rollout 参考。参考完整梯度分别为 -1.5、-2.16666675、-3.25；rank0 空时分别 -0.75、-1.5、-1.5。该探针使用真正 DDP backward，不仅检查权重值。另检查全局 token count=0 的共同 skip 判定；这个全局空检查不冒充完整两 rank actor+optimizer 集成测试。
3. 非空 actual DataParallelPPOActor.update_policy 四方法 CPU 探针：只替代 TensorDict.cuda 与模型 forward，使用真实 Linear/SGD、mini=4/micro=1、真实 actor 调度。参数变化与独立目标参考一致：GiGPO .3250000、Tree .1500000、TOP-D .21666665、CoSearch .1500000；各 optimizer_steps=1。该探针发现并复核了下面的 actor 接入修复，不涉及真正 HF/GPU forward。
4. root 新增 test_paper_ddp.py 的真实两 rank actual actor 更新独立运行通过，覆盖 TOP-D/Tree/GiGPO 每题跨 rank、micro1/2、一空 rank、全局空 mini 参数/已有 AdamW 动量/step 计数保持。其模型 forward 与 CUDA 转移为 CPU 适配，actor/optimizer/DDP 调度为真实执行。
5. 全局空 mini：actor/OPSD 在全 world collective count 决策后共同 skip；局部空而全局非空仍 forward/backward，有 graph-connected 零 policy 梯度并参与同步。worker 只有实际 optimizer_steps>0 才调 scheduler。SP>1 分母因 SP 数据复制取消后可静态推导，未执行本轮真实 SP/FSDP 验证。

## 发现与处理

### R1 / P1：paper_objective 放错函数 — 已修复并复核

第一版把 paper_objective 定义和字段选择加在 compute_log_prob，update_policy 没有定义也没携带 policy_weights/step_ids，任何非空 mini 到 if paper_objective 会 NameError；已有全空 actor 测试提前 skip 所以没抓到。即时通知 root 后，已把检测/字段 select 放在 update_policy，并移除 compute_log_prob 的无用字段。上述四方法 actual actor 探针通过。建议保留非空集成回归，而不只测 helper。

### R2 / P2：最终 mask 不能隔离非有限 logits 的 entropy/log_softmax backward — 已加 fail-fast 防线

独立反例 logits=[[[0,1],[inf,0]]]、mask=[[1,0]]：masked_mean(entropy_from_logits(logits),mask) 前向 loss=.58220303，但第二个被 mask 的位置 logits.grad=[nan,nan]。目前 core PPO 在 exp 前 mask logp/adv 可隔离叶子 logp 非有限；它无法修复此前 log_softmax/entropy 自身的非法 Jacobian。_optimizer_step 新增 nonfinite grad_norm 拒绝，因此不会继续写入非有限参数；这不是“任意非有限 logits 可正常零梯度训练”的证明。

root 已在 entropy_from_logits 开头检查全部 raw logits 是否 finite，非法时在 softmax 前抛 ValueError，无 backward、无 optimizer step。新增 test_masked_nonfinite_logits_fail_before_entropy_jacobian 对 inf/-inf/nan 逐例复核，mock softmax 验证未进入非法 Jacobian，正常但极端 finite 的 masked logits 保持零梯度。该 helper 在 padded 与 remove-padding 两条 forward 路径都会调用：padded 路径虽已计算 log_probs，但 entropy 拒绝后不会 backward；remove-padding 路径在 log_probs 前检查。它不是让非法 logits 继续训练的方案，不作该声明。

性能成本：torch.isfinite(logits).all() 每 micro 扫描 logit tensor，在 CUDA 上 Python 布尔判断会设备同步；独立 torch.compile(dynamic=True, backend="eager") 探针实际确认有限 forward/backward 通过，inf/-inf/nan 均拒绝；torch._dynamo.explain 显示 2 graphs / 1 generic_jump TensorVariable graph break（evidence/review-entropy-compile.txt）。这证明 Dynamo eager backend 的防线语义，不是 CUDA Inductor/flash-attn 性能验证。未实测 3B 吞吐影响，最终报告应保留这个成本，不能声称无性能开销。非监督/填充 logits 也一律拒绝，是较严格的 fail-fast 选择。有效 logits 非有限没有 clamp 或替换，原文有效目标保持。

## 范围与未证明内容

本轮通过的是 finite-valid-input 下的 helper / CPU actor 聚合 / SP=1 Gloo 数学。未证明真实教师身份与训练来源、3B CPU RSS/显存、真实检索与 vLLM、flash-attn/remove-padding、SP>1 FSDP、精确 checkpoint resume。所有论文适配差异须留在最终修复报告；继承上下文不一致按用户明确决定保留。不同权重会改变 policy 与额外 entropy/KL 正则的尺度，当前 actor 对额外项也使用对应 paper_weights；GiGPO 仍是 action 内 token-mean entropy/KL 辅助项，不能说完整正则和论文式逐字等价。

## 最后集成核查

root 已为四种 paper objective 的非空 update mini 添加必须携带 driver policy_weights 的错误门禁，避免掉回旧 token mean；全空仍在门禁前共同 skip，无更新。NCCL 静态设备核查：fsdp_workers.update_actor 在进入 sharding manager 和 update_policy 前执行 data.batch.cuda()；SP allgather 后数据恢复同 CUDA device，因此 global_token_count(mask) 的 all_reduce 使用 CUDA float tensor，不是 NCCL 不支持的 CPU tensor。OPSD 同 worker 入口，亦成立。默认固定微批跨 rank collective 次数对齐；动态长度微批/SP>1 吞吐与协调没有本轮 GPU 执行证据。非有限 forward 的 ValueError 为 rank 本地拒绝，不应称为已验证的全 rank 协同错误恢复；launcher/Ray 应整组退出，不能单独继续该 rank。

root 提供全套 evidence/cpu-all.txt 的 102 passed / 1 skip，并说明新GPU测试使用policyweights；本复审未重新跑全套且未成功连接A800，不能把历史tinyFSDP结果当作本轮新目标GPU验收。
