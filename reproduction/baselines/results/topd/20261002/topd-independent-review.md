# TOP-D 独立审查：数学实现有条件通过，真实效果未验收

审查日期：2026-10-02（北京时间）。审查范围为本线程 `work/topd`，只读审查，不更改共享代码/远端，不启动训练。审查时 HEAD 为 `1620c2e`，算法冻结祖先 `0723dca0c1458482ac3e78f8ed43af04893955d2`；工作树 clean。本结论不把历史报告的旧问题直接移植到现代码。

## 一手来源与可复现范围

使用 paper-glance 的核心提取和 review 模块辅助核查；用户已指定审查任务，因此不询问菜单。已实际读取 [Trust Region Policy Distillation v1](https://arxiv.org/html/2607.04751v1)。方法核查依据 §3 Eq.5、Eq.7、Eq.8 与 Algorithm 1：行为策略生成响应；教师/行为策略的采样 token 概率比转为 proximal reward；每个 token 的当前奖励加排除自身的未来均值；同题所有响应 token 一起白化；固定行为概率和优势，以裁剪 surrogate 重用数据。论文实验为数学推理与 Qwen3 学生/更强教师；Search-R1 Qwen2.5-3B base、搜索观测与同规模 GRPO 教师是本任务适配，不能直接继承论文的效果或单调改善结论。§5.1 列出 alpha .1/.2；本计划选 .1。

未核实 TOP-D 官方作者代码、固定源码 commit 或其运行配方。论文页面中未找到 Code 链接，历史本地审查也明确无已核实官方代码；这不等同于证明官方代码不存在。报告只能声明“对照论文的方法实现审查”，不能声明“与作者代码逐行一致”。

## 默认调用的逐项核验

| 验收项 | 证据与结论 |
|---|---|
| alpha=.1、稳定概率混合 | `advantages.py:69–79` 用 `logaddexp(log(alpha)+teacher-old, log(1-alpha))`，不是 log 概率线性插值；默认 YAML .1，配置预览 .1。masked NaN/Inf 不进入有效值。有效值本身非有限不能作为有效训练证据。通过。 |
| exclusive future 有效 token 均值 | `advantages.py:80–83` 反向累计后减自身，分母也是有效 mask 未来数量；最后有效 token 未来贡献为零。观察 token 和 padding 不拿 reward/loss。通过。 |
| 同题 token 合并白化 | `advantages.py:85–95` 按 uid 汇总所有轨迹有效 token，统一 mean/std；使用 sample std correction=1 与 epsilon。单有效 token 置零。论文没有明确给出 sample/population correction；当前为合理工程选择，应保留记录。通过。 |
| 全 rollout 问题权重 | `objectives.py:17–29` 在完整 rollout 上计算 `B/(Q*Nq)`，其中 B 是行数、Q 是唯一题数、Nq 是该题有效 token 数；空题保留外层零贡献。`weighted_policy_loss` 用 minibatch 行数/DP size 作分母，DP 平均梯度抵消。固定 micro 切分保留权重，不按本地 rank 重算。历史 T02 已修复。通过。 |
| 两次 PPO 内循环 old 固定 | `ray_trainer.py:795–797` 更新前计算 old；`controller.py` 更新前只算一次优势；`dp_actor.py:213–234` 固定 TensorDict 的 old/advantage 重用两次，未在第二 epoch 重算 old。`run.sh:30–34` 默认设 epochs=2，配置预览为2。通过。 |
| masked 溢出防护 | `objectives.py:58–79` 有效值有限性检查，masked logp/优势/权重先归零，再 exp，有效 ratio 溢出明确报错。历史 T03 已修复；不应报告旧 masked delta100 NaN 为现存问题。通过。 |
| 教师冻结、答案隔离 | Controller 中一份 driver CPU FP32 teacher，eval/requires_grad=False；`log_probs` no_grad。默认 privileged_answers=false 时 targets=None，教师见 prompt+重建响应（含检索观察），不附 accepted answers。基准 reward gold 在单独评分中使用，但 TOP-D dense reward覆盖训练奖励。默认通过。 |
| tokenizer 与 label 对齐 | FrozenTeacher 检查完整 token-to-id vocab、BOS/EOS，并统一 pad；teacher_inputs 压紧有效响应，prediction position 指向每个目标前一个 hidden state。通过代码与 tiny HF 模型检查；尚未用真实 Qwen2.5-3B 学生/导出教师核实 vocab/hash。 |
| 已训练教师来源 | 代码只检查显式路径及同路径禁用条件，不能证明教师已训练、权重不同、仅98% train、由2% holdout选择或目录不可变。必须由交付 teacher-manifest 与实物 hash/selection证据验收。当前教师尚不存在：未通过生产 gate。 |

## 现存风险与必要操作

1. **P1：生产启动依赖尚未满足。** 当前没有已验收的独立 GRPO trained teacher 和公共冻结提交。不能启动 TOP-D 算法 GPU smoke、不能把换目录的 base 或 Instruct 代替训练教师。应等待新 GRPO 在更新5/100/200/300/400/500的独立 holdout选择结果，交付绝对 HF 路径、student/teacher 权重 SHA、tokenizer vocab/BOS/EOS、训练/holdout split身份、选择指标、checkpoint update、不可变性和教师训练成本。单凭不同路径不构成验收。

2. **P1：公共部署仍待冻结。** 当前 `preflight.py:35–36` 仍拒绝所有 compute process，与本计划允许双15%以下小进程的规则冲突；当前候选 h100-11 存小进程，不能绕过检查或结束原进程。必须接入公共 owner 审查冻结的门禁与 CPU检索适配，并验证真实query top3/top50一致、正式 Slurm allocation、UUID/lease、资源重查及持久输出路径。这里是已知公共依赖，不建议本 agent 实现第二套。

3. **P2：可覆盖配置需要交付 gate。** 入口允许尾部 Hydra overrides；直接 main 入口也可指定 TOP-D privileged_answers=true、epochs≠2 或 alpha≠.1。validate_config未硬拒绝这些协议偏离；privileged=true 还绕开同路径禁止并实际给教师答案。此次实际配置预览 false/.1/2 正确，因此不能称为当前运行已泄漏。生产前必须检查最终 resolved 配置与 manifest；若加硬门禁，应针对受控复现实验入口，避免改核心数学目标。

4. **P2：继承的 context mismatch 必须量化报告。** `rollout.py:87–99,152–158` 实际生成 rolling history可能与整条轨迹打分前缀不同，默认 report保留用户既定 Search-R1协议。因此 old snapshot参数固定并不保证重建前缀下的概率等于真实行为采样概率；也不能证明重编码后的 target ID与 raw生成完全一致。该偏离已获本计划保留，应报告 context_mismatches计数与实际 Qwen raw/reencoded一致性，不能擅自改变长度/上下文。

5. **P2：计算成本记录不足以直接比较效率。** 当前 TOP-D 每条轨迹串行 CPU FP32完整 transformer forward，虽然词表 head用chunk16，hidden forward仍覆盖完整prompt/响应；一次outer512题×5轨迹=2560次teacher评分。代码有 teacher参数和累计评分token、rollout生成/prefill/search/轨迹计数及 timing_raw/adv；adv墙钟含reward等开销，不能等同teacher纯CPU时间。尚无教师独立CPU time/RSS、retriever CPU time/RSS或本次GPU小时/显存峰值测量。生产需独立采样这些指标，另列GRPO教师训练、holdout选择与TOP-D两epochs成本；不可只按500 outer updates宣称等成本。单份3B FP32权重已约12GB量级，峰值仍含中间状态及其他driver负载，需要按实际Slurm配额实测。

## CPU证据与真实3B边界

独立审查实际运行4项定向unittest，全部通过：exclusive future与mask、不同题不同长度的目标梯度及micro拆分、两epochs全空mini不更新AdamW动量/weight decay、tiny Qwen teacher prediction alignment及教师无梯度。此次测试使用本机CPU环境，不能作为远端Python3.10/CUDA固定环境的通过证据；最后一项学生参数变化是OPSD shared helper的tiny模型测试，不代表TOP-D真实3B更新已成功。

主 agent 的 `outputs/cpu-full-gate.log` 已读，记录134 passed /1 skipped；其reported baseline44 tests、reproduction61 tests另为CPU gate。`outputs/topd-config-preview.yaml` 已读：teacher_path 是PENDING占位，仅配置预览，未模型加载；alpha .1、privileged false、epochs2。仓库历史A800 small-model smoke为历史证据，不是本次h100-11真实Qwen2.5-3B Search-R1教师/学生搜索训练与推理验证。

仍未验证：真实生产资产和路径无关split身份；训练好教师效果/权重差异；H100实际搜索更新及finite loss/logprob、非零参数变化；两次实际outer更新日志；5步保存/加载；500实际outer训练；51,713题最终F1/EM/格式率；与本轮新GRPO等协议对照的提升；完整教师/检索/GPU成本。未完成这些不能声明“TOP-D效果验证通过”。

## 验收结论

静态数学及定向CPU验证 **CONDITIONAL PASS**，当前默认核心路径没有发现需要重写的TOP-D公式错误。训练启动 **WAIT FOR COMMON SETUP + TRAINED TEACHER**，真实3B与Search agentic RL效果 **NOT VERIFIED**。完成上面的交付gate后，应继续按计划GPU smoke→5步checkpoint/loading→500实际outer→统一最终评估，并在REPORT同时报告教师holdout表现、成本和适配差异；若真实结果不提升也应如实报告，不能把实现通过转成效果通过。
