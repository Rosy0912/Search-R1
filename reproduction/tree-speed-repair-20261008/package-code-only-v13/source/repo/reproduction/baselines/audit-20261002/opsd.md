# OPSD 独立单算法审查（2026-10-02）

**结论：CONDITIONAL；当前不能作为完成生产验收的 OPSD 正式训练直接放行。** 核心 full-vocabulary 损失、方向、梯度、预测位置和 DP token 归一化通过本轮独立反例验证。阻碍主要是共享长上下文 on-policy 条件错配，以及未验收的 3B CPU teacher/显存/吞吐与完整恢复能力。没有发现本轮可证明的 P0。这里审查的是固定 teacher 的 Search-R1 适配，不等同论文默认动态同权重版本。

## 固定范围与一手依据

- 本地 HEAD 经 `git rev-parse HEAD` 核为 `7b93fbd01742715dc14290b0a6e2d3b42e6c19f7`；原 Search-R1 为 `b251ba6b503217a59cd3395c1326d7b66b602989`。
- 作者本地缓存 `/tmp/topd_opsd_source_audit/OPSD` 经 `git rev-parse HEAD` 核为 `ae7d2519e94920c4eb6206c0c26de46d9c50abae`。一手来源：[论文 v3 方法](https://arxiv.org/html/2601.18734v3#S3)、[作者 loss/teacher 实现](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/opsd_trainer.py)、[作者固定 teacher 配方](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/scripts/run_opsd_4b.sh)。Web 只读取该 primary 论文；作者代码独立读取固定缓存。
- 独立逐行阅读 `search_r1/baselines/{distillation,actor,controller,rollout,main}.py`，`verl/workers/{fsdp_workers,actor/dp_actor}.py`、`verl/trainer/{main_ppo,ppo/ray_trainer}.py`、两份 Hydra 配置和 `reproduction/baselines/run.sh`。没有直接复用旧 PASS 结论，没有修改实现、远端运行、GPU 分配、kill、commit 或 push。
- 主模型/数据/检索保持项目既定 Qwen2.5-3B、paper-grpo、Search-R1 条件；辅助教师已授权，但其来源及选择仍须登记。应用 paper-glance 的串行方法提取要求；用户已明确选择代码/公式审查，未另询功能菜单。

## 算法实际怎么运行、公式怎样对应

1. `main.py:9` 先验证纯蒸馏：关闭 reference KL/entropy，要求 privileged answers、SP=1、固定 microbatch。`main_ppo.py:184–187` 不建 RefPolicy；`fsdp_workers.py:330–333` 将实际 actor 替为 DistillationActor。
2. Controller 用 SearchRollout 从学生策略生成搜索动作；检索观察保留为后续上下文。compose 拼接 prompt+response，观察对应 info/loss mask 为 0。reward 仅报告，`controller.py:207–209` 令 advantage/return 为 0；它们不进入 OPSD loss。
3. `actor.py:16` 每 actor rank 加载一个独立、CPU、FP32、eval、requires_grad=False teacher；教师不属于 actor optimizer。每 mini 先跨 DP 求有效 actor token 数，再依次切 micro，targets 随 DataProto 行切片保留原配对（`actor.py:32–44`、`rollout.py:21–28`）。
4. teacher_inputs 去除 prompt padding，插入接受答案字符串，再拼同一学生 response，保留观察上下文。若第 j 个有效 response token 的前缀长度为 P，预测 logits 应取 P+j−1；代码 `P+cumsum(attention)−2` 正确（j 从 1 数）。teacher 右 pad，学生 prompt 左 pad/response 右 pad；学生 `logits[:, -width-1:-1]` 正确做下一 token 位移（`distillation.py:27–55,80–87`）。
5. 令 s=softmax(z_S/T)、t=softmax(stopgrad(z_T)/T)，逐词表贡献分别为 t_v(log t_v−log s_v)（beta=0，FKL）、s_v(log s_v−log t_v)（beta=1，RKL）。0<beta<1 时 m=beta t+(1−beta)s，贡献为 beta t_v log(t_v/m_v)+(1−beta)s_v log(s_v/m_v)。学生 mixture 的梯度保留，教师完全 detach。
6. 每 vocabulary component 做 min(component,0.05)，**不是**把整个位置 KL 裁到 0.05；裁剪后对 vocab 求和，然后乘 actor loss mask。与作者 `opsd_trainer.py:441–473` 一致。负贡献不做下界截断，是作者实现本身的语义；不能因裁剪后值可能小于 0 判为新 bug。没有额外 T²，与作者一致（`distillation.py:8–24,88–91`）。
7. 本地优化的是有效 actor token 全局均值：L=(Σ_rank,micro,token mask·D)/(Σ_rank,token mask)。每 rank 使用 denominator=global_count/world_size，FSDP/DDP 平均梯度后恰好得到上述梯度；micro backward 不再除 micro 数。每 mini 一次 grad clip/optimizer step，每 rollout 可按配置多次 epoch；默认 E=1。教师保持初始固定，学生更新后再采下一批。

论文的同模型、特权上下文、学生 rollout、仅学生反传及词表逐项 clipping 是核对依据；其伪代码 trajectory mean 再 batch mean 与作者实际 masked token mean 应分开看。本地匹配作者 reduction，而不是保证每条长短轨迹等权。[论文 Algorithm 1 与 §3.2](https://arxiv.org/html/2601.18734v3#S3)

## 四维检查

| 维度 | 独立结论 | 证据/边界 |
|---|---|---|
| 代码 | 核心接入通过；有配置/空批更新小缺口 | 实际 worker 替换、纯损失路径、metadata 配对均已读；micro 全局到每 rank floor 除法需修门禁 |
| 数学 | 核心数值/梯度通过 | 12 组 FKL/RKL/JSD × 温度 × clipping 与作者函数 values/gradients 一致；真实两进程 Gloo/DDP 不均匀 token+micro 累积匹配 global token mean |
| 逻辑 | 有条件 | 固定 teacher 可对应作者 fixed_teacher recipe；teacher hint 是答案集合适配；共享生成/评分前缀错配破坏真正 on-policy 前提 |
| infra | 未完成生产验收 | 原 FSDP/offload 外层仍工作；OPSD padded forward 绕开 actor remove-padding 优化，SP>1 明确拒绝；每 rank FP32 eager CPU teacher 成本尚未实测 |

## 问题、反例、后果和建议

### OPSD-01 / P1 / 继承缺陷：生成与蒸馏评分前缀不一致

位置：`rollout.py:87,96–100,149–152,177–187`，`distillation.py:61–63,78–80`。本轮独立构造 2049 token 输入、max_start_length=2048，initial_states 得到实际 generation rolling 长 2049、scoring prompt 长 2048，`equal=False`，无需等后续超长轮次即可触发。后续 rolling 保留末尾、response 记录保留头部也会再次分叉。

后果：teacher 与 student 相互比较时都使用重构轨迹，但该学生 response 不是由这一重构上下文策略采样，不能称严格的 on-policy OPSD。默认 `context_alignment=report` 只计数；`error` 可及时拒绝，却不是修复。原 Search-R1 也有该模式，应共同修复/审计所有方法与 GRPO 对照的真实条件，或在正式短验收时使用 error 并证明零 mismatch。详见 [共享审查 INF-02](shared-infra.md)。

### OPSD-02 / P2 / 新路径逻辑缺口：全零有效 token 仍执行 optimizer step

位置：`distillation.py:81,90–101`、`actor.py:37–49`、`dp_actor.py:143–150`、`fsdp_workers.py:229–232`。独立真实 tiny Qwen 测得全零 mask 时 loss=0、所有学生梯度为 0，**不出现其他 PPO 的除零 NaN**。但 actor 仍调用 AdamW step，默认 weight_decay=0.01；probe 在 lr=0.1 的可观测设置下 parameter max delta 为 `6.773322820663452e-05`。有历史 optimizer momentum 时，即使 decay=0 也可能继续移动参数。

后果：完全无监督 mini 被计为更新，可能发生衰减/惯性漂移，不等于“安全地跳过空批”。建议保留原始 global_count，全局为 0 时所有 rank 协调跳过 optimizer step（仍满足 FSDP collective 次序）；单 rank/micro 为 0、但其他 rank 有 token 时仍正常零梯度参与。loss 的有限性已通过，不能把该 P2 混写为 NaN P1。

### OPSD-03 / P2 / 配置校验缺口：允许 global micro < DP ranks，然后得到 local micro=0

位置：`controller.py:55–56`、`fsdp_workers.py:97–101`、`actor.py:29–31`、`run.sh:20`。校验写成 global micro ≤ total GPU；worker 又整除 DP。global=1/world=2 或 global=2/world=8 均 floor 得 0，随后实际 actor 抛 Invalid OPSD minibatch/microbatch sizes。global=8/world=8 才是承诺的每 rank 1。

后果：配置可通过早期验证，却在模型与教师加载后失败。建议在 SP=1/rollout.n=1 的此限定配方要求 global micro=DP ranks，并验证 global mini 可整除、local mini≥1；两卡 launcher 必须显式 override n_gpus_per_node=2 和 OPSD_GLOBAL_MICROBATCH=2，CUDA_VISIBLE_DEVICES 本身不修改 Hydra GPU 数。

### OPSD-04 / 生产 P1 gate / 未验证风险：CPU教师与full logits资源预算不完整

位置：`distillation.py:64–79,104–123`、`actor.py:16`、`ppo_baselines.yaml:90–94`。一个 FP32 3B teacher 仅参数约 12GB 十进制，8 actor rank 仅 teacher 参数约 96GB 十进制（不含学生 offload、optimizer、激活、加载副本）。teacher eager attention 有二次序列成本，micro=1 不消除长 ctx attention 成本。预算检查在 teacher forward **之后**，且只估学生原始 logits，不包括 teacher attention/hidden、float32 softmax、保存的 loss autograd graph、学生 activations/FSDP gather/vLLM KV。

`chunk_size=16` 可限制单次 teacher lm_head 输出，但每块 student loss 直到统一 backward 都有 autograd 保留；不是整体 O(chunk) 显存保证。CUDA logits 在本地 transformers4.46.3 Qwen forward 无 labels 时不主动 upcast，BF16×2 byte 估计契合这一版本；requirements 仅规定 `<4.48`，更旧模型实现 upcast 及实际部署版本需要另验。

建议用正式 3B、固定最大轨迹、每 rank 1、真实 production 版本实测 RSS/CPU吞吐/GPU峰值，建立 teacher前向前的预预算和整机 RAM 预算；评估 eager 替代或合理 teacher placement/sharding。这里不是声称已发生 OOM；是不能凭 tiny evidence 放行正式训练。

### OPSD-05 / P1 gate / 继承保存边界：权重 checkpoint 不具备真实 resume

位置：`fsdp_workers.py:513–539`、`ray_trainer.py:651–665`、`controller.py:251–254`。保存真实 FSDP full actor state→HF model+tokenizer；manifest 只登记 teacher_path，明确 `resume_supported=false`。没有保存/恢复 actor optimizer、scheduler、RNG、dataloader/global step 联动；不保存/哈希冻结教师副本。加载 actor HF checkpoint 再开新任务只是 warm start，teacher 外部路径变化还会改监督目标。

建议正式长任务前实现与验收精确恢复，或明确采用不可恢复的短 run 契约、固定 teacher/hash，并不能把“checkpoint 保存通过”写作“可恢复训练”。这不是新 OPSD 数学 bug。

## 作者代码与原 Search-R1 一致性矩阵

| 项目 | 作者固定 SHA | 本地固定 SHA | 原 Search-R1 一致性/分类 |
|---|---|---|---|
| teacher 何时更新 | `opsd_trainer.py:671–687` 默认 current weights；EMA 可选；fixed_teacher 在 LoRA disable_adapter 下是初始 base。所有 run_opsd*.sh 有 --fixed_teacher | 完全独立 frozen checkpoint，CPU FP32，不同步学生 | 对应作者 fixed_teacher 变体；若要论文默认动态自蒸馏则不一致。full fine-tuning 代替 LoRA 是明确适配 |
| teacher 初始来源 | fixed recipe 同一 base | 任意已授权、tokenizer兼容 teacher_path；未强制与学生同初始权重 | 若选不同模型应命名 external-teacher OPD，不应宣传无外部教师 OPSD；正式输入需登记 |
| 特权信息 | `data_collator.py:97–108` user chat 中 reference solution + rationalize transition | 答案/别名 repr 插于原 prompt 后，可能在 assistant generation prefix 内，无完整解题轨迹 | 答案-only + Search提示是任务适配；论文允许答案，但提示位置/教学效果未验证 |
| loss/clip/temperature | `opsd_trainer.py:441–473` full-vocab、component clip、token mean；脚本 beta0/T1.1/clip.05(4B) | beta0/T1/.05，FKL/RKL/JSD全支持 | loss 与作者数值一致；T1 为保留 Search-R1 配方的有意偏离，无PPO/reference KL |
| 数据与模型 | 数学/Qwen3、LoRA、1024 completion等 | NQ+Hotpot、Qwen2.5-3B、真实搜索动作/观察、full FT | 主模型、搜索服务、共同长度/采样/学习率保留固定 Search-R1；不宣称论文 benchmark recipe 原样复现 |
| observation mask | 作者单段数学 rollout | 观察留上下文、loss只actor动作 | Search-R1 state masking 必要适配；上下文有信息却无直接该位置loss并非缺token |
| DP/micro | 作者 token mask reduction、Accelerate/LoRA | 全局token mean、FSDP average gradient补偿、micro1/rank | 是实际DP规范化适配；micro64共同默认改到每rank1是full-vocab必要资源偏离 |
| remove-padding/SP/offload | 作者不同 infra | OPSD padded forward，不使用 update_policy 的rmpad；SP>1拒绝；worker load/offload仍保留 | SP1的模型未应用SP monkey patch，rmpad=True不会强制该forward变形；性能适配边界，正式offload未验收 |
| 保存与验证 | 作者 Trainer/PEFT/vLLM配置 | 原FSDP HF权重保存，额外manifest；validation无teacher/gold特权 | 保留原Search-R1评测入口；未完成精确resume |

对原基础的 `git diff --quiet b251... -- search_r1/llm_agent/generation.py verl/trainer/ppo/core_algos.py verl/utils/dataset/rl_dataset.py search_r1/search/index.py` 返回0；实际旧入口在无 baselines 配置时走原分支。纯OPSD改变 loss 与teacher是必要适配，不能说原PPO完全没改。

## 本轮实际测试及未覆盖

所有新增probe只写 `/tmp`，不改实现。本地 `.venv-cpu` 为 transformers4.46.3，真实小 Qwen eager CPU；不是 3B、GPU 或真实 vLLM。

| 实际命令 | 结果 |
|---|---|
| `.venv-cpu/bin/python /tmp/opsd_independent_probe_20261002.py` | exit0；AST提取作者 generalized_jsd_loss，12组values/学生gradient全部一致，teacher gradient=None；真实tiny Qwen的不同prompt长度/左右padding/observation mask/chunk1对齐full teacher forward；loss=0.004715222865343094；零mask loss/梯度皆0；AdamW空批漂移反例成立 |
| `.venv-cpu/bin/python /tmp/opsd_ddp_probe_20261002.py` | exit0；实际两进程Gloo/DDP；rank0/1有效token=3/1、两个micro backward；各rank denominator=2、平均后grad=3.5，与(3×2+1×8)/4一致；全零global mask denominator=.5，有限 |
| `.venv-cpu/bin/python -m unittest tests.baselines.test_algorithms.Estimators.test_full_vocab_jsd_direction_clip_and_gradient_mask tests.baselines.test_models_and_driver.TinyModels.test_real_teacher_logprob_alignment_and_student_only_opsd_update` | 2 tests，OK；包括真实FrozenTeacher加载/学生only参数更新 |
| `.venv-cpu/bin/python -m pytest -q tests/baselines/test_algorithms.py -k 'full_vocab or teacher' tests/baselines/test_models_and_driver.py -k 'teacher or config or fit'` | 4 passed,19 deselected；后一个-k实际生效，故full_vocab单独用上述unittest重测；driver test用stub worker，不是实际Ray集群 |
| initial_states CPU构造反例：2049个非pad token，max_start_length=2048 | generation prefix=2049；scoring prefix=2048；equal=False |

一次初始 unittest 命令误用不存在的 `AlgorithmTests` 类名，exit1（AttributeError）；同次另外4个driver/model tests完成。已用实际 `Estimators` 类名纠正重跑，不能将该命令失败藏作全部通过。

既有 [A800 两卡 tiny 记录](../gpu-validation.md) 是 ca95b89、Torch2.9/cu128 下实际 DistillationActor/FSDP CPU教师、BF16 GPU学生、两步更新与权重保存；其结果有帮助，但不是本SHA/production Torch2.4/vLLM0.6.3完整验收。未覆盖正式3B与真实teacher asset、Ray transport、Wiki18/E5真实搜索、完整训练数据/七集指标、remove-padding+offload、生产显存/RSS、超长teacher context、故障恢复、两卡以上实际NCCL全局mask异常条件。

## 正式训练放行条件

先处理或严格阻断 OPSD-01；给出 teacher 与 student 初始权重关系、冻结模式、答案-only和prompt模板适配声明；修正global micro门禁；协调空global-mask跳过更新；完成生产3B/依赖/vLLM/搜索/offload短跑和资源预算。长run还需真正resume或明确可接受的不可恢复短run契约。满足这些前置后，本轮通过的loss与DP数值验证可作为可信基础；目前总体仍是 **CONDITIONAL**。
