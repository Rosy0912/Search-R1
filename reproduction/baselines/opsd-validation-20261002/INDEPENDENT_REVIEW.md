# Agent 4 OPSD 独立审查

审查日期：2026-10-02（北京时间）。审查冻结提交：`1620c2edb13809d19cc3f5af4a298ccd75488104`；独立审查仅新增本目录证据，未修改实现、未启动远端任务、未申请 GPU、未改其他分支。

**结论：数学内核及所测 CPU 路径通过；生产部署和 Qwen2.5-3B 搜索效果验收仍为 CONDITIONAL / 未完成。未发现本轮已证实的新 OPSD 数学缺陷。不能将 tiny CPU、历史 tiny GPU、有限参数更新或蒸馏 loss 下降写成 3B 搜索效果验证成功。**

依据：[论文 v3 §3.2 / Algorithm 1](https://arxiv.org/html/2601.18734v3#S3)、[作者固定源码 ae7d2519](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/opsd_trainer.py)。作者源码经 HTTPS 重新取回并保存于 `evidence/author_opsd_trainer.py`，不是从历史 PASS 结论推断。论文及作者源码用于核对同模型、特权上下文、学生 rollout、全词表损失、学生-only梯度及分量裁剪；本文不复述数学 benchmark 成绩。另逐行读取历史 `review-topd-opsd.md`、`audit-20261002/opsd.md`、`fixes-20261002/review-infra-opsd.md`，只用于识别历史问题，以下结论均以本次代码为准。

## 1. Summary：一轮实际执行顺序

1. `run.sh:25–29` 指定独立 frozen teacher，开启 teacher-only 答案上下文，关闭参考 KL 和熵。`controller.py:60–76` 校验纯蒸馏、SP=1、固定微批、每 DP rank 微批恰为 1；`fsdp_workers.py:330` 接入 `DistillationActor`。
2. `SearchRollout` 从学生生成搜索动作，返回检索观察继续作为上下文；`attention_mask × info_mask` 决定 actor loss mask。观察用于条件输入，不作为蒸馏监督目标。训练 outcome reward 仍用于日志，OPSD 的 advantage/return 为零，不参与 loss（`controller.py:248` 附近）。
3. 每个 actor rank 加载 CPU FP32/eval/no-grad 教师（`distillation.py:122–136`）。教师在原 prompt 后获得 accepted answers，再读取完全相同的 student response 和 observation history；学生输入不注入答案。教师 context 超限直接报错，不静默裁剪。
4. 教师右 padding，学生 prompt 左 padding；教师下一 token 预测位置为 `prefix_length + cumsum(response_attention) - 2`，学生对应 `logits[:, -response_width-1:-1]`。先计算教师 hidden states，再分块计算教师词表 logits，与完整学生词表 logits 对齐。
5. beta=0 计算 FKL `sum_v t_v(log t_v-log s_v)`；beta=1 计算 RKL；中间 beta 计算 `beta KL(t||m)+(1-beta) KL(s||m)`，`m=beta*t+(1-beta)*s`。每 vocabulary contribution 上界裁到 .05，再对 vocab 求和；温度为 1。教师 detach，学生 logits 与 mixture 保留梯度。没有额外 T² 或 PPO surrogate。
6. 每 mini 求所有 DP rank 的有效 token 总数，每 rank 分母为 `global_count/world_size`；micro losses 分别 backward，不再除 micro 个数；FSDP/DDP 平均梯度得到全局有效 token 均值。全局 count=0 时所有 rank 协调跳过 optimizer，避免 AdamW decay/动量漂移。mini 更新后采下一轮学生数据；教师始终保持初始 base。

## 2. Strengths：当前已复核的实现对应

| 检查项 | 当前代码与作者实现 | 独立证据 |
|---|---|---|
| FKL/RKL/JSD、温度、clip、梯度 | `distillation.py:8–24` 对照作者 `opsd_trainer.py:381–478` | AST 抽取固定作者函数，16 组组合（beta 0/.2/.5/1、T 1/1.1、clip null/.05），loss 与学生梯度一致，teacher grad=None |
| next-token 位移、变长 padding、观察 mask | `teacher_inputs` 和 `opsd_loss` | 真实随机 tiny Qwen，2 行不同 prompt/response 长度，带观察遮罩，chunk=1 与整段教师 logits 手工对齐 loss 一致（约 .00215194 vs .00215182，在容差内） |
| 学生-only更新 | 独立教师不属于 actor optimizer | tiny 测试实算有限、非零学生梯度；既有 tiny 参数更新测试重跑通过 |
| 全局 token 归一化 | `global_token_count` 与每 mini 分母 | 真实两进程 CPU Gloo/DDP，非均匀有效长度及一个全空 rank，梯度分别 2.5 / 3.0，符合全局 token mean；不是 CUDA/FSDP 验收 |
| 全空 batch 与微批门禁历史修复 | `actor.py:38–41`；`controller.py:45–53,73–74` | AdamW 已有动量/decay 的空监督回归通过，参数与 optimizer step 不动；配置检查及租约回归通过 |

作者 clipping 只设上界，允许单个 FKL/RKL contribution 为负；裁剪后和可为负，这是作者明确语义，不能当作本实现新增 bug。beta 端点是作者用于选择 FKL/RKL 的约定，并非数学 JSD 在端点的零值。论文 Algorithm 1 用每轨迹均值后 batch 均值，而作者 labels-aware reduction 实际为全有效 token 均值；本地匹配后者，应保留说明。

## 3. Weaknesses：真正缺口及处置

### [W1] P1 部署 gate：公共 GPU 检查仍拒绝低占用已有进程

`preflight.py:33–36` 在满足双 15% 后仍对 compute process 直接拒绝。`test_infra_opsd_review.py` 甚至把拒绝已有进程写成当前正确行为。这不符合本轮 PLAN，必须等 common-setup 的统一修复和独立冻结提交；不要在 OPSD 分支绕过检查或杀已有进程。H100-10 当前候选无进程也不能证明公共代码符合协议。新增测试须覆盖有低占用进程通过、任一阈值超标拒绝和未知状态拒绝。

### [W2] P1 生产验收缺口：3B/FSDP/CPU teacher 资源与吞吐没有本轮证据

`opsd_loss:64–79` 先跑 CPU teacher eager attention，之后才检查学生原始 logits 的估计预算。该预算不包括 teacher attention/hidden、学生激活、FP32 softmax及保存到 backward 的图、FSDP gather、optimizer/offload 或 vLLM KV。chunk16 只分块损失处理，完整学生 logits 已整体分配，不能宣称 O(chunk) 总显存。

8 份 FP32 3B 教师仅权重约 96GB 十进制（估算，非测量），另有 offload/加载/激活；实际限制是 Slurm/cgroup 分配的 RAM/CPU。需要正式锁定 Linux/Python3.10/Torch2.4/vLLM.6.3 等生产环境，实测 3B 两个 outer update、5 步 checkpoint 保存/重载、最长轨迹显存和 RSS、CPU教师吞吐、NCCL/FSDP/offload及 vLLM 同步。本次 Mac CPU 环境 Python3.9.6、Torch2.8.0、Transformers4.46.3、Ray2.41.0、CUDA=False，不具有上述证明力。未声称已发生 OOM。

### [W3] P1 效果/成本验收缺口：默认日志不足以直接产出 PLAN 指标表

冻结版 `_validate():590–595` 只输出 `val/test_score/<source>` 和 count，默认 `RewardManager` 选择 `qa_em.compute_score_generated_em`。不能把这一 score 自动标成 F1/EM/格式率三个独立指标；需真实回答输出和统一评估器或对应新增测量记录。训练已记录 rollout trajectories/generated tokens/retrieval calls/context mismatches，但保留轨迹 response length 含观察等历史信息，不能代替实际生成 tokens。

教师目前只记录参数数和累计 scored tokens（`actor.py:58–59`），缺专属 CPU 时间和 RSS；总 `update_actor` 墙钟混合学生与教师计算，不能冒充 teacher CPU 时间。还需 GPU小时、检索CPU成本、墙钟和真实显存峰值。正式效果判断要使用同协议的新 GRPO，完整500次实际 outer updates及七集 51,713题最终评估；不能引用旧100%训练/不同micro的GRPO作公平对照。

### [W4] P2 provenance 缺口：未交付共享冻结提交/路径无关数据身份/teacher权重身份

当前 `holdout.py:80–84,100–102` specification/ID 含绝对路径，manifest 有产物SHA但没有 PLAN §8 的规范化路径无关 content_identity 与问题/行集合摘要。不同机器原 manifest ID 不一致不是 split 数学 bug；公共适配完成后比较 content identity 和产物 SHA，不改问题分组。

`FrozenTeacher` 验证 vocab/BOS/EOS，未验证与学生初始权重完全相同。PLAN 已指定同一 Qwen2.5-3B base；本轮必须提交教师/学生初始完整资产 hash 与 immutable manifest。只登记 teacher 路径不证明固定同源权重。算法 state 保存 `resume_supported=false`（`controller.py:296` 附近），HF 权重重载不等于 optimizer/RNG/dataloader 精确恢复；沿用用户授权的一次活跃 tmux 运行契约即可，不擅自要求改变数学或实现精确resume。

## 4. 用户已授权偏离：不得列为待修复算法缺陷

| 行为 | 审查分类与报告边界 |
|---|---|
| 初始 base、独立冻结 CPU teacher | 已授权；作者源码支持 fixed_teacher/disable_adapter 变体。本地 full FT + 独立实例与作者 LoRA 实现方式不同，论文默认共享当前权重也不同，均应声明 |
| accepted answer-only，不额外 rationale | 已授权 Search-R1 适配；答案注入格式与作者 rationalize chat prompt 不同，教学效果需实测，不能擅自改模板 |
| T1、Qwen2.5-3B base、真实搜索环境 | 已授权共同协议；不能将 Qwen3/math benchmark 数字迁移为搜索效果预期 |
| 生成窗口与最终训练历史差异 | 用户明确要求保留；`rollout.py:96–100` 诊断 report 数量，不能要求修复/error 模式作为本轮启动门槛。统计并报告 mismatch；严格 full-context on-policy 理论条件可能不满足，仅作结论范围限制 |
| observation 留上下文，loss only actor token；SP1、micro1/rank、无额外KL/entropy | 正确/已授权适配；CPU teacher开销必须单列，不能以8卡GPU数掩盖额外CPU成本 |

## 5. Questions for execution owner

这些是交付核对项，不是重新向用户申请授权：共同部署提交及 review 是否已冻结？真实3B师生权重hash是否相同？跨路径split内容身份是否一致？最长上下文时实际每rank显存/RSS及teacher耗时如何？完整评估中 F1、EM、格式率及同协议GRPO差值是多少？

## 6. 实跑证据与 Summary Decision

- `evidence/independent_cpu_pytest.txt`：复跑 algorithms/models-and-driver/infra-opsd/distributed-normalization/numeric-safety，**32 passed，2 warnings，8.91s**。包含其他方法既有测试，仅称此测试组合通过，不扩大成全仓CPU验收。
- `evidence/independent_loss_probe.py` / `.json` / `.stderr`：新增独立作者损失 value/gradient 数值对照及真实 tiny Qwen 对齐；16组全部通过。stderr 包含 Mac LibreSSL warning，非损失测试失败。
- `evidence/independent_review_manifest.json`：冻结源码及证据 SHA、环境版本和结论范围。

Recommendation：**有条件通过 CPU 数学与代码审查；正式效果验收待完成**。不需要回滚已授权的 frozen-base/answer-only/context适配；必须等待公共部署修复、真实生产3B验收与完整同协议效果/成本证据。后续共同或OPSD实现有变动，需针对最终新SHA增量复核；本报告不能自动认证未来提交。

## 7. 增量复核：教师成本观测（以工作树文件 SHA 为准）

本轮另外读取 `docs/superpowers/specs/2026-10-02-opsd-validation-design.md`，审查 `distillation.py`/`actor.py` 的观测补充及 tiny 测试/fake teacher fixture 适配。F1/EM/格式率将复用共同负责人新增的公共 evaluator；OPSD 未另写独立版本。本附注不自动认证尚未合入/审查的公共 evaluator。

### 数学等效及测试

`evidence/instrumentation_equivalence_probe.py` 从冻结 `1620c2e` 直接读取旧 loss 实现，同一 tiny Qwen 参数、同一答案、同一mask比较新旧实际 forward/backward：**loss bitwise相同，15个学生梯度张量bitwise相同，teacher仍无梯度**。再次执行作者16组loss/gradient对照全部通过，证据为 `instrumented_author_loss_probe.json`。额外用受控时钟验证累计CPU/wall增量，并分别模拟 Darwin/Linux验证 ru_maxrss 到 bytes 的换算。

focused pytest：新增成本指标的真实 tiny 教师更新测试和 fake-teacher 空batch回归，**2 passed**。fake teacher 只新增 `cost_metrics=lambda:{}`，全空 mini 仍完全不访问模型、不走optimizer，未改变该回归目标。完整 instrumented CPU gate 由执行负责人运行，本独立复核不将其未读取的结果写作通过。

### 观测名称和汇总解释

- `teacher/model_bytes_per_rank` 是已加载 frozen model 的 parameter+buffer 张量字节数（去重遍历），包含 buffer；不是额外RSS、加载峰值或含激活的教师总RAM。tiny实算为49056 bytes，独立与tensor逐项相加相符。
- `teacher_worker/peak_rss_bytes_per_rank` 是**整个actor worker进程** lifetime high-water RSS，包含学生/优化器/教师/加载/运行时，不能称 isolated teacher memory。Darwin ru_maxrss 原单位 bytes，Linux单位 KiB，换算分支测试通过。
- load计时包括tokenizer加载/一致性检查、模型加载、冻结与tensor byte统计，排除进入构造器之前的模块导入；作为load区间CPU/wall计时合理。process_time累计该worker所有CPU线程在区间内使用的CPU秒，可能大于wall秒，不能保证独立归因到teacher线程。
- forward计时区间主要覆盖base_model与lm_head，尚不覆盖全部teacher prompt组装、位置计算/hidden gather；`log_probs`的softmax/gather也在区间外。应称teacher forward/output projection观测，不称完整teacher pipeline CPU成本。
- `reduce_metrics`（ray_trainer.py:174–176）对rank值取均值。后缀 `per_rank` 表示输出是每rank平均：固定8rank时，累计CPU秒/model tensor bytes可乘8得到rank和（load另加）；不能沿每步将累计counter再相加。wall rank均值不等于端到端critical-path wall，RSS均值不等于最坏rank峰值；8倍RSS均值也不能等同实际cgroup峰值/物理驻留总量（共享页、峰值时刻不同）。需要最坏rank/cgroup资源验收时另行采集真实记录。

### [I1] P2 初次计时边界发现：CUDA传输在teacher wall区间内

初次观测版本将 `get_output_embeddings()(features).to(device)` 整句放在 `teacher.measure()` 内。CPU→CUDA `.to()` 默认阻塞主线程并同步相应CUDA stream，teacher wall计时会包含传输及可能的先前学生GPU工作等待；因此不能严称纯CPUteacher投影耗时，也与“不同步学生CUDA”注释不完全相符。该判断来自[PyTorch官方blocking/non-blocking说明](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)，本次Mac无CUDA，未声称实测到等待。

已通知执行负责人：把仅CPU output projection放在measure内，退出后执行原`.to(device)`即可保留全部数学行为；或保留现状但明确metric包含传输/等待。下方最终复核附注应以负责人实际修复后的文件与重跑证据更新。此问题是成本归因问题，不是loss/梯度问题。

### 计时边界最终增量复核：I1 已修复

负责人随后将 CPU lm_head projection 与 `.to(device)` 分离：后者在 `teacher.measure()` 区间外执行。独立重新读取最终代码、measure docstring及设计文档，均明确CPU区间包含全部worker CPU线程的process_time，非exclusive attribution；worker RSS仍明确为完整actor进程。**I1关闭，无剩余已证实的观测实现缺陷。**

针对最终修正版再次执行 `instrumentation_equivalence_probe.py`：loss和15个学生梯度张量仍bitwise相同，teacher无梯度、受控时钟累加与两平台RSS单位检查仍通过。最终证据为 `instrumentation_final_equivalence_probe.json/.stderr`，工作树SHA见 `instrumentation_final_review_manifest.json`。这项结论只覆盖本次观测差异；原报告的真实3B/FSDP吞吐/资源、共享evaluator和正式效果验收要求继续有效。
