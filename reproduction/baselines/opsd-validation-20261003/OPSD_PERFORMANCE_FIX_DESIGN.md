# OPSD 最小性能修复候选：共同外侧 padding 裁剪

2026-10-04。本文件是本地设计与 CPU 证明，不是性能修复验收。未改生产源码，未提交 GPU 作业或训练。

## 当前事实与问题范围

核对 `work/opsd` HEAD `fae5e3295c18a0c6b80e314b2fdce61ad571fec6` 的 `distillation.py`、`actor.py`，以及 `OPSD_INFRA_STRICT_REVIEW.md`、`PERFORMANCE_PADDING_PATH_REVIEW.md`、Proposal A 的实际 job2990 原始摘要。当前仍执行学生完整 padded HF 前向和全词表 logits；自定义 OPSD 路径未使用普通 GRPO actor 的 packed/remove_padding 路径。教师上下文已按当前微批过滤 padding 后动态构造，因此不能把学生的全局 padding 开销也归到教师。

当前每 rank/outer 320 样本、mini32、micro2，对应 10 次 optimizer 和 160 次前后向。历史 H20 GRPO micro8 对应 40 次前后向，这只是工作量差异，不是相同算法/硬件的速度 benchmark。教师 FP32/eager、特权上下文和全词表蒸馏也增加实际工作。现有独占 profiler 不足以给这些项目分摊耗时或保证某项改动降低多少秒。

3007 历史最初五步均值生成138.98s、更新407.02s、总步562.03s；只能定位更新值得优化，不可据此把现有慢因归为跨集群或 CPU 检索。当前已是同节点 GPU 检索。Proposal A 的空 loss chunk teacher-head/JSD 省略及窄 connected-zero 已采用：job2990 的短样本真实全8 rank、全参数、双 Adam 严格数值等价证明不覆盖本候选裁剪。

## 具体最小候选

在 actor 取得每个现有微批后、调用 `opsd_loss` 前，只产生这个微批的张量视图，不改 rollout、采样顺序、预算、mask、token denominator、模型、教师精度、默认微批或 optimizer。

1. 在 prompt 内找所有行共同 attention=0 的最左前缀，只裁掉这个连续前缀。必须核对移除 token 全是 pad，不能裁掉任何一行有效 token。每行独有左 padding 留下。
2. 在 response 内找任何一行 attention=1 的最后列。保留宽度向 chunk16 边界向上取整，并且不超过原宽度，裁掉剩余所有行共同 attention=0 的右尾。核对移除 token 全是 pad、loss_mask=0。
3. `input_ids/attention_mask/position_ids/prompts/responses/loss_mask` 对齐切片。**保留原 position_ids**，不重新编号。位置没有改变；只移除了所有行都不参加 attention 的外侧槽位。
4. 不以 loss_mask 决定裁剪边界：tool observation 虽 loss=0，但 attention=1，必须保留。内部 hole、不同长度行的内部 padding、任何真实历史一律保留，不做逐行 unpadding。
5. 全 response 无效而原宽度>0的微批，保留至少一个 pad response 槽位和学生 connected-zero 前向/反向，使其他 rank 非空时仍参与 FSDP 同步。不能直接返回常量零或跳过学生模型。原输入宽度=0 不属于本候选的可接受输入：当前 `opsd_loss` 的 `-width` attention 切片在0时变成全 prompt，需另行定义契约；候选显式拒绝，而不会新制造宽度0。

向16取整是保守必要项：直接裁到最后 attended 列会缩小某个原本非空 teacher-head/JSD chunk，改变 GEMM M 形状；本候选使所有原先有监督的 chunk 形状保持一致，删除的只可能是原先全空 chunk。教师过滤后的完整 token IDs/attention 及保留 response target prediction positions 不变，两种教师 prompt adapter 均核对。

## CPU 可复现结果与边界

独立候选及证明位于 `work/performance-notes/prepare-safe-pad-crop/`。`proof.py` AST 提取当前真正的 `teacher_inputs`，验证 legacy/user_context 两种模式的教师输入、attention、保留目标位置逐元素相同；合成微批包含真实 observation 和内部 attention hole。prompt7→4、response48→32，总学生矩形列55→36（仅这个合成例减少34.5%输入/logits元素，**不是速度测量**）。教师实际上下文长度没有因此减少。

CPU tiny float64 causal reference、全词表 KL、固定 mask/denominator，两次连续 AdamW：loss、全部参考学生参数梯度、更新后参数的 max_abs 全为0，门限1e-12。模型明确按同一活跃 causal key 顺序计算；不是当前真实3B/HF模型。

12项 CPU 测试通过，覆盖有效 observation 尾保留、内部 hole 保留、position_ids 原样切片、无共同左 padding、17 token 保留32及非空 chunk16+16形状、无效 IDs/mask/拼接/binary contract 拒绝、全空微批 connected backward、原宽度0显式拒绝。

复现：使用本地现有 `.venv-cpu/bin/python` 运行该目录的 `proof.py` 和 `python -m pytest -q .../test_proof.py`。原始结果为同目录 `cpu-proof.json`。该 CPU 结果只证明参考数学和张量变换契约。它不证明真实 BF16Flash 全梯度精确相同、不证明 FSDP collective 安全、不证明实际速度或显存峰值下降。

## 采用前必须通过的严格验收

学生整体序列长度改变仍会改变 BF16 GEMM 的 M/布局。历史设备迁移实验已显示小 FP32 教师目标变化可放大 BF16 梯度差异；即使数学目标相同，本候选也可能违反实际严格数值等价。不能用原 precision semantic gate 的容差来放宽本优化门禁。

需在独立受控同节点、实际8 rank/3B/BF16Flash/FSDP/FP32eager GPU教师/相同检索共存、原始与候选相同参数和随机态上验证：相同输入活跃token、原位置、教师历史/target、mask、denominator；所有3.085b学生梯度、loss和两次 Adam 更新数值精确相同；原始/候选同forward重复对照；普通、rank0空微批、全局空 mini 分别覆盖，collective数与非空chunk调用一致。真实最大宽度/不等长/tool尾/内部hole亦需覆盖。

如严格 gate 有任何差异，保留原失败证据、不採用、不宣称修复。通过后另做相同源/硬件/后端/预算且输入形状记录齐全的步时与峰值显存对照，再 fresh5+64题/HF reload 独立审查。不得以 CPU 单元测试代替这些条件。

## 后续方案为什么暂不实施

仅为监督 target 做学生 lm_head 投影，或调用学生 base_model 获取 hidden 再单独 head，理论上减少 prompt/observation 全词表 logits，可能收益更大；但会改变 FSDP forward 包装、参数 all-gather、tied embedding/head 权重处理与反向调度，不能直接绕过 `.module` 包装调用。packed/varlen 路径还涉及每行 response target 映射、教师 observation 历史及模型 kernel 数值。因此不属于当前已可证明的最小补丁。

结论：可进入严格 GPU 配对实验的候选设计与 CPU 证明已完成；**性能问题尚未修复验收**。目前没有足够证据建议直接采纳裁剪源码。
