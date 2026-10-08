# JOB3332 实际概率差异：只读代码诊断 v1

日期：2026-10-08。科学源码 HEAD：`2ddbb0a1619ce7cde28b4c5be110ffe5bd11c963`。范围：官方 Qwen2.5-3B-Instruct，512×5 首批 2560 轨迹，strict joint-action **SUM**，actor rankmicro2 / old rankmicro4 / ref rankmicro4。没有执行测试、SSH、GPU 操作、训练或新模型前向；没有修改实验代码或创建批准。

## 判断

**已定位到真正的模型前向/logits 差异；尚未定位到一个被原始张量证明的具体坏算子。不能以“BF16 通常有误差”或 `max + margin` 宣告通过。** 最有依据的待证假说是：不同打包上下文及 HF/vLLM 不同的数值执行路径，使相同条件概率得到不同的上游 logits。它包含明确的实现差异，不等于已证明误差幅度合理。

发现一个确定的诊断混杂：**complete old 在 `_balance_batch` 重排前计算，current 在重排后计算**。现有 `current2 − old4` 同时改变 micro size、微批的同伴/顺序、可能的执行 rank；它不是隔离 micro2/micro4 内核效应的对照。此问题不表示 source→action 映射错位：现有校验按 `(source, step)` 对齐，且核对了散射后的概率。但将差异直接归因“micro2/4 的 BF16”是不成立的。

静态检查没有发现固定四槽、单 token dummy、roll-next-label 或位置重置导致跨样本注意力的直接代码 bug。保存的 Transformers 实现会把 **每个 `position_ids == 0`** 都作为 FlashAttention 分段起点，因此连续的单 token dummy 也分别有边界。不过现有观察器**没有直接保存内部 `cu_seqlens` 或所选 CUDA 内核**，不能把外部输入一致升级成实际内核边界被独立观察。

## 证据层级与材料限制

本报告读取的是本地真实 JSON、带明确 `PROJECTED` 标记的投影、科学源码、运行层源码和已保存的绑定依赖源码；本地材料目录中 `*.pt` 数量为 0。`METADATA-COLLECTION-RECEIPT.json` 明确写着 `raw_pt_copied=false`、`large_prefix_and_action_arrays_copied=false`、`raw_file_sha_independently_read=false`。故下面的数据结论属于**已保存真实材料的报告/投影证据**，不冒称本次独立加载或重算 H100 原 PT。

依赖代码保存在 `outputs/gigpo-e1-policy-safety/PROBABILITY-MISMATCH-BOUND-SOURCE-v1/`。本次对 HF Qwen2、HF FlashAttention 工具、vLLM Qwen2、vLLM FlashAttention 后端、vLLM sampler、vLLM attention selector、vLLM 自带 FlashAttention init 七个文件计算本地 SHA256，均匹配 `FRESH200-EVIDENCE-REPAIR3325-BINDING-CANDIDATE-v1.json` 的 `asset_files`。绑定的 dist-info 路径给出 Torch 2.4.0、Transformers 4.46.3、外部 flash-attn 2.7.4.post1、vLLM 0.6.3；**vLLM 自带 `vllm.vllm_flash_attn.__version__` 则为 2.6.2**。官方模型 config 中的 `transformers_version=4.43.1` 是模型保存字段，不是本次执行库版本。该 3325 绑定描述了 evidenceclone1 的来源；JOB3332 的实际 identity 记录本次 native binding SHA 为 `2c161a96fd4f1313e1c5de935a98319bf85b9247b4c7632c73afac282ed500ec`。本报告没有宣称读过该 SHA 对应的 H100 BINDING 原文件或实时环境。

主要数据来源：[首批材料目录 receipt](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-instruct200/job3332-first-probability-material-v1/METADATA-COLLECTION-RECEIPT.json)、[实际 identity ready](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-instruct200/job3332-first-probability-material-v1/probability-identity-ready.json)、[绑定来源](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-instruct200/FRESH200-EVIDENCE-REPAIR3325-BINDING-CANDIDATE-v1.json)。

## 实际差异及其含义

| 项目 | 保存材料结果 | 本报告能作出的判断 |
|---|---:|---|
| 轨迹 / actions / retained tokens | 2560 / 5478 / 468211 | 完整首批的报告范围 |
| token、prefix、position、mask、engine prefix mismatch | 全部 0 | 已有结构检查未报告外部输入或 action 绑定错位 |
| perturn HF 原算 / 相同 perturn repeat | 最大差 0 | 原相同上下文可重复；不证明换打包也一致 |
| complete old / perturn old | token max 0.9616069794 | 369 / 5478 actions 非零；340 个位于 step 1 |
| complete current / complete old | token max 0.8290481567 | 70 / 5478 actions 非零；5408 个完全相同 |
| current / old 的 action joint SUM | 最大绝对差 1.1738109589 | source 638 / step 2；不是可忽略的初始 ratio 扰动 |
| complete HF / vLLM | token max 2.4754629135 | 最大差发生在真实已选 target 的上游 logits |
| normalizer / FP32 selected-logit−LSE | max 0.0000038146973 | normalizer 不能解释约 2.475 的差异 |

对投影中的 5478 条 action 检查做了普通 JSON 统计。current−old 非零 action 分 step 为 19、33、16、2，最大 token 差在 source 223 / step 0，178 retained tokens、205 sampled tokens；joint 最大差在 source 638 / step 2，111 retained tokens、112 sampled tokens。fullold−perturn 最大 token 差在 source 213 / step 1，33 retained tokens、34 sampled tokens。这些是定位原 PT 的键，而不是本次已读取相应原 tensor 的声明。[逐 action 投影](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-instruct200/job3332-first-probability-material-v1/full-old-versus-sampled-prefix.PROJECTED.json)

HF/vLLM 最大差的已记录分解：rank 2，source 661 / step 1 / token index 1，target 26865，request 498。HF selected logit 29.875，LSE 36.0025787354，logp −6.1275787354；vLLM selected logit 31.375，LSE 35.0271148682，logp −3.6521158218。于是 selected-logit delta −1.5，LSE delta +0.9754638672，合成 logp delta −2.4754638672，与实际 −2.4754629135 相差约 9.54e−7。**差异在 softmax 前已经存在，单纯把 HF 输出 logp 转 FP32 或替换 cross-entropy normalizer 不会恢复丢失的 logits。**[真实 normalizer 材料](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-instruct200/job3332-first-probability-material-v1/actual-decisive-normalizer-evidence.json)

## 一、HF 微批和重排的真实路径

`DataParallelPPOActor.compute_log_prob` 在第 453 行统一调用 `actor_module.eval()`，第 474 行按 `data.meta_info['micro_batch_size']` 固定切片，第 479 行进入同一 `_forward_micro_batch`。运行层 `calibration_raw_current_log_prob` 第 451–459 行只 deepcopy batch、把 micro 设置成 actor 的 2、再调用 `super.compute_log_prob`。因此 current 这个名称**不代表 train-mode、含 backward 的实际更新前向**；它是与训练 micro size 一致的 eval/no-grad 概率观察。真正更新入口第 495 行会调用 `.train()`。

`calibration_driver_v1.py` 第 498 行先对 `generated` 计算 complete old；第 504 行 `_balance_batch(batch)`；第 512 行 ref；第 517–520 行才对 balance 后 deepcopy 的 batch 计算 current。`ray_trainer.py` 第 774–785 行按轨迹总 attention length 构造八个等数量分区并 `batch.reorder(global_idx)`，Ray 再按位置分发。按 source 比对保证语义身份，**不会消除同一 source 的原本微批同伴、flat row 和 packed offset 已变的事实**。

source 链：[HF scorer](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-fresh200-cpufixturefix-20261007/verl/workers/actor/dp_actor.py:434)、[current 观察入口](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v14-evidenceclone1/calibration_worker_v1.py:448)、[实际 old/balance/current 顺序](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v14-evidenceclone1/calibration_driver_v1.py:494)、[balance 重排](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-fresh200-cpufixturefix-20261007/verl/trainer/ppo/ray_trainer.py:774)。

固定 prefix scorer 以 trajectory 为外层单位，每条轨迹固定四个 action slot。micro2 变为 8 flat rows，micro4 变为 16；每微批 `max_prefix`、`max_action` 根据实际有效长度确定。完成 remove-padding 后模型输入是 `[1, total_nnz]`。同一 action 的有效 token/position 不变，但整次 forward 的 `total_nnz`、最大序列长度、序列数、packed offset 和矩阵 M 维可以变。稀疏不一致可能由矩阵/attention 执行路径的特定分支或尾部产生，**现有 70 条非零数据不足以证明是哪条分支**，也不能只从 shape 变化断言一定存在某个 cuBLAS kernel 切换。

## 二、position、padding、dummy、causal 和未来 token

`prepare_action_prefix_micro` 第 175–204 行：active prefix 左补 PAD，action 右补 PAD，有效性按 length/mask 判定；dummy 只有一个有效 EOS prefix、action mask 为零。位置由 attention cumsum 计算，每个有效 row 是 `0,1,…,row_length−1`。第 193–194 行构建 source/output scatter 索引，第 222–225 行只选真实 action token 的概率回到原 timeline。[prefix 准备/散射](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-fresh200-cpufixturefix-20261007/search_r1/baselines/action_prefix_forward.py:175)

`dp_actor.py` 第 150–156 行用同一 mask unpad IDs 和 positions，第 172–175 行传 `attention_mask=None`、实际重置位置、`use_cache=False`。**sp=1 没有运行 verl 自己的 Qwen monkey patch**：`fsdp_workers.py` 第 377 行要求 `use_remove_padding and ulysses_sequence_parallel_size > 1` 才 patch。因此应审绑定的 Transformers 4.46.3 原实现。[条件](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-fresh200-cpufixturefix-20261007/verl/workers/fsdp_workers.py:373)

绑定的 Qwen2FlashAttention2 第 443–453 行确实把 `position_ids` 传给 `_flash_attention_forward`，`self.is_causal=True` 在第 263 行。内部第 271 行在长度不为 1、位置序列有下降时进入 varlen；第 171–178 行以所有位置 0 的 index 加总长度形成 q/k 相同的 `cu_seqlens`，max length 为 `max(position)+1`。因此 active 长序列后的下一 active/dummy 从 0 重置时存在下降；连续 dummy 的 `0,0` 虽不构成负 diff，但**已进入 varlen 后，每一个 0 仍各自成为起点**。这些构造下没有显见漏掉 dummy 分段的逻辑。[真正绑定的 attention 路径](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-policy-safety/PROBABILITY-MISMATCH-BOUND-SOURCE-v1/transformers-modeling_flash_attention_utils.py:165)

rolled label 在 packed row 最末 token 会指向下一 row，但 retained action t 的评分位置是该 token 的前一个位置。`raw_hf_observation_v1.py` 第 123–127 行明确按 `packed_start + prefix_length − 1 … +action_length−2` 检验 label=action。一个 action 的最后有效 token 对应前一位置，row 最后 token 的跨 row label 不参与 action scatter；dummy 没有 action support。第 133 行同时明确 `internal_cu_seqlens_tensor_observed=False`。故可以说已校验预测 label 的外部 packed 边界；不能说读到了真正 attention kernel 的边界。[观察器边界](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v14-evidenceclone1/raw_hf_observation_v1.py:113)

perturn HF recompute 使用完整 sampled completion；complete action HF 使用 retained action（例如 source223 为 205 sampled vs178 retained），还把同轨迹其他 turns 打进同次 forward。vLLM 对每一预测只处理当时可见 prefix / cache。正确 causal mask 下，后来的 token 在数学上不会影响较早预测，但三者的**总执行长度和 kernel 调用形态仍不同**。fullold−perturn 因此也不是“同一模型同一完整输入重复”。不能把未来 token 当作真实 conditioning 不同，也不能在没有边界证据时排除 mask 错误。保存代码支持 causal 正确，原 PT 可再核 positions 和分段推导，但没有 Q/K 或实际 attention 输出来直接证明 causal 内核行为。

## 三、RNG、eval/dropout、FSDP dtype 和 load

保存 receipts 报告所有 perturn repeat 的 RNG/完整状态未变、最大差 0；所有 current-neutral 八 rank 都 `complete_state_unchanged=true`，没有 Adam 写入。运行层第 383–390 行读取实际模型 config、nn.Dropout modules，要求 Qwen2 attention_dropout=0 且全部 dropout 项为 0；identity-ready 保存 `actual_dropout_neutral_train_eval_verified=true`。这些联合证据使**随机 dropout 或 repeat 改了权重**很不可信。注意当前观察是 eval；真正 train/gradient-checkpointing 的数值路径还没由此直接观察。[dropout 实查与保存](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v14-evidenceclone1/calibration_worker_v1.py:383)

“BF16模型”不等于 actor 的 master 参数和所有 buffer 都是 BF16：`fsdp_workers.py` 第 364–368 行未显式 model_dtype 时 actor 从官方权重加载为 FP32，ref 加载为 BF16；第 420–424 行 actor FSDP forward param dtype BF16、reduce dtype FP32、buffer dtype FP32；ref 第 426–427 行取消 MixedPrecision。`dp_actor.py` 第 142 行再用 BF16 autocast，真实 HF normalizer 输入报告 BF16。初始化保存 official-BF16 parameter digest，Adam 尚空；这表明 BF16前向是实际路径，不是据模型标签猜测。[load/混合精度](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-fresh200-cpufixturefix-20261007/verl/workers/fsdp_workers.py:360)

vLLM 同步第 71–75 行把 FSDP state_dict 交给 `sync_model_weights(..., dtensor)`；Qwen loader 将 q/k/v 与 gate/up 复制进 fused projection，转换为目标 param dtype，tied embeddings 时跳过 lm_head，RoPE inv_freq 不从 HF state_dict 复制。保存官方配置 `tie_word_embeddings=true`，vLLM Qwen2 的 lm_head 确实指向 embed_tokens。因此这些 skip 有相应模型实现依据，没有发现清晰的 wrong-name/错 shard 静态 bug。TP=1 也消除了 TP 切片/归约作为首要解释。但现有本地材料没有**vLLM 同步完成后的全参数及 buffer 值 digest**；“HF官方初始化准确”不等于已逐参数核实 vLLM 同步结果。[同步](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-fresh200-cpufixturefix-20261007/verl/workers/sharding_manager/fsdp_vllm.py:69)、[Qwen loader](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-fresh200-cpufixturefix-20261007/verl/third_party/vllm/vllm_v_0_6_3/dtensor_weight_loaders.py:151)

另一个应保留的边界：HF actor/ref 的 buffer digest 不同，构建时 actor `.to(float32)` / ref `.to(bfloat16)` 可影响 nonpersistent inv_freq；vLLM 则自行生成 RoPE cache。它不解释同一 actor 的 old/current 差异，但不应把 fixed reference 视作已独立证明数值实现完全相同。具体 buffer 值/精度需原 pause state 只读核对。

## 四、HF 与 vLLM 相同 backend 名字下的不同实现

有证据支持的差异包括：

| 算子/调用 | HF 绑定源码 | vLLM 绑定源码 | 证据解释 |
|---|---|---|---|
| Q/K/V | Qwen2 第377–379行三个 Linear | 第154–155行 fused QKV后split | 浮点实现/矩阵形态不同，非参数形状错位证明 |
| MLP | 第223行 separate gate/up、silu、mul | 第65–80行 merged gate-up、SiluAndMul | 融合与rounding步骤不同 |
| residual/norm | 第633、639行显式add；RMSNorm第78–81行FP32统计后cast | 第208、218行 RMSNorm 同时接收 residual | 融合实现可改变中间舍入 |
| RoPE | FP32频率/三角函数，再转x.dtype；第206–207行逐乘相加 | 自建cos/sin cache转BF16；CUDA custom op | 位置相同仍未证明旋转值/融合舍入相同 |
| attention | 同长度Q/K完整retained序列 varlen、无cache | prefill varlen；普通decode第736行 `flash_attn_with_kvcache` | 同FLASH_ATTN不等于同个前向kernel |
| attention 库 | 工具第28行从外部 `flash_attn` 导入，绑定2.7.4.post1 | backend第23行从 `vllm.vllm_flash_attn` 导入，自带2.6.2 | 即使都叫FlashAttention，函数命名空间和库版本也不同 |
| lm_head | 所有输入 token 的 hidden计算 logits，第1179行 | 先selected_token_indices prune，再linear | 矩阵M维不同 |

源码：[HF Qwen](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-policy-safety/PROBABILITY-MISMATCH-BOUND-SOURCE-v1/qwen2-modeling_qwen2.py:377)、[vLLM Qwen](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-policy-safety/PROBABILITY-MISMATCH-BOUND-SOURCE-v1/models-qwen2.py:195)、[vLLM RoPE cache](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-policy-safety/PROBABILITY-MISMATCH-BOUND-SOURCE-v1/layers-rotary_embedding.py:95)、[vLLM attention](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-policy-safety/PROBABILITY-MISMATCH-BOUND-SOURCE-v1/backends-flash_attn.py:674)、[logits prune](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-policy-safety/PROBABILITY-MISMATCH-BOUND-SOURCE-v1/layers-logits_processor.py:57)。

这些是“可能产生差异”的具体代码链；没有 layerwise hidden、Q/K、输出快照或真实 CUDA kernel id，所以**无法从末端 selected logits 确认第几层、哪个融合 op 是根因，也不能据此推出 1.5 logit 或 2.475 logp 在允许范围内**。RoPE backend差异不能独自解释同一HF micro变化。推论应停在“upstream、上下文/实现相关”，不越级为某个算子的已证故障。

## 五、采样 logits 与输出绑定

HF 返回的 logp 是真实 `flash_attn_cross_entropy` 分支输出 FP32；vLLM sampler 第257–271行将 logits cast FP32、按 temperature除法、FP32 log_softmax。采样实际 temperature=1、top_p=1、top_k=−1、n=1，额外 penalty/logits processor 的中立性由被动 sampler observer核查；它还对真实 `compute_logits` 与 normalizer 的 selected rows 做全词表指纹比较，不能用仅 selected scalar掩盖改了其他词的分布。[被动 sampler 观察](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v14-evidenceclone1/decisive_normalizer_v1.py:175)

最终 chain 校验把 selected runner input 与 request/seq owner 绑定，核 target ID、SequenceOutput、CompletionOutput、sampled tensor 和 normalizer值精确相等，全部有效 token报告通过。因此目前不是取了错误top1 logp、wrong target、序列输出错绑或温度未配的典型问题。normalizer残差与 cross-model分解误差分别仅3.8147e−6与5.9605e−6；它们不能成为放大整体概率容差的理由。相关 PT仍在H100；本报告确认观察器校验代码和其实际保存结果，未重跑链。

## 下一步可在原保存材料上完成的只读定位

以下是一项具体证据补全方案，不含模型前向、GPU计算、RNG恢复、采样或训练，未在本报告执行。应在授权的现有只读材料通道读取原文件，按 manifest验证 bytes/SHA，CPU map载入；不能仅再次引用投影结论。

1. 读 complete-actions 的 old/current/ref 八rank原 PT。按 `trajectory_sources` 与action slot映射，逐call保留 serial、rank、local row、packed offset、全部row length、`max_seqlen`、`total_nnz`、dummy数；核调用输入等于保存flat_micro经mask选出的IDs/positions，核每row位置0起点与 cumulative row length精确相同。
2. **首先分解 current−old**：原 PT中 `actual_normalizer_calls` 已保存每个有效 token 的 `pre_normalization_selected_logits_fp32`、`fp32_logsumexp`、`actual_normalizer_selected_log_probs` 和 `normalizer_minus_fp32_formula`（不是只有六条bounded vocab vector）。按 `(source,step,token)` 一一join，报告 `Δlogp = Δselected_logit − ΔLSE + Δresidual`，列出 source223/step0、source638/step2及全部70个非零actions的首个/最大差token。若scalar logits/LSE完全相同而logp不同，才追normalizer；若scalar已经不同，定位上游。
3. 将非零token与 packed tail、行首、长度阈值、M维余数、最大序列长度、不同rank/同伴对应。此步骤可以证实“差异集中于某packing位置/长度条件”，不能单靠相关性宣告某CUDA kernel故障。对perturn fullsampled vs complete retained同样建立表，区分未来token truncation、active filtering/重组和outer balance的变化。
4. 只读 pause/initialization原 state的parameters、local_shards和buffers，核actor BF16有效参数对官方BF16值，核RoPE inv_freq及精度。若原记录不含同步后的vLLM权重，明确该缺口；不能从sampling logp重建全部vLLM参数，也不能承诺原材料足以定位第一层分歧。

当前原材料足以完成1–3的外部布局与末端scalar分解；**不能补造当时未观察的真实 `cu_seqlens`/kernel identity 或layerwise中间值**。第一层/具体坏kernel的最终归因若仍需要新forward，必须另行作为明确授权的诊断工作；本任务不执行。

## 最小具体修复建议与不可替代项

**可执行的第一数值候选：关闭 BF16 GEMM 的 reduced-precision reduction，保留 BF16输入/参数及FlashAttention。** 如未来获得变更冻结运行实现的授权，在每个实际Ray actor/rollout/ref worker进程初始化、任何模型前向前统一设置并记录：

```python
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
```

Torch 2.4.0 的官方 API 已支持这个属性；官方数值说明指出 BF16 GEMM 的 reduced-precision intermediate reduction 默认启用，可关闭以防中间累加精度截断。[Torch 2.4 API实现](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/backends/cuda/__init__.py#L113-L129)、[该版本数值说明](https://github.com/pytorch/pytorch/blob/v2.4.0/docs/source/notes/numerical_accuracy.rst#L87-L97)。这针对的是一条具体因果候选：packed总token数/矩阵M维变化 → GEMM执行方案及中间累加截断可能变化 → 某些hidden/logits舍入变化 → selected-logit/LSE差异；它保留官方模型、训练数据与预算、采样参数、micro2/4、loss和joint SUM。它不是“提高最终softmax精度”的无效替代。

**限制明确如下：**（a）本次材料未保存该flag实时值或cuBLAS所选算法，不能把默认值当作已观察值，更不能声称已证明它造成0.829/2.475；（b）该flag只约束走PyTorch/cuBLAS的GEMM，不统一HF/vLLM RoPE、fused norm、FlashAttention本身及自定义CUDA算子，也不保证不同batch bitwise一致；（c）这是运行数值实现变更，须进入新的持久源码/绑定记录，当前JOB3332的frozen source/run不能热设；（d）不安排额外CPU/native/GPU验证job。若以后允许对原正式batch做数值诊断，应只在同一实际batch、Adam前的已有观察链记录该flag和同一action的selected-logit/LSE/packet；现有old/current前向已结束，不能伪称“无需再forward即可验证此候选”。没有明确的新forward授权时只读原PT完成上节定位。

全FP32不是当前配置下的可直接选项：vLLM 0.6.3 config允许`float32`，但保存的selector第228–232行在 dtype不为FP16/BF16时把FLASH_ATTN改为XFORMERS；不能盲设`rollout.dtype=float32`再宣称backend不变。HF现有BF16 autocast及FSDP param_dtype也会把前向压回BF16，单改master model_dtype不提升实际计算。若真正把Q/K/V变FP32，FlashAttention路径兼容性需要改变；换到eager/SDPA而继续传`[1,total_nnz]`、`attention_mask=None`会丢失按位置重置推导的block边界，必须显式构造block-causal mask或保留独立padded rows，显存/调用路径也随之变化。因此“全FP32+保持当前remove-padding/FLASH_ATTN”不是已支持的一行修复。[真实vLLM selector](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-policy-safety/PROBABILITY-MISMATCH-BOUND-SOURCE-v1/attention-selector.py:220)

局部FP32 lm_head投影也可保留数学模型，但仅把**已经BF16化的logits** `.float()`不能恢复精度；真正局部提升须禁用该投影autocast并用FP32 hidden/weight做乘法，HF与vLLM都需要一致处理。它不能修复之前36层hidden分歧，且增加投影内存/计算及改变执行实现。当前scalar记录只能证明logits已不同，不能区分末层hidden差与lm_head乘法差，所以该方案列为后续被证据指向时的候选，不假定它是最小充分修复。

1. **先修正可证的比较上下文问题。** 在未来被授权修改的实现中，把 `balance` 的唯一重排放到 complete-old 前，之后old、ref、current、实际训练都使用同一冻结的 source/order/rank分配；保留512×5、actor2/old4/ref4、strict joint SUM和现有loss/advantage。记录并核对每role的order/rank/packet identity。这是很小的调度修复，避免把重排效应叫成纯micro效应；**它本身不保证micro2/4数值一致，也不修复HF/vLLM前向差异**。本次job/source/run frozen，不能热改应用到JOB3332。
2. 若已保存scalar证据证实HF自身差异来自不同scoring packet，算法不变的修复方向是让old和current/gradient使用**同一canonical packet布局**。一种具体做法是保持外层old4接口，内部按训练的两个trajectory packet评估并concat，且全部在同一balance后顺序；训练仍actor2。这样不改joint SUM、采样分布或optimizer，但old的实际forward次数由80/rank变成160/rank，观察器及collective计数契约也须同步审查。**若要求“实际old4单次forward”也严格不变，则该方案不满足该运行契约，不能当作已批准修复。** 不存在靠改tolerance能获得相同packet的办法。
3. HF/vLLM的修复不能仅做“normalizer FP32”：实际残差已小，问题在输入logits。应在原scalar及权重/buffer核对后，只修被证据指向的执行语义（若是分段，就显式建立正确cu_seqlens；若是load/buffer，就核验并修同步；若是数值实现，就统一被证实的RoPE/norm/projection实现）。目前没有证据足以保证任一单算子补丁能消除2.475，因此不把“统一RoPE”或“固定kernel”写成已确证解法，也不修改BF16模型配置来掩盖问题。

不能用vLLM sampled old替换HF old来消去观测差异：这会改变所要求的old-policy scorer及初始ratio参照，属于算法/共同契约变更；也不能改strict joint SUM为均值、把当前max当容差、改micro配置或重新抽一批数据。上述候选均未实现。

**当前处理结论：概率一致性仍不通过；最可信待证方向是可复现的上游前向差异，确定需要修正的是old/current比较上下文的混杂。** 首批仍是POST_SAMPLING_BEFORE_FIRST_ADAM，Adam写入0；本报告没有赋予启动、批准、阈值、暂停/取消或改动冻结run的授权。
