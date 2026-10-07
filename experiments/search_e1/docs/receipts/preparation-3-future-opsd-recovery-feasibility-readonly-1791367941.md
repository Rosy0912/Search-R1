# Preparation-3：未来 OPSD 失败后的保真恢复可行性审查

**结论：现有冻结程序不能通过普通重启恢复 OPSD。** 它只在 OPSD update 1 / 5 保存可用于独立恢复的权重与 Adam/RNG 备份，没有恢复入口，也未保存全部控制与指标状态。完整 update 1 / 5 bundle 可以成为保真恢复的起点，但必须验证 bundle、数据顺序、原始选择基线和恢复边界；这不是已经恢复成功的结论。

本次只读观察时间为 Unix 1791367941.0902784（2026-10-07 18:12:21 CST）。实际 run 为 `914bfff786c84ede8d82414dff65f60e`，controller 2459676，阶段仍为 `OPSD100_PREPARATION3_FULL_FRESH_TRAJECTORIES`。原始 mined pairs 和 `runs/opsd` 均尚不存在；**当前没有真实 OPSD 更新、实际 checkpoint、恢复试验或五更新验收**。本审查没有运行 GPU 请求、修改代码/配置/进程或编辑 publisher 主文档。已通过的 tail loss/init 审查不在本次重复测试。

## 实际源码和路径

根目录 B：`/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007`。

- 冻结训练：`B/runtime/opsd-preparation3-xformers/search_e1/opsd.py`，SHA256 `61df9cc630bca7466f3edfb06da18ad92ddff1c713bf7c6c4a104ba88672c037`。下面行号对应本次实际读取的该文件。
- 控制器：`B/source/preparation-3-xformers/pipeline_preparation3_xformers.py`，SHA256 `85c09c3bd5a58a59509b63e7fcc70aeb0cc1b6c139cc11e4ab6a039f4fe48bb4`。
- 外部 V3：`B/source/watch_acceptance_v3.py`，SHA256 `24aabe7d628ad9e87e97df78b7c801157c0ccf3fda2d549fe3cd9c79f8454595`。
- 原完整 pairs：`B/runs/preparation-3-pairs.jsonl`；native output 固定为 `B/runs/opsd`。
- 源 100：`B/assets/archive100`，transfer accepted SHA256 `08b005d8788fbbbc6c57cf40af4c08202b59c0d89160a6f030fad8d555fc007d`。这是新 OPSD 的冻结 base/teacher，**不应把源 GRPO 的 Adam 当作 LoRA OPSD Adam 恢复**。

## native 实际保存了什么

以下为源码的未来输出约定，不是当前已存在的文件。

| 文件/目录（均在 runs/opsd 下） | 保存内容与时点 | 恢复用途和限制 |
|---|---|---|
| `config.json` | 第 112 行记录原 CLI、usable/skipped pairs、world_size | 不包含 pairs/tokenizer 内容 SHA、shuffle 顺序或 cursor |
| `best_adapter` | 第 116 行保存 baseline adapter；后续 epoch EM 严格大于 best 时才覆盖 | 保存选择权重；baseline EM 本身此时没有写入文件 |
| `adapter_update_1`、`adapter_update_5` | 第 168 行仅 rank 0 保存 PEFT adapter；V3 要求存在 `adapter_model.safetensors` | 不保存 base/teacher/tokenizer；其身份须绑定 source100 与固定环境 |
| `update_1_rank_0_state.pt` … `update_1_rank_7_state.pt`；update 5 同样 8 文件 | 第 169 行保存 **optimizer、torch_rng、cuda_rng、updates、rank** | optimizer 含 state、每参数 Adam step/exp_avg/exp_avg_sq 和 param_groups；RNG 为每进程 Torch CPU 与当前本地 GPU，不是 Python random |
| `live-update.json` | 每次真实 optimizer step 与 native finite/change/Adam 审计之后记录 rank evidence、opsd_updates、source100/display step | 运行证据；不是完整恢复 checkpoint |
| `five-update-acceptance.json` | 第 177 行在 update 5 写 native evidence | 不等同于外部 V3 的 CPU 权重/Adam/显存验收 |
| `adapter_epoch_1`、`history.json` | 第 188–193 行在完整 epoch 验证后保存 adapter、baseline/end EM 和全 epoch loss | epoch adapter **没有 Adam 或 RNG**；history 不在前 1/5 更新时保存 |
| `merged` | 第 195–204 行重新加载 selected best adapter 合并入 source base | 推理交付权重；不能据此恢复训练 Adam |

八 rank 是复制 LoRA 模型、共享 all-reduced 梯度，不是 FSDP 的八模型 shard；rank 0 adapter 可加载到所有 rank。过去其他 rank 的 adapter 没有单独保存，因此不能把这些文件称为八 rank 完整模型快照。恢复时必须在所有 rank 加载同一 adapter，并验证恢复后的参数一致性。

此 native 没有 scheduler 或 GradScaler 对象，所以并非漏存了一个实际在运行的 scheduler/scaler。梯度只在 update boundary 保存之外的内存中；从完整 post-step checkpoint 恢复时下一批会重新 zero_grad，无需假称可以恢复半批 accumulation 梯度。

第 167–170 行的写入先保存 adapter 和八份 state，再 barrier；native evidence 随后写。PEFT/torch.save/native write_json 不是具有单一提交标志的原子多文件 checkpoint。失败时可能留下存在但不完整的文件；**不能只看文件名或 mtime 选最近 checkpoint**。

## 缺失的恢复语义

1. **恢复入口和状态加载。** 第 92–100 行每次重新 seed、构造初始 LoRA/Adam；第 119 行将计数清零。CLI 没有 resume。控制器第 93–95 行拒绝已存在 output，日志采用 exclusive create；直接重启整条 pipeline 也不会续跑，且不应绕过保护覆盖旧成果。
2. **Python RNG、epoch 和数据 cursor。** 第 124 行 random.shuffle；保存包没有 Python random.getstate、epoch、start 或排列。原 config 的 usable/skipped 计数不等于保存了实际 usable example 顺序。
3. **数据/编码身份。** 未将 pairs byte SHA、usable example 顺序 digest、tokenizer/encoding skip 决策写入每个 checkpoint。独立恢复必须绑定完全相同的完整 mined pairs、原 tokenizer、8192 context 和编码规则，不能用未完成准备 prefix 或重新 mine 的不同输出替代。
4. **baseline best EM、history 与训练 loss 累计。** baseline EM 在第 114 / 118 行只留在内存，history 到 epoch 末才写。原始 best_adapter 权重不能直接给出该 EM。每 rank 的 total 累计 loss 未保存；rank 0 日志只能恢复 rank 0 的局部累计，不足以还原完整 epoch loss。
5. **审计参照和恢复 provenance。** audit_initial 是第 121 行内存中的原始 tracked LoRA tensor，未保存。恢复不能把 checkpoint tensor 悄悄当作原始 source100 初态，也不能将 discarded failed-attempt updates 当作从 checkpoint 继续的计数。
6. **完整性与参数映射。** optimizer state 数字 ID 不含 parameter-name 映射；必须保持同一冻结构造顺序、PEFT/Torch 版本、target modules、shape/dtype 和 param_groups，不能只匹配 step 数字。V3 当前不验证保存 RNG 的格式或有效性，恢复前需另外检查。

## 若实际失败，最小保真恢复设计

这是方案，不是现在实施的新功能或试跑。

1. **先保存故障和所有权证据，再选择 checkpoint。** optimizer.step 发生在 checkpoint 保存之前；保存失败不意味着没有执行更新。若没有完整 u=1/5 bundle，不能自动把原尝试标为0更新或称其已恢复；只可据真实 evidence 标明丢失状态，并由根执行层选择有明确 provenance 的新尝试。 原 runtime、run/log/parts、native output、内存 observer 和验收 evidence 保留。只根据最近完整的 update 1 或 5 adapter + 全八 rank states 选择 u；CPU 校验所有文件可完整读取、finite、step=u、rank/updates 正确、Adam moments 与 param_groups 全 rank 逐项相同，并记录 SHA。rank RNG 应逐 rank 保留，不要求它们彼此相同。若 update 5 部分写入失败而 update 1 完整，只能恢复 u=1，不能声称恢复了 u=5。
2. **独立恢复副本和输出。** 不修改冻结 runtime，不重启完整 generation/mining/GRPO，不覆盖原 `runs/opsd`。使用独立控制/恢复模块及新 output，仍从相同 source100 构造冻结 base 和原 LoRA 配置，再加载 adapter、原 optimizer param_groups/全部 moments 和对应 rank 的 Torch CPU/GPU RNG。构造/读取/必要验证操作结束后才恢复 RNG，以免这些操作消耗已保存的随机流。科学函数、单 epoch、lr1e-5、r16/alpha32、clip10、chunk32、8×accum4=32 均不变。
3. **恢复原 epoch 排列和 next cursor。** 当前计划是一 epoch。若 usable 数为 N、全局 batch G=32，post-step u 的 next_start 是 `min(32*u,N)`，用于 **原 usable examples 的同一排列**，不是原始 pairs 的行号。必须重建原 Python seed42 的 shuffle 路径并确认其所有随机消费/环境版本与原运行一致；项目源码的验证是 greedy，显式 Python 随机调用仅为该 shuffle，但 checkpoint 并未记录第三方初始化期间的 Python 随机状态。这个重建前提未经证明时，只能说权重/Adam 恢复，不能保证 exact data-order continuation。最后不足 32 的原尾批必须仍按原实际 batch 保留；到 epoch 边界后不能再 shuffle/多跑一轮。
4. **保留 original baseline 和选择规则。** 若原 baseline EM/history 有完整持久 evidence 则恢复它。若早期失败时数值未保存，需要使用保存的 baseline adapter、相同原 512 题和原 native retrieval/evaluation 条件重建该数值，随后恢复 checkpoint RNG；应记录是否与原 evidence 可核验，而不是无证据声称完全相同。不能将当前 trained checkpoint 的 EM 或负无穷当作原 baseline，否则可能把原本应舍弃的有害 adapter 选为 best。缺失的前缀全 rank loss 累计不应伪造；应明确区分已知原日志与恢复剩余段 loss，不能将后者冒充完整 epoch loss。
5. **恢复计数与审计链。** optimizer 与 opsd_updates 都恢复到 u，下一个真实更新是 u+1，display step 仍为100+opsd_updates。原 initial tracked tensor须由相同原初始化/provenance重建或以原 evidence 绑定；原 update 1 adapter 如果被用于新 update 5 验收，必须保留原 SHA/来源。若故障发生在 update 3 但只有 update 1 state，恢复路径会重做2/3；必须保留原 failed-attempt 2/3 日志并分别报告实际执行与当前 checkpoint 链，不能假称这些未保存状态已经恢复。
6. **独立恢复验收，而非沿用旧 continuous 标签。** V3 第 9–10 行硬绑定原 output，inspect_optimizer 硬要求 step5，第 125 行写 continuous_run=True。它没有 resume provenance、RNG、cursor 或恢复前后值校验。恢复副本需有独立外部 checker，绑定旧 checkpoint SHA、新 run、restored step、实际后续更新以及全部 CPU adapter/Adam/显存 evidence；诚实记录 resumed，不能把原 V3 文件直接移到新目录当作 uninterrupted 证明。仍需真实有限/改变权重的更新和全 8 卡直接 memory.free 监测覆盖初始化/恢复及第五次累计更新之后。这里不改变当前正在等候的 V3。
7. **epoch 完成与 final evaluation。** 恢复仍完成同一原 epoch 的剩余批次和原 512 验证、同一 best-selection/merge、同一七基准 final test。若故障只发生在已验证完整 epoch 之后的 merge/evaluation，优先复用完整 selected adapter/merged 权重和证据；无需重新训练。原 epoch adapter 本身不能证明更新5之后的 Adam 已恢复。

## 失败位置对应的可恢复范围

| 未来失败位置 | 可能可用状态 | 不能声称的内容 |
|---|---|---|
| baseline 初始化（0更新），或完整 update1 bundle 之前失败（可能已有未保存更新） | 没有完整训练 state；仅同一 source100 + 已完成固定 full pairs 可作为独立新尝试的起点，不能称完整续训 | 把已执行但未保存的 optimizer step 当作0更新，或声称恢复 Adam/原进度 |
| update1 完整后、update5 完整前 | 完整 u=1 bundle；从原 cursor32继续 | 恢复了故障当时2/3/4的未保存权重或半批梯度 |
| update5 完整保存、但 native/V3 evidence 写入或读取失败 | u=5可能可恢复；checkpoint completeness与外部验收分开核对 | 文件存在即五更新验收已通过 |
| update5之后、epoch结束前 | 最后完整训练state仍可能只有 u=5 | 最近epoch adapter / live-update数字就是完整Adam |
| 完整epoch/selected checkpoint之后，仅merge或最终评测失败 | 完整模型选择/指标证据可支持只恢复交付/评测阶段 | 需要从source100重训、或无历史证明的best selection已保真 |

**边界：** saved weight/optimizer/RNG 完整且可重建控制顺序，是保真恢复的必要前提，不等于数值 bitwise 重放保证。新硬件、GPU kernel、依赖/精度、retriever或数据变化都须另行记录，不能归为算法收益。本次没有实际 checkpoint 可以加载，未新增 resume 功能或执行任何恢复。

详细实际源码 SHA、当前不存在的 output/pairs 与原科学配置见同名 provenance JSON：`preparation-3-future-opsd-recovery-provenance-1791367941.json`。
