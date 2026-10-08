# Tree-GRPO 性能修复独立审查

初审时间：2026-10-08 06:19:42 UTC。审查基线 `6efb60466a233f70bf8f50de26df32012f49b4bc`，初审冻结源码 `0150ee6891e6deb7ff616fd9bf416de42526e029`，最终补修源码 `48a59da5f30a6368e296137281fff1d620149d04`。独立静态审查覆盖 7 个生产文件及新增 driver CPU fixture；补修只核对 F1 的29行增/1行删和最终本地包身份。只读取源码、diff 和已有材料并校验本地文件 hash，没有执行测试、SSH、GPU、checkpoint 加载、提交、训练管理或其他 agent 联系。报告写在源码仓库之外。训练启动与审查并行，本报告不授权取消、重启或热改已启动作业。

## 结论

初审发现 1 项 P2 安全证明缓存缺陷 F1；已在最终源码 `48a59da` 静态核对关闭。最终源码没有本审查确认的未关闭新增 actionable bug。未发现本次 diff 改动 Tree/PPO objective、5leaf 继承/权重、三项 guards、实际写入/skip/credit 或 scheduler 契约。性能优化有静态依据，但普通完整 outer ≤420s 仍须用新作业实际阶段 wall time 证明。最终本地 V12 packet manifest 和全部762个登记文件 hash 已核对，7个核心生产文件与 clean `48a59da` checkout逐字节一致；实际远端启动/迁移仍由 root 观察，独立审查未冒充观察到启动成功。

## 发现 F1 — P2，最终48a59da已静态关闭：同对象嵌套变造仍命中 μ 验证缓存

位置：`source/search_r1/baselines/policy_safety_runtime.py:43-56,189-193`。`_ActorProbabilityValidation` 保存 `meta_info` 与 `non_tensor_batch` 的引用，`matches()` 仅检查字典是同一对象；tensor 身份与 `_version` 不会因为 Python 字典、object-array 元素或嵌套 ledger 被修改而变化。

触发方式：先正常执行 `validate_actor_probability_profile(..., return_context=True)`，随后在原 `DataProto` 的 `tree_raw_projection` / `tree_behavior_mu_binding` / `tree_uid` / `trajectory_uid` 或 μ runtime provenance 中修改嵌套字段，再将同一 data 与 context 传入 `PolicySafetyRuntime`。`shared_validation` 仍为真，完整 `validate_behavior_mu_batch()` 被跳过；只保留 tensor 对比和布尔 corrected marker，旧版第二次完整验证本应拒绝的 row/source/provenance 变造被接受。仅检查现有 binding hash 字段也不能补足，未重新封 hash 的嵌套修改同样需要被检测。

当前生产窗口为 `dp_actor.update_policy:252` 验证返回到 `:289` 创建 runtime，中间经过 `DataProto.select`、`TreeRankAudit` 构造和 `batch.split`。这些当前调用静态看不修改 μ ledger/provenance，因此没有证据证明新训练已经受此缺陷影响；缺陷在缓存的承诺与可复用 API 边界，适合最小修复，不是已授权启动的停止依据。

建议：context 创建时保存关键 metadata 和完整 raw/μ 绑定、row identity 的独立深拷贝或 canonical 内容摘要；`matches()` 对实际内容作类型严格的快照比较或重算摘要，保留 tensor identity/version gate，内容改变就走已有完整 validator 拒绝路径。应覆盖 bool/int 和 signed-zero，不恢复逐 leaf 的 tokenizer serialization。将 context 限于单次 update、可选一次性消费，有助于收窄后续误用。

关闭核查：最终 `policy_safety_runtime.py:47,51-84` 对实际 metadata、完整 `tree_raw_projection` / `tree_behavior_mu_binding`、`tree_uid` / `trajectory_uid` 和当前 runtime provenance环境内容计算 canonical SHA；`matches()` 必须重算一致，格式错误或内容变化回到完整 μ validator。不是比较调用方提供的 binding hash。JSON编码区分 bool/int 和 signed-zero，原tensor身份/version gate保留；没有恢复每leaf tokenizer.to_str。`ACTOR_REPAIR.md` 补充实现方4项CPU变造检查结果，分别 raw selected logprob、μ action receipt、metadata runtime provenance和trajectory UID，均在native forward前拒绝；独立审查未重跑。

适用前提：tensor `_version` 能检测普通 Torch 原地写入，但 `.data` alias 或 CPU NumPy view 的写入可能不增加原 tensor 的 version；当前生产窗口未见此类输入写入，tiny CPU 原地变造证据不覆盖该情况。若承诺对任意存储写入均防篡改，需要实际 tensor 内容比对，不能把 `_version` 描述成所有写入的完整证明。

既有 standalone/fallback边界：`validate_actor_probability_profile` 检查 corrected temperature=1，`PolicySafetyRuntime._prepare` 独立路径的 μ validator本身不复查temperature。最终fingerprint能令temperature变化失去cache命中，但raw-only fallback不会因为temperature单独变化而拒绝。该边界原实现已存在，当前生产窗口没有temperature改写；不列作本次diff新增P2，不作为启动、重建或停止依据。

## 已核对的不变量

- tokenizer/action scope：新 emission 仍执行逐个 native prefix decode；action 缓存是独立深拷贝，命中采用类型严格递归比较，修改/reseal 后回到 hash 与真实 UTF8/semantic 重建。`check_runtime()` 检查真实 file/path、backend 对象、公共 encode/decode/convert 方法、decoder 与 runtime flags；显式 scope 成功返回和 μ compose/validator 完成前检查完整 backend。persistent backend 变造不能以 action-cache 的 hash 或容器身份被接受。controller/rollout 的正式调用均有 boundary scope，继承、奖励与 audit 共享当前 outer 的 action proof。
- UTF8：JSON string hash 按稳定增量字符与 pending replacement 后缀拼接，最终完整 bytes replacement decode 校验仍在；特殊 token、escape、未完成尾字节的 proof 格式沿用。新增 fixture 涵盖这些路径，但审查没有独立执行该 fixture。
- monitor：`measure:319-337` 仅复用 allowed 的 after 结果；committed、全部参数 id/version、固定 monitor tensor id/version、training mode 必须匹配，并经所有 rank MIN 同意后共同跳过 forward。一次真实 after 写后 measure 和最后一次 after 均保留。`_measure` 初始 native repeat、finite/admission、publish 失败路径保留；真实 native call、repeat call 和 reuse 日志分开记录，没有把复用伪造为实际 forward。固定 batch、零 dropout 与相同参数是这种复用的前提；CPU toy 不证明真实 FSDP cache 命中率。
- 数值：只在 reporting 中将原 `.item()` Python float 累加换为 detached FP64 device 累加后集中传回 CPU；原 FP32 scalar 精确提升到 FP64，metric mass 数值不变，loss/graph/gradient accumulation 不受该路径影响。finite 扫描仍覆盖每个浮点参数和 Adam state tensor，按设备聚合并改 chunk 大小；CPU Adam scalar 同样检查，新增首次写入前检查。preclip tensor 布尔表达式与原两个条件的逻辑一致，原 FP64 global preclip 与 measured postclip 保留。
- Tree/PPO/计数：`objectives.py`、`tree_fresh200.py`、guard/scheduler implementation 未修改；5leaf `[3,2]`、继承原 sampled tokens/μ、equal-leaf token 权重、row denominator、DualClip c=3、sample KL=.2、negative tail=.01、exp 前 bound=30、clip=1、20 opportunities、57 warmup 未改。本次 diff 未更改 write/skip ledger、write 后 refusal 的实际计数或每 credited outer 一次 scheduler 规则。
- native5 continuation：硬编码 parent3335、source6efb604、config5e530fc、manifest56832544、8 ranks 各100 Adam、scheduler5、sampler cursor2560/epoch0、next6。manifest 与 PT file/logical hashes、恢复后 live model/Adam/scheduler/RNG、driver/sampler digest 仍严格检查；新加祖先 receipt byte hash07c19b89和完整 provenance equality。child source 要求40位且不同于 parent；记录 executor_equivalence=False / exact_executor_resume=False。此处是源码契约，审查没有读取远端原始 native5 文件，也没有证明实际恢复已经完成。

## 性能风险与 CPU 证据局限

修复减少重复 tokenizer serialization、继承 UTF8 重扫、重复 μ validator、相同参数的 before monitor，以及 unused monitor entropy 和 scalar host 同步。正常20 writes、16个 monitor micro partition 时，理论 monitor native call 从656到352（19次可复用 before）；实际命中必须看真实日志，不能用该预算替代实测。

仍有明确源码热点：每个新 emission 的首次每-prefix native decode 为 O(T²)；每 row/projection 的完整 ledger hash/copy、μ row重构与 action snapshot 比较仍扫描真实内容；每次 Adam 后 exhaustive finite scan、每次写入的两次 FP64 grad扫描、FSDP通信、actor forward/backward、检索与IO均保留。`monitor_forward_wall_seconds` 是 host native-call 区间，不包括所有 CUDA completion/collective wait，不能据此分摊全部 actor wall time。这些是性能风险，未确证其中哪一项导致旧3335耗时。

`DRIVER_REPAIR.md` 报告6项 CPU通过和 tiny5leaf scoped比 separate calls 的 synthetic成本；`ACTOR_REPAIR.md` 报告修正旧 native_stopping fixture 后 targeted7/7、tiny CPU20 SGD writes/19 reuse/44 native calls、标准 tensor mutation失效及 CPU Adam NaN拒绝。上述是实现方材料，driver fixture源码已查看；本审查未独立跑测试，且这些证据不覆盖8rank真实 FSDP/GPU、native vLLM、真实资产、metadata nested mutation或新run的速度。旧普通4–7.7min、3335约47.3min是任务提供的观测，不是本审查重新测得。

速度验收仍待新run的普通完整 training outer 实际 wall time，目标每个≤420s；评测、保存和恢复成本须另列。不得以源码优化或 synthetic CPU倍率声称目标达成。

## 外部入口边界

已只读检查 `build_speed_packet.py`、`submit_speed_once.py`、`runtime-template/current_chain_runtime.py` 和最终 `package-code-only-v12`：build 强制实际 clean HEAD 与 supplied child SHA相同；模板 pin native5，next6，planned195，源码映射显式parent→child；submit使用独占intent、已验证字节、unknown不重试；metadata validator不冒充 PT重开，只有核心启动产生实际 cold equality receipt 后才记录 migrated。

最终本地身份核验：manifest SHA `6b061fd4d0aa7cc6ad6f48c13a671014e5dbfee6f630a1ca80d94785cccdbe5e` 与 `deployment.json` 一致；762个manifest登记文件全数hash一致；7个核心生产文件和当前clean源码逐字节一致；`production_contract.SOURCE` 为48a59da完整SHA，dispatch script SHA `abc94d4df6ac1902864bdde343f0bd6d63ff6d645ce357ed9442cca99e1e9dad` 与deployment一致。release为`cdd3fa016c6c024c68fdcc88a57af402621127341af7e3a0b66afa2e77e9ed13`，authority为`04ff97a35f71435c13541d40d1f176c311854697fd59196bb696e9712561fbc3`（后两项来自deployment，未单独重算）。最终实际child job启动身份/实际冷迁移receipt与normal outer速度尚未观察，本报告不把本地包一致性写成远端启动通过证明。
