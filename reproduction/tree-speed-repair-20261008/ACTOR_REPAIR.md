# Tree actor 性能修复

基准：`6efb604`；隔离源码：`source/`；分支：`codex/tree-speed-repair-20261008`。
本 agent 仅修改 `verl/workers/actor/dp_actor.py`、`search_r1/baselines/policy_safety_runtime.py`、`search_r1/baselines/tree_behavior_mu.py`。未修改 driver/rollout/controller/objectives；未提交、未 SSH、未下载模型、未执行 GPU 训练。提交/推送和真实运行由 root 负责。

## 已消除的确定重复

1. Actor 启动的 `validate_actor_probability_profile` 与 safety `_prepare` 曾各完整重建 μ/raw/UTF8 绑定。现在第一遍实际验证后建立进程内私有上下文，第二步只在同一个 source 对象、同组 tensor 对象且 mutation version 未变，且关键 meta/raw ledger/μ binding/row identity 实际嵌套内容的 canonical fingerprint 未变时复用；绑定当前环境六项 runtime provenance，不依赖 caller 自填 binding hash 或仅依赖 dict 对象身份。不接受 driver/meta receipt，也不跨 outer 复用。Runtime 被独立调用或任何关键内容变化时仍完整验证。
2. μ validator 曾逐 row/field `.cpu().tolist()`。现在每个完整 tensor 只转 CPU 一次，之后沿用全部原有逐行比较、FP32 scalar bits、绑定 hash、sample provenance、mask/prefix/position 检查。实际 μ FP32 bits 只组装一次。
3. μ compose 与 validate 各共享一个 batch 的 tokenizer contexts 字典，传入 driver 的 `trajectory_projection(..., _contexts=...)`；成功末尾逐 context `check_complete()`，保留 live tokenizer/backend 终边界校验。
4. 同一 optimizer 写入后 monitor 的结果，曾在下一次写入前对相同参数再计算一次。现在仅当 committed 计数、全部参数对象/version、全部固定 monitor tensor 对象/version、model training mode 都未变，并且所有 rank 经 MIN reduction 同意时复用。每次实际 Adam 后仍完整 measure；末次 Adam 后仍完整 measure。日志明确 `monitor_measurement_reused`；实际 native forward 计数、初始 repeat 计数均独立累加，未伪造。
5. Safety monitor 不消费 entropy，现只跳过其 unused 全词表 entropy reduction；模型 forward、输入、partition、temperature、selected-token logprob 路径不变。训练默认继续计算 entropy，训练 loss 不变。
6. 必要参数/Adam exhaustive finite 扫描改为设备内汇总，每个设备仅转移一个 flag；CPU Adam step 标量集中检查；chunk 从 262144 改为 4194304 元素，布尔临时分配仍有界。首次实际写入前补上参数/Adam finite 检查，此后沿用每次写入后的 exhaustive 检查。
7. Reporting scalars 改为每个 minibatch 一次集中 CPU 传输，以 FP64 累加对应原 Python float 运算；不参与 loss/安全 admission。preclip gate 的两个 Python tensor boolean 改为一个 tensor boolean表达式再同步。

正常 20 次写入、每 monitor 16 个原 micro partition 的理论实际 monitor 次数：原 `(20 before + 20 after + 1 initial repeat) × 16 = 656`；现 `(1 before + 20 after + 1 initial repeat) × 16 = 352`，19 次相同参数的 before 复用。若版本变化或任一 rank 不同意，就重新完整监控；不能承诺始终命中该预算。

## 数值与控制边界

- 三项 PPO 防护完整保留：DualClip `c=3`、sample-policy KL `0.2`、negative ratio tail `0.01`。未改阈值或 loss 公式。
- `exp` 前有限性、log-ratio bound `30`、梯度 clip `1`、FP64 global preclip 与实际 measured postclip、backward 前 loss finite、Adam 状态/参数 finite 均保留。未以 cache 伪造 finite 证明。
- μ 分母仍来自实际 sampled FP32 scalar；actor β 仍独立；初始相同 monitor partition 的 native repeat 仍执行，μ KL baseline 不被减掉。
- mini/micro、leaf、epoch、reward/advantage/policy weights、所有训练超参数不变。loss 的训练 forward、图、梯度 accumulation、optimizer 操作不变。
- 20 opportunities 与实际 writes 的 ledger、安全 skip、write 后 refusal 的真实计数、scheduler 每 credited outer 一次逻辑未修改。
- 精简对象为重复 provenance 重建、相同参数的重复 monitor、unused entropy、reporting/finite flag 的重复 host 同步。未关闭审计失败 gate；审计 JSON 的复用事实单独记录。

## 唯一 targeted CPU 检查轮次

解释器为已存在的真实 CPU venv：
`/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-3/work/fresh200-local-test-env/bin/python`。

第一启动在旧 collector fixture 构造时退出：该 fixture 没传基准已强制的 `native_stopping`，7 项均在进入 actor 前失败（0.017 秒），不能当 actor 回归结果。未更改生产 guard；在本轮 fixture capture 包装中补齐明确 synthetic EOS/stop schema 后重跑同一 targeted 检查，7/7 通过（unittest 0.500 秒，进程约 2.27 秒）：

- μ DualClip 梯度、Tree policy weights/global row denominator 保持。
- 初始 KL/tail 超界均拒绝零 writes。
- 有限参数但 postwrite KL 超界、write 后 NaN 均保留实际一次 write 计数。
- 原有真实 row binding mutation 在 native forward 前拒绝。
- tiny CPU actor 20 次实际 SGD writes：19 次 monitor 复用、44 次真实 monitor native forward（旧预算 82）、2 次初始 repeat；actor 内完整 μ validator 恰好调用一次。
- 参数 in-place mutation 改变 version 后缓存失效并重新 forward；sampled μ in-place mutation 后概率验证上下文失效并重新校验拒绝。
- CPU Adam step scalar 为 NaN 时仍拒绝。

Fixtures 使用 synthetic tokenizer/native receipt、tiny real Torch optimizer；没有 stub 生产安全判定，也不构成真实 vLLM/FSDP/GPU 等价或速度证明。未执行长测试 pipeline、分布式/GPU测试或完整旧 suite。

## 独立审查发现与最小补修

对冻结提交 `0150ee6` 的独立审查指出：概率验证上下文当时只绑定 meta/non_tensor 容器身份，原对象内的 nested metadata 变化可能继续命中。因此已在 `policy_safety_runtime.py` 绑定其完整关键内容 fingerprint，context 创建与 `matches` 各进行一次 batch JSON 扫描；不恢复逐 action tokenizer serialization 或 prefix 证明扫描。

一条针对该发现的真实 CPU venv 小检查已通过四种 case：raw ledger 内的 selected logprob、μ binding 内的 action receipt、meta runtime provenance、trajectory UID 原对象变造。每种均保持原 tensor 对象/version，但缓存失效、完整 μ validator 实际重新调用一次、在任何 native forward 前拒绝。生产 guard 无 stub；仅继续使用前述明确 synthetic fixture native stopping metadata。该补修之后待 root 提交新 SHA；不能把 `0150ee6` 当最终已修复版本。

## 仍需真实新 run 定位

742 秒 actor 时间不能全部归因于 69.6 秒 host monitor native-call 区间；该计时不包含所有 CUDA completion、collective wait 与其他 CPU 工作。以上修复只针对源码中已确认重复，不承诺 outer ≤ 7 分钟已达标。root 应读取新 run 正常完整 outer 的实际 wall time；若仍超过 420 秒，继续划分 actor training forward/backward、FSDP 通信等待、FP64 grad scan、参数/Adam finite scan、rollout/CPU proof、IO 等真实区间，不放宽防护、不改固定训练预算。
