# Tree driver 性能修复

隔离源码基线 `6efb604`，分支 `codex/tree-speed-repair-20261008`。仅改 driver 文件；没有 SSH、GPU/模型调用、训练、提交或原 session 源码修改。

## 实现

- `search_r1/baselines/tree_raw_tokens.py` 增加显式 `tokenizer_validation_scope(tokenizer, identity, _context=None)`。完整真实 encoder/file/decode identity 在 scope 入口和成功返回前检查，嵌套 scope 复用上下文。上下文再次用于同一 outer 的奖励/审计 phase 时，使用时检查真实 file/runtime flags/decoder/public encode/decode 方法，成功返回前完整 backend 复检。没有 tokenizer 对象/identity-only 的永久信任缓存。
- 每个已经验证的 action 保存排除计时字段的独立深拷贝。继承 action 必须与该不可变快照类型严格一致；修改 bool/int 类型或 signed zero 不能以 Python 宽松 equality 命中。缓存不命中仍检查 JSON 绑定并重建真实 UTF8/semantic 证明，重新封 hash 也不能替换真实语义。
- 每个新的 native emission 仍检查每一个 native prefix decode。UTF8 对原 bytes 的重复完整 decode 和 prefix JSON/hash 改成增量 UTF8 decoder 与增量 JSON string hash；pending replacement 后缀单独处理，最后完整 bytes decode 仍校验。proof 字段与原协议一致。
- `rollout.py` 给整个 run、独立 advance/compose 加明确 scope；生成行共享同一个 validation context，最终叶子共享 contexts 字典。正常 run 的 advance/compose scope 嵌套复用，避免逐叶/逐 action 完整 tokenizer serialization。
- `controller.py` 在新的 outer 建 fresh scope，并让默认 RewardManager 与 actor audit 复用该 outer 的 action proof。每个独立 phase 完整 backend 检查后才返回。semantic reader 不再重复生成 projection 的深拷贝/hash，也不对同一个实际 tokenizer 重复 UTF8/native decode。
- `objectives.py` 未修改。原始 sampled IDs/logprobs、raw/semantic 双表示、Tree 继承和叶子预算/权重、μ修正协议、PPO 防护、4turn/2048/4096/500/500、5leaf `[3,2]` 均没有改变。

## CPU 验证

`python3 -m unittest test_tree_driver_speed_cpu -v`：6 tests passed，0.107s，无 skip。

覆盖 sampling→继承→semantic reader 的 scan 重用、原始 token/μ 与输出一致、无效 UTF8/特殊 token/JSON escape 的增量 hash 一致、cached/resealed 语义及类型变造拒绝、真实 backend vocabulary/normalizer/runtime flag/decode 方法变化拒绝，以及新 emission 的每 prefix native agreement 拒绝路径。

五条 500 token 继承轨迹的小型 CPU 比较（改后 standalone 调用与改后显式 scope；不是旧训练 benchmark）：

| 模式 | CPU 秒 | 完整 backend serialization | UTF8 重扫 |
|---|---:|---:|---:|
| separate calls | 0.035977 | 5 | 5 |
| scoped batch | 0.011924 | 2 | 1 |

两种模式完整 projection 输出相同。新采样→继承→semantic 同一个 scope 测试实际只进行一次 scan，scope 入口/出口共两次完整 serialization。最后给独立 advance/compose 补相同 boundary scope；语法检查和 diff whitespace 检查通过，未另跑扩展回归。

## 剩余耗时与正式目标

每个新的 native emission 的第一次每 prefix native decode 仍是 O(T²)，保留这些检查；重复继承叶子和语义消费已避免重复执行。每个输出 row/projection 的绑定 hash 与必要 ledger copy、实际生成/检索、μ以及 actor 防护的真实 forward 仍有成本。μ helper 使用共享 contexts 后，成功末尾的完整 `check_complete()` 仍是真实 boundary 成本。不能把 synthetic 3×CPU改善外推成正式 outer 达到420秒。

有效人工目标是普通完整训练 outer ≤420s；评分/保存额外成本单列，不能隐藏。如果正式续训仍超过420s，须按新的真实阶段耗时继续定位。首次 perprefix proof 和 row 绑定仍是本次明确保留的 driver 热路径。
