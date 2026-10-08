# 3339 实际训练轻量速度诊断

当前已确认 v13 的逐前缀文本校验存在实际 CPU 串行热点。尚未拿到 outer6 完整耗时，不能声称普通 outer ≤420 秒已经达标，也不能用启动耗时判定失败。

- 作业：3339；节点：retained H10018 / h100-10；run：`/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-current200-continue-3339`。
- 源码：`62271081ba1bb74d8fc3f4958d623758efa74fa5`，`package-code-only-v13`；加载 native5 后当前推进 outer6。
- 最后观察：2026-10-08T06:42:55.433658+00:00；已生成的训练 rollout cost / 完成 outer6 计时尚不存在。

## 真实热点证据

实际 own cgroup `/slurm_h100-10/uid_0/job_3339/step_batch` 确认 `ray::measured_task` PID305810，`/proc` starttime 为1850185226。三次附加前后均重复核对，身份未改变。py-spy SHA与授权值一致，仅执行 `dump --nonblocking --full-filenames --json --pid 305810`；没有暂停、信号、locals 或 native 模式。

第一张快照在 `rollout._advance:166 -> generate_sequences -> ray.get_objects` 等待生成。随后两张快照相隔15.053秒，均命中：

`sampled_action:536 -> _scan_utf8_prefixes:245 -> tokenizer.decode:4002 -> transformers.to_py_obj -> infer_framework_from_repr`。

同期已有 CPU 记录覆盖26.863秒，driver 累积 CPU 增长26.95秒，约100.3%单核。该窗口已有GPU采样5次，八卡每次全部0；foreign MEM/SM记录均0。它说明存在真实的 CPU 处理热点，不能据此计算整轮热点占比。

从加载状态记录后的参考窗口，已有72次资源采样，56次八卡全部0，各卡所有采样利用率均值12.29%，中位数0。采样不是完整阶段计时；state migration 的 mtime 也不是精确 outer 起点。

## 剩余候选优化

1. 第一优先：在已绑定的 tokenizer backend 内批量解码全部原生前缀，减少每个 prefix 经过 Hugging Face `to_py_obj` 和框架探测的开销。必须保留每个原生前缀 agreement、Frozen ByteLevel/special-token/cleanup绑定、UTF8完整证据、prefix hashes、动作边界和概率信息。禁止跳过 `check_native_prefixes` 或缩短扫描。候选尚未实现或验证，须核查 wrapper/backend 精确语义后在未来入口生效；本次没有热改3339。
2. 次优先：在 tokenizer 身份绑定的 context 内缓存不可变 token ID→byte piece，去掉重复 `convert_ids_to_tokens` 和 alphabet转换。单项耗时尚未测量，不能把它列为已验证收益；不能跨outer复用证明缓存或弱化漂移校验。
3. 主线继续读取真正完成的普通 outer 和现有 `semantic_prefix_scan_*` cost字段，区分剩余 CPU扫描与GPU生成时间。只依据实际训练，不补 GPU基线、短训或新作业。

Ray日志另有文件系统>95%的告警，但仍约539 GB空闲；没有观察到 spill/object creation失败，因此不是已确诊速度原因。长observation警告也不是减数据、context或token预算的授权。

## 诊断边界

共3张快照，工具自身累计耗时0.139秒；快照不构成吞吐测试。只读当前own资源/阶段记录和已有metadata，未读取模型/PT或训练原始数据，未发起GPU查询或模型调用，未安装工具，未改运行源码或当前LIVE/STATUS，未提交/取消作业，未暂停goal。报告JSON保留完整三张快照、身份核验及资源时序。
