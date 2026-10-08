# GiGPO Instruct200 故障原因和处理记录

更新时间：2026年10月8日。本文记录目前这组官方 Instruct200 实验的实际故障、原因及处理状态。所有计算位于武汉 H100 集群的 h100-8；本文保存在 Mac 本地。

**最新状态：作业3337完成2轮训练、累计40次实际成功参数更新后，在第3轮生成响应时发生GPU非法内存访问并退出。当前没有训练在运行，也没有启动后继作业。** 前2轮日志和代码均保留；尚未到第5轮保存点，没有完整模型与Adam检查点可用于精确续训。

## 已发生的问题与处理状态

| 作业 | 实际训练进度 | 错误与原因 | 处理办法 | 当前结论 |
|---|---|---|---|---|
| 3323 | 0轮，0次Adam | 基线评估前后的训练状态身份比较失败。已确认初始FSDP本地分片摘要为空，首次读取模型状态会触发分片初始化。原失败未记录完整的比较后身份，无法确认所有差异仅来自分片。 | 在官方初始状态快照前完成原生FSDP初始化，并在拒绝前保存前后身份与具体差异。 | 启动比较问题已修改；不能将它记录为训练数值故障。 |
| 3325 | 0轮，0次Adam | 保存首批采样证据时，PyTorch拒绝序列化同一底层storage通过不同dtype访问的张量。故障发生在首次参数更新前。 | 对用于记录的张量逐个detach、转CPU并clone，解除记录副本的storage共享；不修改参与训练的张量。 | 后续3332完成了首批证据保存；这条记录路径已实际通过。 |
| 3332 | 0轮，0次Adam | 首批采样后，HF重算与vLLM采样的选中token概率有明显差异，外部数值验收没有放行。实际最大log-prob差约2.47546；目标位置的token、prefix和行映射核查没有发现对应错误。具体数值根因未确定。 | 关闭BF16矩阵乘法中间低精度累加，保留原权重精度与算法；按较新人工指令取消额外检查对健康训练的等待，后台继续诊断。 | 该作业在首次Adam前由本任务主动结束以应用新入口，属于技术性停止；概率一致性和尾部阈值仍未宣称通过。BF16设置是数值处理假设，不是已证明的完整根因。 |
| 3336 | 1轮，20次Adam | 完成第1轮后，写JSON日志时把采样器的随机状态Tensor直接交给JSON编码器，触发“Tensor不可JSON序列化”。这是记录代码缺陷。 | JSON报告仅保存该随机状态的dtype、shape及哈希；可恢复训练状态继续保存真实Tensor。 | runtime18已应用。3337的第1、2轮均成功写出对应JSON和完成记录，这条故障路径已实际通过。 |
| 3337 | 2轮，40次Adam | 第3轮vLLM生成时，rank1出现CUDA非法内存访问。CPU同步读取采样结果时暴露错误，真正出错的更早GPU内核尚未定位。 | 保留完整原始日志；核对权重切换和KV缓存生命周期。准备仅改变CUDA执行同步方式的隔离正式入口，让错误定位到真实内核，不改GiGPO算法或训练预算。 | **尚未解决，后继尚未提交。** 当前方案是诊断与执行层处理，不保证一定修复。 |

每个作业的已完成轮数单独记录。3336的1轮不能与3337的2轮相加，旧实验也不能拼入当前200轮。

## 当前GPU错误的直接证据

3337的第1、2轮都实际完成了每rank20次更新机会，并各成功写入20次Adam。第3轮在响应生成阶段失败，尚未进行该轮策略更新。原Slurm记录为FAILED、退出码1:0。

rank1原始日志最早在本机时间13:54:16报告模型执行失败；失败输入转存也被同一个CUDA错误中断。异常最终显示在vLLM sampler的 `random_samples.cpu()`。CUDA会异步执行，所以这个同步位置不能证明采样器就是根因，随后清理时的 `empty_cache()` 报错也不是另一项独立根因。

内核日志记录了同一进程1939501的NVIDIA Xid31：GPU虚拟地址读取发生MMU页表错误。这支持确实发生非法GPU地址访问，但不能单凭Xid31判断显卡损坏。原始rank stderr没有更早的device assert或NaN错误。

后台只读审计显示第1轮各rank的loss、梯度均有限；第2轮已有记录的PG loss约0.0216506、梯度范数约104.626，裁剪后约1。现有证据不足以将本次错误归因于GiGPO损失、显存不足或硬件损坏，也不能把前两轮有限更新视为长期稳定性证明。

## 下一步处理方案

1. 保留3337原始代码、日志、已完成更新回执和失败现场。已核实原driver及8个worker均已退出、GPU计算进程为空、原cgroup已移除，清理记录为空。
2. 准备隔离入口，在CUDA初始化前将 `CUDA_LAUNCH_BLOCKING=1` 传到driver及全部Ray工作进程，以同步执行定位实际失败内核，并排除部分异步释放问题。该方案尚在准备，尚未执行；不能称为已确认修复。它可能增加运行耗时。
3. 最新人工指令已明确取消本次独立审查等待。保存并推送同步执行入口后直接提交正式训练；跳过训练前基线，不另外开CPU或GPU测试作业。此处不记录或伪造新增审查通过。
4. 若新正式训练再次失败，按同步栈定位具体内核，再做针对性修复。禁止把非法CUDA上下文当作可以跳过的普通优化器机会，禁止偷偷降低micro、响应数或训练预算。

3337只有driver数据状态与诊断材料，没有完整模型和Adam检查点。因此不能从它精确恢复2轮训练状态。任何后继正式作业都必须单独标明初始化来源、作业号和进度，不能声称从第3轮无缝继续。

## 避免继续浪费时间的执行规则

训练前512题基线已经跳过，直接复用历史3332的结果：F1约24.29%，EM约18.36%。这是旧结果，不是3337的新评估。**每轮训练仍然是512题乘5条响应，共2560条轨迹**；训练批次不是重复基线。

额外诊断交给后台agent。健康训练不等待后台测试结果，也不因后台诊断重新启动。第5轮及每50轮的阶段验证、模型保存与最终51713题评估仍按训练协议执行。

同步数值防护允许跳过已知不安全的更新机会；一轮的20次机会全部处理完、至少有一次实际成功Adam写入且没有中途放弃，才计为完整训练轮。实际成功写入数单独报告。未知CUDA、FSDP或通信错误必须保留现场并修复，不能伪造完成轮数。

## 文件与证据位置

当前失败实验的远端绝对路径：

`/mnt/data/shaoshitong/search-r1-rl/gigpo-e1-instruct200-safety-20261006/runs/gigpo-e1-fresh200-live-job3337-1791437756282298127`

当前科学源码提交：`ebe91bfebd56c3bd445219dcd94342d815f899f0`。失败运行使用runtime18，新增同步执行入口尚未执行。

Mac本地材料：

- [当前实验完整报告](/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/gigpo/gigpo-e1-instruct200/REPORT.md)
- [配置和兼容性记录](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-instruct200/CONFIG-COMPATIBILITY.md)
- [第2轮完成原始事实](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-instruct200/JOB3337-SECOND-COMPLETED-ITERATION-ACTUAL-FACTS-v1.json)
- [3337终态与原始日志身份](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-instruct200/JOB3337-TERMINAL-FACTS-v1.json)
- [原始错误和资源现场](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-instruct200/RETAINED18-JOB3337-ACTUAL-STATUS-v14-terminal-original-logs-and-resources.json)
- [后台只读指标诊断](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/outputs/gigpo-e1-instruct200/JOB3337-BACKGROUND-READONLY-DIAGNOSTICS-v1.json)
