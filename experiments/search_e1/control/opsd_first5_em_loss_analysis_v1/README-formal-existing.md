# 已保存 formal512 的本地证据提取（实际执行源码）

`formal_existing.py` 保留本次成功本地 Python stdin 执行的原始源码，未将未执行的重构称为实际实现。它是固定路径、固定四个已闭合输入的执行记录，不是通用新评估入口。源码不包含原问题、题目 ID、响应、token 日志、权重或凭据。实际执行记录和产物位于私有分析目录；这些私有数据不提交 Git。

实际成功执行：本机 `python3 -`，2026-10-09，return code 0。脚本直接读取本地已验收的 BF16zero/C0/C1/M records 与 native captures；C1/M 只逐成员解压现有 gzip tar，并验证所选成员的 SHA 和 gzip EOF/CRC，无全量解包，无重传。共验证 2048 条正式 capture，四模型 512 个 qid/row SHA/dataset 逐槽位一致。没有新的模型推理、QA 评分、Torch/GPU、Node 写入、资源/队列操作或 Git 操作。

实际输出：
- 私有 2048 行 outcome table 与 512 行 paired table；
- 2048 capture fresh SHA 清单；
- 来源与概况 summary，保存 EM、严格最终动作、搜索次数、终止和四模型配对翻转；
- 概率缺口明确保存为 unavailable/null。所有原 capture 均明确未保存 generate 时 processed per-token scores，model_generate 仅 raw IDs/kwargs/processors；不能从 IDs 构造生成概率或“解析置信度”。

正式 C0/C1/M 是 **10 outer 末 raw 权重**（CP100/101/100），并非第 5 outer 权重评测。单次 greedy 每题结果只能给固定 cohort 的 EM，不能估计单题 pass probability。TRAIN 5 次 fresh stochastic sampling 的 s/5 与 observed pass@5 必须另取真实 TRAIN reward，不与不同 qid 的 heldout loss 合并。

初次相同审计在写 JSON summary 时遇到 failure_class=null 的 histogram key 与 sort_keys 冲突。失败执行和已产生私表保留在第一版私有目录。当前 V2 显式将 histogram 的 null key 序列化为字符串 `null`，原 outcome table/原评分的 null 值保持不变；V2 全部本地重新读取验证并成功退出。它不是训练失败、评分变化或生产源码修复。

这个 recorded script 使用不可覆盖的固定新输出目录（`mkdir(exist_ok=False)`）。本次输出已经存在，直接再次运行将拒绝覆盖；如需另一次读取应先审查新的独立路径与输入契约。未声称此固定记录支持自动恢复、任意输入或单因素因果归因。
