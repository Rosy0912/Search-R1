# Preparation-3 小时只读运行审计

审计区间：2026-10-07T18:45:53.413943+08:00 至 2026-10-07T18:45:58.708367+08:00（Asia/Shanghai）；按要求在 18:45 CST 后执行。原 run `914bfff786c84ede8d82414dff65f60e`、controller 2459676/starttick915357855、child2460237 仍绑定原身份与私有进程组。结果：**RUNTIME_PREPARATION_AUDIT_PASSED**，失败检查为空。

固定文件大小快照共 **169600 / 845515** 完整轨迹，八分片为 21440/21120/20800/21120/20800/21760/21120/21440。全部完整 JSON 与原 question 字段/sibling 序列一致；没有中间无效记录、重复位置或顺序错误。本次尾部 partial bytes 与不完整 k5 sibling 都为0；审计按冻结文件大小读取，正在追加的未完成尾部不会被误作故障。此有限 prefix 一致性不证明全部将来输出正确。

八 worker 的 UID、PPID、PGID、精确 start ticks、argv、E1_RUN_ID/路径/GPU/seed 环境及实际 XFORMERS/22671 GPU +7281 CPU cache 日志均正确。实际 controller/helper/config SHA 保持原值，305 个 runtime 源文件实际重 hash 全通过；原 train169103/validation512/test51713 实际 SHA、行数与路径身份通过。

source100 三个大 safetensors **没有再次读取13GB计算 SHA**。本次记录 realpath/device/inode/size/mtime_ns/ctime_ns，并复核审计前后 stat 不变、ctime 不晚于 transfer acceptance；小配置/tokenizer 文件重新 hash 匹配原 accepted SHA。大文件以已有 transfer/startup 内容 SHA、当前 config/argv 和文件 metadata 绑定，这是 metadata provenance，不是 fresh 大权重 content hash，也不是防止高权限 metadata 篡改的证明。上次独立 receipt 未记录完整 inode/mtime baseline，所以本次不声称做过跨小时 inode 逐项比较；此新 receipt 为后续提供该基线。

全八卡直接 memory.free 当前为 3718/3718/3714/3756/3756/3714/3714/3179 MiB，当前最低3179 MiB、连续历史最低3099 MiB。检索当前574 MiB、连续峰值654 MiB。直接显存 observer receipts 年龄为 1.726/0.374 秒；对应进程存活、身份正确。V3 仍存活，在原 `runs/opsd/five-update-acceptance.json` 无限 observation wait 中，当前 sleep 状态正常。

可读 dmesg 在新启动 16:31:23 CST 之后没有新增 Xid/NVRM/CUDA 匹配；journalctl 提示系统 journal 可见性不足，其空输出不作为证明。旧80,320条失败准备、GPU3 Xid31/MMU fault和 SHA 绑定的故障证明完整保留；超过旧行数不能证明其根因修复或 bitwise 复现。

实际阶段仍为 FULL_FRESH_TRAJECTORIES；`runs/opsd` absent、实际 OPSD 更新 **0**、native/external 五更新 receipt absent。未将准备进度或未来状态设计算作100→105验收。本次没有远端写入/代码、配置、affinity、作业变更，没有停止进程，也没有 GPU workload 或 HTTP 推理请求；只使用健康 GET与只读状态查询，未重复已通过的 loss/init 测试。

完整证据：`preparation-3-hourly-runtime-readonly-audit-1791369951.json`，SHA256 `a026b58343115a6bfdda1d902f3146d40f3623ceb89d9df66fcf3cf6fc031850`（192792 bytes）。本文为本地独立分析，不编辑 publisher 主文档。
