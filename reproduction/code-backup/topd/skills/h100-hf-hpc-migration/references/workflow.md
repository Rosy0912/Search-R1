# 迁移操作与故障恢复

## 状态与记录

每组记录 `group`、源归档路径、`archive_bytes`、`archive_sha256`、所有 `files[{path,size,sha256}]`、来源与 revision。保留源读取、HF 上传、每跳下载/传输、目标归档/解包、fresh 全文件验收的独立回执。更新计划要保留旧记录，不能用新总数掩盖漏迁资产。

每个组件用专属任务目录、controller 和 flock；启动前验证进程命令/启动时间与锁持有者，PID存在不等于身份相符，进程活着不等于有增量。目标另需按组协调单写；不能只靠“不同控制器 PID”判定安全。观察实际目标部分文件和近期进度。

## H100→mgmt-host→HF

1. 源端归档就绪、文件 manifest 和 SHA已完整。读取归档路径和大小，确认它是真实 checkpoint/官方初始模型，不能把空学生训练目录当已训练 checkpoint。
2. 在 mgmt-host 上传进程中复用源 SSH 路线，实现有界、可 seek 的内存读取；源 seek/重新开流支持 HF LFS 分段上传。不要在管理机 materialize 整片，也不要让无界 read 把大型分片读入 RAM。
3. token从stdin读取，调用官方 HF API 确认 private；上传源 archive_bytes/SHA，逐组保存 commit 回执，不上传自动误选的凭据/环境。
4. official paths-info 的 `size` 与 `lfs.oid`逐项对源 manifest；提交成功不代表目标迁移完成。失败只重试本组件的未验收组，不盲目重新提交全部资产。

## HF→HKUST2：测速与下载

先清点已有进程/部分文件，后续任务分配不同组。官方与镜像比较同一私有归档、相同 Range、有界样本，读取数据到内存；记录HTTP状态、实际接收字节、耗时、样本SHA。先小样本连通，再用64MiB或限定20–30秒长样本；镜像401/超时不当作0MB/s有效下载速度，更不由它推断官方已卡死。

单流续传：part命名含归档SHA；从实际offset发Range，要求206和正确Content-Range，拒绝超长/重复数据，flush/fsync后核对完整SHA，再原子改名。若Range被忽略，保留已有part并修复请求；不要直接追加200完整响应。

### hfd/aria2

原始代码：[padeoe hfd](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f)。使用前读当次源码，版本可能变化。2026-10-09版本的两个落盘点：

- `.hfd/last_download_command`包含token原文。
- aria2下载清单包含 `header=Authorization: Bearer ...`。

安全版保留hfd筛选/恢复和aria2多连接，将指纹token字段改为是否设置；磁盘清单不含认证header；curl配置走内存FD，aria2输入经stdin即时插入header。token仅在RAM，不使用 `--hf_token` 实参。日志不得落 token/header或带授权的签名URL；本案例aria2文件日志关闭。

从4–8连接、每次1个归档起步，不与现有任务抢同一个文件。可以用 `--include`只取已分配组。镜像失败可在不同控制目录测试官方，不能指望切HF_ENDPOINT能解决401。核对列表确实只有计划组；禁止自动把整个仓库317GB当本次173GB计划。

aria2多连接文件可能稀疏，stat大小甚至接近总长也不等于完成。启用仅loopback RPC；只读取 `status,totalLength,completedLength,downloadSpeed,connections,gid`，不要输出URI/header。用两个 `completedLength`样本之差计算平均速度。hfd表面Done只表示下载工具结束；还要完整归档 SHA 对源/HF。保存独立download-verified回执。

重跑hfd前检查同组进程已退出及`.aria2`控制文件；保留控制文件与部分归档。原版会删除错误大小且无`.aria2`的文件，因此不要把其他下载器的线性part直接交给hfd覆盖。

## HKUST2→HKUST3→HPC3

按用户要求逐跳执行。每跳先核实身份、可写盘和目标无同组写入者：

1. HKUST2归档已下载SHA验收，再把它复制/续传到HKUST3本任务目录。rclone/SFTP先实测认证；TCP端口成功不等于认证成功。可用现有SSH双端有界流，Mac只转发RAM，不落模型。
2. HKUST3核对完整大小/SHA后保存独立回执，才向HPC3写入该组。不要省掉该跳后仍报告“整条路线已验证”。
3. HPC3以含SHA的`.part`恢复，写入从实际offset开始。完整归档SHA通过后改为最终归档名；解包时验证路径无绝对/`..`逃逸、manifest内无重复成员，只写本任务bundle，用临时文件+原子改名逐文件核验大小/SHA。
4. rclone可用 `copy/copyto`，不要用会删除目标多余资产的 `sync`；SFTP不保证服务端提供SHA，因此单独在远端算SHA。rclone操作成功不能代替内容验收；断点能力以所选后端/当次配置实测为准。

已有直接路线若用户仍授权且正常，保持其同组单写；新增路线推进不同待迁组。恢复只操作本任务组件，先确认归属，再结束失败/无增量组件。正常迁移不为测速中断。

## 最终验收与巡检

所有目标组回执存在后，fresh逐文件重算共享存储SHA；将源扫描的新checkpoint与计划比对，必要时追加组。保存真实complete.json和验收报告。来源元数据、最新运行代码也属于资产，源码变更需提交推送和远程SHA核验。

进度公式（十进制单位）：

- GB = 实际compressed bytes / 1e9。
- 百分比 = 实际compressed bytes / 当次选定总bytes ×100。
- 平均MB/s = 两次目标实际compressed bytes差 / 两次实际测量epoch差 / 1e6。

同一归档final与part重复、临时控制文件不应重复计入；按计划归档及合法part汇总。不得把HF提交字节、HKUST下载字节与HPC3目标字节相加。teacher通过数只包含归档和全部原文件SHA通过的目标回执；fresh验收完成另记。

状态读取失败：保留上次实测及时间，写当前读取错误与无法计算速度，不把旧数值当新值。连接/网络暂态用有上限的单次超时和可追踪退避重试，保留partial；凭据拒绝、空间不足、SHA mismatch或连续无增量时停止该组件的盲目重试并诊断，不暂停整个已授权迁移，不删除副本。
