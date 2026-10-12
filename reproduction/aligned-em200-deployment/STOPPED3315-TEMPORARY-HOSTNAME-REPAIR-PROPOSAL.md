# 3315 临时名称解析修复提案（待额外授权，不执行）

建议候选：仅在 **h100-5 的 `/etc/hosts` 临时新增一条带独立标记的映射**，让已停止的3315 daemon 有机会向当前同集群 controller 发出完成通知，然后撤销新增行。它是根据日志推测的最小修复，**不保证旧 daemon 的缓存、NSS/DNS 重试、端口或完成 RPC 一定恢复**。本提案不实施任何修改。

实际21:10:34Z证据是 `work/evidence/aligned-em200-old3315-controller-resolution.actual.json`，SHA256 `7c6d76924d74651e464295d20eed83788f73fb382260fc12b5ab1ea7ab788999`：ClusterName=`wuhan-h100`、当前 controller=`h100-8`，ping UP，解析地址=`10.33.0.8`、控制端口=`6817`；`h100-16` 的 getent 返回2。当前 `/etc/slurm/slurm.conf` SHA为 `8098a7a9eb5436522948607f18fccd891dc0b1ed3df14bb51432eb8d4b0fc0cc`。21:01原件显示旧3315重复报 h100-16解析失败、controller地址建立失败、job complete发送失败；已知 job cgroup不存在，daemon没有子任务。installed21.08手册确认TERM被忽略。此前node observation2也确认当前controller持有原3315的exact source/comment/旧RUN及COMPLETING记录；不再追加诊断循环。

## 授权边界与具体变更

原2026-10-08T18:07:39.855509Z指令仅停止3315并禁止其恢复。`/etc/hosts` 改动会影响整个节点对旧名称的解析，因此需要**用户明确授权本节点这一条临时映射及其撤销，并由有管理权限的执行者实施**；旧停训授权、root UID或新训练goal都不能代替这项扩授权。

唯一待审新增行（包含末尾LF）：

```text
10.33.0.8 h100-16 # BAPO_STOPPED3315_RESOLUTION_REPAIR_20261009_V1
```

保持现有 `10.33.0.8 h100-8` 行和所有其他原字节完全不变。不更改 Slurm 配置、DNS服务、controller、其他机器或既有hosts行。不TERM/KILL任何进程、不restart服务、不DOWN/RESUME节点、不requeue/resume3315、不替换或重复提交3345。

## 先准备完整备份和可审阅差异

执行前只读准备并留存：`/etc/hosts` 全部原字节、完整SHA/长度、设备/inode、mtime/ctime、owner/group/mode、ACL/xattrs/安全标签和挂载情况；备份置于新ROOT内一个独占、权限受控的 evidence目录，fsync文件和目录，并将清单/哈希回读保存。保留旧3315日志、原停止/取消记录和当前controller证据；不删除旧实验源码、checkpoint或工作区。

确认hosts是普通、非symlink、单link、支持同目录原子替换的文件且以LF结尾；现有文件没有任何h100-16映射或本提案标记。若是bind mount、不能安全保留metadata、缺末尾LF或已有冲突行，拒绝本方案，不能转用truncate/in-place写入或改现有行。核对 `/proc/755855/root/etc/hosts` 与节点hosts确为同一文件，避免改错daemon所在mount视图。

用完整原字节加唯一新增行生成待审候选文件，记录其SHA、长度和只有一行新增的diff。具体发布前必须已有可回读备份和这份候选。当前证据只有部分hosts行，**本提案没有虚构完整hosts的before/after SHA**；准备阶段补齐后，授权/执行记录应绑定这些实际SHA。

## 写入前不可放宽的复核

确认当前集群/hosts配置仍为上述证据，配置文件SHA不变、h100-8 ping仍UP且解析10.33.0.8；当前CLI仍显示原3315身份和COMPLETING。刷新节点无其他运行/挂起allocation、NVML为空、旧job cgroup无任务的事实，确认3345仍是原单一排队作业。若3345或其他作业已启动，停止此维护候选，重新审查影响。

核对 boot_id=`9471cbc1-9ae1-4be5-8edb-1e84e850e2cc`；旧PID755855/start_ticks1433986891、PPID1、UID0、完整argv=`slurmstepd: [3315.0]`；exe=`/usr/sbin/slurmstepd-wlm`，device64769/inode288378、266032字节，SHA=`be7770948fcfd24b074745d8add09005f77c1ae0ddb380b86abceb71d2bfdd6a`。任何身份变化或进程已退出都不添加映射。不得把共享slurmd.service cgroup当作3315专属归属。

管理员需在短维护窗口协调所有hosts写入者，取得独占维护锁；flock只是协作锁，不能阻止不遵守它的编辑器。紧邻提交时复核原文件的SHA、inode和时间/长度均与备份一致，才能用保留metadata的同目录临时文件进行原子替换并fsync目录。出现并发修改则不写，重新生成差异并审查；不声称一般POSIX rename能对任意其他写入者实现无条件原子compare-and-swap。

## 有限观察与恢复

添加后立即验证hosts实际字节及metadata、h100-8原映射未变、h100-16现在只解析10.33.0.8。只等待原daemon自然的完成重试，建议最多300秒、每60秒一次有界观察；不发送额外信号，不靠增加工具或无限轮询代替修复。

成功条件是独立实际证据：旧3315完成RPC/epilog结果、原daemon退出或完成状态、3315退出COMPLETING、节点恢复正常且3345仍同一个作业；单纯getent成功或daemon消失不等于完成验收。若3345由Slurm正常启动，记录实际转换并尽快撤销旧名称映射，继续保留其原资源/源码/硬件门禁。

不论成功、超时还是解析/RPC失败，撤销本次新增行：若hosts仍精确等于已发布候选SHA，则原子恢复完整备份字节和metadata，回读SHA必须等于原SHA。若发生别人的并发编辑，绝不能直接用旧备份覆盖；只有新增行仍唯一且字节/标记完全一致时，重新审阅“仅删除该行、保留所有其他现有字节”的候选及当前before/after SHA后再撤销。若标记被改动、出现多行或无法确定归属，停止自动写入并立即交管理员处理，不盲删行。

恢复后验证原h100-8映射保持、独立标记消失；在无其他合法并发改动时整个文件SHA应回到原值。保存实际操作、观察、恢复和所有异常证据。如果恢复失败，必须明确报告仍有临时映射及其影响；不能把维护当成功或静默结束。超时/失败**不得升级**为KILL、restart、DOWN/RESUME、永久alias或其他配置修改。

## 审查结论与限制

这条变更有明确目的、单行范围、完整备份、并发拒绝和回退条件，可供管理员与用户审阅。它仍是节点级操作，且可能把节点上其他程序对h100-16的访问导向h100-8；因此必须额外授权并核对管理员确认的同一controller身份。缓存配置、NSS负缓存、网络/auth/端口或controller对旧job的拒绝仍可能导致失败。本文只审查方案及实际只读证据，未改hosts、未执行修复，也不承诺能解除3345启动阻塞。
