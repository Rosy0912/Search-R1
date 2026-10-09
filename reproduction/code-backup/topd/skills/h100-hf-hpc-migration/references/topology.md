# 连接与存储

## 路线与权限

数据路线：H100 源归档 → mgmt-host 内存流 → 私有 HF → HKUST2 实盘 → HKUST3 实盘 → HPC3 共享存储。新路线在远端完成下载和传输，Mac 仅控制 SSH；禁止使用、修改或恢复 Mac 代理。不要随意省掉用户指定的 HKUST3 中转；若用户另授权 HKUST2 直接转目标，记录例外。

旧全局记忆可能禁止跨集群；本次用户后来明确授权的特定迁移覆盖该任务范围内的旧冲突，不扩展到其他资产/任务。一般下载记忆的 Mac 落盘回退也不适用于本迁移的明确约束。

## H100 入口

TOP-D 现有源读入口是 `work/remote_topd_retained18.py`；先核实并复用，不写回旧 h100-16/Tailscale 路线。管理机公共配置：

`/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json`

管理机已有 key 经保留18的 SSH `-W` 到目标，目标认证也在管理机进行，不把私钥复制到18/Mac。内部命令清理 `LD_LIBRARY_PATH`；`IdentitiesOnly=yes`、`BatchMode=yes`、严格 host key 和既有可信公开 host keys。

相关记忆：`/Users/shaoshitong/Documents/remote_codex_control/memories/h100-retained18-connection-20261008.md`。里面时间和退役日期不是当次连接成功证据。

## 校园 SSH

复用别名和现有身份，可按已验证的 Mac 网卡条件执行：

```sh
ssh -T -B en0 -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes hkust2 hostname
ssh -T -B en0 -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes hkust3 hostname
```

`en0` 是本案例实测，其他设备先查实际接口；失败先排查系统路由/TUN，不能盲目改全局 VPN/SSH。`ProxyCommand=none` 不保证系统路由不走 TUN。HPC3 配置通过 HKUST3 跳板：

`/Users/shaoshitong/Documents/remote_codex_control/access/hpc3-20261008/ssh_config`

主机、用户、端口、可信主机公钥可以作为元数据传递；登录私钥不复制。认证缺口优先沿用已有 SSH 通道或限定作用域的 agent forwarding，先验证；不要尝试旧密码或关闭 host key 校验。

## 空间

在每个中转/目标读取完整 `df`、`findmnt`，定位真实挂载来源，使用当前用户创建很小的本任务写探针并移除。记录可用字节，预留归档、部分文件、解包体积。不能仅看根目录或某个小盘就推断整机空间。

HPC3 必须确认共享目标，不把计算节点临时盘当最终成品。rclone 远端配置不要包含新复制的私钥；先测试真实认证与仅本任务目录的读写。

## HF 远端自身出口

先测远端官方 HF 直连；若失败，调查用户授权范围内可读的远端工具、服务状态和配置路径。可以检查其他用户/root 可访问的文件元数据与下载方式，但不能打印凭据、接管外来服务或绕过权限。优先复用用户自身已授权出口。本案例找到 HKUST2 的 `/home/shaoshitong/project/clash-for-linux/runtime/config.yaml` 和 `runtime/bin/mihomo`；原服务 inactive，以独立本任务进程和 RAM 配置恢复出口，不改原服务。

本案例使用 HKUST2 loopback `127.0.0.1:17894` 下载；供 HKUST3 使用的校园出口绑定 `10.120.17.102:17895`，仅允许 HKUST3 `10.120.17.122/32`。控制端口只绑定 loopback，配置位于 `/dev/shm`、权限0600。地址、端口和节点均须重新核实，不能直接覆盖已有监听者。API secret、节点认证和 HF token只在RAM使用，不输出配置全文。

清除下载子进程继承的大小写 proxy 变量，再设置已测通的远端出口；必要时给 aria2 显式指定本次远端代理。校园 SSH 仍直连。节点按真实私有归档 Range 测速选择，历史节点或测速不保证当前可用。出现 TLS EOF/连接重置时检查本任务出口的脱敏上游超时证据，采用有限单次超时与退避重试，保留断点；禁止关闭 TLS 验证或回退 Mac 代理。

## 两跳远端传输

先从 HKUST2 实际验证到 HKUST3 的 SSH 认证，再在 HKUST2 执行 rclone SFTP `copyto`；归档在 HKUST3 落盘并完整SHA通过后，才在 HKUST3 执行到 HPC3 的直接 SSH 续传和解包验收。两跳都不经过 Mac 数据流。

使用既有远端身份及严格主机密钥校验。缺少主机公钥时，可以从已核验 known_hosts复制公开主机密钥；这不授权复制登录私钥。TCP22可达、工具退出成功或字节传完，都不能代替实际认证与独立SHA回执。历史已授权旧流单独核实当前组和单写状态，不因切换文档而盲目中断。
