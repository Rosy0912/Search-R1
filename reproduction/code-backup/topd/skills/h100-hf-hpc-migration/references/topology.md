# 连接与存储

## 路线与权限

数据路线：H100 源归档 → mgmt-host 内存流 → 私有 HF → HKUST2 实盘 → HKUST3 实盘 → HPC3 共享存储。Mac 仅控制 SSH、可转发字节而不落盘，并可为 HF 公网建立 HTTP 反向隧道。不要随意省掉用户指定的 HKUST3 中转；若用户另授权 HKUST2 直接转目标，记录例外。

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

## HF 公网代理

若远端直接访问官方 HF 失败，但 Mac 已有 HTTP 代理实测可用，可以建立该节点专用反向转发：

```sh
ssh -T -N -B en0 -o ProxyCommand=none -o ProxyJump=none \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 \
  -R 127.0.0.1:18092:127.0.0.1:7892 hkust2
```

端口先检查，使用绑定127.0.0.1的现有代理，不向公网暴露。保持隧道并记录真实 PID/身份；不重复建同端口。先清除下载子进程继承的 `http_proxy/https_proxy/all_proxy/no_proxy` 大小写变量，再设置本次代理。旧小写 `https_proxy` 曾覆盖新大写设置，导致请求连错7890。

requests/curl/aria2 对环境代理处理不同。aria2 必要时显式 `--all-proxy=http://127.0.0.1:18092`；校园 SSH 仍直连。不要在全局 shell 配置写 token。
