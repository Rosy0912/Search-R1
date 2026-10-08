---
name: h100_retained18_route_20261008
description: 2026-10-08已验收的H100保留18转发路线、密钥选择、主机校验及节点退役时间；替代旧16默认入口。
type: reference
---

# 当前H100连接：保留节点18

北京时间2026-10-08 10:37–10:45实际验收。mgmt-host→root@10.33.0.18，hostname=h100-10；经18转发成功到8(h100-8)、5(h100-5)、14(h100-11)。Slurm可从18直接查询。8和11控制器实际回读各8张H100成功。

必须使用mgmt-host已有 `/home/lichen/.ssh/id_ed25519`。专用 `id_ed25519_claude_h100` 被18拒绝。不要套用旧16/Tailscale默认入口，也不要依赖18上的worker别名或18自身登录worker的私钥（已测到DNS/认证失败）。使用SSH -W转发，最终worker认证在mgmt用同一已有key完成，不复制私钥到Mac、18或其他节点。

所有SSH层：env -u LD_LIBRARY_PATH、IdentitiesOnly=yes、BatchMode=yes、StrictHostKeyChecking=yes、UpdateHostKeys=no。不能为连通而关闭主机校验、更改全局SSH/auth.json/VPN或反复尝试旧密码。

14的管理机全局known_hosts原来没有条目；现在使用从已严格认证的18读回的既有公开host keys，并与实际14的严格SSH成功连接核验。只参考公开主机密钥；不是复制登录私钥。完整非秘密路由配置同目录 `reference_h100_retained18_route_20261008.public.json`，含该可信公开host key参考。

## mgmt-host上复用已验收路径（默认只读）

以下只读例子可把target换为h100-10、h100-8、h100-5、h100-11；command换成已授权的只读查询。不要直接把裸h100别名交给mgmt SSH，别名未配置。

```python
import json, shlex, subprocess
from pathlib import Path
cfg = json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
target, command = 'h100-8', 'hostname'
ip, gateway, user = cfg['targets'][target], cfg['gateway_ip'], cfg['user']
prefix = ['env', '-u', 'LD_LIBRARY_PATH', 'ssh', '-T', '-i', cfg['management_identity_file'], '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=8', '-o', 'ConnectionAttempts=1']
args = prefix.copy()
if ip != gateway:
    args += ['-o', 'ProxyCommand=' + shlex.join(prefix + ['-W', '%h:%p', user+'@'+gateway])]
public_keys = cfg.get('trusted_target_host_keys', {}).get(target)
if public_keys:
    args += ['-o', 'KnownHostsCommand=' + shlex.join(['/bin/echo', '\n'.join(public_keys)])]
subprocess.run(args + [user+'@'+ip, command], timeout=40, check=True)
```

%h:%p由mgmt端目标SSH展开一次，不改为%%。Mac最外层到mgmt的已有7892路由沿用；不要把内层%h/%p放进Mac的外层ProxyCommand。

## 退役和失败分类

- 10.33.0.13=h100-9（Tree-GRPO），10/8释放；10.33.0.16=h100-16（CoSearch／旧跳板），10/8释放。当前现有凭据均拒绝，实时Slurm节点表已不含二者。未核实源盘数据是否保留；本记录不等于已完成迁移。
- 10.33.0.8=h100-8，10/20释放；10.33.4.14=h100-11，10/21释放；10.33.0.5=h100-5，10/23释放；10.33.0.18=h100-10保留。用户未提供当日具体释放时刻。
- 当前Slurm只列5、8、10、11。10:38看到3304(OPSD)/3315(BAPO)/3329(Search-G1) RUNNING；3332(GiGPO) PENDING(Resources)；3224(TOP-D学生) PENDING(DependencyNeverSatisfied)。3333/3211/3228不在队列。accounting storage disabled，不能查出其历史终态。
- 队列活跃不等于GPU训练；需读GPU/进程/日志/产物，过期证据标未核实。不发消息或唤醒其他实验agent，不停止或热改训练，不因退役通知擅自暂停整个goal。
- 旧17份模型导出的H20/HF验收不覆盖全部最新checkpoint/optimizer/检索资产。迁移与最新实验源码Git提交推送须单独核对远程SHA和文件哈希；本记录没有宣称最新实验资产全部安全。

旧reference_cluster_access/project_training_cluster/project_network_env中16入口、旧专用key和旧备用密码方式仅为历史记录，当前连接以本文件为准。未来真实验证失败应保留副本、记录错误，不强制覆盖host keys、不重新提交训练。


## 实验输出持久保存（用户2026-10-08明确要求）

所有实验输出必须保存到经实际验证的同集群共享存储，禁止仅存于计算节点本地盘。启动前核验真实挂载来源及共享服务端释放日期；详情见 [共享存储强制规则](experiment-outputs-shared-storage.md)。
