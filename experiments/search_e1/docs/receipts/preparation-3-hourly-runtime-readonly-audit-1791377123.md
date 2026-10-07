# Preparation-3 20:45 CST 小时只读审计

审计实际区间：2026-10-07T20:45:24.864726+08:00–2026-10-07T20:45:34.075017+08:00。结果：**RUNTIME_PREPARATION_AUDIT_PASSED**，没有失败检查。实际阶段仍是 FULL_FRESH_TRAJECTORIES，原 controller2459676/starttick915357855、child2460237、run914bfff786c84ede8d82414dff65f60e 均正确；未把未来正常的阶段转换/生成父进程退出当作故障。

八个固定文件大小快照合计 **322240/845515** 完整记录，分片 40640/40000/40000/40320/40000/40960/40000/40320。各分片在审计过程中依次定界，不是跨文件原子瞬时快照。完整 JSON 的 source question 字段和 sibling 顺序正确，无中间无效/重复位置/顺序错误。本次 partial bytes、不完整 k5尾与未完成64题批尾均为0。

八worker实际UID/PPID/PGID/starttick/argv/E1环境、XFORMERS和GPU22671/CPU7281缓存正确。控制器/helper/config实际SHA、305冻结源码实际SHA与三个原数据SHA/行数/realpath通过；未重做loss/init/split的科学测试。

source100的realpath/device/inode/size/mtime_ns/ctime_ns，与19:45基线 `preparation-3-hourly-runtime-readonly-audit-1791373522.json`（SHA256 `9dc0b10b048212f21a90d0166012ab1e135a68c6d94f36bb5b11c593fd4ddb26`）逐项一致。以精确Python整数比较纳秒字段，审计前后stat也稳定；小配置/tokenizer实际重hash匹配。三大权重未重复13GB读取，继续绑定accepted/startup content SHA与当前metadata；未声称metadata就是fresh content hash。

检索候选identity保持原PID2239529/digest/ready/GPU ready，未请求HTTP推理或检索。directfree当前 3718/3718/3714/3756/3756/3714/3714/3179 MiB，最低3179 MiB，历史最低3099 MiB；retriever当前574/峰654 MiB。两个directfree observer存活，receipt年龄 1.002/0.147 秒。可读dmesg在16:31:23 CST新启动以后无新增Xid/NVRM/CUDA匹配；journalctl系统日志可见性不足仍说明，旧80,320条失败/Xid31证据保留。

**仍0次真实OPSD更新**，runs/opsd absent，native/external五更新文件 absent；V3正常存活并无限等待native五更新文件。准备prefix健康不是100→105验收，也不是旧CUDA根因修复/bitwise复现证明。

本次仅写独立本地JSON/分析。没有远端写入、修改job/runtime/env/affinity/publisher、停止进程、新GPU工作或检索负载。证据：`preparation-3-hourly-runtime-readonly-audit-1791377123.json`，SHA256 `34b44ac43facc45ffddcff47464fd69adb7f70f209cd46cf39ba5747fb3c74c6`（202320 bytes）。
