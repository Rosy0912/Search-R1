# Preparation-3 19:45 CST 小时只读运行审计

实际区间：2026-10-07T19:45:24.340140+08:00–2026-10-07T19:45:32.319352+08:00。原 controller2459676/starttick915357855、child2460237、run914bfff786c84ede8d82414dff65f60e 未改变。结果为 **RUNTIME_PREPARATION_AUDIT_PASSED**；没有失败检查。

按每个分片固定文件大小读取的八快照合计 **246179 / 845515** 完整 JSON，分片为 30965/30400/30720/30720/30400/31360/30720/30894。各文件在约8秒内依次取大小，不是一次全文件系统原子快照。中间记录的 source question 字段和 sibling 序列全部正确，无重复位置或顺序错误。无 partial-line bytes；rank7最后一个 question已有4条有效 sibling，正处于追加中，不按缺失第5条判故障。rank0/rank7也有正常的未完成64题批尾，详细 remainder已单列，未把当前LIVE边界误当作退出时的完整批证明。

八 worker实际 UID/PPID/PGID/start ticks/argv/E1环境/XFORMERS/cache22671+7281 均匹配。controller/helper/config、305个冻结文件实际SHA与三数据实际SHA/行数/路径身份检查通过。未重复已通过的损失/初始化/验证split测试。

source100逐文件 realpath/device/inode/size/mtime_ns/ctime_ns 与18:45新基线完全一致；基线receipt SHA256为 `a026b58343115a6bfdda1d902f3146d40f3623ceb89d9df66fcf3cf6fc031850`。nanosecond整数按Python原始JSON精确比较，未经过V8 Number舍入。本次前后stat也不变，小配置/tokenizer实际重hash通过；三大权重没有再次读取13GB，而是绑定既有transfer/startup内容SHA和上述未变metadata。metadata稳定不等于新鲜content hash，也不是防高权限篡改的内容证明。

全8卡直接memory.free当前 3718/3718/3714/3756/3756/3714/3714/3177 MiB，当前最低3177 MiB、连续历史最低3099 MiB。候选retriever当前574/峰654 MiB。两个直接显存observer存活，receipt年龄为 2.144/0.150 秒。可读dmesg在新启动16:31:23 CST之后无新增Xid/NVRM/CUDA匹配；journalctl系统日志可见性不足照实保留，旧prep2的80,320条故障与Xid31/MMU证据未动。

仍处于完整新轨迹准备；actual OPSD updates=0，runs/opsd absent，native和external五更新receipt absent。V3保持原路径存活，正常无限等待native五更新文件。有限prepare prefix健康和超过旧行数不证明根因修复或100→105验收。

只写本地新receipt/本文，不改publisher文档或远端code/config/affinity/job，不停止进程，不发送GPU推理/检索负载；只有只读状态查询与health GET。完整receipt `preparation-3-hourly-runtime-readonly-audit-1791373522.json` SHA256 `9dc0b10b048212f21a90d0166012ab1e135a68c6d94f36bb5b11c593fd4ddb26`（198135 bytes）。
