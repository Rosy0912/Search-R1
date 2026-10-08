# C1 已完成 outer0/1 的原字节压缩备份

当前人工授权要求保留所有实验输出并持久备份，且备份与GPU启动/训练无依赖。本项沿用已有可执行的 source100 直连 tar→Mac gzip→完整成员SHA方案，只替换固定C1闭合范围。它不改变实验、采样、损失或评分。

选择单个压缩容器保存全部非缓存普通文件，包括 JSON/JSONL/log 和可能存在的 NPY。NPY也只按原文件字节归档、完整解压成员SHA核验，完全不调用数组反序列化或重新保存。相比裸rsync，可避免再放一份数百MB的裸文本；相比分别处理文本/二进制，单一完整清单减少两套复制之间的覆盖缝隙。原Node所有文件保留，Mac仅新增容器及证据，没有删除资产。

固定范围是同一 C1 run a1a56efd32e5456d8c6813454975e38f、config13fdbd51…，outer0/1 的真实完成票据必须引用 CP11 SHA09a5e605…及CP21 SHA681e246b…。读取完成票据及固定CP/config的原字节身份，核accepted/run/outer/实际更新，再列举该两闭合目录全部普通文件。只排每轮 `merged-model-cache` 可重建权重子树，保留合并的输入/输出身份、退出及设备证明。为缩小实现，不扣除任何已备份的小CP闭包文件；重复小控制文件允许。所有失败attempt和日志仍纳入，活跃outer2及global mutable状态不纳入。

原 V2 backup.py 的路径/固定句柄SHA/前后metadata与stream_archive保持不变。库存预检每个USTAR相对名字可编码。完整source SHA库存→固定stream中再核每个句柄SHA→Mac gzip level1→Node闭合目录再列举与metadata稳定→独立从Mac完整解压逐成员SHA与严格普通USTAR/唯一覆盖/尾零/CRC→原子改名和fsync→独立完成票据。动态实际数量/字节用于空间判断；不沿用source12或full192数量常量。

空间门限保守按原始tar上界与gzip膨胀上界加5GiB。中断保留partial与Node原件；复制或空间问题仅形成备份待办，不影响正在运行的C1及后续10轮/512流程。不新增模型、GPU或重复科学测试，只做语法与独立源码审查，以及这一真实闭合范围的复制验收。后续范围需新的固定已完成轮次与检查点输入，不覆盖本次scope或给mutable latest猜身份。
