# W 十轮闭合与持久证据状态（2026-10-09）

16:31:46–47 CST 的原始只读记录确认 W 完成 10 个 online outer、100 次实际 Adam 更新、3018 对可用监督、550282 个有效 token。原 group controller 的父进程实际 wait 返回0，原终态及 owned/after release 票据通过。W 末检查点为 update-00000100-outer-00009，manifest SHA256 `2db5b57e9fd60d4ab034cf75870ddb9c3b17ebc59e69dc5d94d9734cccd7536d`。固定 source100 没有重新训练 GRPO。

全研究当前真实闭合训练为 50/70 outer、500 次 Adam、15115 对曝光、2947399 个有效 token。C0/C1/M/R/W 各十轮，P/J 各十轮仍待执行。W 的同条件 native512 已由现有 study driver 自动启动；本次快照仅36/512、八worker运行，尚无完整 EM。已有完整正式结果仍为6模型、3072/5632预测、8/18比较、24/54分层；不能把部分评估算入正式结果。

末 CP100 已仅一次独立 CPU 实读：八rank实际 Adam100 的 moments/groups/步数逐项一致且有限；504个 FP32 adapter 张量全有限且相对共同初始适配器全部改变。只验证此保存点，不声称每个历史更新均独立读过，也不证明 CUDA/NCCL/RNG 实际恢复。原运行源码、科学参数和自动 W评估→P→J 顺序未变。

R/W 保护检查点0/1/5的原已验收 tar 已分别压为499521972/499795968字节 XZ。两者均从磁盘独立完整解码，单CRC64流、原 tar 整体 SHA/字节精确一致。实际压缩父进程返回0后，只清理各自42份已验收 loose 重复成员和原 tar；86项删除均有持久日志。当前持久表示为 XZ 与原验收/解码/清理链，旧 receipt 描述的 tar/loose 路径不再存在，不能继续声称它们仍在。逻辑净减少2359644077字节，不冒充 APFS 实测净增。Node原资产、失败partial及其他组文件保留。

W全部十轮非cache原始轨迹、HTTP、teacher/student、逐token日志及8份NPY，连同7项原闭合配置/终态引用，已实读固定清单4088文件/2426412462字节并启动独立本地压缩传输。当前只有清单及启动证明，不是备份完成；实际传输、source-after和完整本地成员验收仍待完成。其他未列入本 scope 的全局记录及研究词表ledger保持最终交付待办。备份和CPU工作均不阻塞GPU评估或P/J启动。

新增 `control/opsd_ablation_recorded_precision_history_v1` 是只读本地报告入口。待P/J真实闭合且固定原始备份验收后，汇总所有保留attempt的 precision JSON、已提交更新scaler、retry manifests及原调用/退出记录。缺rank、缺调用、失败记录均标PARTIAL；未读取capsule PT，不把引用关联当作优化器、CUDA或RNG恢复证明。只有两份AST和一个缺rank/缺调用的小fixture，未新增模型、生成或训练测试，尚未运行真实P/J聚合。

16:35 CST再次验证本机五台HKUST临时无代理入口均实际登录0、stderr为空、耗时0.50–0.58秒；普通别名仍被旧ProxyCommand立即关闭。八份本地连接记忆按实际结果更新，全局SSH/VPN配置未改；连接失败不能解释为实验故障。HKUST4原archive和他人进程保持原授权范围不变。

目标继续ACTIVE：W完整512结果→P/J各十轮及相应评估；完成70outer/11模型5632预测、18比较54分层、全debug/词表ledger/检查点及轨迹持久索引，再交中文分析和曲线。不从稀疏保存点插值精确退化时间，也不将目前小差值表述为已证明算法收益。
