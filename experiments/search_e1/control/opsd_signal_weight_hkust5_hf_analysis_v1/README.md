# K5固定响应概率与EM分析资源入口

继续原四组100轮、九个同协议HF端点后，复用源224b795的数学、原编码/词表reverseKL/entropy/targetlogp、区域份额、覆盖/NA、Pearson/Spearman/AUC和预先指定的paired McNemar/Holm/bootstrap。frozen-files.json列出十个原文件实际SHA，它们逐byte保留。新增entry只映射四张K5 0–3、远端K4 oracle（绝无本地同PID豁免）、真实all100及原G单独wait0/目标H wait0、资源锁与释放。仅事后teacher-forcing重算已保存响应，不产生新轨迹/评测/训练，不恢复Adam。teacher缺B明确NA；标量loss区域贡献不是梯度或性能因果。

先用terminal_inputs.py保存两个K4原G controller exit（仅2868B）到目标原路径，保留源与unknown原K4整体监督退出。全部100后build_terminal.py只据真实当前K5监督terminal0和四25-row原ledger生成小桥，不填造GP/GN或H退出。adapter/replay读这种新明确lineage包，绝不假冒旧four-group-terminal。目标driver若换身份，独立新版本，不放宽旧守卫。

所有目标脚本从entry.py --publication新pub --publication-sha256实际SHA --target-script自身绝对路径 <原script参数>执行。worker自动带相同entry/pub，parent argv显式保留目标replay.py路径。子进程原数学文件和request不变；父CPU不importTorch，原study/group/shared/replay leases加targetexperiment锁，0–3实际UUID/free和ownrun/父进程祖先及remoteoracle准入。大产物在K5 study/analysis；Mac只留代码/配置/文档/小摘要。

当前只CPU准备，没有九个端点时不得GPU运行。Node integration只检四配置、原helper namespace和两个G退出，不能冒充formal100/effect/replay完成。GP/GN全量训练token/selection等仍需同集群远端可读，不能把仅四CP与两个exit的转移说成全部原始证据已经迁移。全量备份或分析准备不阻塞训练。
