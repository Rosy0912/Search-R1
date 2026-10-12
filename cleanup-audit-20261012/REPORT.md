# H100-8 备份与清理记录

对应会话：01a0fcbb-863f-7450-926d-db537f648265（当前 Search-G1 mixed，历史 GiGPO）。

已更新 GitHub 独立备份分支 `codex/backup-gigpo-cleanup-20261012`，源码、仓库外运行入口及失败摘要已推送。提交 `be1de45ba73bbb90741770638ea793fe86ab196f` 已 fresh fetch 核验，共 7918 个文件。最新4文件×4条持续SSH读取实现、恢复配置、一次性清理控制器已包含。原始会话 checkout 和分支保留。当前远端源码 840 个文件与最新会话源码比对，仅一份 REPORT.md 文档不同，代码一致。

已删除 18 个明确失败实验的 actor checkpoint，删除前逐个核查真实运行命令和文件描述符依赖；没有删除候选被其他进程占用。实测释放 allocated 541,145,194,496 字节，磁盘可用空间从 142.90 GB 到 684.05 GB，使用率从 100% 到 96%。完整路径见 `clean-failed-result2.json`。

保护：现行 Job3381 全部 root、当前恢复源 checkpoint5、当前输出 checkpoint10、所有共享初始模型/检索资产、历史 GiGPO 最佳 GPU512 checkpoint25（EM 26.171875%）、全部待验证备份。旧训练日志保留，另生成远端必要日志归档。

清理前后真实训练从第35轮中间更新到第36轮完整结束，8个 rank 各720次 Adam 写入、所有参数/状态 finite=true，无新增 failure 文件。未停止、重启或热改训练。

HF备份仍进行中：目标私有仓库 `Sutoonq/TMP_MODEL`，前缀 `h100-cleanup-20261012/gigpo/`，checkpoint 文件46个，共101,325,264,987字节；必要日志归档5,272,709,120字节，均采用 H100→mgmt-host有界RAM→HF，Mac没有落权重。已完成的小文件逐项官方 paths-info 或固定 commit 下载SHA验证；大模型上传尚未验收，不能称备份完成。全部原始文件大小/SHA见plan.json。

2026-10-12 15:24 验收更新：历史最佳512测试集 checkpoint25 的三个模型片、tokenizer/config/index及algorithm共13文件、13,604,380,261字节，全部官方固定commit SHA/size验证；必要日志5,272,709,120字节已全SHA验证。再次 fresh source SHA核验并扫描活动命令/FD依赖后，额外18旧checkpoint已删除726,313,934,848字节，无blocked依赖；两轮累计释放1,267,459,129,344字节，删后可用1,319,905,951,744字节（91%）。清理后的训练已从完整outer42正常推进到outer43，8个rank finite，无failure。历史最佳完整训练state及当前checkpoint5额外恢复副本仍在后台上传，remote进程4056726正常，四个state实测读取2.379GB→3.290GB、无新增错误，不能称这部分完成。保护范围不变。
