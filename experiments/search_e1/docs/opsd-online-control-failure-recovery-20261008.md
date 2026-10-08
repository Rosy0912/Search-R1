# 在线 OPSD 首轮控制交接与恢复

2026-10-08 人工已确认 KL 词表分量上限10、200外层轮次以及HKUST节点本地输出。正式run `1d1d67b60f0e46a2afc0536a36ab69b1` 的首轮八分片各完成320条，512题共2560条，独立核验输出、生成trace、完整检索HTTP和原生评分。CPU预览327对；尚未创建训练输入或执行Adam。

原控制器退出于 `wait()`：它将运行检查异常锁存为failed，最终拒绝进入下一阶段。旧代码未持久化具体检查异常和实际子进程退出码，因此不能认定是进程退出竞态，也不能把完整worker产物等同于真实exit0。日志未见CUDA/OOM异常；直接显存观察最小余量3175MiB，低于准入下限的采样数为0。

保留原source、immutable config、runtime305、attempt0全部产物及旧日志。外部恢复控制器仅修复等待与证据记录；逐次记录门禁异常，真实非零退出仍拒绝，门禁未知时保持资源锁，最终退出与GPU释放审计通过才能继续。原Config SHA `d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17` 和全部科学函数保持。

恢复使用同run/config的显式resume。原首轮没有可靠退出证明，所以新attempt完整重新生成同512题×5，不拼接旧prefix，不冒称已训练续步。当前0Adam，完整训练尚未验收。五真实更新验收必须检查实际八rank Adam5、有限且改变的adapter以及鲜活显存证据；随后同一完整run继续200outer。

已实际核验旧controller和八sampler全ABSENT，原tmux dead=1/exitstatus=1，GPU仅私有18004检索2239529。原门禁和完整同UID进程释放扫描均通过；terminal proof SHA `b8978990c3c276570383e82f280eef0d52a7f39d3b73992f8e11e595e8a0484a`。未发送任何恢复停止信号。K4 archive继续原作业，最新只读GRPO169，reward0.488/gradnorm0.651有限；检查点和512最新150，不能当作169验证。

本地私有失败证据：`work/search-e1-analysis-output/opsd-online-launch-20261008/first-sampling-control-failure/`。原始问答/轨迹/token及权重不入Git。此前每小时跟进现已按最新人工重启与原小时监控授权恢复ACTIVE，仅重要失败、恢复、验收和评测通知。
