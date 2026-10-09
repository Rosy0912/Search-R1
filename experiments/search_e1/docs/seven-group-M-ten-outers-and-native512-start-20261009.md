# 七组 OPSD：前三组训练闭合，M 进入评测

2026-10-09，原八卡队列保持运行。C0、C1、M 均完成十个在线 outer；M 的原控制器实际 wait0、完整释放后已自动启动同协议 native512，八 worker 已产出正式题目记录。M 512 尚未完整验收，不能给出 EM 或把训练 reward 当评测收益。备份与执行并行，未成为启动条件。

原生闭合训练记录覆盖30/70 outer：C0 100次、C1 101次、M 100次实际 Adam，共301次；累计9067对可用配对曝光、1849553有效策略 token。M单组3032对、637380 token。R/W/P/J尚未开始，空槽保留。各 outer 的 TRAIN cohort 不同，图中 reward 不是固定验证曲线。

[原生训练曲线](../reports/seven-group-recorded30-outers-progress-20261009/seven-group-recorded-curves.png)与[70槽记录](../reports/seven-group-recorded30-outers-progress-20261009/seven-group-70-slots.csv)来自原提交记录，无新训练或重新生成。M 的目标是 mixture，目标 KL 和 C0/C1 的教师目标 KL 不能按绝对大小直接比较。M第十轮307条配对，目标 KL 为0.00651303，学生对教师 KL 为0.40667690；小目标 loss 不能解释为教师匹配更好或 EM 提升。全 token 的 clip probability mass 与完整 raw 证据对账仍待闭合备份后处理，现有原生 component clip计数不能替代它。

已完整接受的正式评测仍为2048/5632：BF16零适配器239/512，FP16零适配器245/512，C0 BF16为227/512，C1 BF16为236/512。M及后续结果未计入完整覆盖。各组最后保存状态的独立 CPU tensor 读取与文件哈希验收分开记录；M最终100步已用原只读工具独立CPU实读：八rank实际Adam100与moments/groups一致，504个FP32适配器张量全部有限且相对共同初始值改变，原RNG/scaler/cursor检查通过，CUDA未初始化。它是保存状态数值验收，不能替代512效果结论。

新闭合 raw 备份实现已提交 adc92f57ed9c9b5e105374c3bfb5d62ba65a09de 并从远程逐字节回读。实际生产训练仍871dc46f及538ac99a精度修复副本，评测仍72e1e127；文档/备份提交不代表热改生产。原节点、失败日志、源模型和检查点保留，完整70轮与5632题及中文配对统计尚未完成。
