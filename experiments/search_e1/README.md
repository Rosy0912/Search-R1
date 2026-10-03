# Search E1 复现与训练配置对照

archive/ 保存用户提供的 search-e1-reproduction-20261003-233328 压缩包文件；以 PACKAGE_MANIFEST.json 的 301 个 SHA256 为准，原文件内容未修改。内含固定 Search-R1 上游 commit 598e61bd1d36895726d28a8d06b3a15bed19f5d3。模型、训练数据和Wikipedia检索资产不在代码包内。

对照实验都运行 Search-E1 的 GRPO→OPSD→GRPO→OPSD 算法，统一 Qwen2.5-3B-Instruct 初始权重、训练数据划分和评估题集。A 采用压缩包训练配置；B 采用 search-r1 分支训练配置中的 mini-batch、entropy、loss reduction、warmup 与更新预算。两组共同修正最终回答与观测来源隔离评分，历史奖励另行记录。因此新实验不是旧曲线逐点复刻，不能把配置组合对照解释成单个超参数的因果贡献。

代码、资源检查、运行配置和每阶段结果记录在 docs/。实际权重、数据和完整训练日志只保留于任务所属HKUST集群，禁止跨集群导入或调用检索。只有通过真实GPU训练与评估验收才报告训练已开始或效果已复现。
