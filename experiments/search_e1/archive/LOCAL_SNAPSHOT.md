# 本地代码快照说明

代码来源：Lumia /home/jmzhang/Workspace/Search-E1-20261001，状态核查时间：2026-10-03 23:31（北京时间）。

包含当前 search_e1、scripts、configs、tests、pyproject.toml，以及固定 commit 598e61bd1d36895726d28a8d06b3a15bed19f5d3 的 Search-R1 源码与许可。LUMIA_INSTALLED_PACKAGES.txt 记录服务器环境版本；CUDA/FlashAttention 依赖需匹配目标服务器安装。SOURCE_MANIFEST.json 是服务器原文件校验值，其中 README.md 保留为 README.lumia_snapshot.md；README.md 使用本地最新运行记录。PACKAGE_MANIFEST.json 是最终代码包文件校验值。

模型权重/checkpoint、完整训练/测试数据、Wikipedia 检索库与索引、虚拟环境不在包内。脚本保留服务器绝对路径，换机器运行前需配置模型、数据、检索服务和 Slurm 参数。本包不能直接在 macOS 上完成 CUDA 训练。

首轮137/200步，训练异常观察中；OPSD作业133236质量hold仍有效，等待150步独立验证后决定释放或修复。七个完整测试集没有最终结果。不要直接重复运行 submit_phases.py 对当前实验重复提交。
