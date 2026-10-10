# HKUST5 原精度批量评测入口

迁移仅改变节点资源和原监督终态。此入口取代误准备的HKUST5 vLLM评测候选，未启动任何GPU。遵循原计划HF SDPA/BF16底座、未合并FP32 LoRA；八逻辑64题分片，每卡batch32，rank seed42，普通greedy/repetition1.05及原搜索预算不变。evaluate、worker、metrics、bank、HTTP和dependencies源逐字节同已发布541075b，原64125f9批量core不变。仅编译映射物理0–3、原K5真实all100/退出及较长释放等待。

使用新source-manifest及publication；原b77绑定cdd和同tmux监督终态不变。先训练四组100轮，之后共同zero及八raw15/25端点。无需CPU合并权重，不能使用best/cache替代raw。每卡至多32题批量、每worker至多4独立单query HTTP并发保持原接口。实际生成与模型加载/HTTP耗时、全512覆盖、八worker退出及GPU释放必须实测；CPU准备不是速度或效果验收。

父进程CVD空，不importTorch；持原共享study/group锁及target experiment锁。大产物HKUST远端；Mac只有代码、文档和小摘要。GPU开始前同绑定/实际UUID/directfree/无外来任务/oracle核验。若监督身份变化，独立新绑定，不放宽旧终态。

正式入口entry.py --binding原路径 --binding-sha256 cdd... --mode run --study-manifest原路径 --study-manifest-sha256原SHA --publication新路径 --publication-sha256新SHA。子进程自动带绑定；原训练不改。后续仅使用原HF批量analysis；误准备vLLM candidate保持历史且禁止正式运行。
