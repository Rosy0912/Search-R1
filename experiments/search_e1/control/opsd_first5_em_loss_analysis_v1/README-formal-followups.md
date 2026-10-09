# Formal512 既有结果的两项追加分析

`formal_retention.py` 和 `formal_overlap.py` 是本次成功本地 `python3 -` stdin 执行源码的逐字持久副本。二者均与私有 execution record 中的原 stdin bytes 精确一致，并实际 return code 0。它们不是参数化通用 CLI，也不声称未执行版本已验收。当前固定输出已经写入；需要重复分析时应使用审查后的新独立输出目录，保留既有产物，不能直接覆写本次证据。

`formal_retention.py` 读取已核 SHA 的四模型 512 同题配对表，按“共同起点正确、C0 错误”定义事后探索子集，统计 C1/M 仍正确时的严格最终行动、搜索次数与终止。这里只描述原 10 outer 末 greedy 行为；不将其当作前 5 outer 的评测，也不从搜索次数同现推断因果。保存无题目 ID/原文的聚合与私有槽位清单，后者不入 Git。

`formal_overlap.py` 读取已经实际闭合的本地 7,680 条 TRAIN cohort features、其原 SSH exit0/end summary 及 formal512 的原语义身份。它核每组五轮的 2,560 unique qid、三组同轮 qid/四字段 semantic SHA/dataset 映射相同，以及与 formal512 的 qid/完整行 SHA 交集为 0。四字段 semantic SHA 包含 qid，因此这不证明不同 qid 不会有相同问题文本；该独立文本重复检查未完成，不声称文本完全分离。没有将不同 cohort 的训练 loss 配给 formal EM。

两项均只读取已经存在的本地私有证据，并生成本地私有结果。没有额外 HKUST 读取或大文件传输，没有 QA 重新评分、模型加载、Torch/GPU/HTTP、训练/队列/进程操作、Node 写入或 Git 操作。正式 greedy 单次每题结果不提供单题 pass 概率，也没有 generate 时的 logprob 或完整分布；TRAIN s/5 是外轮采样权重下五条原生生成的经验频率。
