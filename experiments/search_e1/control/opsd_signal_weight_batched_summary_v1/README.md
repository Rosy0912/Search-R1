# 批量端点结果复用原配对统计

只在共同zero与八个15/25端点完整结束后，CPU读取其完整512分片。没有新增推理、训练或显著性比较。转换输入结构后直接调用原冻结训练helper的result_analysis.analyze：第25轮G_POS−G_NEG、H_POS−H_NEG两项exact McNemar/同两项Holm家族；seed42、10000次共享NQ/Hotpot分层题级配对bootstrap。总体及逐集EM、source→15/25、15→25与互斥失败分类均沿用原实现。

旧HF239不参与本次比较；所有4608行必须来自vllm-greedy-rep1p05-bf16-merged-fixed512-v1。实际八worker wait0、每分片64、完整512唯一问题/gold、输出SHA和末raw checkpoint与request绑定必须成立。统计正确不代表推理已经完成。训练loss/信号关系由已有train_signal_analysis.py单独提取，不把新测试题和每轮不同TRAIN题硬连为逐题因果。

运行：原HKUST4 env Python，CUDA_VISIBLE_DEVICES空，summary.py --evaluation-complete 新batch根目录/complete.json --evaluation-complete-sha256 实际SHA --validation 原validation.jsonl --validation-sha256 75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76 --training-helper 原source/training-00fec71/opsd_signal_weight_v1 --output study/analysis/独立新summary.json。大输入/输出只留HKUST4；本地只复制小型聚合结果。当前只做CPU准备，不抢占训练GPU。
