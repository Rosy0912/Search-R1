# 同精度HF批量端点的最终分析

仅接收hf-sdpa-greedy-rep1p05-unmerged-fp32-lora-batch32-fixed512-v1的9个真实完整端点、八worker实际wait0和每波GPU释放。adapter.py还核实际loaded504张量FP32身份，不接收合并BF16/vLLM结果。原训练100轮先完成；不添加生成、检索、训练或优化器。

复用已发布batched-replay-v2的输入转换，改协议身份和实际加载检查。replay_math.py/replay_worker.py逐字节不变：普通student entropy包括无成功参考的题；teacher/g/KL缺B或上下文超限明确NA。共同参考bank只使用新同协议source100正确响应，实际8端点前冻结；不补造参考，不复用旧HF239 bank。事后重编码的HF概率不是greedy时含repetition处理的分数，也不是历史实时训练loss。完整tokens和分布仍远端。

summary.py保留原统计：EM与gap/entropy/KL的Pearson/Spearman/AUC、控制长度/搜索/数据集、区域贡献与覆盖。常量或单类别返回NA；不同自主响应不能按token位置配对。区域loss贡献不等于梯度或因果，成功参考筛选使baseline参考子集只有正确类，这个AUC必须NA。训练s/5关联仍用原train_signal_analysis.py，与heldout512分开。

em_summary.py复用原result_analysis.py的两项主对照G_POS25−G_NEG25、H_POS25−H_NEG25，exact McNemar、两项Holm及seed42/10000次分层共同配对bootstrap；15轮为中间端点，25轮为最终主终点，不自动增加E控制或重训。

运行顺序：所有100轮和9个完整同协议端点后，用adapter.py生成私有请求的新analysis目录；独立run及原资源锁下执行replay.py；最后summary.py与em_summary.py。CPU准备/fixtures不是GPU完成。全部大产物留HKUST4；本地只留代码、配置、文档和小摘要。
