# A800 GPU4 全量 FP32 index 独立结果审查

结论：**支持进入独立 CPU encoder + HTTP 服务 gate；尚不接受生产切换或训练。** 固定H1009 CPU embeddings的index-only对照已取得真实结果，服务脚本仍须完成独立反馈的资产/embedding绑定与platform条件检查。

| 查询批次 | 实际k | 顺序全同 | 集合全同 | 最大matched score差 | search墙钟 |
|---|---:|---:|---:|---:|---:|
| 14 | 3 | 14/14 | 14/14 | 5.96046448e-07 | 0.195017s |
| 14 | 50 | 11/14 | 14/14 | 5.96046448e-07 | 0.150155s |
| 512 | 3 | 512/512 | 512/512 | 8.94069672e-07 | 0.971029s |
| 512 | 50 | 402/512 | 512/512 | 1.1920929e-06 | 0.980332s |

14query和512重复query的top50变动发生在同三个原query；每query在512中的重复输出只有一种顺序，没有观察到重复行间随机漂移。其余11query的top50排序一致。真实k3调用均完全同序，但不能用k50前3代替它。

| 原query | top50换位 | 文档ID | CPU两个同分值 | GPU14两个同分值 |
|---|---|---|---:|---:|
| 4: Where is Mount Everest located? | 7,8 | 15622197,3199837 | 0.84088164567947388 | 0.84088170528411865 |
| 8: When did the Apollo 11 mission land on the Moon? | 42,43 | 11729919,9602779 | 0.84788930416107178 | 0.84788918495178223 |
| 12: What is the tallest building in the world? | 1,2 | 16864745,7009180 | 0.88336646556854248 | 0.88336658477783203 |

特别是第12题最高建筑：GPU top50把16864745与7009180的第1/2位交换，而独立search(k3)保持CPU的原顺序。因此该候选不满足top50排序等价；Tree生产top3必须实际执行k3，不可先取50再切片。换位文档在CPU与GPU14分别同分，属于并列排名变化。得分差在FP32路径仍存在，不能称逐值精确相同。

资产index SHA69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166、corpus SHA85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9与2854一致。GPU4 UUID固定、useFloat16=False、scratch512MiB，query向量仍是2854 CPU FP16 E5输出后转float32，并未在A800运行encoder。本review重新计算了1052行k3/k50结果的ID集合、rank与matched score delta，核summary准确；对应内容相同来自独立相同corpusSHA和相同ID的推断，没有在该index-only探针重新读取每篇raw文档。

原始cost/samples核对：probe body 44.758524秒、单选定GPU约0.012432923 GPU小时；44条NVML采样均无foreign，最大间隔2.208623秒。观测峰值67521019904 bytes（67.521GB，约62.884GiB），80GiB卡在此观测峰值尚余17.116GiB。这是周期采样最大值，不能称完整瞬时allocator峰值。CPU自身RSS峰65.376GB；CPU index加载32.6695秒、GPU传输6.7404秒。费用范围排除了env准备及既有8GPU平台整段空闲分配；不是财务账单或整个平台作业GPU总成本。

下一服务gate需真实A800 CPU FP16 E5与相同model文件身份，比较新embeddings与2854、实际k3/k50及内容hash；用回环HTTP和既有SSH隧道实发请求，核协议/结果、CPU编码耗时/RAM、GPU驻留峰、foreign守卫和生命周期。该index-only结果不证明A800 CPUencoder parity，也不证明512 unique query吞吐或Search-RL效果；14向量重复到512仅覆盖较大batch执行路径。不得因本报告自动写production acceptance、启动训练或修改当前服务。
