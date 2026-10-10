# 原精度的后置四卡批量512评测

训练先完成四组各25个outer；此入口要求原续训监督实际exit0、100条完整训练EM和四张获准卡释放，才会获取资源锁并使用GPU。准备代码和CPU检查不阻塞正跑。它不启动训练，不停止进程，不修改科学配置。

每个端点仍是HF4.46.3/Qwen2 SDPA、原BF16 checkpoint100底座，加未经合并的实际FP32 LoRA。复用原raw身份读取与504张量/252层的加载检查。原teacher和训练数学不变。旧CPU合并BF16/vLLM候选不用于本次正式测试。

8个逻辑分片每片64题，物理GPU4–7分两波，每卡每次最多32题，一片分两个batch。全部逻辑rank的seed仍42。原greedy/repetition_penalty1.05/EOS151645或151643/500token每轮/8192context/5turn/4search/top3/480obs保持；原每个query独立HTTP、CPU E5请求形状和timeout120保持。只批量模型生成，逐题停止/预算/观测插入遵循原SearchAgent，人工padding不会额外惩罚EOS。

依赖已经发布且不可变的hf-batch-core-0f226d2。该core的3项执行器对照和2项Torch数值测试不重复改写。新入口额外核64题分片、raw完整身份、严格EM及父进程不加载Torch；Node原环境用微型随机Qwen2 CPU模型执行实际HF批量/逐题生成比较，无下载、无实验模型、无GPU。正式运行才确认真实模型耗时、峰值和完整512结果；CPU检查不代表GPU完成。

先新共同zero512，再固定G_POS/G_NEG/H_POS/H_NEG的15、25末raw，共8次正式测试。只有zero的rank0有3个额外单题控制：原SearchAgent、批量入口size1、关闭零LoRA；不进入512分母。各正式输出保留实际ID、完整query/doc、响应、严格EM及相对新zero新增/丢失；每波记录真实wait0和GPU释放。没有逐题重新hash源模型或运行review。每个worker记录模型加载/生成/完整耗时，以实际运行检查提速。

批量padding和GEMM形状可能改变数值舍入，因此全部9端点同一批量协议；不强迫复现历史逐题HF239，不把不同协议相减。新成功参考bank在任何训练后端点评测之前冻结，后续固定历史概率分析使用这批真实响应。实际生成分数未保存，事后retokenized概率不能冒充生成时processed score。

输出全保留HKUST4的study/hf-batched-endpoint-evaluations。Mac只留源码、配置和小摘要。完整终态后运行：

```
CUDA_VISIBLE_DEVICES='' python evaluate.py --mode run \
  --study-manifest /weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261010/receipts/study-manifest.json \
  --study-manifest-sha256 86db5c8e1a86f6b5fd27f880ca062d193b4623e29869e8fa2aa3dba0e8e46c8e \
  --publication ACTUAL_PUBLICATION_PATH --publication-sha256 ACTUAL_PUBLICATION_SHA
```

部分输出或故障保留，在独立恢复副本中继续未完成分片/端点；本入口不伪称已实现自动resume，不重启原训练或填缺失EM为0。

旧GP15的32个保存案例（rank0–3、slots0/1/10–15）累计模型生成400.97秒、HTTP等待1061.89秒，等待含排队，不是服务纯扫描耗时，也不是512总体墙钟。为解决该实际瓶颈，每worker至多4个单query HTTP并发，保持请求边界/FP32 E5/检索参数。日志使用atomic request IDs/写入和明确case绑定；请求可乱序完成，不假称序号代表完成顺序。原训练与旧CPU候选保留。
