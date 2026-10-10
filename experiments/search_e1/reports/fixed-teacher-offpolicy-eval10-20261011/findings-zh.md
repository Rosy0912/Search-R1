# 固定teacher普通轨迹 BF16 RWM：第10轮512评测

实际完整512：243/512（47.4609375%）；NQ113/256，Hotpot130/256。8worker实际wait均0，完整释放accepted。

同协议source100既有239/512：净多4题、+0.78125个百分点；尚未计算逐题配对置信区间，不能认定确定提升或归因单一机制。旧student采样BF16 RWM第10轮239/512是另一实验结果。

本次权重是10个完整outer后的实际raw CP101/outer9，非best或采样cache。

远端summary：/data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-010/summary.json
SHA256：f9062e599b2c61de517080dedac77236df328ea20b9b7f9c658135e48ef888bd
Checkpoint manifest SHA256：99143040eed32abc670e9979a34b79fbc62993553e81c885fbcbf0a49e5ad38c

新鲜实读1791648830：同owner967452/run5dd5f0d68c4249ae9cb273bed38dabe3已完成17轮、171真实Adam，第18轮采样，无报错。总200及20次正常512目标继续。
