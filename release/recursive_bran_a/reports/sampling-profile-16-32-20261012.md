# 16/32条完整轨迹的实际速度诊断

两次独立GPU0短实验均已正常退出（tmux原pane dead=1/status=0），summary完整；最后实查只剩获准的GPU7 oracle，GPU0已释放。没有训练、Adam更新或512评测。

模型：同一官方Qwen2.5-3B-Instruct本地权重，BF16/XFORMERS/eager，GPU利用配置0.4、temperature1、top_p1、max_tokens500；同一前32题原始parquet提示、seed42、最长4turn、真实18004检索。每条件先16条，再32条。输入prompt SHA为`a9872b57893efb7193bc79225b02c3214d5deb55e47a4644cba302912468d1a0`。上下文上限4096；这是单GPU微实验，不代表整个八rank/FSDP/Ray树构建耗时。

| 条数/停止方式 | 总秒数 | 实际生成秒数 | 交互/检索秒数 | 前缀扫描秒数 | tokenizer绑定秒数 | 生成token | 丢弃尾token |
|---|---:|---:|---:|---:|---:|---:|---:|
|16 / EOS或500上限|32.204|14.339|17.001|0.272|0.315|4473|2077|
|16 / 行动标签stop|30.853|11.973|18.199|0.137|0.263|2607|0|
|32 / EOS或500上限|49.197|19.504|28.517|0.498|0.316|9197|4305|
|32 / 行动标签stop|39.580|11.221|27.423|0.215|0.350|4620|0|

## 支持的定位

1. **等待检索返回是此短实验的最大项。** 32条原路径交互28.517秒，占总时间58%。标签stop后的16条首turn的cProfile中，交互9.671秒，`requests.post → socket.recv_into`9.646秒，观测文本tokenization仅0.018秒。不是将全部交互时间猜成HTTP，而是实际调用栈计时支持。原f667 manager没有独立HTTPtimer，旧摘要中的0是缺项，不能解释为HTTP零耗时；新helper将它记录为null并保存实际cProfile。
2. **生成标签后的废弃尾部显著增加decode工作。** 原路径16/32尾部比例46.44%/46.81%。32条标签stop对照实际生成耗时下降42.47%，整体下降19.55%；16条整体仅下降4.19%，检索时间波动抵消部分生成收益。因此标签停止是有效的局部优化，不能宣称它独自解决35分钟或保证8分钟。
3. **逐token完整前缀decode有O(L²)实现，但本组不是大头。** 原32条扫描0.498秒，占总时间约1.01%；首turn2961次native decode累计0.095秒。优化它合理，但不能把20分钟主要归因于它。

## 边界

- 真实生成输出只在远端进程内使用；仅小型aggregate和cProfile表持久保存，无问题、token IDs、HTTP文档、权重转到Mac。
- 字符串stop只是诊断对照，还不是生产native mu停止证明验收。当前canonical parser有SEARCH-first语义，EOS路径可能在首次行动标签后继续生成其他标签，因此stop也可能改变后续history；不能宣称逐题轨迹bitwise相同。两组32条active序列分别为32/32/2/1和32/32/3/1。
- 此处KV常驻；未测FSDP权重同步/Ray collectives，也未独立测每次KV重建成本，不能将其算作已实测的主因。
- 测量冷启动load约9.7/9.9秒，与表中的条件时间分列，不能当纯采样耗时。

## 原始小摘要身份

- EOS：`/data/shaoshitong/recursive-bran-a-8gpu-20261011/small-profile-v3/runs/current-eos-16-32/summary.json`，SHA `02e1ea3dd4828cbbad00652504989c9eaa1fef43100d45f012c840758378a078`；原pane `%107`，实际exit0。
- 标签stop：`/data/shaoshitong/recursive-bran-a-8gpu-20261011/small-profile-v4/runs/current-tag-16-32/summary.json`，SHA `b2a905e75cd48ee2377060288cc790aa23495ce6882118318ab0ded5c6fe038a`；原pane `%108`，实际exit0。
- 独立helper `release/recursive_bran_a/profile_small_sampling.py`，EOS版本`4b5fa037`；标签对照版本`8e791348`（仅额外交互cProfile/缺项null）。原生产模块仍来自`source-mu-meta-f667a2d`，没有热改。
- 首helper标准vLLM CachedTokenizer特殊ID集合与生产HF身份合同不符的失败保留在small-profile-v2；修复只使用同模型真实HF tokenizer，没有伪造身份属性。
