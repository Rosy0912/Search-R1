# Prep3后续OPSD主机RAM只读检查

结论：未发现当前主机RAM实质不足。MemAvailable约885.8 GiB；基于320条已存在轨迹的4倍长度/对象裕量、并保守叠加八rank加载暂存，规划额外峰值约622.8 GiB，小于本次可用快照。它不是未来真实峰值保证；严苛上下文上界另列，不能自动当作OOM证据。当前仍未进入OPSD，无新训练结果。

## 当前实测与限制

/proc/meminfo：总1007.5 GiB，MemFree97.9 GiB，MemAvailable885.8 GiB，Cached781.4 GiB，SwapFree7.97 GiB。大量缓存可回收，不能只用MemFree判断失败，也不把8GiB swap作训练主要容量。

同run-id的生成parent和8worker共9进程RSS48.17 GiB、PSS45.72 GiB；controller2459676另外RSS0.038 GiB/PSS0.029 GiB。相同retriever根标识的5个服务/监控进程RSS69.55 GiB、PSS69.31 GiB。evidence.json逐PID保存cmdline/status/smaps_rollup，仅检查本用户且匹配本run或本检索root的进程。RSS跨进程共享页会重复计数，所以以PSS辅助解释，不把两者相加。

controller所属cgroup及可读父层memory.max/memory.high均max，无额外已读到的硬RAM限制（controller-cgroup.json）；cgroup.memory.current含页缓存/其他同session进程，不等于本作业匿名RSS。所有统计为顺序读取快照，不是原子全主机测量。

## 源码与阶段峰值

pipeline-source.py:117–133对子作业child.wait，134–140先完整generation结束，再另起mine，148–152等mine退出后才另起8rank OPSD；没有让mine全量rows列表与八rank encoded列表常驻叠加。frozen-parallel-rollout.py:52–54等所有8worker退出才merge，随后parent退出；其finally只处理已拥有worker。源码顺序支持生成进程资源释放后再mine；实际转换时仍应由root现有监控核对已退出PID，不能把GPU/CPU后台异常残留当作正常释放已实测。

mine.py:10–12与io.py:4–6一次把845515条JSON全部载入RAM，core.mine_pairs每question_id至多生成1个pair。原输入169103问题×5且ID覆盖校验，因此mined pairs确定上限169103（可少或零）；这不是845515个pairs。分组/选中strings多数引用已有rows内容，mine并非每rank复制；按样本row对象浅层估计均值约6.4 GiB，4倍裕量约25.7 GiB，另有嵌套对象、分组和pairs小容器。此值非深度heap测量，4倍为规划裕量。

opsd.py:95–100各rank先加载同一底座到CUDA、构建LoRA/Adam，再101–103每rankread_jsonl全部pairs、计算全部encoded并建立全部examples，124再shuffle。没有rank提前按问题分片，也没有共享CPU encoded cache；local分片只在130训练batch中发生。因此不能把数据RAM除8。encoded与examples是列表引用，examples不再复制token内容；原pairs（含ref/student字符串）和encoded一直保留到训练后，未显式删除。无encoded磁盘缓存。

## 样本估算，不代表实际挖掘pairs

每rank在5个固定文件字节位置各取8完整行，共320条；仅CPU AutoTokenizer，无模型加载、GPU、检索、额外generation或mine。记录不包含问题/答案全文。用该行response同时作参考长度几何，计算student无reference前缀、teacher带reference前缀；这不是实际选择的pair，教师通常选最省搜索参考、学生按距离选，后续分布可能不同。

响应token平均821.5、p95 1753、max 2376；字符平均3490.0、max 9601。三份token IDs列表＋mask pointer缓存样本平均74.2 KiB、p95 147.0 KiB、max 195.5 KiB；使用实际CPython3.10 int28字节/list header56字节，mask0/1用共享小整数、每项只计指针。max169103 pairs下encoded平均估算每rank12.0 GiB、八rank95.7 GiB；所有pair都按样本max时八rank252.2 GiB。

| 规划额外峰值分项 | GiB |
|---|---:|
| encoded_4x_sample_mean | 383.0 |
| pairs_4x_two_sample_rows | 82.2 |
| model_load_private_fp32_plus_bf16_allowance | 151.9 |
| metadata_and_tokenizer_allowance | 4.0 |
| simultaneous_adam_cpu_serialization_allowance | 1.8 |

保守将整个源FP32 checkpoint加一份BF16转换体积算作每rank私有加载暂存，即151.9 GiB；现实safetensors文件页可共享且底座随后在GPU，CPU加载暂存先于encoded，但不同rank进度不同，规划仍故意把二者叠加。已保存基线/updates1/5时CPU序列化Adam暂存另按八rank合计加入（真实optimizer常驻GPU；不能把GPU模型/activation直接算作主机RAM）。未来参数/RNG/checkpoint内存并未实测。

## 严苛上界与8192上下文

encoded_pair在返回有效缓存前要求两条prefix＋同一response分别≤8192，并过滤response空/zero政策mask。设P学生prefix、T教师prefix、R响应token：P+R≤8192、T+R≤8192。只用上下文条件的宽松IDs/mask包络约785.1 GiB（八rank×169103），未含raw pairs等。这是假设所有pair都达到上界，不是当前占用。

额外只读流式统计train问题字节长度（非split测试）：169103条中max UTF8 655 bytes，固定student模板593 bytes，完整prefix字节最多1248。实际Qwen2TokenizerFast使用ByteLevel BPE，合并不增加初始byte项数，特殊标签计数也小于原字节，所以P≤1248是保守token上限。有效IDs数量P+T+R≤1248+8192=9440；mask长度粗界8192。按ID项38 bytes（28字节PyLong＋10字节含list过分配指针裕量）、mask项10 bytes、固定512字节容器，每pair包络441,152 bytes，八rank有效encoded约555.8 GiB。None项、examples引用容器约几十MiB，不改变数量级。

这个更紧的encoded包络仍不是整个OPSD RSS严格上界：未界定原始JSON字符串的Unicode储存、tokenizer临时offset tuples、加载allocator残留、其他进程未来用量。pair_loss:29–30两份input_ids是逐pair的GPU int64张量，不是把全169103缓存转成Tensor；CPU构造临时最多约2×8192×8 bytes/rank，不随pairs总数常驻增长。teacher/student隐藏层、logits是GPU内存，另由既有GPU门禁处理。

规划额外峰值比当前MemAvailable低约263.0 GiB，且未借用当前生成worker退出可释放的约46GiB。没有观测到实质RAM不足；理论极端不能当自动停训依据。建议root沿已有监控在generation退出→mine→八rank读pairs/encoded时留意MemAvailable与PSS趋势；本审查不新建monitor、不改运行/affinity/publisher、不停止任何进程、不重跑loss或split测试。
