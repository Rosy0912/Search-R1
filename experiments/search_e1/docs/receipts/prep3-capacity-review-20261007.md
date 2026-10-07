# OPSD preparation3 存储与后续容量只读检查

结论：本次快照未发现磁盘容量实质不足。/data挂载/dev/sdb1（ext4），普通用户可用1.594 TiB（1752.7 GB），inode可用456,259,335。df虽然显示95%占用，但应按f_bavail而非root可用f_bfree估算。没有quota命令，挂载选项未显示usrquota/grpquota；未安装或修改软件，无法据此确认是否存在未暴露的账号/项目配额。空间为共享快照，不能保证未来其他用户不消耗。

## 实际本作业文件

基准run914bfff786c84ede8d82414dff65f60e，controller2459676、生成parent2460237，evidence.json保存只读/proc命令身份。采样开始UTC 2026-10-07T11:26:52.771624+00:00。8个parts合计0.839 GB，worker日志识别到221,760/845515条；比委托时219200推进了一批。stats和日志在持续写入中顺序读取，非原子snapshot；日志只统计固定question_id/sibling进度行，不扫描全部轨迹文本。每条观测均值3785.0 bytes。

| rank | parts bytes | worker日志已完成轨迹 |
|---:|---:|---:|
| 0 | 106,069,473 | 27,840 |
| 1 | 104,732,321 | 27,520 |
| 2 | 103,727,822 | 27,520 |
| 3 | 105,420,087 | 27,840 |
| 4 | 103,961,459 | 27,520 |
| 5 | 106,224,220 | 28,160 |
| 6 | 104,574,277 | 27,520 |
| 7 | 104,654,321 | 27,840 |

本次merged目标目前0 bytes；runs/opsd目录未产生，因此0OPSD，未把估算称为已保存模型。当前runs/preserved/logs/receipts非symlink文件逻辑总量1.570 GB（含早期失败parts，未删除），既有资产与旧输出已包含在df的占用中，不重复扣减。

## 按实际源码估算后续磁盘

frozen-parallel-rollout.py:23先创建空merged，59–70等所有worker成功后逐行写入同一内容；没有删除parts。因此训练完整parts的线性预测为3.200 GB，最终parts＋merged约6.401 GB，非“合并只剩一份”。早期失败结果仍保留，已由当前df计入。pipeline-source.py:134–158是一轮全量generation→mine→native OPSD→最终51713条k1八rank评测，没有额外GRPO或第二次同规模训练generation。

以轨迹文件均值放大4倍作容量计划（是保守裕量，不是token字节硬上界），完整训练单份按12.80 GB预留。mine.py:10–13整体read_jsonl后选pair写JSONL；每问题最多1pair、上限169103，含reference和student两条选中轨迹。为避免未知成功率/长度估计，pairs磁盘同样预留12.80 GB，远高于通常按5条取2条、每问题一pair的平均值预测。

frozen-opsd.py:101–103的pairs、encoded、examples只是每rank RAM中的Python列表，没有持久化encoded cache、没有此阶段数据下载。磁盘encoded cache额外0；不能把RAM缓存误算成硬盘文件。模型是LoRA训练，不是全底座Adam。按config层36、hidden2048、kv256、intermediate11008、rank16、七个target线性层计算LoRA共29,933,568参数，FP32约0.120 GB；Adam两份FP32矩约0.239 GB/每rank。

opsd.py:167–170只在update1和5保存8rank optimizer/RNG state，共16份，预计Adam矩约3.831 GB，另加小量step/metadata/RNG；不是每个更新保存8rank Adam。baseline best、update1、update5、epoch1四个adapter目录约0.479 GB。best被更新覆盖，不新增无限目录。CPU merge保存一份全模型（199–203）；实际source100 safetensors13.588 GB，为保守起见按这整个大小给merged预留，而非只按BF16体积减半。tokenizer及metadata由1GiB杂项裕量覆盖。

最终评测parallel_rollout同样保留parts＋merged。按训练平均字节×51713/845515再放大4倍，两份预计1.566 GB。题材和终止长度不同，属于估算，并非已生成test结果。

| 从本次快照起新增预留 | GB |
|---|---:|
| remaining_training_parts | 11.962 |
| training_merged_copy | 12.801 |
| mined_pairs_allowance | 12.801 |
| eight_rank_adam_two_snapshots | 3.831 |
| four_adapter_directories | 0.479 |
| merged_model_conservative_allowance | 13.588 |
| final_seven_parts_and_merged | 1.566 |
| logs_metadata_allowance | 1.074 |

新增总预留约58.10 GB（54.11 GiB），扣后仍有1.541 TiB快照裕量。可用空间约为新增预留的30.2倍。没有要求清理文件或停止进程。

## RAM与估算边界

encoded不会落盘，但mine一次载入845515条、OPSD每rank复制pairs与encoded，CPU RAM需另考虑。/proc/meminfo的实际MemAvailable见evidence.json，当前约0.865 TiB。极端上界包络：若169103题全部可用，双prefix与response/mask总列表项以每题16384项、每项36 bytes粗算，八rank约797.9 GB，尚未计pairs字符串/Python容器与其他作业；这不是真实encoded占用预测或严格进程上界。通常长度远低于此极端。此处仅提示RAM和disk不能混同，没有新测模型或重跑已通过测试。

quota不可读、共享磁盘未来消耗、Python RAM对象开销、未来轨迹长度均是估算限制。建议root沿已有监控在生成结束/进入mine和OPSD时复核df及MemAvailable；本审查不建立新observer、不改controller。只CPU/文件读，无GPU、检索HTTP、生成负载、作业信号、source/runtime/config/affinity/publisher修改或commit。
