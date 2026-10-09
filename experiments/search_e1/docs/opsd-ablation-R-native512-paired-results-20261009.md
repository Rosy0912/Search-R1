# R 随机配对消融：完整512题结果与同题比较

2026-10-09，R完整评估后的阶段报告。R 已完成同一运行的10个在线 outer，以及原生512题评估。R 答对240/512（46.875%），相对 C1 净增4题，相对共同 BF16 零更新起点净增1题。两项按题配对 bootstrap 区间均跨过0，目前不能认定随机配对带来了可靠提升。

## 设置与实际训练预算

R 从共同 archive GRPO100 原始权重和共同零B LoRA开始，fresh Adam；LR1e-6、reverse KL、词表分量上裁剪10、每轨迹有效token平均后实际全局轨迹平均，BF16 base/compute 与 FP32 LoRA。相对 C1，改动是 D 在本题全部五个 sibling 中确定性均匀选择，允许 D=B；R不加正确性权重，也不混合旧student目标。

R 的10个 outer 产生3030次实际pair曝光、547564个有效policy token、99次真实Adam更新。C1为3023 pair、101次Adam，因此相同10轮并非相同优化器更新数或样本曝光数。每轮采样使用本组当时最新student，不借用其他组后续轨迹。

真实末checkpoint：`/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-R/runs/checkpoints/update-00000099-outer-00009/manifest.json`，SHA256 `0d53584c33a17f6ab7d1465d269e93268351bd8255f5d1eea796c102d39ad2bc`。末点独立CPU实读通过八rank Adam99、完整moments/groups一致且有限，以及504个FP32 adapter张量有限且相对共同初始全部改变；并未重新计算全历史logits。

## 同协议原生512题

普通提示、共同BF16 base与未合并FP32 raw adapter，冻结HF SDPA greedy/SearchAgent、repetition penalty1.05、8192上下文、每轮500 token、5 turn/最多4 search、top3/observation480，以及原18004检索。512题为NQ256和Hotpot256，八逻辑rank各64题。最终raw模型参与评估，没有best回退。

| 模型 | NQ /256 | Hotpot /256 | 总计 /512 | EM |
| --- | ---: | ---: | ---: | ---: |
| BF16零更新起点 | 110 | 129 | 239 | 46.6796875% |
| C0：LR5e-6 | 109 | 118 | 227 | 44.3359375% |
| C1：LR1e-6 | 109 | 127 | 236 | 46.09375% |
| M：概率混合目标 | 109 | 128 | 237 | 46.2890625% |
| R：均匀随机D | 110 | 130 | 240 | 46.875% |

FP16零更新起点另为245/512；它属于不同精度执行条件，不能作为这张BF16表的训练收益基线。W、P、J及P/J的FP16部署评估尚未完整产出。

R的512个最终行动全部协议有效并执行过搜索；272题为合法最终答案但gold不匹配，最终协议无效为0。所有八worker与原评估controller的实际wait返回码为0，原生phase/controller/父进程释放票据接受。此结论使用原始评估记录与终态，不从PID消失推断成功。

## 同题变化和不确定性

每项以题为统计单位，10000次配对非参数百分位bootstrap、seed42、95%区间；下表为未做多重比较调整的边际描述区间。

| 比较 | 新增正确 | 丢失正确 | 净变化 | EM差值 | 95%区间（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 → R | 15 | 11 | +4 | +0.78125 pp | [-1.171875, +2.734375] |
| BF16零更新 → R | 13 | 12 | +1 | +0.1953125 pp | [-1.7578125, +2.1484375] |

C1→R在NQ净增1题、Hotpot净增3题；BF16零更新→R在NQ净变化0、Hotpot净增1题。丢失题在R端均为协议有效但gold不匹配，不能把这些丢失归因为标签格式失效。区间覆盖0也不能证明效果等价；本次只有一个seed与有限题集，尚不支持随机配对优于原配对的确定结论。

## 完整目标仍在执行

已完整训练C0/C1/M/R共40/70 outer，真实400次Adam、12097次pair曝光、2397117个有效token。已接受6/11个固定模型、3072/5632条正式预测、8/18项比较及24/54个分层。W由原study driver自然接续，P/J随后按原队列运行；每组只跑10轮，完成即评估。备份及CPU归并不构成GPU启动前置条件，没有重启R或把五步门禁当作完整效果验收。

另据15:10–15:11实际读取，W首outer已提交11次Adam、327 pair和62314个有效token，正在outer index1的教师诊断阶段。因此该次已闭合训练累计为41/70 outer、411次Adam、12424 pair和2459431有效token；上述四组曲线仍严格对应原40轮快照。W PID3516415/UID1005/starttick932059559、run `dc8a4e85822547548c0f79be4c87e13e` 与原启动身份一致；显存observer新鲜，430次采样无低于获准阈值记录。

随后独立CPU实读W的CP1/CP5，八rank分别为实际Adam1/Adam5，完整moments/groups逐项一致且有限，504个FP32 adapter张量全有限。相对共同零B起点，CP1有252个张量改变、CP5有504个改变；前者符合首步零B适配器的更新结构。读检查点进程未初始化CUDA。此处只覆盖保存的第1/5步数值，不能称全部11步均经独立CPU实读，也不触发从零重跑；同一W运行继续原10轮。

40轮TRAIN曲线已按原始提交标量生成。该图使用不同TRAIN题及更新前policy，不能当固定heldout曲线；图例只展示C0/C1/M/R，W/P/J的30个待完成槽是缺失值，不是零。少量全词表快照的全study账本仍须等全部writer结束后闭合，不能以当前四组的有限性检查替代全study验收。

## 身份与证据索引

- R训练run：`779e853e37b540e8847336c142129377`；训练config SHA256 `8cc81ae28882f0fb32a924d7a4d96fc6f7f0702b210ef62847e409c404d36ab9`。
- R评估run：`214e1c0ab8f4488e9f2781e349cb4296`；评估config SHA256 `88eab71ba3bab8e2d51528597707be903be5c7dfad803eddb2ba21673f1a43c8`。
- 原生summary SHA256：`6d2e975ef087f76cfa1040cd6ee4e88933d09e5e3742f5ddaf6dd5c7fa8d0a39`；原逐题records SHA256 `6969b37bb4f1adb11baf1de5d2d826e038b0386b4ad879ab1a58a0937e64bef9`。
- 六模型同题统计summary SHA256：`0624bf1f9fc4cdcdbab23143104f5731c3dde5e512377845214877ff51c76625`；使用已发布统计实现，没有新增推理或重复评分。
- 私有root验收：`/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output/opsd-ablation-execution-20261008/goal-continuation-20261009-33/R512-terminal-and-paired-root-accepted.private.json`，SHA256 `9637dfa4e496bd3e81e5bc76ebeb9fcf53480fffec5d45415041883e9c5c28f4`。
- R完整原生评估持久备份：`/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-analysis-output/opsd-ablation-execution-20261008/R-closed-native512-compressed-32ebb86-20261009/complete-sha-verified.json`，SHA256 `b4d28290125422d23c74d1ac3f8ddbb2a3d7f0013a746409f43eb89a4305604f`。591个文件、33092905成员字节；传输、源端复核与全部成员SHA验证实际退出0。
- 训练源码仍为已部署`871dc46f`，dtype实现`538ac99a`，评估实现`72e1e127`。本次备份SSH入口修复源码`32ebb8690296298b28ed0f2332908d0c9a77584f`只改变外部传输命令，未热改训练/评估或科学设置。

原题、完整生成/HTTP/token/权重证据保持私有。本报告是已完成阶段的结果，完整七组交付仍待后续实测。
