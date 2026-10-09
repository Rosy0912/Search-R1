# C1 + P + R：首10轮和完整512题评测（2026-10-10）

10个online outer已实际完成，99次Adam更新、3002条可用轨迹曝光和512724个有效policy token。源模型为archive/Search-R1 GRPO checkpoint100；OPSD从新Adam开始。P保留FP32主权重、梯度和Adam，FP16主要计算/采样；R从五条轨迹（含成功参考）均匀选学生D。教师目标和单位样本权重沿用C1，LR1e-6、reverse KL词表分量上限10。

| 同协议固定512 | Source100 | 10 outer raw（CP99） | 差值 |
| --- | ---: | ---: | ---: |
| 总EM | 239/512，46.6797% | 232/512，45.3125% | −7题，−1.3672pp |
| NQ | 110/256，42.9688% | 106/256，41.4063% | −4题，−1.5625pp |
| HotpotQA | 129/256，50.3906% | 126/256，49.2188% | −3题，−1.1719pp |
| 最终协议无效 | 4 | 1 | −3 |

按同一问题配对：保留222题正确，新增10题正确，丢失17题正确，263题均错。17个丢失全部为最终行动协议有效但答案不匹配gold；没有丢失题归于最终协议无效或不搜索。raw CP99的512条均实际执行过搜索，511条answer终止、1条invalid_or_budget终止。

按问题单位、10000次/seed42 paired bootstrap，EM差值95%区间为[-3.3203,+0.5859]pp，跨0。这次样本上净少7题，不能据此认定总体统计确定退化，也不能归因给P或R中的单项设置。

TRAIN是每outer不同512题×5个新采样轨迹，十轮严格正确12326/25600=48.1484%；第1/10轮为50.6250%/49.53125%。逐轨迹token均值后等轨迹平均的记录postclip loss为0.5516057→0.4944656；这些记录不构成固定验证曲线或每次Adam的EM。逐轮/逐真实Adam记录和静态四面板图保存在本目录。

本次512普通提示HF SDPA/BF16 base＋未merge FP32 raw LoRA，greedy、rep1.05、原EOS、8192上下文/每turn500/5turn4search/top3/obs480、固定NQ256+Hotpot256、rank[::8]/seed42。基线239是已实际完成的同协议评测，不是历史数字填充。

原始checkpoint绝对路径：
`/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010/runs/checkpoints/update-00000099-outer-00009/manifest.json`
SHA256 `1ed5329cf6f84c518cf01dcc219dddf254500137a0f3164410f053cecd710258`。
完整512汇总：
`/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010/periodic-recovery-e72eb3c/eval-010/runs/summary.json`
SHA256 `25a43c5359f83f7a29d9946847161abdc0c07e2bdf1a6d4a7ba8ce102ebfab44`。
独立只读CPU汇总：
`/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010/analysis/first10-training-native512-20261010-0417`。

评测原controller实际wait0、全部八个GPU worker正常完成64题并释放资源；首10轮＋512运行验收已接受。2026-10-10 04:17 CST实际续训controller1664464/UID1005/starttick936743594使用同run `c6a79477161a4a059e4fd68fc7567f87`、原config651dfb…/原dc08aad…代码、`--resume --until-outer20`，现已开始outer_index10的新采样。原科学设置不变。初始化和采样阶段不冒充后续已提交Adam；继续至100、其余9次512评测尚未完成，完整目标保持进行中。

大文件、原逐题评测/轨迹/token/状态全部保留HKUST；本目录仅聚合CSV/JSON/说明/PNG，不包含模型、optimizer、原问题或逐题预测。
