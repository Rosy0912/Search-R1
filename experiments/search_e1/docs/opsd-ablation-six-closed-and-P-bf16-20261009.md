# 六组完成记录与 P 组 BF16 评测

2026-10-09 的实际记录覆盖 C0/C1/M/R/W/P 各 10 个 online outer，合计 60/70；J 尚未开始。完成链包含 599 次唯一 Adam 更新、18,120 次可用配对曝光及 3,532,531 个有效 policy token。完整七组、11 个模型的 5632 条预测、debug 和持久备份仍未验收。

## 实际优化预算

| 组 | 完成 outer | Adam | 配对曝光 | 有效 token |
| --- | ---: | ---: | ---: | ---: |
| C0 | 10 | 100 | 3012 | 583074 |
| C1 | 10 | 101 | 3023 | 629099 |
| M | 10 | 100 | 3032 | 637380 |
| R | 10 | 99 | 3030 | 547564 |
| W | 10 | 100 | 3018 | 550282 |
| P | 10 | 99 | 3005 | 585132 |
| J | 0 | 未开始 | 未开始 | 未开始 |

十个 outer 不代表相同 Adam 次数、有效 token 或配对曝光。W 的实际加权曝光为 3030.562981169141，原始曝光仍为 3018，不能按权重和重新归一化训练目标。

## P 的固定 512 题结果

P 的 raw 第十轮 adapter 使用共同 BF16 base 的主要评测协议：普通提示、HF SDPA greedy、repetition penalty 1.05、同一 NQ256/Hotpot256、同检索和严格最终行动评分。该 BF16 phase 的八 worker 实际退出码均为 0，release 已接受；这不等于尚在执行的整个 P 评测控制器已经退出。

| 对象 | 整体正确数 | EM | NQ | HotpotQA |
| --- | ---: | ---: | ---: | ---: |
| BF16 zero | 239/512 | 46.6796875% | 110/256 | 129/256 |
| C1 raw10 | 236/512 | 46.09375% | 109/256 | 127/256 |
| P raw10，BF16 评测 | 242/512 | 47.265625% | 111/256 | 131/256 |

P 的 270 个失败均为协议有效但 gold 不匹配；最终行动无效 0，未执行搜索 0，512 条均以 answer 终止。搜索次数为一次 264、两次 199、三次 43、四次 6。

| 基线 → P | 新增正确 | 丢失正确 | 净变化 | 95% 配对区间，百分点 |
| --- | ---: | ---: | ---: | ---: |
| C1 → P | 18 | 12 | +6 题 / +1.171875 pp | [-0.78125, 3.3203125] |
| BF16 zero → P | 18 | 15 | +3 题 / +0.5859375 pp | [-1.5625, 2.734375] |

按问题单位、seed42、10000 次 percentile bootstrap；两个区间均跨零。因此 P 的点估计较高，仍不能确认改善或把变化直接归因于精度机制。逐集结果和置信区间保存在原远端分析凭证。FP16 部署评测尚未完成，必须另与 FP16 zero 比较，不能减 BF16 zero 作训练收益。

## 训练曲线与解释边界

六组逐轮记录与 70 个显式槽保存在 `../reports/six-closed-recorded-curves-20261009/`。J 的槽为空，不补零。图展示不同 TRAIN cohort 的 reward、可用 K、raw reverse KL、加权裁剪目标、对冻结 outer-old 的 KL 及累计 Adam；这些不是固定 held-out512 的效果曲线。

P 按实际配对曝光加权的记录 raw/post 目标为 0.4765983648/0.4575197813，对 outer-old 的记录 KL 为 0.00005417873。C1 相应为 0.4026358140/0.3940404729 和 0.00077687475。两组在线生成轨迹及计算精度不同，这些标量差异不能单独证明根因。M 的目标含 0.9 old-student，不能把其较小的混合 KL 与纯 teacher 目标直接解释为更好答题效果。全部数值是原 committed 训练记录的后处理，没有重算 logits 或梯度。

P 的最终检查点独立 CPU 实读已接受：八 rank 的真实 Adam99、完整 optimizer 状态一致，504 个 FP32 adapter tensor 全有限并相对共同初始权重改变；RNG/scaler 状态已验证。该证据不替代全部 token/debug 的本地完整备份。

## 来源与未完成项

- 共同七组 goal SHA：`c65bea46dbb4ef6451ed56b2c4e26adafde5fd247e6b67fb5bea593b8689c94f`。
- 已发布只读事实收集器 SHA：`50fc6bf8609ef93f4ca68907b9f688252010d950e59188efc467552b5c2c220c`；本次 60 轮原始事实 SHA：`7c42bd6cfaa91bef45d9af5801360c6f0e42a0aeaae774bc7718e1a7981cc902`。
- 已发布曲线工具 SHA：`b41944178bfda97514cf3179c5651d350fa97a552e3af9b48b5162d718e69583`；派生产物 manifest SHA：`c5f0e1551b9ace181e0f12c1de810f03259c7703807bab4801ff8d01dce2a1c3`。
- P BF16 分析：HKUST1 `/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/receipts/P-bf16-interim-analysis-1791548311765683763.json`，SHA `365da82553c130efcc1a73961e7e5f0a6ddffc4e070f4dd1a3884c7220d161dc`。
- P99 数值验收：同目录 `P-final99-independent-state-readback-1791548728870034808.json`，SHA `d954864551546839424d1ca9b436e6c280a51a2bacb19cf50facb042bf0ae56c`；最终 CP 为 `group-P/runs/checkpoints/update-00000099-outer-00009/manifest.json`，SHA `b14ee6240e182291f16699816cf3a605e4d0a7bde89a28d747ca8c12e3614dbe`。

仍需原 driver 完成 P FP16、J10 及 J 两种评测，并完成所有 18/54 配对统计、全 study 词表快照账本、完整 debug/执行终态及持久备份。Mac 空间不足的问询仍待回复；备份不阻挡 GPU 队列，原节点资产保留。后来的人工指令已取消账号中转；本次直接查询没有重启、热改或重复评测。
