# W 十轮后的 512 题结果与 P 接续（2026-10-09）

W 完成十个在线 outer 后，原八 worker 对固定 512 题的评估已自然结束，正确 **237/512（46.2891%）**。相对随机配对但不加权的 R，新增答对 9 题、丢失 12 题，净少 3 题；相对共同 BF16 零更新，新增 16 题、丢失 18 题，净少 2 题。两项总体配对区间均跨零，本次没有可靠证据表明正确性加权提高 EM，也不能据此认定加权等效或无效。

原 study driver 在 W 评估释放后自动进入 P；没有从零重跑已完成组，也没有等待本地分析、备份或发布。P/J 及其 BF16、FP16 末点评估仍属于必须完成的剩余范围。

## 固定条件与实际完成

七组使用同一 Archive GRPO100、同一 504 张量 FP32 初始 LoRA、fresh Adam 和十个预先冻结的 TRAIN512 cohort。W 与 R 均为 LR1e-6、随机 D 配对、BF16 base/compute 加 FP32 LoRA、reverse KL 分量 upper clip10。W 在 clip 后的轨迹 token 均值之外乘 D 正确性权重，最后除以实际 pair 数，未按权重和重新归一化。由于权重改变后续学生与在线采样，两组实际轨迹、K 和更新数可以不同。

W 十轮共有 **100 次真实 Adam、3,018 次 pair 曝光、550,282 个有效 policy token**；R 为 99 次 Adam、3,030 次 pair 曝光。十 outer 是相同采样轮数，不是相同优化器步数或完全相同训练样本。W 原最终 CP100 的八 rank Adam/moments/groups 一致、LoRA 有限且相对共同初始值改变，已有独立 CPU 实读证明，本次未重复读取这些大张量。

主要评估使用共同 BF16 base 加各组未合并 FP32 raw10 LoRA、普通提示、HF SDPA greedy、repetition penalty1.05、原 EOS、8192 context、每 turn500 token、5 turn/4 search、top3、observation480。每逻辑 rank 处理原 validation 的 `[rank::8]` 64 题，NQ/Hotpot 各256题，seed42，检索及严格 authored-final-action scorer 保持原协议。采用 raw 末权重，未回退 best。

17:35 CST 原 W 评估八个 worker 的实际 wait 均为0，原父进程 PID3796207/UID1005/starttick932605493 的实际 wait 也为0；原 phase/controller/result accepted 及 owned/current release 齐备。512 个 validation index 唯一完整、两数据集各256题。这里的计数来自已保存的严格评分记录，本次派生没有重新推理或重评分。

## 同题变化

下表方向为右侧 W 减左侧对照。CI 是 seed42、10,000 次按题配对 percentile bootstrap 的95%描述性区间，未作多重比较校正。

| 对照→W | 分层 | 新增 / 丢失 | 净差（百分点） | 95% 配对 CI（百分点） |
| --- | --- | ---: | ---: | --- |
| R→W | 全512题 | 9 / 12 | −0.58594 | [−2.34375, +1.17188] |
| R→W | NQ256 | 1 / 4 | −1.17188 | [−3.12500, +0.39063] |
| R→W | Hotpot256 | 8 / 8 | 0 | [−2.74414, +3.12500] |
| BF16zero→W | 全512题 | 16 / 18 | −0.39063 | [−2.53906, +1.75781] |
| BF16zero→W | NQ256 | 5 / 8 | −1.17188 | [−3.90625, +1.56250] |
| BF16zero→W | Hotpot256 | 11 / 10 | +0.39063 | [−3.12500, +3.90625] |

W 的最终行动协议无效数为0，未执行搜索数为0；275 个错误均为最终行动合法、答案不匹配 gold。R→W 的12个丢失和 BF16zero→W 的18个丢失也均属于合法答案不匹配，不能把本次少答对归为最终标签格式崩溃或停止搜索。这个分类仍未定位事实知识、检索选择或推理决策的具体因果。

所有区间跨零。单次训练 seed 与有限验证集不足以作等效性、跨 seed 稳定收益或机制结论；top32、标量 loss 和局部漂移也不能补足这些对照。

## 当前完整模型与剩余范围

| 已完整模型 | 正确 / 512 | NQ / 256 | Hotpot / 256 |
| --- | ---: | ---: | ---: |
| 共同 BF16 零更新 | 239 | 110 | 129 |
| 共同 FP16 零更新 | 245 | 111 | 134 |
| C0 BF16 raw10 | 227 | 109 | 118 |
| C1 BF16 raw10 | 236 | 109 | 127 |
| M BF16 raw10 | 237 | 109 | 128 |
| R BF16 raw10 | 240 | 110 | 130 |
| W BF16 raw10 | 237 | 107 | 130 |

已完成五组共 **50/70 outer、500 次 Adam、15,115 次 pair 曝光、2,947,399 个有效 token**。七个模型完整产生 **3,584/5,632** 个正式预测；同题派生覆盖 **10/18** 个对照、**30/54** 个分层。共同 FP16 零更新245题是部署精度对照，不能减共同 BF16239题当成训练收益。

P 的原 run 为 `c0a425974d2142b1955429c85a4fb9f3`，controller3966861/UID1005/starttick933005288。17:45:47–49 实读首 outer0 已闭合：K313、10次真实原生 Adam、313次 pair 曝光、64,118有效 token；现在 outer1 使用末点 CP10 的最新 raw adapter 采样。全目标已闭合 **51/70 outer、510 次原生 Adam、15,428次 pair 曝光、3,011,517有效 token**，剩余 P9轮、J10轮。P 的1/5次更新已于17:50通过独立 CPU 实读：原两次 reader实际退出0、八rank完整Adam/moments/groups与step1/5一致；各rank保存的RNG、scaler和cursor通过检查，504个adapter张量均FP32有限，相对共同初始值分别改变252和504项。scaler均启用，scale65536、growth tracker1/5。此为保存状态核验，没有恢复RNG、初始化CUDA或证明全历史无溢出。

P 原 CPU merge 元数据记录938个 FP32 PEFT参数在 CPU 合并后一次转为434个 FP16 canonical参数，前后 CUDA均未初始化。八个当前采样 worker 与获准 oracle 是当时全部 GPU compute 进程，直接最小 memory.free3133MiB。首轮八 rank精度记录均为 FP32主参数、scaler65536/growth tracker10；本次只检查首轮和当前轮的指定 retry路径，不能据此宣称全历史 overflow为0。详细 dtype events有cap，完整十轮精度/溢出历史仍待闭合审计。

P/J 尚须各完成十轮；P末点及J的1/5和末点仍须实际 Adam/参数/scaler/dtype/恢复验收、共同 BF16 和 FP16 部署各512题。全部70轮后的全 study NPY空间账本、debug/恢复资产持久清单和最终中文归因报告也未闭合，目标保持 ACTIVE。旧 K1 forward/离线训练及旧七基准评测不重启；旧 K4 archive 已按其单独较新人工授权停止，不依据过期 heartbeat 自动恢复。

## 证据与发布身份

- [同题聚合 JSON](../reports/seven-closed-native512-20261009/stratified-paired-summary.json)、[全部对照分层 CSV](../reports/seven-closed-native512-20261009/stratified-paired.csv)、[派生说明](../reports/seven-closed-native512-20261009/stratified-paired-notes.zh.md)、[产物与输入 SHA 清单](../reports/seven-closed-native512-20261009/derivation-manifest.json)。四个文件按实际派生原字节复制，不含原题、题ID或逐题 trace。
- 新 W 原评分记录 SHA256 `d4f106e49c7dadb84e666012e02ccfbaf54701de70a47fbb916dcb214789515b`；原 summary `f8f215afd9fddfc30231eeb866fa1670af7302e4c9268af1f3fd3b16b2e5f073`。七模型聚合 SHA256 `90d6ccadedaa9ef978c83d058a18167451538d8707ce90d8d058f8dd4f3cbf24`。
- 使用既有已发表 `opsd_ablation_stratified_effects_v1/derive.py`（SHA256 `588bc7da3e782d6ae06e666e1099f0b232cf6a566b6b54af9ec90c33b5966eed`）和原 pure counts/bootstrap，实际 CLI0、stderr 空；没有增加模型、数学或归档测试。
- W 原训练目录：`/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/group-W`；末点 `runs/checkpoints/update-00000100-outer-00009`，manifest SHA256 `2db5b57e9fd60d4ab034cf75870ddb9c3b17ebc59e69dc5d94d9734cccd7536d`。
- W 原评测目录：`/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851/W/runs`，eval run `fa3f3ed66e1e405fa35acabeb22b0ba1`。
- W 十轮原始训练/debug 和受保护初始/1/5/末点已持久验收；本次闭合评测 trace/日志的592文件、31,992,844成员字节也完成独立备份，inventory/transfer/source-after/verify的实际wait均0。压缩包4,907,580字节，SHA256 `c92524cc695ce56c3beeeff7621477d3125c033b8d45ddce384ca5d8cf4eb63a`；每成员SHA、连续USTAR header、gzip/footer及末尾padding通过，未声称逐member内部padding独立核验。原 Node资产保留，备份未成为 P/J 启动门禁。
- 正跑训练仍为 `871dc46f`，dtype 修复 `538ac99a`，评估 `72e1e127`。本报告的提交是结果发布，不冒称新的训练部署。原题、完整轨迹/token、模型权重、环境和凭据不入 Git。
