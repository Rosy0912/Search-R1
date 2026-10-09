# R 十轮训练完成并直接进入 512 题评估（2026-10-09）

R 同一运行已完成全部 10 个在线 outer：5120 个 TRAIN 问题、25600 条新生成轨迹、3030 个可用 pair 曝光、99 次真实 OPSD Adam 更新及 547564 个有效 policy token。原训练父进程实际退出码为 0，自有后代完成释放，原 driver 随即启动 R 的 native512；CPU 数值核验与本地备份没有成为评估或后续 W/P/J 的启动依赖。

R 保持 reverse KL、LR1e-6、共同 source100、BF16 base/compute + FP32 LoRA。与 C1 的预定区别是 D 在五条 sibling 中均匀随机选择，包含成功参考 B 本身；不加正确性权重。每条轨迹先对有效 token 平均，再按实际全局 pair 数平均，词表分量上裁剪仍为 10。没有恢复 GRPO Adam，没有回退到 best，也没有从零重复训练。

| outer 索引 | 实际 K | 本轮 Adam | 累计 Adam |
|---:|---:|---:|---:|
|0|327|11|11|
|1|305|10|21|
|2|298|10|31|
|3|284|9|40|
|4|286|9|49|
|5|308|10|59|
|6|298|10|69|
|7|308|10|79|
|8|298|10|89|
|9|318|10|99|

最终原始 CP99 已独立 CPU 实读一次：八 rank 的实际 Adam 步数均为 99，完整 moments/groups 逐项一致且有限；504 个 FP32 adapter 张量全部有限、全部相对共同初始模板改变，完整 RNG/scaler/cursor 检查通过，CUDA 未初始化。这是最终保存状态的数值验收，不能代替 heldout 效果，也不声称重新计算了全部历史 logits/token。

十个 outer 的 teacher 诊断父票据实际各接受 8 个案例，共 80 个，原单 child 的实际 wait0 均有记录。此计数来自原票据；完整轨迹/HTTP/token 的派生覆盖分析另行进行，不能把父票据计数当作所有 debug 语义重新核验。

13:57 CST 的 R 正式评估快照为 8 个 worker 均 RUNNING、83/512 个正式结果；14:16:46 实读同一 run 已有 267/512 个正式结果，八 worker 仍 RUNNING、controller 实际 UID/starttick/父进程吻合。尚未接受完整 512 覆盖，尚无 R 的最终 EM；不能将这 83 题外推或作为训练收益。正式评估绑定 R 第十轮的 raw CP99，使用与共同 BF16 起点及 C0/C1/M 相同的普通提示、HF SDPA greedy、repetition_penalty1.05、固定 NQ256/Hotpot256 和原搜索预算。

已完成 C0/C1/M/R 四组训练，共 40/70 outer、400 次真实 Adam、12097 个 pair 曝光、2397117 个有效 token。已接受的正式评估仍为 2560/5632 个预测槽：共同 BF16/FP16 起点以及 C0/C1/M。W/P/J 各 10 outer 与 R/W/P/J/P-FP16/J-FP16 的完整评估继续由原顺序队列执行，剩余 30 个 outer 和 3072 个正式预测槽。四组的原生标量事实已收集 40 行；13:57 快照时完整 clip/debug 派生仅此前 30 轮，14:14 完成 R 派生后四十轮子集闭合，仍不能混称七十轮全部完成。

现有完整 BF16 评估为 source100 239/512、C0 227/512、C1 236/512、M 237/512；R 未完成。M 相对 C1 净增 1 题、相对起点净少 2 题，配对区间跨 0，不能认定可靠提升。10 outer 各组的 K/Adam/token 曝光并不完全相同，最终报告需保留这些差异。

## 实际身份与保存路径

- 主目录：`HKUST1:/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008`。
- R 训练 run：`779e853e37b540e8847336c142129377`；原 controller3069437、UID1005、starttick931125352。正式训练 terminal 由原 driver1894801 保存，实际 exit0，未发送停止信号。
- 配置：`group-R/receipts/authorized-launch-20261009/config.json`，SHA `8cc81ae28882f0fb32a924d7a4d96fc6f7f0702b210ef62847e409c404d36ab9`。
- 最终 CP：`group-R/runs/checkpoints/update-00000099-outer-00009/manifest.json`，SHA `0d53584c33a17f6ab7d1465d269e93268351bd8255f5d1eea796c102d39ad2bc`。这是 R 新运行的 CP99，不能与旧 forward 运行的同编号检查点混淆。
- 评估 run：`214e1c0ab8f4488e9f2781e349cb4296`；controller3332309、UID1005、starttick931652101。
- 评估配置：`eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs/03-R/evaluation-config.json`，SHA `88eab71ba3bab8e2d51528597707be903be5c7dfad803eddb2ba21673f1a43c8`。
- 评估原输出：`eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851/R/runs/raw10_bf16`。
- 生产训练 source 仍为 `871dc46f`，精度恢复包装仍为 `538ac99a`，评估仍为 `72e1e127`；本报告没有热改这些版本。

R 十轮 raw 的独立压缩备份于 14:09 完整验收：4240 文件/2413211382 字节，gzip 362796005 字节；实际 transfer/source-after/member verifier 均 rc0，源固定集合及全部成员 SHA 通过，完成票据 SHA `b8c59151c2b7d2c0ad342578ee0d7a9d4fff2c9f07c9bdf2f5bfceecb2ff1854`。R 十个 outer 末完整状态也均持久备份，原队列累计 C0/C1/M/R 四十个 outer 末包完成。上述备份证明不等于每个中间 CP 的 Adam 张量均重新实读。已接受的旧 CP0/1/5/11 本地无损备份另完成独立 XZ 解码核验并观察到原父进程真实 exit0，随后只移除一份 6239805440 字节的本地重复 tar；可恢复的完整 XZ、58 成员验收票据及 Node 原资产均保留。资产操作不阻挡 GPU 队列。

R 的 CPU recorded clip/debug 派生于 14:14 完成，未加载模型、重算 logits 或新生成：547564 个唯一有效 token 中，465644 个仅保存标量，81920 个在同一标量记录中另外带 top32，二者相加才是总有效数。1170 个位置/1172 个词表分量触发 upperclip10，位置占比约 0.21367%；命中位置的 student mass 均值 0.8591565、teacher mass 均值 3.75248e-6。这些是记录的损失/概率质量，不能解释为梯度或因果收益。按轨迹等权的 raw/post loss 均值为 0.52430055/0.51590681。独立自主 teacher 共 80 案例，不能计入训练或 native512 分母。

R 保存的两处完整词表位置共 8 个 FP32 NPY、4862976 字节，原 member SHA/header/finite 通过；这不是全词表 KL 算术重建，也不替代全 study 24GiB ledger 的最终验收。四组四十轮 recorded clip/debug 子集目前均已闭合，最终七组七十轮汇总、完整评估与归因仍待完成。私有 R 派生 manifest SHA `1a2ec8e09797ef08a996fac85774e4c40ed92a885014f9cd553fdb5a31b716fd`，root 窄验收 SHA `0012d34bade2717c256d115dd46f84eba4fa954b4c9744c05d24600f36dfe43e`。

本地私有根核验：`work/search-e1-analysis-output/opsd-ablation-execution-20261008/goal-continuation-20261009-30/R-ten-root-accepted.private.json`，SHA `8c3f9a82d02341897a8b37292939a3e4522c26d8166f267361f70ea55550ebb0`。最终 numeric stdout SHA `15323695add06be1cea2cbab44f1c929ac6f048875cfbb78b77139ca3f8d2daa`，40 行事实 stdout SHA `2ea37319249acb70324828da8b7ec0487740ebe901f446b2664fb201d8447c48`。原题、完整轨迹、token、权重与环境仍只保存在私有输出，不进入 Git。
