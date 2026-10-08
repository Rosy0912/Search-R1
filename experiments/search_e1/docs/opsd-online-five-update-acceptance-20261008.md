# 在线 OPSD：五更新验收、首轮完成与真实续训

2026-10-08 用户明确选择保留 KL 词表分量上限 **10**、**200 个外层轮次**，允许 HKUST 节点本地存储。本实验按指定配置文档采用 LR5e-6、逐轨迹有效 token 均值后全局实际轨迹均值、micro1/global32，冻结 source100 教师。每轮 TRAIN512×5，最新 raw student 重新生成，沿用同题原矿选规则，一轮可训练 pairs 更新一遍。200 是外层轮数，不是 Adam 次数。

## 实际验收

- 节点 HKUST1，目录 `/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008`。
- 同一 run `1d1d67b60f0e46a2afc0536a36ab69b1`，配置 SHA `d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17`。
- 源 GRPO/Search-R1 checkpoint100 权重；fresh OPSD Adam。展示“100→105”表示此源权重后五次新 OPSD 更新，没有恢复 GRPO Adam。
- 官方外部 CPU 验收 `receipts/five-real-updates-accepted.json`，SHA `7e12f7e0c86623dd45787471ee708a96c78615725e7cbdd48c6dae01a003287d`；验收时间 2026-10-08 17:36:52 CST。
- 实读 checkpoint0/5：八 rank Adam 均为5，全部 moments 和参数组精确一致；504 adapter 张量全有限且504均改变，最大绝对变化 `2.5111432478297502e-05`。校验不初始化 CUDA。
- 独立审查另核 CP0/1/5，以及更新1/5的16个真实训练 token 日志、4130有效 token；各轨迹归一化与 global32 贡献正确，clip10 实际触发7个词表分量。抽查不代表全日志/全词表均复验。
- 首外层完整生成2560轨迹、512题，P=K327，全部327 pair 可训练，真实 Adam11（十个32批＋一个7尾批）。CP11 完整八 rank 权重/Adam/RNG与采样、矿选、编码/order/cursor身份均留存。
- 独立显存监测一直运行，至五更新验收的全部实际采样无 `<1024MiB` 余量事件，历史实际最小余量3175MiB；训练阶段验收附近各卡均大于15GiB。原失败采样和原有历史证据保留，不抹除失败。

## 已执行的下一轮边界恢复

首轮11步后，original `sampling.merge_latest` 在 controller 内调用 PEFT。虽 base 参数在 CPU，该父进程实际出现 GPU0 500MiB CUDA context，因而 strict release gate 持续 hold。八 sampler 与 trainer 已退出，未发现 OOM。CUDA context 本身不能证明旧合并执行了 GPU 算子；旧实际 adapter 位置未留 after proof，属于未知。

已保留 CP11 和旧 cache/receipt，仅停止实际 UID1005/PID660173/starttick924349236 控制器；原17个子进程均实际不在运行，未停止检索或其他任务。终态凭据 `receipts/cpu-merge-parent-terminal.json` SHA `e16f6b9e032112b0da65594d902d535a7afa7cbd061320c0ee83e44f87866a31`。旧两件合并产物及原控制状态保存在 `preserved/cpu-merge-parent-cuda-1791453514`，CP11 完整状态未改。

外部恢复入口提交 `51eb6c574dd726b929dffcd6c0abfdc7b441a5d4`，8文件 manifest SHA `c349ab69d2939bf6bdd4e59eb32ae0023ec009796f85089a945d792b963d607e`；节点入口、源码、发布凭据逐 SHA 验证后启动。当前控制器 UID1005/PID720753/starttick924526765，tmux `e1-opsd-online-cpu-recovery2-200-20261008`。它将原合并函数放入 `CUDA_VISIBLE_DEVICES=''` 的 CPU 子进程，保留原函数、BF16 PEFT 方法与所有调用参数。原23文件科学 helper、305 runtime、config、采样/矿选/损失/训练/RNG逻辑不热改；同一 run 继续 outer1，完整 Adam 不重置。

实际 CPU 子进程 PID721247/starttick924528294 正常退出码0；938个 PEFT 参数与434个合并模型参数均实际在CPU，CUDA前后未初始化、可见CUDA设备数0。父进程实际等待退出并通过全部自有进程/GPU释放检查，唯一获准共享GPU进程仍是18004检索。完整凭据在 `receipts/cpu-merge-recovery/0b2b155ee839478db5e8a6b79b09ecc4`。新旧合并产物11个文件字节SHA全部相同，`receipts/cpu-merge-old-vs-isolated-byte-comparison.json` SHA `fcf45567cdf91418148129fded8256bfc2dc57dd0f70eb2b603ed4ff863d0c54`。恢复修复了父进程CUDA上下文阻塞，没有改变该轮采样权重。

第二轮真实重新生成8×320=2560轨迹，P=K306；它使用上一轮CP11的raw adapter，训练从实际Adam11继续。独立CPU实读最新CP19的8个完整状态文件：每rank504个Adam参数状态、1512个优化器张量全有限，实际step19；moments、step、参数组逐项跨rank一致，累计样本曝光583=327+256。审计 SHA `306343152c58c36ce9ddec44c5b3b32a4bddb89edb94a0c54183552e8bb04bfe`，私有本地 `work/search-e1-analysis-output/opsd-online-cpu-recovery-review/outer1-live-state-readback.json`。该快照只证明续训到19，不表示200轮结束。

初始恢复CP11/outer1按已授权“仅保留最新两个中间状态”规则，在独立复核完整moments之前被正常回收；保存的原manifest及retention tombstone中，八rank导出的adapter/optimizer摘要均精确匹配永久保护的CP11/outer0。恢复源码还在初始导出前断言恢复的adapter与Adam张量及参数组摘要一致；这不表示新轮次消耗后的游标与RNG字节仍与前一轮相同。这里明确区分上述原生导出/断言证据与CP19的独立CPU实际读取，不声称重新读过已经回收的初始moments。永久首轮CP11未回收。

18:11 CST节点实读：第二轮306对已全部完成，累计633对曝光、21次真实Adam更新；两轮共11+10更新，尾批分别7和18。永久保护CP21/outer1 manifest SHA `4c13412f7d94c63b12bbd2fa2c5c5a89e5c0aa00333e750d64446f414beb7899`，outer-complete SHA `84b58def0c0de74dc584ca6c840db6d9a85d8155be213b8254e43b1e5816ae9c`。同一控制器已在第三轮SAMPLING，以最新CP21权重采样；因此当前是**完成2/200轮、累计21次真实更新**。显存observer此时1819次实际采样，低于1024MiB余量事件为0，全部历史实际最小3131MiB；该轮8个采样GPU进程与原oracle仍正常。数值与源验证范围分别按上述独立CP19审计和当前原生CP21完成凭据记录。

首200个TRAIN512 cohort的CPU预检共102400个唯一问题，与原native512验证集的ID和字面问题碰撞均为0；各轮仍保留自己的实际选择/身份校验。此预检不替代每轮采样与训练实读。

## 持久化与后续

源码、运行入口、部署/恢复配置与报告已推送 `search-e1`，每次发布均实际读回远程文件SHA；模型、环境、凭据、原始问题和完整token/轨迹不入Git。HKUST节点原始输出保留，Mac的CP0/1/5/11完整状态与训练输入备份预计6239760772字节，当前仍传输，只有完整成员SHA验收后才会标为完成。不会因为备份进行中释放节点或删除原始资产。

每小时自动跟进同一200轮运行，核真实更新/采样/完整状态、检索身份、直接显存余量与异常恢复。正常进展保持安静，重大恢复、失败、验收或新可靠评估才通知。当前没有新EM评估，五更新与续训验收不证明性能收益。

外部边界脚本与只读本地 checkpoint 备份工具见 `experiments/search_e1/control/opsd_online_boundary`，CPU合并入口见 `experiments/search_e1/control/opsd_online_cpu_merge_recovery`。验收receipt与原始token/轨迹在私有输出目录。后续完整200轮与统一评估仍需完成。
