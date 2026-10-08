# 在线 OPSD：五更新验收及完整首轮

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

## 下一轮边界恢复

首轮11步后，original `sampling.merge_latest` 在 controller 内调用 PEFT。虽 base 参数在 CPU，该父进程实际出现 GPU0 500MiB CUDA context，因而 strict release gate 持续 hold。八 sampler 与 trainer 已退出，未发现 OOM。CUDA context 本身不能证明旧合并执行了 GPU 算子；旧实际 adapter 位置未留 after proof，属于未知。

恢复方案保留 CP11 和旧 cache/receipt，只窄停止实际 UID1005/PID660173/starttick924349236 控制器，技术恢复不暂停实验目标。新外部入口将原合并函数放入 `CUDA_VISIBLE_DEVICES=''` 的 CPU 子进程，记录 PEFT及合并模型实际所有参数CPU、前后CUDA未初始化、真实rc及释放。原23文件科学 helper、305 runtime、config、采样/矿选/损失/训练/RNG逻辑不热改；同一 run 继续 outer1，完整 Adam 不重置。

外部边界脚本与只读本地 checkpoint 备份工具见 `experiments/search_e1/control/opsd_online_boundary`。验收 receipt 与原始 token/轨迹在私有输出目录，模型/环境/原始问题不入 Git。五更新验收只证明运行和真实学习成立，尚不证明 EM 提升；后续完整200轮与统一评估仍需完成。
