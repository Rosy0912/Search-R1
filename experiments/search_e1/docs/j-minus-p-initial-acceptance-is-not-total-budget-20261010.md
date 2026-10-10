# 人工纠正：五轮是初期验收，总预算200

2026-10-10人工明确纠正了上一回合的错误：五轮只是确认长程训练正常的门槛，不能将它作为总预算或在门槛后停训。用户要求写入持久记忆，并明确此次续训不插入512评测、补测、重复五轮验收或多余诊断。

记忆已写入 `/Users/shaoshitong/Documents/remote_codex_control/memories/task-progress-verification.md` 的“初期验收不是总训练预算”节。适用于以后“先跑N轮/初始化验证”等要求：分别记录验收点与总预算；验收后保留同状态继续完整预算，除非用户明确要求停止。

原五轮/Adam50/1497配对/CP50完整保留于HKUST1 `/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-20261010`。前次“五轮完成”报告只作为初期验收历史，不是200轮目标完成；旧自动跟进被关闭也是误解的结果，已恢复为 `search-e1-bf16-rwm-200` 长程自动跟进。

当前不可变续训目录 `/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-continuation200-20261010`，同run `d040ebac9fa24ec28d43976299fd4f71`，config SHA `3b3e632813271020d5674da06eecc797df94fd28e55d5c01a7279620734fee09`。从第6轮/index5和精确CP50 SHA `5bf7588f3182a0b46f858f7f85b6699a461665f566366eec6a3832cba06b46c4`恢复原八rank权重、完整Adam/RNG/cursor；不重新生成前五轮，不重置Adam。只有这一个已验收CP50允许跨预算配置身份，原state文件未修改，新checkpoint绑定新config。

实际源 `842fee6cd749459f0f802c846bd58411f6b418ac`，helper原目录下 `source/continuation200-842fee6/opsd_j_bf16_200_v1`。只有预算/200cohort/明确prefix恢复与BF16 merge检查改变；R/W/M/objective/precision/Adam科学代码与前五轮相同。200个TRAIN cohort仍是原seed42 shuffle前102400题，前10个与原冻结cohort完全一致。验证数据身份仅用于保持原cohort分离，未启动任何验证推理。

当前tmux `e1-j-bf16-continue200-20261010`，$84/%84，controller3129175/UID1005/starttick939523776。实测已第6轮SAMPLING，最新raw来自CP50，observer独立运行。启动本身不算有效推进；结束本次回合前确认原生新更新和完整第6轮，持续200的目标保持进行中。只保留关键checkpoint/必要日志；不再为初期验收做隔离测试或评测。
