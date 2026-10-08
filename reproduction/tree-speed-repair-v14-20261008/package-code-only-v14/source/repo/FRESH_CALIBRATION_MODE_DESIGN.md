本次只实现 fresh200 的零写入校准模式、fresh 初始化事实和直接 rank 调度器记账，未运行任何远端/GPU/真实模型或权重。

校准先将实际 actor 的全部子模块设为 train，用原生 monitor 调用观察；成功及异常都逐个恢复原始 training 标记，保留 Torch CPU/所选 CUDA RNG fork。模式安装错误在 matched collective 处同步拒绝，模式恢复/文件 IO 错误也同步拒绝。独立 mode JSON 与最终 rank JSON 的 SHA、身份、全部 train/成功恢复状态由 finalize 核对。

fresh_base_zero_v1 与默认 legacy healthy_checkpoint_parent_v1 分开。fresh 需要真正空的整份 Adam state、scheduler0，拒绝非空 step0。它使用 fit 初始 global0 的 provenance；观察时保留真实逻辑 outer1、实际更新0、sampler cursor512。legacy 的 parent/Adam/scheduler 校验仍保留。fresh 的初始化 RPC 要求 driver 传入已验证的官方 model path；返回 configured load path 证据，不声称完整模型参数 bytes 独立核验。

每次 fresh actor 返回后先检查各 rank 的直接 optimizer20/skips0 和 scheduler-before==outer-1。MAX 错误同步及 outer MIN/MAX 身份同步通过后才 step；随后检查 scheduler-after==outer，并保存 fresh-state/outer-xxxxxx-rankXX.json 的本 rank 计数、LR 与调度器前后状态。19/21 的平均20反例全部拒绝。部分写入、推进后 IO 失败均保留实际状态并拒绝完成 outer 记账，不回滚或伪造零更新。验收必须读取全部直接 rank 文件。首轮 LR0 仍有真实 Adam20、参数Δ0；公式不变，参数变化证明由原 TreeRankAudit 提供。

实际 CPU：8 个模式边界、14 个 fresh/worker 边界、2 个 fresh driver 边界通过，共24个新 case；因 fresh 分支接入，另外仅复验旧成功/异常2个受影响 case。实跑包含两 rank Gloo、tiny AdamW/LambdaLR。fresh driver 首次 deepcopy toy __getattr__ 的 KeyError 保留为失败日志，修 toy fixture后2个通过。未重跑旧完整套件。

所有测量阈值仍需正式标定；本收据不接受原 sampler/actor 概率门禁、真实 FSDP 校准、短五训练或200轮效果。没有提交、部署或改 scheduler 公式。root 的 trainer/config 等其他并行改动不属于本收据 source_files 范围。
