# PR100 前10轮与评测控制恢复

HKUST1 新 PR run `c6a79477161a4a059e4fd68fc7567f87` 已完成10个 online outer、99次真实 Adam、3002条可用配对和512724个有效 policy token。原训练父进程实际 wait0、全子进程资源释放，raw CP99 位于 `/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010/runs/checkpoints/update-00000099-outer-00009/manifest.json`，SHA `1ed5329cf6f84c518cf01dcc219dddf254500137a0f3164410f053cecd710258`。

TRAIN 采样正确数为12326/25600（48.1484375%）；outer1/10分别50.625%/49.53125%，每轮题组不同。每轨迹等权记录的裁剪后 loss 从0.551605705到0.494465619；全部3002条配对平均0.521936085。它们不是固定验证 EM，也不是每次 Adam 后重新评测的 EM。首次完整 native512 尚未产出，不能据此声称效果收益。

原首个评测控制器实际返回2，未开始GPU/正式512。控制器启动的 CPU 检查点镜像子进程已经退出，却直到outer100才会被wait；其 zombie 被正常资源审计阻挡。仅技术停止原 CPU supervisor，保留既有10轮、完整状态、原源码和实际失败日志；观察器仍存活。旧 supervisor 的真实 wait 返回码无法获取，tmux只确认原前台pane已dead，故如实记 UNKNOWN；pidfd发送SIGTERM不冒充实际返回码。旧mirror父进程未wait，亦保留UNKNOWN。

独立 `opsd_pr100_periodic_recovery_v1` 通过后台CPU实际wait及时回收镜像进程，在所有admission/观察节点收集返回码。活跃副本仍异步，备份不成为GPU启动依赖。四项针对性CPU回归通过；gates/scanner规则不放宽。原训练配置、科学源码、source100、PR精度/选样/损失和Adam保持不变。

恢复首动作是从已验收TRAIN_BOUNDARY10/CP99做512评测，随后原训练run按完整状态`--resume --until-outer20`，继续20…100与每10轮评测。启动和初始化不算512验收；最终100及10次完整评测仍待真实产出。所有原始状态/轨迹/损失留HKUST，本机只保留代码及这份小型聚合报告。

图表和CSV位于 `../reports/pr100-first10-training-20261010/`。远端原始小型报表 `/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010/analysis/first10-training-20261010`；reporter源码commit `176f50f41659bdf0361e64eb30b76eaaacb4d556`。

恢复实际已启动：新source commit `e72eb3cdbf444625824c3d8369890e1c4d76c688`，pipeline run `5e5c6851462d43b5822721ffaa58446e`，supervisor1465428/UID1005/tick936338578，tmux `$76/%76`。新eval run `740dc6a8b01140c297cb24a16be3f200`，controller1465564/UID1005/tick936338806；八GPU worker实际/proc/run/父进程匹配、RUNNING，未完整512。新mirror1465550已实际wait0回收，证明生命周期修复真实执行。原observer1146725保持运行/PPID1，当前实际最小free16495MiB、历史below0。此为该次快照，不替代后续新读。新不可变pipelineconfig SHA `e1f9ec99ade143f3b07f6018995d76a280d6c2b720536c19a8b8b697ec6da185`，小型配置/实际启动记录位于 `configs/pr100-recovery-e72eb3c-20261010`。此恢复启动不算效果或全100验收。
