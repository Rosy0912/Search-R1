# PR100 前10轮与评测控制恢复

HKUST1 新 PR run `c6a79477161a4a059e4fd68fc7567f87` 已完成10个 online outer、99次真实 Adam、3002条可用配对和512724个有效 policy token。原训练父进程实际 wait0、全子进程资源释放，raw CP99 位于 `/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010/runs/checkpoints/update-00000099-outer-00009/manifest.json`，SHA `1ed5329cf6f84c518cf01dcc219dddf254500137a0f3164410f053cecd710258`。

TRAIN 采样正确数为12326/25600（48.1484375%）；outer1/10分别50.625%/49.53125%，每轮题组不同。每轨迹等权记录的裁剪后 loss 从0.551605705到0.494465619；全部3002条配对平均0.521936085。它们不是固定验证 EM，也不是每次 Adam 后重新评测的 EM。首次完整 native512 尚未产出，不能据此声称效果收益。

原首个评测控制器实际返回2，未开始GPU/正式512。控制器启动的 CPU 检查点镜像子进程已经退出，却直到outer100才会被wait；其 zombie 被正常资源审计阻挡。仅技术停止原 CPU supervisor，保留既有10轮、完整状态、原源码和实际失败日志；观察器仍存活。旧 supervisor 的真实 wait 返回码无法获取，tmux只确认原前台pane已dead，故如实记 UNKNOWN；pidfd发送SIGTERM不冒充实际返回码。旧mirror父进程未wait，亦保留UNKNOWN。

独立 `opsd_pr100_periodic_recovery_v1` 通过后台CPU实际wait及时回收镜像进程，在所有admission/观察节点收集返回码。活跃副本仍异步，备份不成为GPU启动依赖。四项针对性CPU回归通过；gates/scanner规则不放宽。原训练配置、科学源码、source100、PR精度/选样/损失和Adam保持不变。

恢复首动作是从已验收TRAIN_BOUNDARY10/CP99做512评测，随后原训练run按完整状态`--resume --until-outer20`，继续20…100与每10轮评测。启动和初始化不算512验收；最终100及10次完整评测仍待真实产出。所有原始状态/轨迹/损失留HKUST，本机只保留代码及这份小型聚合报告。

图表和CSV位于 `../reports/pr100-first10-training-20261010/`。远端原始小型报表 `/data/shaoshitong/search-e1-opsd-pr100-8gpu-20261010/analysis/first10-training-20261010`；reporter源码commit `176f50f41659bdf0361e64eb30b76eaaacb4d556`。
