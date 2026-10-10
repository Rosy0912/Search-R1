# J去P：BF16＋随机配对＋正确性加权＋混合目标，五轮验收

已完成5个真实在线outer，共12,800条新轨迹、1,497个可用配对、50次Adam更新。正常退出：tmux $83/%83 的实际dead_status=0；原控制器2972706（UID1005/starttick939218401）已结束。训练GPU全部释放，仅原获准检索2239529保留。

| 轮次 | 可用配对 | 真实Adam更新 |
| --- | ---: | ---: |
| 1 | 327 | 11 |
| 2 | 307 | 10 |
| 3 | 289 | 10 |
| 4 | 279 | 9 |
| 5 | 295 | 10 |

从共同Search-R1 archive100权重fresh OPSD Adam开始，BF16 base/main/rollout，原C1 FP32 LoRA/梯度/Adam；保留R均匀五选含参考、W正确性权重、M 0.1固定特权teacher＋0.9冻结outer-old。LR1e-6/cap10/每轨迹有效token均值后全局pair均值及其余冻结预算保持。生产科学源8c2032abd19eefe02ac67cfaa5e0a6eb00e70d79，config SHA b7df2888ee69cf0aed475a39996e92119c8d9ed8e5d512d9afc46e0a223ef165；run d040ebac9fa24ec28d43976299fd4f71。

最后一次独立CPU读取：八rank实际Adam50的完整moments/groups一致且有限；各rank的完整RNG及cursor/scaler按原manifest独立核对通过（不要求不同rank RNG相同）；moments实际FP32，LR实际1e-6。保存的504个FP32 LoRA张量全部有限且相对初始权重全部改变。CPU审计未初始化CUDA。原生逐轮actual wait=0和释放记录保留。本验收确认运行正确完成，不是新的512测试EM或收益结论。

最终checkpoint：`/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-20261010/runs/checkpoints/update-00000050-outer-00004`；manifest SHA `5bf7588f3182a0b46f858f7f85b6699a461665f566366eec6a3832cba06b46c4`。

主目录：`/data/shaoshitong/search-e1-opsd-j-minus-p-bf16-20261010`。CPU验收：`receipts/five-outers-accepted-cpu.json`；终态：`receipts/five-outers-terminal-snapshot.json`；最终运行日志：`logs/controller-recovery4.log`；各轮训练日志：`runs/outers/outer-0000N/trainer-logs/attempt-00000.log`。Node保留关键状态和必要日志，Mac只保留代码/配置/此类小摘要。

恢复记录：首轮完整2560轨迹/八sampler原actual wait0保留；误挂在训练控制器下的CPU observer造成release hold，只精确停止原0Adam held controller2920599和observer2920754并独立启动monitor。旧`observer-detached.json`的only_observer_signaled字段不准确，实际范围同时包含该held controller；此处明确纠正，原记录保留。首恢复完成第一轮CP11后，旧J标签检查错误要求FP16 cast，拒绝实际正确的BF16 CPU merge。外部不可变恢复入口仅将该检查按实际precision判定，复用原CPU child accepted/actual wait0和CP11继续，未重采样/未重置Adam。当前入口`source/operations-17f90f4/resume_completed_sampling.py` SHA e53b81d377c780b2387964f8c21aaeb37f2e540027e24fffc480eddce8175cc6。原生产科学源码/config未热改。

原性能下降PR100八卡已精确停止，原runs/logs/periodic pipeline/eval输出按最新人类授权删除；源代码/配置/小停止证据保留。五轮验收后独立CPU显存observer已结束。没有启动旧训练、旧评测或新消融。
