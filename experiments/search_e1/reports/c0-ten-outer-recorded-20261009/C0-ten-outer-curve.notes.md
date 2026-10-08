# C0 逐外轮派生曲线


10个完整外轮、100个连续实际 Adam 更新、3012对曝光、583074个有效 policy token，均从已有已提交记录派生。CSV保留原浮点精度；图用于阅读。没有读取节点、重新打分或重算 logits。


每外轮使用不同的512题 TRAIN cohort，采样2560条轨迹。TRAIN reward 的分母为2560条采样轨迹；它不是固定512题 heldout 正确率，不能从该曲线推断评测收益或退化时间。


C0 的原始目标方向为 student→teacher reverse KL，mixture_alpha=0。记录先对每条轨迹的有效 token 平均，再等轨迹平均；外轮汇总按实际轨迹数合并各更新，包含真实尾批，不是对更新数平均。C0每对权重为1，weighted_post_loss与post_loss相同。逐词表贡献clip后的post_loss不是原始KL，也不是梯度贡献；clip次数没有完整词表/概率质量分母，不能补出比例。


梯度面板范围是本外轮各真实更新记录的全局 clip 前范数 min–max，不是置信区间；虚线1为原 grad norm clip 阈值。图不证明梯度或权重的新增数值验收，原验收证据另存。


全10轮按轨迹曝光合并：raw reverse KL=0.37264755582515885，post loss=0.36394943558509507。记录中被clip的词表贡献计数1021，已提交overflow=0；该零值不代替所有历史失败尝试的完整审计。
