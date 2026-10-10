# 固定普通提示教师轨迹的 off-policy OPSD

Goal：HKUST1八卡，从source100权重fresh Adam开始，BF16 R+W+M训练200个outer；完成前5个outer并确认第6轮真实推进后交回初期验收状态，后台继续200。每10轮顺序评测固定512，不额外评测第5轮。

1. 独立新目录及源码版本，保留当前训练最后完整checkpoint、日志和评测；只停止本任务当前owner和其子进程，保留检索服务及其他任务。
2. 每outer固定source100 teacher用普通提示fresh生成512题×5轨迹。teacher生成不含正确答案或成功参考；检索及生成预算不变。R从五条均匀随机选择D，原规则选正确B，原W权重保持。
3. 在同一D历史上计算当前student普通分布、fixed teacher特权B分布及outer-old普通分布；M=.1teacher+.9outer-old，原reverse KL/cap10/LR1e-6/轨迹均值/global32/LoRA/Adam设置不变。分离采样teacher身份和训练student/outer-old身份。
4. 保留teacher普通完整轨迹、检索和实际逐token preupdate loss。每轮固定少量同题诊断另行生成student普通自主轨迹与teacher特权自主轨迹，明确无训练/评测分母贡献，避免全量冗余。诊断上下文不进入训练。
5. 发布源码/入口/配置，必要CPU检查后启动。亲自核前5轮完整sampling、mining、真实Adam、finite loss、checkpoint和实际退出；第6轮真实采样或更新后结束初期跟进。持续每10轮512和200轮最终结果。

不改变BF16为FP16，不恢复GRPO Adam，不重启旧实验。大产物只留HKUST，Mac仅代码、配置和小摘要。
