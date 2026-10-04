# OPSD loss 日志修复 — 仅本地候选，未部署

2026-10-04。父提交9ed9c3c470f5e0071ddeda0a7e09837c667162b2。已取消3007冻结源8e保持不变；本次不改teacher prompt、clip、backward、全局分母、optimizer或训练配置，不启动GPU。

## 设计与修复

每micro使用同一完整mini全局有效token分母；其loss仅是该micro对mini目标的部分贡献。此前把这些部分贡献交给最终列表mean，日志随micro拆分数量缩小。本次在每mini累加float(loss.detach())，只append一次完整mini贡献，其后仍对各mini等权平均。比日志末尾乘固定16更稳健：末小mini、不同micro和多epoch不会依赖固定拆分数。

此量是每rank的全局mini目标贡献乘world_size；相同mini的全rank平均才是全局mini目标。legacy未带rank键可能来自rank0，不自动是全局KL。各mini等权均值也不是整个outer按token数加权均值。目标仍含component clipping，不保证非负，不能标作真实未裁剪KL。旧3007日志量纲与本候选不同，旧证据不能重写或静默同比。

## CPU验证

新增test_opsd_loss_logging.py实际调用DistillationActor.update_policy、真实DataProto/TensorDict/global_token_count/SGD，用可微解析损失代替昂贵模型。六行有不均匀有效token，mini4与末mini2；micro1/2/4必须记录两个完整mini目标，且梯度及更新权重一致。该测试覆盖真实actor循环和记录流程，不验证3B teacher语义。

修复前实际测试失败：micro1产生6个部分loss，预期2个mini记录。修复后新增测试及既有infra/空监督Adam状态回归共4 passed；原始输出opsd-loss-log-fix-cpu.txt。这是日志修复验收，不是教师提示修复、实际退化归因或效果提升。

独立审查另核旧父提交与候选同micro的梯度/权重一致，具体结果以OPSD_LOSS_LOG_FIX_REVIEW.md为准。尚未部署H100、启动新训练或完成500/HF500评估。后续改变提示或目标仍需真实教师分布对照及独立验收。
