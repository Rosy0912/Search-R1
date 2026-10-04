# OPSD loss 日志最小修复独立审查

结论：**候选通过CPU独立审查，只改变日志聚合；未部署，不代表GPU或效果改善验收。** 冻结8e训练历史不被重写。

## 源码范围

审核当前候选actor.py SHA256 `92cb9237170260c1d625007dc006db86ba82b66a7e264e004901147c502a99e3`；测试 SHA256 `bec81b955b4e665a2c87ac6993da555ffe9f49d96b11366a9ae2aa2212308cc4`。生产候选diff仅在每mini初始化Python float mini_loss，将原每micro detach标量append替换为累加并在mini末append一次。loss构造、teacher、分母、backward、zero_grad、step及grad_norm保持同次序。没有保留autograd graph或新增tensor反传；原每micro float同步仍存在，不声称性能提升。

## 独立执行证据

CPU环境CUDA_VISIBLE_DEVICES为空。真实DataProto与DistillationActor.update_policy测试通过：`tests/baselines/test_opsd_loss_logging.py` **1 passed / 1.34s**。使用不均匀监督mask、SGD、6行/mini4及末mini2，micro1/2/4验证每mini日志、每rank均值、累积梯度和最终权重；此测试模拟可微loss而非真实3B teacher，能有效检查actor控制流和聚合问题，不证明新数学目标有效。

另外独立从固定8e AST提取原 `_update_policy`，在同一Python进程执行原/候选，使用完全相同DataProto、权重初始化、micro和SGD，扩展到ppo_epochs=2（4次optimizer）。每次step梯度与step后权重均 `torch.equal`，所有micro1/2/4通过，优化器次数均4且无改动。原loss条目分别12/6/4；新条目各4。新完整mini目标分别约 `[3.0, 4.9866667, 1.9599999, 3.0222218]`；不同micro存在正常浮点累加末位差异，测试使用数值close而非宣称跨分区字节一致。原/新在**同分区**下每次参数更新字节一致。

## 指标准确含义

修复后为每mini微批贡献之和，再等权平均mini。多rank下每rankloss使用全局token分母/world，故是rank局部贡献×world；全rank平均才还原全局mini目标。rank0 legacy key不能称完整全局KL。末小mini同等权重，因此也不是全outer监督token加权均值。该候选修复微批大小造成的日志缩放，不改变既有rank/mini观测语义。

component_clip=.05仍可能产生负目标；修复后的opsd_loss仍不是保证非负的unclipped KL，不应据日志下降判定行为改善。旧micro2/mini32日志约缩小16倍仅用于历史解释，不倒填旧数值或把所有不均匀mini统一乘16。

## 边界

无阻断项。允许作为仅观测性本地候选继续主审；完整CPU gate由root执行，本独审没有把root尚未读取的全集输出计为已通过。未修改生产、测试、冻结8e、远端环境；只写本报告，未提交Git，未运行GPU。
