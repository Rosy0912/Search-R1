# C0 已提交 token 日志的本机只读聚合

保存实际执行的独立CPU读取器及收尾来源工具，原执行上下文位于本机私有分析目录。沿固定事实/库存的准确 committed 引用重聚合，不访问 Node、GPU、模型或 HTTP。不是新训练/推理入口，也不自动重跑。

源码中的原持久本机布局仅用于还原历史执行。原始问题、token日志、完整引用文件清单和带私有路径的 proof 不纳入Git。可公开的聚合数值和输入内容digest在 reports/c0-ten-outer-recorded-clipmass-20261009。结果描述被记录的 forward 标量，不重建 logits、梯度或因果。
