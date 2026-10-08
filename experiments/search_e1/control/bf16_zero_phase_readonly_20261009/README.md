# 已完成 BF16zero 单阶段的只读 CPU 来源

保存本次实际成功执行的 stdin 原程序，SHA8453f087022a75ee9e180b9777c3f5ec9430805f6fc7b5998e74d8c353b5e8b5。仅按明确已闭合 phase、8个实际worker终态、固定源/模型/heldout/原生轨迹及冻结评分函数核 BF16zero512，并复用现有 summary.py 的计数和按题bootstrap与C0比较。不import Torch，不加载模型、generate、启动GPU、写Node或发信号，不等待FP16控制器整体结束。

这是历史实际只读核验的源码副本，不是重新运行旧训练、M4或评测的入口，也不自动执行。原生逐题和最小身份材料在本机私有proof中，不纳入Git。调用和实际rc0绑定由source-manifest记录；本机原execution JSON含原stdin及参数，可完整核对。不改变已运行72e科学或控制代码。
