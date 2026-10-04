# 实际3146无裁剪 KL 独立复核

**结论：同一批真实保存分布的256个配对，无裁剪梯度的0.01归一化logit步全部降低真KL，零变化0、上升0。** 这验证该诊断位置的目标方向，尚不证明实际参数更新、搜索完成率或reward改善。

本次只读使用 actual3146 已冻结的64个有序训练初始状态、base/HF5学生与legacy/user_context固定teacher，不新增模型、数据、CUDA、生成、检索或optimizer。原生 Torch2.4 CPU单线程，独立逐bytes重哈希全部256 NPZ，manifest SHA `3618cc9e2ccefe5be052b2b2ee13adaac25734f1895d0879011518c64ba11002`。

独立计算：p=softmax(student)，q=softmax(teacher)，KL=Σq(logq−logp)，梯度 `q.sum()*p-q`；显式保留FP32概率和的归约误差。比较 `student - 0.01*g/||g||₂` 前后的未裁剪KL。未调用候选损失函数或autograd。

| 学生 / 教师 | 下降 / 零 / 上升 | 平均KL变化 | 最小变化 | 最大变化 |
|---|---:|---:|---:|---:|---:|
| student_base / teacher_legacy | 64 / 0 / 0 | -0.0044715498 | -0.0078191757 | -0.0024042130 |
| student_base / teacher_user_context | 64 / 0 / 0 | -0.0015920406 | -0.0049881935 | -0.0005803481 |
| student_hf5 / teacher_legacy | 64 / 0 / 0 | -0.0044628568 | -0.0074887276 | -0.0024135113 |
| student_hf5 / teacher_user_context | 64 / 0 / 0 | -0.0015899903 | -0.0049939156 | -0.0005954951 |

与先前原生autograd重放最大绝对KL变化差 1.43051147e-06，符号计数全部相同，不宣称不同浮点路径逐位相等。逐行原始结果：`job3146-independent-unclipped-cpu.json`，SHA `e9bf2c5e5ef92f5c0907934eaaac18906a929d13b091462bb7b66226b7a08f04`。

相同分布下旧 `.05` 裁剪，user_context base 30/64、HF5 34/64负梯度logit步反而提升真KL；legacy两个条件各0/64提升。无裁剪消除了这256配对中的诊断反向现象。不能把初始位置结果扩大为真实Adam或晚期历史的因果结论；teacher新提示的完整首答案候选仍55/64降低，这项监督内容风险未被删除clip自动解决。

生产候选入口审核尚待root配置变更落盘。本报告未修改主源码或冻结8e历史数据，不放行训练。后续审核应确认 generalized_jsd默认None、YAML null、controller传值不隐式恢复.05、checkpoint路径保持None，显式.05仍可复现历史；标准teacher KL解析梯度及micro累加/mask边界需CPU核对。
