# OPSD 教师提示候选修复与分布诊断

2026-10-04，父提交5cae101。停止的3007源8e、历史日志及step5保持原样；候选尚未部署或训练，不宣称已恢复效果。

## 采用的解决顺序

先修角色语义而不同时替换算法：在明确Qwen ChatML user结束之前插入accepted answers与独立推理/遵循工具协议的教学过渡，然后保留assistant generation suffix和完全相同的学生历史。学生问题不加入gold，tool observation mask/目标位置/分母/教师权重/clip保持原设定。答案只进入训练教师上下文。

候选配置teacher_prompt_mode=user_context；legacy可供配对诊断。函数缺省legacy保证旧调用可复核。精确限定当前Qwen2.5非thinking模板，验证generation True/False、sentinel唯一、实际suffix IDs；无模板/异常边界/特殊token答案/超长/未知模式拒绝。不是任意模板通用解析。TOP-D无答案既定路径保持不变，本候选只验收纯OPSD；不声称未来TOP-D privileged调用自动采用新模式。

仅按正确角色插入，比继续在assistant内容内追加注释更合理；这项结构修复仍不能证明冻结3B教师在搜索任务具备有效教学能力。LoRA/全参数、单/多采样、10mini滞后等原文适配差异未被本次修复消除。

## 已完成验收

根任务CPU新提示、loss日志、算法与模型/driver回归 **33 passed**，原始stdout `opsd-teacher-prompt-fix-cpu.txt`。独立新提示+日志 **10 passed**；另外2row不等长、search/observation/answer历史的16个教师LM-head chunk与完整HF reference最大差2.98e-8，teacher无梯度、student梯度finite。独立报告 `OPSD_TEACHER_PROMPT_FIX_REVIEW.md`。

在h100-10仅CPU加载真实tokenizer（未加载模型/索引/GPU）的候选函数探针 **exit0**：真实tokenizer_config SHA `c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9`；hint在user中、学生prompt未改、每个response预测位配对、控制token注入拒绝均通过。候选函数SHA `ceb955404a2e5c5be8b3397c80ed388752847774a89eb76e4a33e810f31ce02f`。原始stdout/stderr `opsd-teacher-prompt-native-cpu.txt`，可重放脚本 `opsd_teacher_prompt_native_cpu.py`。

这些证据只验收模板/对齐/CPU路径，不是3B教师概率或Search效果验收。新条件分布允许改变梯度，不能宣称与旧8e数学目标等价。

## 下一项：真实3B分布对照，暂不启动训练

准备同一h100-10、自有Slurm单GPU、固定64个**训练问题初始state**的只读诊断：base与已有HFstep5学生，固定base教师，legacy与user_context提示，对照clip=.05/None。不是validation/test gold、不是新训练、不是64题阶段评估。学生输入不得含gold；保存问题身份、输入tokens、完整下一token logits和tag序列logprob，以便CPU重放。

测量未裁剪KL、clip激活及teacher概率质量、clipped/unclipped logit梯度夹角、小logit步之后真正KL变化、完整think/search/answer tag条件概率。tag概率不是合法完整轨迹率，logit小步不是真实参数优化，初始state不是工具交互全轨迹。不存在80轮checkpoint，不能冒充检查80轮模型；现有5轮结果也不能归因80轮崩溃。

完成数据/源码/资源门禁与独立审查后才提交该诊断；不自动启动500轮。若真实证据支持，再安排受控fresh5与一致评分64题、行为退化停止门禁；保持所有GPU编码/检索/教师/训练同节点，无A800或跨集群RPC。clip是否移除由配对证据决定，不把论文同款机制直接称抄错。
