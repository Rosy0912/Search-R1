# OPSD 固定8e：teacher 对齐、动作行为与检索原因只读审核

2026-10-04。范围：实际训练3007固定8e60976d364ab7d7ae51714415813c1359dafe9a的teacher位置、GT/observation/mask、工具完成检测及现有检索证据。未修改任何训练源码/配置、未调用GPU、未重启/取消/提交训练。更新梯度/clipping另由独立review负责。CPU原证据：work/OPSD_ALIGNMENT_CPU_CHECK.json；审查源码来自git show8e，对执行的distillation/rollout函数还核对本地文件与8e字节完全相等。

## 结论

**动作格式/完成行为发生明显退化，不能用“infra验收通过”说训练效果正常。** 现有证据暂未发现teacher预测位off-by-one、GT行错配、检索observations被作为直接监督token、或跨集群/CPU检索回归。真实teacher在相同轨迹状态下是否偏好合法搜索/答案，以及privileged hint实际是否改善这些token分布，尚未做实测，不能把teacher能力/上下文假说写成已证实根因。

根提供当前77/500：旧reward记录显示0.000（四舍五入，不能据此断言精确为零）；finish0.050、valid action ratio0.050、searches0.054、actions3.882。只读本地progress snapshot03:22:58UTC支持下列持续趋势，非只看一个随机batch：

| step | finish | valid action ratio | searches/trajectory | actions/trajectory | response tokens mean |
|---|---:|---:|---:|---:|---:|
|1|0.404|0.528|0.928|3.180|1223.359|
|10|0.421|0.536|0.939|3.166|1211.481|
|20|0.342|0.442|0.787|3.304|1296.412|
|40|0.266|0.253|0.292|3.450|1403.818|
|50|0.117|0.118|0.149|3.752|1648.680|
|60|0.072|0.073|0.085|3.837|1690.925|
|77|0.050|0.050|0.054|3.882|1729.922|

这些是归一化展示snapshot里的训练行为指标，原始正式all8验收独审另行保留；此处不将展示snapshot当原始byteSHA完成证明。初始/step50阶段64题旧EM0.125→0.09375、F1约0.173780→0.1189，只能作为旧计分历史；已知旧reward协议漏洞使其不能替代fixed最终配对效果证据。行为指标是独立的原生动作解析统计，不依赖EM/F1 correctness。

## 已核查：预测位、GT与mask

teacher_inputs先去prompt pad，再追加私有GT hint，再保留response_attention为1的完整学生response（包括检索/invalid反馈observation）。对每个响应槽位j，positions=P+cumsum(response_attention)-2；第一个有效响应token对应teacher hidden的P-1位，也就是预测第一个响应token的位置。内部attention空洞后的预测位按有效token计数恢复。student logits切片[-width-1:-1]对应相同响应next-token位。CPU两行/不同prompt长度/内部空洞/GT hint测试逐个检查8个有效槽：ids[position+1]==对应response token，全部通过。

observation不从teacher上下文删除：因此后续动作teacher能看到同一检索/invalid反馈；这符合自回归环境上下文。直接监督valid取loss_mask；ray_trainer的loss_mask来自info_mask响应段，rollout生成动作model_mask1、observation0、pad0。opsd_loss对未监督槽的student/teacher logits同时置零，无直接observation目标损失。已有实际all8 actor/loss_mask_valid=1仅证明该mask结构约束，不证明teacher目标质量。

actor mini/micro使用DataProto select_rows，tensor与reward_model非tensor按同一索引切；targets取micro.reward_model.ground_truth.target。balance_batch同样按DataProto行重排，不能仅凭某个tensor的重排推测GT未同步。CPU反向排列[1,0]验证Rome target随Rome response行移动。此toy测试不等同保存真实77步轨迹逐token重放，当前未获得完整实际teacher logits/输入trace。

固定入口配置：max_turns4、force_final_answer=false、max_response_length500、max_obs_length500、max_prompt_length4096、max_start_length2048、privileged_answers=true。没有第五次强制回答机会。rollout/context_mismatches在已读1/10/20/40/50/60/70/77指标都0，response_length/clip_ratio通常0；未发现由滚动上下文截断失配导致这一退化的当前证据。force_final_answer=false是已记录实验配置，不能说这是随机丢失了一轮调用。

## 原生完成检测与计分漏洞

原生postprocess_predictions用首次regex <(search|answer)>(.*?)</同tag>。完成条件是解析到answer，既不查正确性也不依赖奖励；search则继续，invalid则添加格式重试反馈继续，最多4turn。finish_ratio=1-mean(active_mask)，valid-action-ratio是每条合法动作数/turns的均值。

直接执行8e原生parser/execute_predictions（检索接口CPU stub，未联网）：完整<answer>Paris</answer>完成；普通Paris或缺closing tag的<answer>Paris不完成；完整search合法但不完成。嵌在think里的answer以及空answer也被原生解析为完成。这是已证实原生解析偏宽松的边界，不是当前5%finish的虚假判零机制：它更可能高估格式完成。_postprocess_responses优先遇到任意closing-search字符串便截断，否则closing-answer；与最终合法外层协议的严格解析不同。缺真实晚期响应文本时不能断言哪些错误tag分布触发了该边界。

旧qa_em.extract_solution需要两次以上answer匹配并取最后一次（通常prompt示例+输出），可接受think/search内部answer；fixed新协议已独立修正。这影响历史评分，但OPSD compute_advantages明确zeros，DistillationActor直接使用full-vocab teacher KL，不将EM/F1/合法动作奖励作为loss或重采样筛选。因此旧奖励漏洞不能解释训练梯度改变，也不能靠仅修计分修复已有动作退化。

## 真实teacher目标质量：假说，不是已验证bug

teacher与初始student均源Qwen/Qwen2.5-3B同一原资产；teacher冻结FP32/eager，训练答案通过字符串[Teacher-only training context: accepted answers: ...]直接追加已tokenized prompt末尾，再续相同学生动作历史。代码没有把这段hint重新包装进用户/system chat角色，亦不生成/筛选teacher合法工具演示。若原prompt末尾是assistant generation prefix，则hint语义上是该续写前缀的一部分；字节next-token对齐正确仍不证明它产生合法搜索/答案教师分布。

当前训练包括学生invalid action token，并在invalid反馈上下文继续蒸馏。纯KL没有显式correctness、finish、search合法性奖励纠偏；一个数学上执行正确的蒸馏目标也可能把不适合工具格式的teacher分布传给student。该机制与观察到更长、更多invalid重试、搜索/完成减少一致，但需teacher实际概率/生成证据才能验证因果。不能宣称base teacher一定不具备能力、hint一定被忽略、或全部原因都在此处。

## 检索与infra证据边界

当前固定部署证据是h100-10本节点E5 CUDA+GPU FAISS+loopback18084；同allocation teacher CUDA update/CPU storage与training共存。已读current3007 pretraining retrieval独审，包括真实query CPU/GPU top3/top50 ID/content匹配、live HTTP top3/top50、bulk24相符；CPU/GPU微小分数/尾部顺序差异已记录。它证明该检查时本地GPU检索实现有效，不证明整个77步期间每个生成query语义有效或命中文档正确。最新作业继续正常outer，没有当前检索超时/跨集群RPC/CPU E5回归的证据。本review未新查询GPU，也不以当前no-exception等同所有infra绝对无问题。

search次数从约0.9降至0.054与原生valid-actions/finish一起降，说明模型越来越少发出regex合法search；不能将所有search减少归因后端检索慢或坏。已存在本地完整索引数值验证，未发现这次失败由远程A800通信造成的证据。具体语义故障需原始response/query/returned-doc trace，而不是从泛化CPU计时推测。

## 历史验收保证与本轮后续

fresh5/all8、结构mask/finite logits与grad、teacher frozen/device/dtype、checkpoint变更/HF5 reload、periodic64、源码/资源lease/数值实现一致性是执行与协议证据。**不保证目标算法改善EM，也不保证teacher比student更会搜索或完成。** 本轮未扩大测试/未给现有run无条件健康结论。

最小成本后续是保留冻结run原始结果，并在授权独立CPU或未来本节点自有allocation中，用实际一小组相同question/prefix/observation/目标别名对teacher hint有/无、teacher/student的合法tag token概率、下一动作类型/最终完成进行配对只读诊断；同时结合独立loss/clipping反向梯度审核。固定协议重评是评价纠正，不是证明动作collapse已修复。任何下一训练修复需新冻结源码/独立验收，不能把此只读报告改为已修复或热改现有3007。
