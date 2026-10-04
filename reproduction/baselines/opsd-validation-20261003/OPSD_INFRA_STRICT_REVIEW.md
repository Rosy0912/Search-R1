# OPSD 固定8e严格infra/性能与teacher上下文复核

2026-10-04。审查对象：8e60976d364ab7d7ae51714415813c1359dafe9a、正式3007。已读真实scontrol终态work/job3007-slurm-final.txt：CANCELLED、ExitCode0:15，运行13:06:47，h100-10；最新本地展示snapshot最大step80。终止按用户授权由根任务执行，本review未取消任务、启动训练/GPU、修改生产源码或提交Git。80不是500，原训练未完成正式效果验收。只读本节点CPU tokenizer探针原始stdout/stderr保存work/OPSD_TEACHER_HINT_TOKENIZER_CPU_RAW.txt。

## 最重要的结论

当前update约400秒有确定的工程与工作量差异，**不能归因于已经修复的跨集群/CPU检索问题，也不能说这是所有OPSD实现必然速度**。固定8e仍使用padded学生全词表head、localmicro2、FP32/eager完整teacher、chunk16完整KL/检查点重算与逐chunk同步；这些是真实代码事实。现有计时不是独占profiler，不能把7.32倍慢全部精确归因某一项。

teacher答案hint确实处在assistant generation header之后，本轮CPU真实tokenizer验证已确认其结构；这与普通next-token错位不同。**结构位置可以证明，导致完成率退化的因果影响尚不能证明。** 未做实际teacher logits/合法tag概率或hint对照生成，不能说teacher一定被误导/忽略答案，更不能把clip toy方向反例说成实际80步全部退化的已证原因。

## 1. 当前已排除范围与尚未排除范围

| 核查项 | 当前证据支持 | 不支持的扩大结论 |
|---|---|---|
| A800/跨集群检索RPC | 当前3007实际retriever属于自身allocation/h100-10，localhost18084，E5 CUDA+8GPU FAISS；原始retrieval comparison/live top3、top50/bulk24独审匹配。无当前跨集群调用或CPU E5回归证据。 | 不证明每个80步query内容合理、召回语义正确或整个期间服务完全无抖动。 |
| CPU teacher计算瓶颈 | 固定代码on_device在update将完整frozenFP32 teacher放rankCUDA，结束返回CPU；实际每rankCUDAentries递增/device/dtype/freeze审计齐全。CPU storage不是CPU forward。 | CUDA stream interval不是exclusive kernel profiler；不能用processCPU秒代替GPU归因。 |
| targets/mask/off-by-one | 固定源码及先前CPU原函数8有效槽、内部padding洞、targets同步行重排证明基本对齐；observations attention1/context保留但loss0。实际mask valid/有限更新审计通过。 | 不证明teacher target distribution适合工具格式，不等同实际80步全轨迹/logits逐位重放。 |
| 无更新/teacher污染optimizer | 已有actual all8 fresh5/first50观测：每rank320samples、10optimizer、0skip，teacher trainable/grad/membership0；参数/HF5变化验收。 | 不证明参数变化方向提高EM/finish；console loss0.000/actor LR0.000是舍入，非无信号。 |
| 配置验证爆炸 | 当前初始/per50阶段64题，普通前5stage_testing0，最终七source51713尚未完成。 | 不可把BAPO每步3393验证或旧作业CPU配置用于解释当前400supdate。 |
| 完成检测 | 原生finish由动作regex是否answer判定，独立于reward correctness；从初期约.4退到step77 .05是真实格式行为异常。 | 旧计分修正不能被宣称修好了生成/训练；解析本身过宽（空answer/think内部answer接受）可能高估完成。 |

执行/finite/device/资源门禁通过只证明这些具体执行事实。本轮不写“infra绝对无问题”“训练正常”或“仅算法自身慢”。

## 2. 400秒相对H20 55秒：可核查差异

实际3007前5步汇总work/job3007-partial-timing.json：gen138.9838、update407.0216、step562.0284秒。H20数字由用户提供：gen69.41、update55.59、step153.44秒。比值分别约2.00、7.32、3.66。这是跨硬件/不同目标实现的诊断比较，不是同硬件同预算benchmark。

固定8e reproduction/h20/train_paper_grpo.sh:26,40-44,54仍给原GRPO train512、mini256、micro64、n_agent5、8卡、param/grad/optimizer offload=true；当前实际resolved YAML是同train512×5、globalmini256、globalmicro16即local2、SP1、ppoepochs1。每rank320轨迹，localmini32给10optimizer/outer；当前每rank160次学生forward/backward，H20 local8对应40次。**四倍调用数**增加FSDP调度/模型检查点重算/同步次数与小GEMM开销，但不等于四倍墙钟因果证明。micro4实际headroomgate失败、micro1/2通过，不能声称可直接micro8且稳过，也不能只提高batch忽视teacher/index/vLLM共存。

固定distillation.py:68-71每micro跑一遍完整teacher base，106-107又按chunk16投影完整词表。teacher独立3B、FP32/eager、TF32关闭，存储约11.50GiB/actor rank，update前后双向转移；GRPO没有这套带答案特权的full-vocab teacher训练计算。H20 Reference log-prob13.89秒另计，OPSD teacher计算在update_actor内，因此两个actor时间还不是相同计时范围。FP32/eager选择为当前数学/精度协议的一部分，不能未经新验收换BF16教师宣称同算法等价。

固定distillation.py:82-84直接调用student_model(paddedinput, attention, position_ids)，绕过dp_actor._forward_micro_batch的unpad_input/packed(total_nnz)路径。use_remove_padding=True配置不会自动作用于自定义opsd_loss。原GRPO dp_actor.py约72-105在packed有效token上算模型/full-vocab logits。**H20也算完整词表**；区别是OPSD保留paddedhead并增加完整分布teacher/KL，不能用“H20不算词表”解释。FlashAttention内部可能剔除attention padding，故只把embedding/MLP/head矩形布局冗余作为确定代码差异，不声称全部attention都算pad。

固定loss_checkpointing=true：student HF gradient checkpoint开启；JSD chunk又用非重入checkpoint重算FP32softmax/KL/clip，节省显存但增加算力。每个非空chunk还有source/target isfinite().all的Python条件，GPU Tensor布尔转换会同步host；每micro chunk_has_loss.cpu.tolist也有一次host read。8e空loss chunk已跳过teacherhead/JSD，但**没有跳过整个学生forward/full logits或teacherbase**。float(loss.detach())日志也逐micro等待；teacher事件在flush时同步。数量必须按当前真实非空chunk/物理shape算，不能沿用旧2960所有232chunk作为3007事实。

3007原始教师计时区间如firstouter各rankCUDA约105–107秒、双向转移约12–14秒（其余rank/步以原日志为准）。这些区间包含stream/同步边界且可能重叠；不能从407减105再把差值叫已测学生耗时，也不能写独占百分比。缺当前teacher base/head、student forward/backward/collective、JSD/recompute、host同步的独占profile，最终归因是待测。

## 3. teacher hint的位置：本轮真实CPU证明与边界

固定RLHFDataset.__getitem__约127-129：若tokenizer有chat_template，apply_chat_template(chat,add_generation_prompt=True,tokenize=False)；原数据为role=user，内容来自qa_search_train_merge.make_prefix(base)，要求think/search/answer并包含答案示例。固定teacher_inputs约41-51取得已经模板化的prompt有效tokens，随后tokenizer.encode私有hint直接追加，不调用chat_template重新增加role或封闭assistant turn。

本轮在h100-10原资产目录只加载AutoTokenizer、CUDA_VISIBLE_DEVICES空，没有加载模型、索引或teacher forward。实际tokenizer_config SHA c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9，与原资产content记录相同；chat_template_present=true。独立问题audit?生成的实际模板末尾为：

```
<|im_start|>user
Question: audit?<|im_end|>
<|im_start|>assistant

[Teacher-only training context: accepted answers: ['audit answer']]
```

这证明当前hint在assistant续写前缀里，没有新的system/user role边界。真实问题内容不同不改变这段构造顺序。teacher在该hint之后继续条件化相同学生已生成history；普通next-token位置仍正确，不能把上下文语义问题叫index off-by-one。把hint移入user/system或构造author原collator teacher_prompt将改变教师输入分布，需新版本/配对验证，而不是本轮已做修复。

影响的合理假说是：冻结base同3B teacher在不同特权前缀下偏好普通文字/结束/invalid续写，学生全词表蒸馏可能跟随；invalid动作也是model-mask1，环境invalid反馈仅context，没有显式correctness/finish罚项，因而不能靠reward0约束该方向。**需要实际相同state的teacher有/无hint分布、合法tag/正确答案概率与teacher生成对照，才能证明。** 当前CPU tokenizer证明的是角色/字符串布局，不是概率、行为或因果效果。

## 4. 作者代码一致性与Search适配的真实边界

已读保存的author_opsd_trainer.py:compute_loss约632-703：student/teacher分别用student_prompt_length/teacher_prompt_length slice前一预测位；完整teacher_input_ids本身来自作者数据/生成流程。作者教师context可动态同model、EMA或PEFT disable_adapter。当前PLAN明确冻结同Qwen base、answer-only私有上下文、无rationale、base主模型、四轮Search；当前独立frozenFP32 teacher是在该计划下的Search适配，**不是作者默认动态/EMA/PEFT workflow的逐项复现**。仅generalized_jsd函数值/grad等价不能证明teacher prompt与Search控制逻辑也等价。

保存作者trainer没有其完整原collator构造在本次审查范围内，因此不凭函数变量名臆造作者把答案放哪个role，也不说该hint布局一定违反作者协议。严格原文对应项还需由主任务原文/原collator审查完成。用户批准的PLAN本身会限定teacher能力/precision/上下文/teacher冻结策略；它不能被当作作者效果已保证的证据。

已知component clip=.05反教师梯度toy反例与micro平均loss口径16倍问题由另一独审记录：前者证明某些合法输入的目标性质风险，不证明实际80步clip触发程度；后者是记录量纲而非梯度重复归一化错误。此infra报告不将它们归因百分比或替代真实teacher/clip概率审查。

## 5. 必须保留的现实结果与最小后续证据

正式3007已经CANCELLED0:15/80步，不能填写500/HF500/final51713完成receipt，也不能追加旧2999轮次。行为退化、旧阶段负面结果和历史评分缺陷都应真实保留。前5/50执行验收不能被改名效果通过；新fixed协议工具的CPU/default78审查也不证明任何实际配对效果。

本轮到此只读复核结束。若后续获授权，先固定少量真实history与teacher/student权重，用独立CPU或本节点自有allocation采集teacher有/无hint、clipped/unclipped损失与梯度方向/clip触发比例及合法动作概率，保持问题/history/mask/target完全一致；分别做packed学生/更大micro/headstreaming等数值与内存验证，再同硬件同检索同预算测速。未做profile不再声称某个环节独占400秒，也不在当前被冻结源热改继续训练。

## 与根任务综合稿/原文审查交叉复核

已读work/OPSD_STOP_AND_STRICT_REVIEW.md及work/OPSD_ORIGINAL_STRICT_REVIEW.md。根独审取得actual80完整all8原日志，并给step80 gen180.873/update401.573/step597.641秒；这些是step80，不替换本报告独立读取的前5均值。step80 update/H20约7.22倍；前5均值约7.32倍。综合稿已区分额外工作和非公平benchmark，没有将teacher区间相减求独占成本，也没有把旧55GiB snapshot说成cancel后仍占用。资源停止状态由专门JOB3007_STOP_REVIEW验证，此infra报告没有重新采样GPU或据历史值称仍占用。

原文独审已直接核固定作者data_collator:95-109和transition:37-43，确认作者在user角色加入solution/过渡后再apply_chat_template开assistant。这个新增primary-source核对补足本报告之前只读trainer、未单独取得collator的范围；本报告自己的实际CPU tokenizer原证据也独立证明本地hint在assistant头之后。作者4B主脚本fixed_teacher+PEFT且论文实验固定initial teacher，故不能把冻结初始teacher本身判错；真正差异是独立FP32/eager实例/全参数学生以及teacher条件构造和Search任务适配。综合稿提示结构差异已证明、实际因果待probability对照的表述与本review一致。

没有发现根综合稿infra/性能/提示部分的新增阻断事实错误。严格原文/目标、停止状态与本报告分别保留其保证范围；这仍不是修复或效果通过声明。
