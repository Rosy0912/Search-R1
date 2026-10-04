# JOB3146 独立实际分布诊断审查

结论：**只读诊断的执行与证据完整性通过；提示修复的效果不通过自动放行，不能据此启动训练或宣称效果恢复。** 当前生产源码仍为 `8e60976d364ab7d7ae51714415813c1359dafe9a`。诊断外部候选是 `user_context` 结构修复，并未写入该生产源码。

## 实际执行与来源

实际 Slurm 3146 在 h100-10、自有 UID0、独立单 H100 上 COMPLETED / 0:0，2026-10-04 14:01:26–14:05:11，225 秒。外部审批绑定真实 job、UID、GPU UUID `GPU-2c3bdc08-b976-0fb5-79bd-04def458046b`，allocation SHA `7c14fa00d566bdd8cfc902c8f59c305913ec71887232094a4f110b6b4604a129`。实际 wrapper 的继承 FD 为 `[9,3]`，当前作业和 GPU 租约未借用其它作业。没有模型 optimizer、生成或检索调用。

实际 manifest SHA 为 `3618cc9e2ccefe5be052b2b2ee13adaac25734f1895d0879011518c64ba11002`。本审查通过只读 SSH 独立读取、逐 bytes 重算全部 **256 NPZ** SHA，核对 manifest、runtime、三个实际模型阶段、初始/最终 job 与 UUID、原始 child exit0/bootstrap0、无 failure 文件，以及生产 commit。未仅依赖下载摘要。模型 stages 为 base BF16/Flash、实际 formal3007 HF5 BF16/Flash、固定 base FP32/eager teacher；每阶段实际唯一参数 3,085,938,688、全参数 finite、frozen、tied、加载四类异常均空。base 序列化434 keys与HF5的435 keys仅 tied lm_head alias差异，不能把HF序列化重复 alias计成额外独立参数。

原始 NPZ 保留远端，13份小证据在 `job3146-actual-evidence/`。本机核对 replay SHA `537b88efb691beded856f8c66bf0ecf28d3e348ea7ccdbd24f0e2e35d3bb6633`，其原生 CPU 重放 exit0。重放严格重新核对原资产、原训练前64行、完整151936词表、全部候选 token、位置及各条件概率，不是只读取自报 aggregate。

## 独立数学复算

另以原生 Torch2.4 CPU、单线程，从保存的真实 initial logits 复算解析梯度；没有调用候选 generalized_jsd、autograd、模型、CUDA、optimizer。令 p 为学生概率、q为教师概率、c_i=q_i(log q_i-log p_i)，a_i=1[c_i<=0.05]。未裁剪梯度为 p-q（精确实数下）；裁剪梯度为 (Σa_i q_i)p-a_i q_i。计算两梯度 cosine 与沿裁剪梯度负方向、L2长度0.01的假想 logit步后的未裁剪 KL。

| 学生 / 教师 | 问题数 | 激活clip | 梯度与未裁剪KL反向 | 小logit步使真KL上升 | 平均真KL |
|---|---:|---:|---:|---:|---:|
| base / legacy |64|64|0|0|1.984459|
| base / user_context |64|61|30|30|0.527396|
| HF5 / legacy |64|64|0|0|1.970490|
| HF5 / user_context |64|62|34|34|0.528849|

所有计数独立一致。独立公式与原 autograd 重放最大绝对差为 KL 1.91e-6、clipped loss 2.39e-7、cosine 2.07e-4、小步KL差1.20e-6；不同计算路径/CPU线程的浮点归约结果不宣称逐位相等。新提示 cosine 离零的最小绝对值分别0.05510和0.003073，上述差异不改变反向计数。独立原始结果见 `job3146-independent-analytic-cpu.json`。

这表明 **结构正确的 user_context 与保留 component_clip=.05 结合，存在实测初始状态目标方向风险**。假想 logit步不是全模型参数梯度、FSDP clipping 或 Adam 更新，不能把30/34例直接当真实训练伤害比例。legacy 在本批首位置没有反向，也不能排除晚期历史/其它生成位置上的问题；不能据此证明旧3007完成率下降由 clip 引起。`.05` 是已批准 PLAN 和作者主脚本选项，风险不能写成工程擅改。

## 候选身份与概率

独立核对全部64行的四个 case、候选字符串及逐 token logp求和。完整答案只取真实 `targets[0]`，候选是 `<answer>`+首个 accepted alias+`</answer>`，无跳过；不是全部别名的成功概率。下表均为初始 prefix 后，teacher-forcing 的完整候选 token概率乘积之算术均值。

| teacher 新提示相对旧提示 | 概率提高的行 | 旧均值 | 新均值 |
|---|---:|---:|---:|
| `<think>` |21/64|0.000352166|0.000239782|
| `<search>` |42/64|0.005699359|0.007092148|
| `<answer>` |63/64|0.004926750|0.017141031|
| 完整首个答案候选 |9/64|0.000184947|0.0000691874|

完整答案55/64降低，平均 logp变化 -0.962877；更强的答案起始标签不代表更强的参考答案监督。新教师 search 标签均值仍低于学生 base 的0.014075748，不能把新旧教师之间的提升当作有效工具策略监督。新提示更接近学生首 token分布（KL下降）也不自动说明搜索目标更正确。

HF5学生相对base，think候选63/64提高，search46/64提高，answer32/64提高，完整首答案31/64提高；没有晚期HF80/100/500模型，不能外推晚期退化机制。直接在初始状态立即回答的候选不包含思考、搜索及观察后的最终回答，完整候选概率降低不等于最终准确率降低；标签概率提高亦不等于真实搜索次数或完成率改善。

## 接受边界与下一步

本报告只接受本次 **诊断完整性**。仅64个有序训练初始状态，非随机代表性评估，未测真实学生历史的监督位置、工具结果上下文或最终 reward/完成率。HF以BF16/Flash直接载入，不能冒充生产FSDP FP32 master + BF16 autocast的逐位重演。225秒单卡分配仅是0.0625 GPU分配小时，不是利用率积分或独占模型计算成本。

因此 root 的 `JOB3146_ROOT_DIAGNOSTIC.md` 核心事实与因果限定准确。建议在下一次另行授权的 fresh5之前，先验证真实历史上的教师提示、参考内容与工具策略，并对裁剪/不裁剪作受控比较；当前证据不放行自动训练，不声称提示修复已恢复效果，也不追认已停止正式500轮的效果验收。
