# OPSD 行为退化独立审查（初步可审查结论）

审查对象严格固定为 `8e60976d364ab7d7ae51714415813c1359dafe9a`；通过 git show 读取，不使用 eval 分支 e3f 的代码。当前正式任务3007不热改、不重启、不取消，不运行新GPU任务，不提交Git。CPU反例和原始结果见 [OPSD_BEHAVIOR_AUDIT_EVIDENCE.json](OPSD_BEHAVIOR_AUDIT_EVIDENCE.json)。

**结论：已复现一个确定的目标函数风险：component clip=.05可以让优化方向背离教师；还发现loss日志的量纲容易误导。没有发现本轮能复现的普通next-token错位、反向梯度符号、mask或FSDP分母错误。当前reward不是优化目标且存在旧评分协议缺陷；不能用零reward直接判断训练未执行，也不能把“作者实现一致”当作“算法能提高搜索完成率”。实际裁剪激活比例和教师的格式/答案质量未采集，因此退化主因尚未因果证明。**

## 1. 确定数学风险：截断正分量可产生反教师梯度

固定源 `search_r1/baselines/distillation.py:12–28`：beta=0先计算每个词表项 `t_v(log t_v-log s_v)`，在第27行仅做上界clamp(max=.05)，再对词表求和。这与真正完整KL不同：各分量本来可以为负，原始正负相抵才保证KL非负；把正项的值和梯度截去，却保留负项，不再保留KL的非负性或朝教师收敛的性质。

CPU直接提取固定源 generalized_jsd 原函数（不是手写替代损失）复现：教师概率(.9,.1)，学生(.1,.9)。两分量截断后为(.05,-.21972246)，损失=-.16972245。对学生logits的梯度为(+.010000,-.010000)；正常梯度下降一步后，教师偏好的第一个token概率从.1降到.09982015，真实KL从1.75777984升到1.75937974。相同输入不截断时梯度=(-.8,+.8)，第一token概率升到.11534964，真实KL降到1.63098192。优化器符号没有反；错误方向来自该截断目标本身。

这是PLAN第35行明确指定component clip=.05的数学风险，不是工程人员擅自增加的超参数。保存的作者固定源码 `ae7d2519` generalized_jsd_loss也在词表求和之前做此上界截断（author_opsd_trainer.py:442–464），其可选token_clip默认None。因此“与作者启用该选项的代码一致”并不能排除这个风险；作者注释称per-token clip也不改变实际分量级算子的性质。此处不宣称违反作者实现或私自改变PLAN。

**已证实的是存在反教师梯度的合法输入；未证实的是实际3007有多少词表项/监督位置触发裁剪，以及它解释了多少完成率下降。** 当前未记录 unclipped KL、正项clipped比例/质量、clipped与unclipped梯度夹角。实际75–77的console loss=-.001不能作为真正KL下降的证据，也不能单独量化这一机制。

## 2. 确定日志口径问题：0.000不代表零信号

`actor.py:50–59`每个micro的loss使用整个mini的全局token分母，backward将这些部分相加；第59行却记录每个部分，然后collect_observations对部分做mean（约97–101行），console再四舍五入三位。当前local mini32、micro2，每mini有16个部分；所记平均loss比平均mini目标小约16倍。该日志不是完整全局监督token平均KL；改变micro大小后其量纲还会变。**梯度累加本身正确，问题在日志解释。** 当前grad_norm非零、全部rank真实optimizer调用成立，不能从0.000断言没有学习信号。

## 3. 实现路径核对：暂未发现这些具体工程错误

| 项目 | 固定源证据和判断 |
|---|---|
| 基本梯度方向 | distillation.py:15–18为正确的KL(teacher||student)顺序，teacher.detach；actor.py:58 backward后调用正常Adam step。未clip时CPU反例梯度正确。风险是clip，而不是全局负号。 |
| token对齐 | teacher_inputs:41–58去左padding、加私有hint、保留有效response；第一个response的prediction position=prefix长度-1。student:84取最后width个前一位置logits；普通左prompt/rightresponse布局对应。观察保留context、loss_mask排除观察；padding位置无需监督。此前真实tiny-HF和全梯度对照支持这条路径。 |
| teacher特权及泄漏 | teacher_inputs:46–50只在teacher前缀加入答案；student:82仍取原input_ids，teacher no_grad且冻结/独立optimizer。targets按DataProto微批对应reward_model行获取（actor:44–55），select_rows同步索引tensor/non-tensor。没找到gold泄漏进student或target乱序的证据。 |
| mask及归一化 | rollout.py:146–147把action标1/observation标0；controller生成attention×info mask；opsd_loss:110–124仅监督mask位置，A仅跳过全空chunk，连接零保持FSDP参与。global_count/world与FSDP平均抵消，micro部分求和，没有额外再除micro数。 |
| 采样context窗口差异 | rollout.py:150–152保留报告模式，理论上滚动cap可能造成生成/scoring前缀不同；这是已授权适配且记录context_mismatches。当前75–77该计数均0，不能把它当当前退化的已证实原因。 |
| update是否真实 | 当前快照75–77 grad_norm=.076/.074/.071；历史all8rank审计每outer320样本、10次Adam/0skip、teacher冻结/finite/mask。当前77不是“全部没有更新”。正式501预算warmup142，第77仍warmup内但学习率已经为正；不是一直LR0。 |

## 4. 为什么reward/完成率不受直接保证

`controller.py:257–259`明确将OPSD advantages/returns置零，outcome reward仅报告；DistillationActor不用rewards，也不读取old_log_probs来构造surrogate。故reward提高不是当前损失直接优化的约束，格式合法率和完成率也没有单独惩罚。冻结初始Qwen base + answer-only hint是用户批准的适配，不等于教师是正确的Search-R1策略、会遵循search/answer标签或能在学生错误历史上输出正确分布。教师在完整学生历史上逐位置预测，并非生成一条正确答案轨迹再做SFT；正确答案提示不会自动保证合法动作概率提高。这里是待验证的目标/教师能力限制，不能凭源代码断言教师确实不好。

缺失importance ratio不是已证实的PPO实现bug：这里是给定采样state/history的全词表条件蒸馏，每个条件内显式对词表求和，不是采样动作reward策略梯度，因此不需要仅为了“有old_log_probs”而加入PPO ratio。512题×5轨迹=2560，每rank320，mini32产生10次更新；同一outer轨迹一次生成、之后mini使用逐渐旧的state分布，确实不是每次optimizer后重新生成的严格on-policy分布。其可能导致state-distribution偏移、重复强化坏历史，但是否应改为fresh-per-update或importance校正属于算法设计决策，不能未经证明称必需的缺失项，更不能直接往损失加ratio热修。

监督action包括无合法标签的action tokens（rollout:147），invalid action本身没有单独loss权重。正确mask不等于正确教学信号；它防止学习环境observation，不保证修复格式。若教师在这些历史下偏好普通文本、结束或拒答，纯蒸馏也可能降低合法完成率；该机制需要真实teacher分布/格式可观测证据。

## 5. 实际观测与评分边界

本地已保存原始3007快照，sample03:22:58UTC，实际77outer：finish=.05、valid_action_ratio=.05、平均动作数3.882、context_mismatches=0；reward mean/max本轮都显示0，loss=-.001、grad_norm=.071。75/76的max reward仍为1（mean打印.000/.001），因此不能把整个训练历史说成每次reward都严格为零；console三位小数也不能还原很小的均值。

step50原64题历史协议EM.09375/F1约.1189低于初始.125/.17378；新sourcepilot2994此前5outer也出现negative stage趋势。由于已报告旧reward/eval评分协议缺陷，这些旧EM/F1不能充当corrected efficacy最终结论。动作解析finish/valid率下降及反复四轮仍不完成是实际运行的生成行为退化，不能仅用评分问题否认。仍须在明确新协议的独立评估上验证真实答案质量，不改变当前3007训练路径。

本轮未发现通信/检索/教师冻结/有限性/执行计数层面足以解释退化的新infra错误；已有gate通过只证明运行和有限更新，不证明行为目标正确。未重新启动模型、下载或改变正在运行的环境。

## 6. 历史独审的边界和待闭环项

历史作者probe/tiny/full3B checkpoint/device/A配对测试主要证明“公式实现一致、内存优化保留梯度、教师冻结、更新实际执行、serialization正确”。已明确pilot效果negative，但对component-clipped目标的非负性、teacher-match全局最优和更新是否朝教师靠近没有做反例检查。这一审查覆盖不足本次被发现：过去允许负loss的说明准确描述了作者实现，却不足以论证目标适合搜索行为优化。

建议后续单独、经明确授权的诊断/对照（当前不执行）：固定真实history比较unclipped KL和clipped loss；测每监督位置的clip激活比例/正项质量及梯度夹角；测冻结teacher对合法结束/search/answer与正确答案的概率及格式质量；独立进行clip=None对照和一致协议评估，保持其它变量冻结。任何换clip、teacher prompt、teacher更新方式、ratio或reward都改变PLAN/算法，须作为新实验source重新审查，不能热改3007或把旧500轮结果改名为新算法。

**本轮可交付结论：目标存在可复现的反教师梯度风险、loss观测存在缩放口径问题；尚无足够证据认定实际退化唯一由clip或某个infra错误导致。教师目标有效性及真实clip激活诊断为PENDING。**
