# OPSD 退化原因分析计划与验收条件

日期：2026-10-08。负责人：当前 Search-E1 会话。此计划承接用户要求“针对三个疑点保留 teacher/student 轨迹和每个 token 的 loss”，以及“好好制定 plan 去做”。它定义持续分析的顺序、对照和交付，不授权重启已停止的八卡训练或旧七基准评测。

## 目标和当前判断

目标是解释这一次固定实现的 OPSD 为什么降低问答表现，区分代码执行错误、教师条件与学生条件不匹配、样本选择偏差、固定轨迹上的训练偏移。需要输出能被反例推翻的判断和逐题证据，不能用后台启动、loss 降低或工程验收代替效果结论。

已确定的事实：

- 相同 native512 协议，source100 为 239/512（46.6796875%），raw epoch1 为 203/512（39.6484375%），净减少 36 题 / 7.03125 个百分点；一 epoch 完成 3156 次真实 Adam 更新。
- 全量 100974 对中，56895 对是正确 reference＋正确 student（56.35%）；44079 对是正确 reference＋失败 student。169103 道训练题中 68129 道五次 strict reward 均为零，不进入蒸馏。失败包含协议失效，不能直接称为事实无知或固有难题。
- 首个已独立验收的 correct/correct 样例，原响应的固定检查点 loss 由 1.223577 降至 0.201201；普通提示实际生成却由 strictEM1 变成 strictEM0，未闭合 think，且没有执行搜索就声称已经使用初始搜索结果。目标优化与有效回答之间存在实际背离的例子。
- 首三类样例 42 probes / 29383 token rows 全有限，18 次 native pair_loss 对照绝对误差为零。已检部分没有显示 token 错位或数值失控；这不等于完整实现或教师目标已证明正确。

2026-10-08 12:36 UTC+8 后的固定 64 题快照只作探索性证据：source100 46、update1 47、update5 41、epoch1 38 题 strict 正确。epoch1 相对 source100 有 13 题正确→失败，5 题失败→正确；13 个退化样例中 3 个协议无效，10 个最终 action 合法但答案与标注不匹配。执行前缀为 NQ61 / Hotpot3，不能当作完整192、native512或总体退化率。教师自主生成的61个可用参考样例有17个协议无效；自主生成教师与训练中沿原 student prefix 的 teacher-forcing 分布是不同对象。

当前最可信判断：这版蒸馏目标存在与实际问答行为背离的证据，协议失效只解释部分诊断失败。教师条件依赖、数据选择和长训练偏移的因果贡献尚未量化；不能直接认定学习率过大，也不能宣布 OPSD 算法普遍无效。

## 分析路径的选择

采用“现有固定检查点＋逐题证据＋固定前缀反事实”，先充分使用已保存权重和轨迹，额外计算集中在能区分机制的对照。仅看case和loss相关性虽然快，但不能测教师纠错或定位保存点上的退化；直接重新训练多个超参组合成本高且容易混入样本/预算/目标的同时变化。完整七基准能判断外推表现，却不能单独解释机制；它不作为重启已停止评测的理由。先完成机制诊断和native512，只有固定模型证据不足以识别训练因果时再提交最小单因素训练方案。

## 固定对象与禁止混用

- 原训练对象：HKUST1 `/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007`；source100、adapter_update_1、adapter_update_5、adapter_epoch_1 分别作为四个固定模型。raw epoch1 与 selected best 必须分开，best 仍是 source100。
- 当前诊断：`/data/shaoshitong/search-e1-opsd-debug-20261008`，run `894021c8f55f436f96bb3cb146ab05bb`，原192题清单 SHA `1703b15b1a1b3c5b91b13422fa314ab9a3b83f7bf7bf8a85dd7afcdaecdbf082`；六个 dataset×group 单元各32题，既有作业继续。
- 精确科学源码：305-file runtime，outer manifest `3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc`。当前15-file诊断源码已经冻结；新分析或对照另用目录和 run，禁止热改它。
- 教师是冻结 source100、关闭 adapter，看到成功原始 reference；学生看到普通提示，使用对应固定 adapter。训练教师沿同一预存 student response 逐 token 条件预测，并非给成功 reference 硬标签，也不是训练期间重新生成的教师路径。
- 原训练 loss 是 teacher→student 的逐词表 forward-KL 贡献上截 clip10，再有效 policy-token 平均，再全局轨迹平均。information 被 mask 掉但仍是上下文；think/search/answer 均有监督，没有独立正确性奖励、最终答案加权或普通提示原策略 KL 约束。裁剪后的量不能无条件当作未裁剪的标准 KL。
- 实际 mine 先选成功轨迹中搜索最少、再 token_count 最小的 reference；在失败候选中取与 reference 完整字符串归一化 Levenshtein 距离最大的 student，无失败时取另一成功轨迹。信息文本虽不计 loss，仍参与距离。不能误写成“最近轨迹”。
- 本文“final action协议有效”专指冻结QA对最后authored policy span中终结答案的接受，不证明整条transcript所有think/search标签都平衡。首source100例的前一search段也有未闭合think，最终新policy段的answer仍获接受；epoch1零搜索时think与answer落在同一最后span而被拒绝。要单列这些span边界/搜索路径变化，不能径直写“OPSD才学会不闭合think”。成功reference同样不是完整推理/全部结构均正确的保证。
- 历史 online per-token loss 未保存；现有记录是固定检查点重算原响应和新生成响应的 posthoc teacher forcing。不得伪称原 Adam 步的在线记录。
- 训练soft目标使用原head raw logits的温度1概率；HF实际生成还有generation_config、logits processor和停止/结构截断。既有原生审计发现source generation_config的repetition_penalty=1.05；新评估必须实际读取并保存完整配置，不能只标greedy就称条件一致。

## 三条假设与可证伪检查

| 假设 | 当前证据 | 必须追加的检查 | 何种结果会削弱该解释 |
|---|---|---|---|
| H1：特权教师目标依赖学生推理时不存在的参考信息，或不能纠正失败前缀 | 有无搜索却引用参考/搜索结果、think未闭合、loss降低而 strictEM失败样例；实际教师自主生成并非全部有效 | 在同一个原 student prefix 上比较普通 source100 与有 reference 教师；再在固定前缀边界实际续写。固定模型，分别用原参考、无参考、去掉 reference 最终 answer、确定性错配参考。记录 closing-think/search/final-answer 概率、实际 action 与 strictEM。去掉最终 answer 不等于移除了思考或文档里的全部答案信息 | 有参考教师稳定改善失败前缀续写会削弱“教师纠错无效”；它仍可能强烈依赖特权参考。共同历史上分布/行动对参考条件不敏感才会削弱“参考依赖”。两者分别报告 |
| H2：配对偏向已会题，距离含检索文本导致监督选错重点 | 全量56.35%成功/成功，40.29%无参考题排除；选最大完整字符距离已核实 | CPU重算各原 sibling 的完整文本距离、去 information 后 policy文本距离、实际搜索/最终答案差异；报告原选择与 policy-only选择的不同率。固定192分层统计正确→失败/失败→正确、每段有效token与loss/漂移。全量比例与诊断样本比例分开 | 选择很少变化，易题没有更多退化，难题原本就没有可用监督不能解释已有题下降。任何统计相关都不足以证明该选择规则的训练因果效果 |
| H3：固定旧轨迹上的3156次纯蒸馏使策略逐步偏移，关键行动/答案约束不足 | native512 epoch末下降；有保存的update1/5/epoch1，原响应loss下降却实际生成变坏 | 四模型同题比较原响应loss、未裁剪/裁剪KL、普通source100→student漂移、答案/搜索/思考区段和真实生成变化。以完整同协议512评估保存的检查点，确定最早“已保存且能观察到”退化的点 | 早期已有同等失败、epoch末没有进一步移动会削弱“长期累积主导”，不反驳蒸馏本身引起早期drift；共同状态分布稳定而只有decoder/检索差异才削弱分布偏移解释。精确退化步不能由只存1/5/末尾反推 |

这些反事实只测试固定权重的条件依赖与纠错能力；即使结果支持机制，也不能直接量化“重新训练去掉该机制后会提高多少”。学习率、样本过滤、蒸馏时长的训练因果效果，需要独立受控训练消融，不能用固定模型诊断冒充。

## 执行顺序、负责人和验收

### M0：身份和原始结果基线（已完成，持续保护）

Root 维护源、模型、数据、评分身份；独立审计复核数值与模型标签。保持 HKUST1 resident 完整 FP32 GPU检索、原oracle和公共路由；保持 HKUST4 archive 自有GPU4–7作业。旧K1八卡训练/七基准停止范围不扩大到新诊断目标，也不以旧heartbeat重启它。

验收：源和权重清单一致；native512两端结果来自相同协议；epoch1/best不混用；原样本/失败/代码和检查点仍存在。原历史源与新辅助脚本完整本地备份、Git提交、远程SHA读回。

### M1：把当前192题诊断完整做完（进行中；Root＋独立逐题审计）

无需用户逐次催促，跟进同一个已有作业至192个唯一完整case，不重做已经完成case。先完成观察，再按固定六单元报告四checkpoint配对翻转。每一对模型都给正确→失败、失败→正确和净变化；strict失败主类互斥拆成协议无效、协议有效但答案与标注不匹配。no_answer/budget/context等termination单独作为另一轴标签，不重复计入主类。仅严格合法final action计EM，嵌套tag字面匹配另列。

同时按原轨迹/新轨迹、dataset/group、think/search/answer分别汇总loss与普通策略漂移、搜索调用和长度、context/empty-mask/full-vocab-cap等skip。沿同一原轨迹比较loss；不同新生成文本的均值不得装作同对象loss曲线。区域划分有明确token offset和mask，跨tag token边界单列，不篡改原mask。

每区段同时报告区段内token均值、该区段loss总和除以整条轨迹有效token数；只有后者之和还原真实轨迹loss。它是loss贡献分解，不能冒充梯度贡献。hard的五siblings先题内聚合，每题为统计单位；按总体六单元人数可另给训练题加权结果，但不将其转成验证/测试总体估计。token不能作为独立显著性检验样本。

验收：192/192唯一清单覆盖；三worker exit0；源/模型前后稳定；每个已接受probe有限、mask正确、native比较通过；token行数与summary对账；所有skip有理由。首三类独立全词表CPU核验已经完成，追加抽检按固定case hash而非专挑失败例。输出完整配对表、六单元表、loss/漂移表和带证据的反例集。192来自训练题且平衡抽样，不能估计验证集总体EM。

### M2：补齐样本选择与损失区域审计（CPU并行；数据审计＋数学审计）

复用已有845515轨迹/100974pairs，不重新生成。先读实际数据结构和已存统计，避免反复全文件hash。一次受控流式扫描，核实 sibling 覆盖、原选reference/student与源码规则；比较包含/移除information的字符距离排序，按数据集与原成败组分开报告。该变体是CPU分析，不改训练pairs。

另统计成功reference的非最终policy段标签闭合、搜索前缀和最终span的差异；只作结构/数据质量审计，不改变原strict reward，也不将旧评分接受自动升级为完整轨迹语法正确。调查零搜索后原有结构缺陷是否更容易落在最终span，与“新产生结构缺陷”区分。

已有token records用于逐轨迹先平均的区域贡献和KL漂移统计；若全量需tokenizer，单列原mask实现与context条件，不用字符长度替代policy-token数。对clip10报告饱和token/词表概率质量、未裁剪与裁剪量差异，检查是否大量目标被裁剪改变；不凭clip参数值认定失控。

验收：全量结果分母、稳定输入文件identity、坏行/重复/缺失计数、全量可重算入口齐全；选择变体仅分析；每个结论附代表性与反例样本。能确认偏差与监督分布，不能宣称训练消融已完成。

### M3：检查教师在相同失败前缀上的纠错能力（M1身份通过后，独立固定权重对照）

M1完成后M3/M4都可准备；GPU顺序优先M4的原native512两端核验与update1/5定位。M3固定前缀概率分析先复用已存token records，追加生成在资源空闲时进行，不能为这些新对照暂停正在运行的192诊断或K4 archive。

原192题已在观察中，因此这一批是明确的探索性机制诊断。成功参考四单元各按固定seed/hash取16题，共64题；独立保存选择清单，不能看到对照结果后换样本。all5_wrong组维持“无成功参考”，不虚构教师，作为普通模型能力的另组证据。

先在64清单中每单元前4题（16题）验证状态重建、普通base对照、容量和记录完整性；通过后同一run继续剩余48题，已完成16题不重跑，不因某个效果方向好看而停止/换样。另列少量已见退化＋同单元稳定/改善的case-study，用于首次动作分叉和解码追踪，不混入64固定诊断效应比例。

在原轨迹的两个可执行边界（首个完整policy段后、最终回答前最近完整段后）比较普通source100与特权teacher的分布及实际续写。已消耗的search次数、已插入文档和context预算全部继承，不能重置搜索预算或任意截断到残缺tag/token。边界不存在或context超过8192则显式skip。四种参考条件只改变teacher上下文，冻结base、同一student prefix、生成预算、检索实现和评分；错配reference用确定性跨题映射，仍标记为人为错误参考。

续写必须重建完整executor状态：已消耗turn及剩余5-turn预算、已用search次数/剩余4次、完整已有retrieval observation、末轮forced预算提示是否已经插入。search段后的边界必须在原真实observation返回并插入之后；不能停在等待检索状态就开始新生成。若边界已经terminal answer/invalid/context/budget，或首段本身已answer，标记terminal_boundary_skip；不能越过终态继续。最终答案前边界应是原最后policy生成之前的可执行状态，不能截出生成中的标签。保存各边界状态及是否有失败student历史，普通与teacher条件共用同一状态，不多给5轮、不重复插forced提示。

reference删除或错配同时改变prefix内容、长度和token位置，结果只解释为整体prompt干预效应，不独断为纯答案泄漏效应。成功reference仅证明最终strict正确，不证明其中全部推理正确。新实际响应需交叉重评：source100与epoch1各自响应都在同一组四checkpoint下测量，剔除不同history/observations造成的loss可比性混淆；原固定student历史上已有teacher/base分布直接复用，不重复计算。

对首次协议破坏位置追加少量解码核验：从实际turn context_text和保存generated IDs重建输入，分别记录raw logits与实际HF processor之后的scores/argmax、generation_config、停止条件。尤其比较closing-think与opening-answer的排序，检查repetition penalty或停止/结构cut是否改变行动。它是解码机制诊断，不改原生成配置、不使用修复后的宽松评分；posthoc native重tokenize概率不能代替这份真实生成前缀的processed-score证据。

验收：同一prefix的条件差异、query/doc/action和逐token记录可追溯；去ref的冻结teacher须通过普通source100身份/分布对照；实际续写strictEM/协议有效率/搜索变化与概率变化分开。结果若表明teacher自主表现好但失败前缀续写无改善，也必须报告。反事实结果只支持条件依赖，不等于OPSD训练因果贡献。

### M4：在独立512验证集上定位保存检查点的退化（固定权重评估，不重新训练）

先检查原native512是否保留了完整逐题输出；当前已确认有总体history，不能假称已有逐题轨迹。缺失时在独立评估目录复用原native SearchAgent与512数据，使用source100/update1/update5/epoch1。优先保存首次完整输出，之后不重复重跑同一条件。保持BF16 base、实际adapter dtype、HF SDPA、greedy、8192上下文、500token/turn、5turn/4search/top3/info480、评分和模型身份；逐rank分片/种子执行方式、GPU检索切换及数值身份验证必须记录。

先用source100/epoch1检验与原239/203是否复现。完整generation_config及实际processor、tokenizer/chat template、adapter加载、检索数值、后端、分片等均须留身份；若不一致，保留差异并定位条件，不将新协议结果拼成原曲线。随后才报告update1/update5，给逐题翻转和配对差值/不确定性，避免把同题两组当独立抽样。

验收：512唯一覆盖、四模型条件一致、原基线/末尾核验或明确解释未一致处；完整逐题轨迹；以保存点1/5/3156展示结果。只可说“最早在保存点5观察到”等；没有50/100/500等中间adapter时，不插值捏造退化发生在某个具体更新，也不从GRPO100/150检查点冒充OPSD中间点。

### M5：根因报告与下一实验选择（前三条机制有证据后）

产出中文结果报告，三条假设逐条标为：机制事实已核实 / 固定权重实验支持或反驳 / 训练因果贡献未识别。将512实际下降分解为协议失效与合法答案不匹配的配对变化，展示loss改善但回答退化的例子、相反例子以及教师条件依赖。报告所有有限、零、负面结果和skip；不只展示最坏案例。

若固定权重分析足够定位代码错误，先制作独立修复与最小验证，不热改历史运行。若要识别训练因果，提出最小单因素消融：同一source100、同一可比样本/曝光预算、同一优化设置，分别只改reference条件或pair选择/早停。过滤后样本量和更新量变化必须分离。降低学习率、改样本、加奖励、早停不能在同一实验同时改，不能当原算法复现；新增训练参数/预算与原停止授权冲突时，先交付具体实验配置供人工决定。

验收不是“列出三条可能性”，而是每条都有支持证据、反证/控制、效应范围和剩余缺口；固定权重机制证据与训练因果明确分层。完整七基准属于后续效果外推；旧baseline部分38592条不得充数，不自动恢复已停止作业。

## Debug 保存清单和容量原则

- 每题：源row与SHA/offset、五条原轨迹、原pair/选择规则、参考条件、四模型身份、每次实际prompt、生成token IDs、段边界、检索query/top3 IDs/score/content、480token截断、forced信息、context/search预算、真实终止、strict评分及拒绝原因。
- 每token：ID/text/charoffset、policy/information mask、teacher/student/普通base的target probability/logprob、entropy/top32、全词表计算的未裁剪/clip10 KL、clip计数/概率质量、轨迹均值贡献。原响应与新生成posthoc记录分开，统计目标是逐轨迹mean。
- 完整词表：六case、source100/epoch1只是预选完整分布的候选范围，FP32快照总上限24GiB保持；已发生cap_skip，不能说六case×两权重都已完整dump。top32不能还原完整KL。将实际成功保存的少量完整分布用于独立CPU重算，普通case保存全token统计；原首correct/wrong全分布已经独立验收。cap只限制npy，JSON/JSONL与新反事实另查容量和备份余量，不能让debug写盘耗尽中断诊断。
- 原始问答和token正文存节点＋持久本地分析目录；源码/入口/配置/compact统计和hash进入search-e1。原模型权重、数据、环境、凭据不进Git。现有首三例102files/242864102bytes已逐SHA同步本地，完整诊断采用清单分批备份，留出容量余量。

## 推进和汇报方式

同一诊断正常运行时小时巡检保持安静；作业异常、有效新结果、阶段验收或必需人工决定才通知。巡检不只看PID：同时核对case/token实际增加、最后日志时间、direct memory.free、源/模型identity、完整case覆盖。无进展先定位，保留失败，在独立副本恢复未完成工作，不把failure当作停工理由。

每次阶段汇报回答四件事：发现了什么；依据多少题/哪个协议；支持或排除了哪条假设；还有什么未识别、下一步哪项对照会解决。Root负责整合和发布；数学审计、数据/资源审计、逐题独立验收分别交叉检查。不得要求用户反复指定下一项常规工作。

当前里程碑：M0通过；M1固定协议快照64/192，12:43 UTC+8 live确认82/192（三worker27/27/28正常、无exception），实时进度以live receipt为准，不能用82冒充已重算统计；M2既有全量比例通过，距离/区域分析待做；M3/M4设计已经过逐题、数学/因果、资源/数据三路独立审查并修正executor状态继承，全部无剩余must-fix；M5最终归因未完成。GPU4–6须现有worker正常退出并实际释放后才进入下一批，GPU0–3 resident、GPU7 oracle与K4保持。源Git基准152c5ed已远程核验；本计划发布SHA由本次提交后的独立receipt记录，不能先写成已发布。

依据：[native512结果](receipts/opsd-epoch1-native512-result-20261008.json)、[全量配对统计](receipts/opsd-epoch1-full-pair-statistics-20261008.json)、[64题独立逐题快照](receipts/opsd-independent-paired64-analysis-20261008.json)。独立快照包括分母/哈希/配对统计，不含问答或token正文。
