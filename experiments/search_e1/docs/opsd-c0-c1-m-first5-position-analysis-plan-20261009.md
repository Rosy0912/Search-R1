# C0、C1、M 前五轮：位置分歧与性能关联分析计划

> 时间顺序更新：2026-10-09用户明确要求“核心是最终EM（或者pass概率）和这些loss以及概率的关系程度……完成细致的分析，不要使用brainstorming”。当前按[EM/loss关系执行协议](opsd-first5-em-loss-relation-protocol-20261009.md)直接推进，原确认等待已被取代；本文保留为区域映射/对账设计历史，不作为暂停分析或现有队列的依据。

2026-10-09。人工新增目标：分析三组前五个在线 outer 的不同，以及 think、search、答案等位置与性能变化的关系。本文是分析设计及已核对的起始事实；**区域全量结果尚未产出，不把计划写成完成报告。** 当前训练、十轮后原生512评测及其余消融队列继续；本分析不成为 GPU 启动依赖。

## 1. 要回答的问题及证据等级

1. C0→C1：学习率从5e-6降为1e-6后，哪些区域的 teacher 分歧、裁剪、outer 内偏移及行为差异改变？
2. C1→M：相同学习率下，用0.1 teacher＋0.9本outer冻结旧student的概率混合目标，是否主要减弱某些区域的更新信号？是否保留搜索行动、也保留了错误行为？
3. 哪些位置的异常与合法但答错、最终行动格式失效、搜索次数/查询内容变化相伴？是否能用它们解释已保存评测上的 gained/lost？

明确分三种结论：

- **直接事实**：实际训练前向的全词表标量、mask、分母、日志与检查点身份可核验。
- **关联证据**：位置分歧与同题行为/最终EM变化共同出现；可能受题目、历史、矿选与长度影响。
- **因果证据**：保持其他条件不变的区域监督干预或独立训练重复。当前日志不含逐token参数梯度，不能以loss占比替代梯度或因果；本文不自动启动新的训练消融。

优先方案是全量CPU读原始记录，再关联已有固定评测。仅看top32/少数高KL样例会有抽样偏差；立即重训区域屏蔽组会增加因素和预算，均不作为第一步。

## 2. 冻结比较边界

| 项目 | C0 | C1 | M |
|---|---|---|---|
| 学习率 |5e-6|1e-6|1e-6|
| 目标 |teacher reverse KL|teacher reverse KL|到0.1q＋0.9p_old的reverse KL|
| 其他条件 |共同source100、BF16、相同原矿选、clip10、轨迹均值、global32；保持原科学配置|同左|同左；p_old每outer冻结|
| 本次窗口 |outer索引0..4，五个在线外轮|同左|同左|

C0→C1与C1→M分别是预设单因素组间比较；C0→M同时改变学习率和目标，不能当作单因素。

共同cohort按原配置与逐题身份核验，512 TRAIN题/outer、每题5条自主生成。问题ID相同不意味着五条轨迹、成功参考B、入选学生D、student历史或token序列相同。不得将不同组第t个token硬配对。第一outer的轨迹/配对/编码是否完全相同，要核原SHA后才能称为共同响应；同reward、K或token总数不能代替该证明。

训练样本与固定validation问题禁止连接为同一实例。训练题上的区域指标不能直接回归到另一道验证题的EM，不能把训练样本重复视为独立验证样本。

## 3. 已完成的首层核对

输入为已发布的`../reports/seven-group-recorded25-outers-progress-20261009/seven-group-70-slots.csv`（24175字节，SHA256 `b4bc7e85fbd50e515a0e6195a2f4be45b9a1d5d18bf3815dc929f1d8e7da574f`），只取三组`COMPLETE_FACTS`且outer0..4的15行。下面KL/loss是**每轨迹先平均，再按实际K汇总的轨迹均值**，不是将所有token混池后的均值。五轮实际采样各12800条。

| 前五轮 | C0 | C1 | M |
|---|---:|---:|---:|
| 可用pair曝光K |1512|1514|1503|
| 实际Adam |50|51|50|
| 有效policy token |305446|318739|316110|
| CC / CW配对 |839 / 673|833 / 681|824 / 679|
| 全五条失败、没有B的题 |1048|1046|1057|
| student→teacher原始KL |0.390349720761|0.411244373916|0.405164738074|
| 实际优化的裁剪后目标loss |0.380959310110|0.401658880401|0.006661280478|
| student→本outer旧student原始KL |0.001167289843|0.000773877215|0.000765205401|
| 被上裁剪的词表分量总数 |578|648|0|
| 更新前grad norm范围 |0.564180..1.608271|0.592456..2.645512|0.038295..0.153485|

共4529条训练pair、940295个有效token、151次真实Adam。三组无原生committed overflow/retry；这只描述这些票据，不冒充每个Adam tensor或全历史异常均重新实读。

当前可支持的判断：

- M优化的混合目标明显接近本outer旧student；小loss有目标几何原因。其teacher KL并没有同量级缩小，不能称为“已贴近teacher”或“效果更好”。梯度norm较小也不等于Adam实际参数更新同比例较小，应另看实际adapter变化。
- C0的outer-old KL均值较大，提示更强的outer内分布变化，**不是从source100到第五轮的累计漂移**。小KL需要核初始/同权重数值底噪，不能直接把约0.00077都视为学习造成。
- 各outer题目和组内生成历史变化，不能把C0的teacher KL从第一轮到第五轮下降写成同一固定响应的学习曲线。
- 578/648是词表分量计数，不是命中token数、概率质量或异常率；M的0只针对实际混合目标触发集合。

混合目标的结构界：令`a=0.1q+0.9o`，`KL(p||a) <= KL(p||o)-log(0.9)`，因为逐词概率`a>=0.9o`；当`p=o`时上界约0.10536。另有凸性界`KL(p||a)<=0.1KL(p||q)+0.9KL(p||o)`。**实际优化的是概率混合后的KL，不是两个KL的加权和**，这些上界不证明EM改善。

已实际只读核验第一outer：三组原始rollouts完全同SHA（`87547406a3ded2f99ea9405a9f557dcc1e435ab0db240968c771a42de16c5135`）；327条pair的question/reference/student/reward/spans/weight等科学字段规范化后同SHA（`09641e02612518c4e7e9631dd746e050b4d3a6e0690b5ce419fa77c22943327a`），矿选元数据唯一差异为variant；原pair文件原始SHA仍不同。327项pair_indices列表亦完全同序。由此第一outer可做真正共同D/reference比较，但各pair的pre-update权重随实际优化路径分叉，不能宣称这些forward发生在相同模型状态。

既有固定512：source100 BF16为239/512；C0十轮raw为227/512；C1十轮raw为236/512。**它们不是第五轮评测。** C1相对C0净多9题，已有按题bootstrap区间跨0；不能认定学习率是退化根因或C1显著提升。M第五轮和三组第五轮固定512目前不作为已得结果。

## 4. A：输入身份及一次全量对账（首先执行）

**输入链**：各组冻结config → 闭合outer票据 → 采样/矿选/编码文件 → committed update票据 → all-rank pair的token_log及其SHA → 原B/D文本、policy spans、HTTP/生成原始IDs。检查点用显式outer4末路径与manifest；不从mutable latest猜第五轮。

1. 只纳入outer0..4已提交成功的更新，按run/outer/update/rank/pair/encoded SHA去重；保留失败重试，但不计入接受样本。记录每组1512/1514/1503分母，并核50/51/50更新及尾批真实人数。
2. 使用token header里的完整ID、offset、mask，与原D和编码证明对齐。仅有效policy token有逐token数值行；补齐非policy区的身份/位置表，但不为其编造loss。检索/自写information及非authored文本是否被mask以冻结规则为准。
3. `token_text`是单ID decode，不能强制等同于原文本offset子串；区域映射使用原student文本和实际offset。零宽/跨边界、不明或不合规offset独立统计，不擅自修mask。
4. 原始文件不改、不解包覆盖；本地已验收压缩包可流式读成员。CPU小内存逐轨迹聚合，避免再保存约94万行大副本。HKUST节点只读、Mac产出独立目录；备份与训练无依赖。
5. 核finite、raw/post方向、target kind、概率质量范围、clip计数、observed logprob/entropy、模型与更新前adapter/outer-old身份。逐轨迹重建post/count与原trajectory loss；逐真实batch重建sum(w_i L_i)/actual人数及尾批，核与native票据一致。容差继承原reader数值合同并在产物中写明，不能为通过任意放大。
6. 对第一outer另外核问题/五siblings/B-D/pair/order/ID/mask的SHA，找出真正可跨组对齐的共同历史子集。后续outer按共同题ID比较覆盖、成功数、矿选与行为，loss仅在历史确实相同时作token配对。

**验收**：三组前五轮全部有效token对账至940295、4529 pair；所有缺失/skip/不确定/重复明确列出。失败保留具体文件与位置，不把失败样本静默剔除后宣称全量完成。旧192题的forward/posthoc region程序schema与本次reverse实际训练不同，不能直接套用或改写旧结果。

**产物**：私有输入清单、SHA、reader身份、覆盖/分母/对账报告；公开仅compact计数。训练原题、完整轨迹、token、权重均不Git。

## 5. B：位置地图、真实损失贡献与高分歧点

主分类互斥且穷尽：`think内容`、`search query内容`、`最终answer内容`、`结构/行动tag`、`other`、`跨区域boundary`。最后answer是否可被strict scorer接受另设标记；标签内字面答案不等于EM。结构tag再分think开闭、search开闭、answer开闭；每个区域保留token数和无该区域的轨迹数。

另设互不替代的上下文轴：turn编号、执行第几次search、第一search前/观察后下一policy段、最后行动段、有效policy位置分位数。检索本身无policy loss，但其内容会影响后续分布；只分析“观察后”不能声称检索内容本身有梯度或造成变化。

每组×outer×区域输出：

- raw teacher KL、outer-old KL、实际目标raw/post；M的teacher、old、mixed必须三列独立。
- raw减post的裁剪削减量、命中token比例/词表计数/学生与teacher及混合目标被裁剪概率质量；M质量沿**混合目标的触发集合**，不误写teacher KL被裁剪。
- observed D-token的student/teacher/old/target logprob和entropy；它们描述已选D-token，不把teacher的高概率当该行动正确。
- token均值/中位数/90与99分位、区域token占比、每轨迹归一化贡献、真实batch归一化贡献；top1%位置集中度和raw-post差异。
- 保留正/负post贡献、signed总量与绝对贡献。裁剪loss可能有负值，signed区域占比可能负或大于100%，不可强行变成正百分比饼图。

主统计采用等轨迹的区域贡献：`C_i,r=sum_{t属于r} loss_i,t / M_i`，区域求和必须还原整轨迹loss。另报区域内部token均值：`sum loss / 区域token数`；前者看优化信号分配，后者看该类位置平均分歧，不能混用。batch贡献额外除实际batch人数；跨batch总和/更新平均与pair平均分别标明，不能被K加权掩盖尾批。

分层预注册：NQ/Hotpot、CC/CW、参考/学生search数、D长度档、是否已有非法标签、同组eligible/三组共同eligible/专有eligible。全cohort都报告资格和行为，不能只分析三组共同eligible后忽略矿选差异。其他数据源单独列，不假设所有TRAIN都是NQ/Hotpot。

top32是最早选定有效位置的有限覆盖，不能用于总体分布估计或全KL重建。完整词表快照仅预选outer0少数位置；可验证已保存位置的公式，不对未保存的词表作插值或声称答案/search区域都有full dump。

**验收**：所有区域贡献和分母还原A的原生数值。真实轨迹没有某区域时，该区域对整轨迹的贡献为`sum(empty)/M=0`，必须保留在等轨迹平均；区域内部token均值因分母0标NA。文件/字段缺失则是未验收/NA，不能补0，也不能当作合法区域absence。同题比较按question聚类；不将五siblings/token当独立样本。

**产物**：区域×轮次矩阵、数值分布、裁剪质量、累计pair/token/Adam曝光四种横轴、按题bootstrap区间、区域heatmap。静态科学图使用标准绘图工具，原始值和样本数随图保存。

## 6. C：跨组比较与行为证据

两条分析线保持分开：

**在线事实线**：真实产生的三组轨迹各自包含其优化与采样变化。先比共同cohort的五siblings strict成功数、是否全错/全对、eligibility、B/D、长度和search行动；再按CC/CW、长度、turn、数据源分层展示区域分歧。可做描述性标准化，但不把事后矿选交集或回归系数当随机化因果结果。

**共同历史线**：仅使用已核SHA相同的原D与teacher参考、mask/offset/prefix；不同组当前权重的实际记录若来自不同更新点，还必须标真实Adam和pre-update identity。共同outer0采样相同不保证训练到该pair时参数相同。无共同历史的区域均值差异不能单独解释为参数变化。

优先检验：

1. C0与C1：old偏移的差异集中在think还是行动/答案？是否随着实际adapter变化、grad norm clip触发、尾批共同出现？从现有CP提取的参数变化另列，grad norm不是参数更新幅度；不额外读取全base或重复8rank验收。
2. C1与M：混合目标缓和了哪些高teacher KL位置？同时检查teacher KL是否仍高、old与current是否接近，以及改变/保留的search、答案行为。M小混合loss不作效果排名。
3. CC与CW：已正确D是否承担多数区域贡献？合法答案区是否信号少，而错误查询或格式位置存在高分歧？仅描述“优化了什么”，不自动断言“没有优化答案导致下降”。
4. 高KL/高削减/高clipped student mass样例与按source、CC/CW、长度、turn匹配的低分歧对照。先冻结hash抽取规则，人工核原HTTP、B/D、实际D prefix及发生位置；不只展示最显眼失败。自主teacher生成每轮最多8例，selection非总体随机，单独报告覆盖，不用40例估512 EM。

统计单位为问题；主比较只有C0→C1、C1→M。bootstrap按outer/cohort和问题分层复用同题抽样，保留同题全部sibling/pair；10000次、seed42。唯一预注册主指标为**等轨迹的区域outer-old原始KL贡献**，三类主区域为think内容、search query内容、最终answer内容：五轮汇总的两对比×三主区域共6项，若报告检验p值则按此family作Holm校正。这描述轮内普通策略分歧，不是累计漂移。teacher/mixed/post/clip/entropy、逐outer、tag子类及其他探索全部标描述性，不能把它们追加到未校正的六项family或从多张图挑显著项。原512已有区间是未经多重比较校正的单比较区间，不改称同时置信区间。只有单次训练seed，区间衡量该题集的抽样不确定性，**不衡量独立训练run的方差**。

## 7. D：与固定评测连接，分清第五轮与第十轮

先复用已有accepted十轮raw512/source100的逐题记录，M十轮自然完成后自动纳入原队列结果，不打断其他组。分NQ/Hotpot、gained/lost/preserved/still-wrong，失误轴为最终行动invalid或valid但gold不匹配；search/termination另轴。按照评测自身文本/真实HTTP定位行动变化，不把teacher-forcing的训练概率当自主generate processed score。

目前三组第五轮末有显式受保护检查点候选C0 Adam50、C1 Adam51、M Adam50，实际路径/hash需A确认。**现有11模型正式5632槽均是十轮端点；不能用十轮512填第五轮曲线。**

第五轮性能是关键缺口。要回答“前五轮哪组更好”，需要独立新增三权重×512=1536预测，复用共同BF16零更新512及原普通提示HFSDPA/FP32 adapter/greedy/完整generation_config、repetition1.05、同18004检索、8192/500/5turn4search/top3/480。不新增训练、不重生成TRAIN。作为追加固定评测配置单独记录，不能混入原5632完成数；在原GPU队列空闲/完成后独立run，若用户选择优先重排需明确范围，当前不抢占正在运行作业。

先做CPU位置分析与已有十轮证据；第五轮新增GPU对照和任何共同前缀重算均不作为CPU分析或原训练的前置门禁。没有新增评测时，最终报告明确“第五轮EM未知”，提供十轮关联而不推断精确退化发生轮次。

训练/验证题已隔离，因此loss与验证翻转只能在组/阶段的假设层面连接，不能从某训练题的位置值证明某验证题损失原因。必要的第二阶段共同前缀固定权重实验应另建新分析run、事先固定问题/上下文/预算，不恢复旧M3/M4、也不做新Adam；作为待选择增强证据，不伪装成已有训练日志。

## 8. 交付与验收顺序

| 阶段 | 具体交付 | 完成标准 | 依赖 |
|---|---|---|---|
| A 身份/覆盖 | 输入清单与15轮对账 | 4529pairs/940295tokens/151Adam、mask/尾批、缺失全部明确 | 原已闭合数据 |
| B 位置统计 | 3组×5轮×区域的KL/loss/clip/entropy/贡献矩阵与图 | 区域和还原native轨迹及batch，分母和NA清楚 | A |
| C 行为关联 | 全cohort矿选/搜索/答案变化，分层/同历史结果，高分歧与对照轨迹索引 | 同题/同history边界与选择偏差明确 | A、B |
| D 效果 | 现有10轮paired512；独立5轮512列为追加对照 | 严格逐题完整512、真实权重/协议/EM及CI，5/10不混用 | 已有评测；新增GPU另排 |
| E 中文结论 | 每个假设的支持、反证、未知和最有价值下一对照 | 每条发现有数字、样本/协议、来源SHA和反证边界；不把loss当梯度/因果 | A–D可用部分 |

优先级：A→B→C并行利用现有评测→E第一版；M自然完成的10轮512补D。第五轮追加评测随后补齐位置关联与前五轮性能，避免“只巡检、等全部七组才开始分析”。正常既有训练/备份无需新增反复检查，CPU helper只做必要schema/数学/边界校验。

任何新reader/region adapter都在独立目录，先提交推送并核远程SHA再读节点数据，不热改冻源。原source为871dc46、dtype恢复538ac99、eval统计72e1e12，不能拿分析文档commit冒充运行源码。原题/完整trace/token不入Git；compact统计、方法、source和匿名例证索引可发布。所有新分析输出唯一目录，保存失败与执行rc，源数据仍保留。

## 9. 本文状态

- 已实际阅读实验矩阵、objective/trainer、旧region helper及现有15行原生事实，完成第3节聚合。
- A/B/C的新全量区域读取、第五轮512、区域因果干预均**未执行**。
- 该计划不授权改变学习率、KL方向、clip10、矿选、混合alpha、训练预算或停止当前队列。
- 新区域reader需要把reverse实际训练schema映射到本方案；不得沿用旧forward字段标签或把posthoc probe宣称为历史训练forward。
