# Search-E1 OPSD：十轮消融与固定512题验收

设计日期：2026-10-08，北京时间。状态：**设计与交接文档；本线程不实现、不启动新实验。**

最新人工范围明确为：“你只是设计给出完整的需求文档和prompt，执行我交给别人去做”。因此本线程仅完成需求、公式、矩阵、验收与执行prompt；下文的实验里程碑是交给执行agent的要求，不是本线程继续跑实验的授权。完整需求及独立prompt另见同目录的 `2026-10-08-search-e1-opsd-ablation-requirements.md` 和 `2026-10-08-search-e1-opsd-ablation-executor-prompt.md`。

**方向修订：用户随后明确要求reverse KL。本版本所有新组使用学生→目标的reverse KL。先前已发布的forward-KL设计被本版本取代，不可按旧版本执行。原历史实现确实计算教师→学生forward KL；历史资产保留，不改名、不倒写结果。**

## 1. 最新人工需求与实验目标

本次目标是判断教师条件、混合目标、配对选择、正确性权重、学习率及精度执行链对OPSD早期退化的影响。每个实验完成10个连续在线outer，第10轮后评估同一512题验证集。10 outer不是10次Adam。

最新人工修正优先于最初的前缀设想：**教师和学生均使用同一D的历史，不做B/D前缀替换实验。** B仅作为教师额外的完整成功参考。teacher-forcing continuation在预测D的第t个token时只能取D_{<t}的因果隐藏状态，不能从continuation未来隐藏位置取信息。完整B参考始终保留；允许D=B时，B文本与当前／未来D重合属于明确允许的教师特权信息，不能因此删除或截断B。

人工已明确选择：

- 采用混合分布KL，不移植完整TOPD的dense reward、future mean、group whitening及PPO。
- KL方向为reverse：学生分布在前，教师或冻结混合目标在后。
- 混合概率中alpha=0.1乘教师，0.9乘学生。
- 正确参考B与从五条中均匀随机选出的D配对，允许D=B。
- 使用随机D的非负正确性权重，正确高、错误低但非零。
- 新实验学习率1e-6；另保留原5e-6对照以识别降低学习率的影响。
- 测试FP32基座及FP32 LoRA主参数、FP16主要计算与激活、FP16推理权重。

设计落实选择：学生混合项取outer开始时的冻结普通学生；非负权重采用下文标准化reward的指数映射；设置七组；10 iter按10个在线outer解释。这些选择均在本文件给出精确定义与理由，不能误写成用户已逐字指定其数学形式。只运行组合组较省预算但不能分别归因；完整全因子设计需要更多实验。此处采用七组以分别识别主要因素。

原有clip10、LoRA16/alpha32、global batch32、micro1、gradnorm clip1、无warmup及其他未更改科学设置保留。下文给出了设计采用的具体权重映射和精度例外。指数权重、七组矩阵及outer冻结旧学生是落实需求的设计选择，不能说成用户逐字指定过；执行方如需改变必须先说明影响，不能自行改成带符号advantage、标准答案SFT或完整TOPD。

## 2. 实验矩阵

推荐采用“共同对照＋单因素分支＋组合组”。仅运行组合组无法归因；完整全因子组合会明显扩大预算，本次不采用。

| ID | 变化 | LR | 配对与目标 | 精度 |
|---|---|---|---|---|
| C0 | reverse-KL基础对照 | 5e-6 | 原成功B＋最远错误优先D，reverse KL并保留分量clip10 | 原BF16基座／计算＋FP32 LoRA |
| C1 | 仅降低学习率 | 1e-6 | 与C0相同 | 与C0相同 |
| M | 混合目标 | 1e-6 | 与C1相同配对，0.1教师＋0.9冻结旧学生 | 与C1相同 |
| R | 仅随机配对 | 1e-6 | 成功B＋五条内均匀随机D，不加权 | 与C1相同 |
| W | 随机配对＋正确性权重 | 1e-6 | 与R相同，再乘D的非负组权重 | 与C1相同 |
| P | 仅精度执行链 | 1e-6 | 与C1相同配对和目标 | FP32主参数／FP16主要计算及采样权重 |
| J | 用户要求的组合方案 | 1e-6 | 随机配对＋正确性权重＋混合目标 | 与P相同 |

各组10 outer，共70 outer上限、179200条训练采样轨迹。**C0必须fresh运行reverse KL，不能复用原forward-KL在线前10轮。**

另外允许附加L：原在线forward-KL前10 outer的历史对照。仅在共同源权重、初始LoRA、题目顺序、生成／评分／矿选配置、全部10轮及原始末轮adapter完整核验后，补做同协议512评估；不符合就报告不可用，不自动增加第八个训练实验。L不替代任何新reverse组、不热改原运行。原本地证据显示前10轮末点为Adam99／outer index9，执行方仍须实读原checkpoint，不能以此文档当现时权重验收。

组间对照：C0→C1为LR；C1→M为混合目标；C1→R为随机配对；R→W为正确性加权；C1→P为精度执行链；J评估组合效果，不能单凭J把收益归给某一因素。L若完整可用，L→C0比较forward／reverse方向，明确新旧运行及验证证据边界。

## 3. 共同起点、数据与预算

- 所有组来自HKUST Archive GRPO100同一原始检查点的12文件身份，不从GRPO最新轮次或退化OPSD模型开始。不恢复GRPO Adam。
- 使用同一已核验零效果初始LoRA模板：504个FP32 A/B张量，B全0，逐张量身份相同；不以“同seed”替代实际初始化一致性。各新组fresh Adam／RNG，L身份核对时须与该模板一致。
- 固定TRAIN169103题，按原seed42顺序选前10批，每批512题，各组同outer的题ID及顺序相同；验证集不进入采样／配对。
- 每组每outer使用本组最新student重新生成每题5条，共2560完整轨迹。固定worker／题目／sibling种子协议。不同权重的采样内容可以不同，不能强行复用其他组后续轨迹冒充online。
- 保持temperature1、top_p1、context8192、每轮生成500token、5turn、最多4search、top3、obs480和同一获准HKUST检索索引／FP32服务身份。P/J只改变批准的精度路径。
- 每outer仍一题最多一个pair；五条全错无成功B，跳过且完整记录。共同上下文／空mask规则不变，实际K独立记录。
- 每outer对可用pairs做一遍优化；Adam次数为ceil(K/32)，尾批按实际人数。不得为了配齐batch补造轨迹或丢尾批。每组10轮最多5120pair曝光，不将25600生成量当成训练曝光。
- 不增加GRPO、warmup、熵奖励、答案SFT或其他保护项；这些不属于本次已指定变量。

## 4. 同一D历史上的教师与混合目标

令q_t为冻结source100在“问题＋完整成功B＋D_{<t}”下的全词表分布；p_old,t为本outer采样时固定学生在“问题＋D_{<t}”下的分布；p_theta,t为当前可更新学生同普通输入的分布。

```text
m_t(v) = stopgrad[0.1*q_t(v) + 0.9*p_old,t(v)]
```

必须混概率，不混logits。q和p_old均冻结、无梯度；p_old在本outer所有Adam更新期间保持原始采样时adapter，不随当前forward刷新。下一outer才更新p_old。教师权重始终source100；B仍从该轮成功学生轨迹选出，因此冻结教师权重不等于每轮q数值不变。

p_old具体表示合同为：绑定outer起始完整checkpoint的旧adapter，在该组训练精度合同下、普通提示＋同一D历史的未合并replay分布；冻结adapter身份、dtype、tokenizer／mask及目标版本随配置和恢复状态保存。它不冒充vLLM merged模型采样时的精确processed概率，也不宣称两种前向逐位一致。由于本实验采用分布KL而非PPO，不使用这个replay分布伪造采样行为概率比。

沿用原clip10的上界形式，但词表分量改为学生概率加权的reverse方向。混合组实际逐token目标为：

```text
raw_reverse_KL_t = sum_v p_theta,t(v) * [log(p_theta,t(v)) - log(m_t(v))]
loss_t           = sum_v min(p_theta,t(v) * [log(p_theta,t(v)) - log(m_t(v))], 10)
```

非混合新组将m换为q。学生的p及logp都保留梯度，只冻结q／m／p_old。不能detach学生概率当固定权重，不能沿用 `q*(logq-logp)`，不能只交换日志名字。记录raw reverse KL与裁剪后目标；clip10触发后不能把目标称为严格普通reverse KL。负词表分量保持，不另加floor0、绝对值或总KL上限。稳定log-space计算混合概率。每条轨迹内按有效policy token平均，再按实际global pair数平均。

对于固定target的未裁剪reverse KL，若z为学生logits，梯度为 `dL/dz_v = p_v * (log(p_v/target_v) - raw_KL)`，不等于forward的 `p_v-target_v`。需要CPU有限差分／autograd fixture同时检查loss数值和梯度。冻结target且p=target时梯度为0；任意非相等分布的非零方向须正确。旧forward设计中“动态混入当前学生仅缩放0.1梯度”的推论**不适用于reverse KL，已撤回**；仍固定outer旧学生，避免目标随更新移动。混合初始状态p=p_old时raw reverse KL应不超过 `-log(0.9)`；这是初始同history的检查，不保证后续更新仍满足该界。

本次计算的是本outer采样／筛选后固定D历史上的逐token全词表reverse KL，优化时不对已生成的离散history反传。它是在线刷新历史的局部分布蒸馏，不等同于完整序列级 `KL(P_student(y)||P_teacher(y))` 的策略梯度估计；不能冒称已实现TOPD论文的序列目标／全部定理保证。

这是**TOPD启发的混合目标OPSD**。所指TOPD实际代码采用混合分布的sampled-token log-ratio奖励，再做future mean、组内白化和PPO。本次人工选择KL，故不宣称复现完整TOPD，也不引入其32B教师或任何其他集群数据／权重。

## 5. 随机配对与非负正确性权重

reference选择仍为：正确轨迹中搜索最少，再按token_count（缺失才用字符长度）最短；确定性first-tie规则保持。D从全部五个sibling均匀随机，包含B；用独立的seed42／outer／question ID哈希流和确定性拒绝采样确定sibling索引，避免直接mod5偏差。哈希算法、编码、版本及拒绝次数明确保存，使R、W、J的选择规则一致且不依赖rank或训练RNG调用顺序。

D=B时依然使用有参考教师对普通学生的分布损失，不认为self-pair自动等于零。

本设计把“GRPO式归一化、正确高、错误低且非零”具体定义为：

```text
r_j in {0,1}, j=0..4
mu = mean(r_0..r_4)
sigma = sample_std(r_0..r_4, correction=1)
A_j = (r_j-mu)/(sigma+1e-6), when sigma>0; otherwise A_j=0
w_j = 5 * exp(A_j) / sum_k exp(A_k)   # temperature固定1
weighted_pair_loss_i = w_selected_D * mean_valid_token(loss_i)
global_loss = sum_i weighted_pair_loss_i / actual_global_pair_count
```

使用稳定softmax等价实现。全部权重严格为正，五条的平均权重为1；全对组权重自然为1，继续蒸馏；全错组虽权重为1，仍因无B而不配对。不得把全对A=0误写成全对不训练，不除以advantage和，也不在minibatch重新归一化权重。

| 五条中正确数 | 正确D权重（约） | 错误D权重（约） |
|---|---:|---:|
| 1 | 3.503 | 0.374 |
| 2 | 2.013 | 0.324 |
| 3 | 1.505 | 0.242 |
| 4 | 1.217 | 0.130 |
| 5 | 1.000 | 无 |

这是GRPO启发的非负指数权重，不是原带符号GRPO目标。负advantage乘KL会推动学生远离教师，违反本次明确选择。它也不是正确答案硬标签SFT或EM策略梯度，不宣称自动“保护答题能力”。

记录五条原reward、mu/sigma/A/w、随机种子及索引、D是否等于B、D原reward、权重曝光、批次权重和及actual人数。R和W用相同随机规则；各组轨迹因模型不同可以不同，不能声称完整内容配对完全相同。

## 6. FP32主参数／FP16执行合同

- 直接从原FP32 source100加载，禁止将BF16 merged升精度当成原FP32源。
- 基座冻结且真实存储FP32，LoRA主参数、梯度及Adam moments FP32；teacher同一FP32基座关闭adapter并冻结。
- 主要matmul、attention及主激活使用FP16；norm、softmax、log-softmax、KL／混合概率和必要稳定累加允许FP32。记录实际embedding／residual／QKV／MLP／head／checkpoint重算dtype；不得只开autocast便声称所有activation均为FP16。
- teacher与student使用同一规定计算合同；不暗中给teacher改为完整FP32前向，使“目标分布变化”成为额外未标记变量。
- 同步GradScaler：八rank scale一致；scaled gradients SUM，unscale后finite检查、clip norm1，再实际Adam。overflow导致跳过时不计真实更新、不推进已提交数据cursor；保存scaler及完整RNG，可在同配置恢复。
- 前向FP16溢出不由GradScaler解决。失败保留case与dtype／数值证据；可以改变物理内存实现或使用已获准空闲48G卡，但不能暗改科学batch／长度／clip或回退BF16后冒充P。
- 采样／FP16评估模型从本组准确base表示＋FP32 LoRA在CPU合并，最后一次转换FP16；核权重finite、绑权及原源SHA。记录base究竟是原FP32值还是原BF16表示转FP32，禁止混淆两者。
- 24G卡只按实际最长轨迹及GPU7检索共享峰值验收。必须记录nvidia-smi实际memory.free、峰值、GPU UUID及进程身份，不以八卡总显存证明每rank能放下。

## 7. 固定512评估与归因

主要比较沿用同一原生512验证协议：NQ256＋Hotpot256，普通提示、greedy、同HF／SDPA／BF16 base＋FP32 unmerged LoRA、原generation_config和repetition_penalty1.05、相同预算／检索／严格评分，各logical rank64题。在所有组训练后使用共同原BF16 base表示加载对应adapter，隔离所学adapter的效果。P/J的此结果不冒充FP16部署效果。

此外，按第5项需求必须评估FP16推理链：同一512、同一HF greedy生成配置，分别评估FP16零更新source100、P末轮及J末轮；保存实际FP16模型身份。原BF16零更新baseline与FP16零更新baseline分别报告。不能拿FP16训练后分数直接减历史BF16的239/512归因为训练收益。

预算：七组各512主要评估＋共同BF16初始512＋FP16初始/P/J各512，共5632验证预测；若L历史对照完整可用，再增加512，总上限6144。L只做已存在历史模型的补评，不新增forward训练。旧七基准partial不纳入，不自动重启旧七基准评估。

各组保留：source100及0/1/5真实更新、每outer末、最终10outer的原始adapter和完整恢复状态；不通过best选择回退后冒充训练模型。每轮训练reward是该轮更新前新TRAIN轨迹的严格EM均值，不是独立heldout曲线。

最终报告至少包含：

- 10轮逐轮reward／raw与postclip loss／K／CC-CW／全错／实际Adam／权重曝光。
- 512逐题严格EM、NQ／Hotpot、合法但错答案／最终行动协议无效、搜索次数和零搜索。
- 共同baseline及各单因素分支按题paired gained/lost，分层及按题bootstrap区间；不把同题五sibling当五个独立样本。
- 普通策略漂移、teacher／mixed target分歧、clip触发数及命中集合上的student质量、teacher质量、mixed质量，三者分别命名；标量loss贡献不当成梯度因果。
- 各组不同K／Adam／token／加权曝光，不将同10outer当成完全相同优化预算；P包含训练及采样精度执行链变化。
- 支持、削弱、未知分别陈述；不要求某组必须有提升才能验收，但数值／运行验收不能替代效果结果。

## 8. Debug、资源、发布与恢复

每组独立目录、run、不可变配置／源码清单、锁、日志、receipt；继承既有轨迹／HTTP原始记录和全部有效token训练前向日志。保留原生generated IDs／执行历史及policy来源、B/D完整文本和身份、teacher/q_mix/p_old/p_theta目标／top32／熵、clip前后loss、mask／offset、区域贡献及skip。top32不能重建完整KL，少量完整FP32词表dump沿明确24GiB上限，cap_skip记录，不宣称所有词表已dump。

全量概率、原题／轨迹和权重仅私有存储，不能入Git；源码、启动／恢复／评分脚本和必要配置必须持久本地、提交、推送指定远程并逐文件SHA读回。

仅HKUST资源；TOPD只阅读本地机制代码，不转移其其他集群实验资产。启动前核SSH/node/UID/PID/starttick/GPU UUID、资源锁、磁盘、检索服务和输出保存授权。原K1在线200轮与K4 archive均不因本设计自动停止／替换，也不热改；无空卡则完成CPU实现／审查并排队。不得停止他人或通知七算法线程。

另为实际teacher推理debug，每组每outer在有B的pair固定分层选最多8题，teacher＋B自主greedy生成，隔离RNG、不进入训练，完整trace/HTTP/IDs保留；无B不补造参考。最多560诊断轨迹单独计数，具体合同见完整需求§14。

输出默认遵守同集群共享存储规则；已有“HKUST本地保存”例外明确绑定原在线run。新组启动前核该例外是否覆盖本goal；如果范围未确认且无已验证共享挂载，保留待输入，不拿超时当批准。所有原数据与唯一资产保留，Mac持久同步、节点释放日期与清单SHA核验独立记录。

运行故障在授权内保留失败原日志／源码／checkpoint后恢复同一已提交cursor、Adam／scaler、Python／NumPy／Torch CPU-CUDA RNG、outer／题目／pair顺序及hash；不得重采样旧prefix、重置Adam或从零重跑并称续训。有限但没有改变参数的调用不能算真实学习验收。前1/5真实更新在同一10轮作业内审计后直接继续，不做短测试后从0重跑。

## 9. 执行里程碑与完成条件

1. **设计审核**：本文件、权重映射、混合old-policy语义与评估精度合同明确；独立数学／实现审查完成。
2. **CPU实现与发布**：独立副本实现M/R/W/P/J及fixtures，目标方向／detach／mask／尾批／八rank缩放／group退化／FP16scaler与恢复检查通过；持久副本、Git推送及远程SHA通过。
3. **资源准入**：获准空闲HKUST卡、输出持久保存条件和检索身份实查通过后启动，不占用现有作业。
4. **真实10轮**：每组全部25600采样轨迹、10outer提交和实际更新／finite changed参数／恢复状态／显存验收；错误或空pair轮如实记录。
5. **512与分析**：全部规定模型的完整唯一512覆盖／逐题评分／按题差值、不同精度初始baseline和最终结果产出。
6. **交付**：中文消融报告、配置／源码与远程SHA、实际节点／run、所有checkpoint／日志／debug／评估绝对路径和持久备份证明。

**只有上述必需实验与评估真正完成，才可把goal标记complete。** 后台启动、初始化、五更新门禁、一次巡检或loss下降都不能代替完整验收。正常进展继续推进，故障必须恢复或保留明确输入依赖，不反复只报告失败后停工。

## 10. 依据

- 最新人工：本线程五项消融要求，以及“不是，都用D_t”“混合分布KL吧”“用随机D的非负权重：正确高、错误低但非零”。
- 随后人工：本线程只交付需求文档和prompt；明确纠正KL方向为reverse。
- [TOP-D论文§2.1式(2)](https://arxiv.org/html/2607.04751v1#S2.SS1)：该文OPD定义为学生→教师reverse KL。新设计方向修正依据该式；本实验局部full-vocabulary目标与论文序列策略梯度不同，不能直接继承其结论。
- 原E1矿选与prompt：`experiments/search_e1/control/opsd_debug_frozen_runtime_20261008/runtime/search_e1/core.py`。
- 当前在线目标／八rank更新：`experiments/search_e1/control/opsd_online/objective.py`、`trainer.py`，原配置文件SHA `d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17`。
- TOPD本地机制：所指线程 `01a0fcbe-15f8-7600-ae84-9487d924560a` 的 `work/topd/search_r1/baselines/advantages.py:64`，alpha乘teacher，原机制为dense reward及PPO，本设计不称完整TOPD。
- 原FP32来源／LoRA／显存估算只读核验：`work/search-e1-analysis-output/grpo-outer-comparison-20261008/fp32-master-fp16-compute-case5-readonly-review-20261008.json`。

## 11. 可直接执行的goal文本

交给执行agent的目标：持续完成本文件定义的Search-E1 OPSD七组十轮消融。以用户最新共同D历史、学生→目标reverse KL、混合概率alpha0.1、正确参考与五条内均匀随机D（含自身）、D非负正确性权重、LR1e-6、FP32主参数与FP16执行／推理为准。先完成独立CPU实现及远程源码核验，再在获准HKUST空闲资源和明确持久保存授权下执行。所有新组从同一source100及初始零效果LoRA开始，fresh Adam，固定同10个TRAIN512题cohort，每轮各组自身最新student生成5轨迹；reverse C0不得复用旧forward模型。按本文件精确公式和矩阵运行，不引入B前缀、完整TOPD/PPO、带符号KL权重或额外GRPO。持续保留实际轨迹／检索／token loss／target／mask／权重／dtype／scaler／显存／全部恢复状态；故障保留并在授权内恢复同一cursor，五步门禁后直接续至10outer。完成所有规定固定512共同评估及FP16部署评估、按题收益／退化分析、配置／代码／checkpoint／报告绝对路径和持久备份证明后才验收执行goal。不得自动停止既有K1/K4或他人任务，不跨集群、不通知七线程、不热改正跑、不把不同批次reward当固定验证或把初始化当完成。当前设计线程仅交付需求与prompt，不负责启动该执行goal。
