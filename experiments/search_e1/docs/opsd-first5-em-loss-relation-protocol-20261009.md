# C0/C1/M前五轮：EM与成功概率和loss/概率的关系

2026-10-09最新人工指令：核心是最终EM或pass概率与loss、概率指标的关系程度，要求完成细致分析，明确不使用brainstorming。此指令取代上一方案中的确认等待和“先以区域占比为中心”的顺序；不改变现有训练/评测队列，不新增Adam。

## 核心结果变量

1. **选中学生轨迹D的最终strict EM**：完整自主生成结束后，冻结严格scorer给出0/1；不是teacher-forcing的答案概率，也不是termination=answer。
2. **同题五次采样的经验成功比例**：`s/5`，其中s为五条strict正确数。这是temperature1协议下的有限样本pass@1估计，单题只有五条，不冒称精确真实成功概率。
3. **五次中至少一次成功**：`1[s>0]`，称observed pass@5。所有被矿选进入训练的题都有成功B，因此该标签在有loss的样本中恒为1，不能对这个常量做相关/AUC，也不能给全五次失败题编造loss。
4. **实际固定512 EM**：零更新和C0/C1/M十轮末普通greedy、相同协议的已接受逐题结果。三模型总体数仅三个点，不能据此给“模型loss与最终EM相关”的可靠统计结论；TRAIN与validation问题分离，不做错误逐题join。第五轮512未产生时不借第十轮结果填补。

## 两条分析线

**主线：题/轨迹层面实测关系。** 读取三组outer0..4全部4529个实际pair、940295个有效token及全cohort7680个group-question记录。用D的strict EM与s/5作为结果，量化teacherKL、当前目标raw/post loss、outer-oldKL、实际D答案token的student logprob、student-teacher logprob gap、student entropy及区域对应指标的关系。答案概率主量为逐token平均logprob/几何平均token概率；整段概率积另列长度，绝不当整题正确概率。

**补充线：固定heldout结果。** 复用已有同条件512逐题gained/lost、最终行动合法性、search/termination；若它未保存logprob，不虚构置信度或逐题loss。仅分析现有确实同题且同协议的输出；不将三个组的平均loss散点拟合成因果结论。

## 关联强度的具体度量

- 六个预设主特征：teacher raw KL、实际目标post loss、outer-old raw KL、答案student平均logprob、答案student减teacher的logprob gap、student平均entropy。报告逐组与五轮合并的均值差、点二列/Pearson、Spearman、raw方向AUROC及题级bootstrap区间。
- 将特征按题目分位分箱，展示每箱D最终EM与平均`s/5`，报告人数、缺失与长度。分箱是描述，不称已校准未来正确概率。
- 分区域think/search query/final answer/tag，问“哪个区域的指标对EM/成功比例最有关联”，而不只问该区域占多少loss。区域贡献、区域内部均值分开；区域缺席贡献为0，内部token均值为NA。
- 采用question-block五折OOF，四个逐级模型分别为：①group、outer、data source、学生长度/search次数、轮内更新进度；②加student普通分布指标；③再加成功参考的长度/search元数据；④再加特权teacher分布指标。另对think/search/final分别加入区域分布指标。train fold内标准化、缺失处理和固定L2正则，报告OOF AUROC/Brier/logloss及增量。相同四模型另对`s/5`做fractional回归并报OOF RMSE，禁止用CC/CW或五次成功数作为EM模型控制变量，因为它们会泄露结果。
- pooled指标同时给逐group/outer和分层结果，排查Simpson现象。OOF按共同question_id分组，三组同题必须同fold；bootstrap保持同题全部组/行。区间是固定这次运行/这套题的题目不确定性，不是多次训练seed的方差。

## 必须实测的选择限制

原矿选失败优先D、全对fallback，预期`D正确 <=> s=5`；要在实际数据核验，而不能先当成立。若成立，“D EM判别”实际上区分全五次成功与仍有失败的题，只代表该矿选分布，不代表所有自主轨迹。全五条失败题无成功B，没有teacher loss；报告其比例和特征缺失，不能删去后将结果外推整个TRAIN分布。

同一outer各组题目相同；第一outer已有科学内容一致的327个B/D及同训练顺序。后续最新student产生不同轨迹与入选样本，因此必须控制题目/长度/搜索/数据源和展示资格交集。精确共同历史子集与不同历史的在线描述分开。

## 时间与信息可用性

loss/logprob来自已记录的实际训练前向，D则由outer起始采样模型生成；该pair训练前current可能已经经历本outer前面的更新。即使OOF分类很强，仍是**完整轨迹的事后诊断**，不是生成前预测。

teacher看过成功参考，teacherKL和teacher给D答案的概率可直接包含正确答案信息。将student-only与加入teacher的模型分开，避免把特权信息的判别力当学生自身置信度能力。full reverse KL是分布差异，不是输出正确概率；低KL可能只是共同确信一个错误。

数据已保存的概率是固定D历史上的原始模型分布，不是vLLM/HF生成processor处理后的行为概率。没有逐token参数梯度，也没有同题更新后再次采样，故不能由关联宣称某区域loss造成下一次性能变化。

## 实施与产物

独立`opsd_first5_em_loss_analysis_v1`只读CPU extractor及stats；从显式闭合manifest/committed token refs读取，核token SHA、offset/mask、轨迹和实际尾批目标重构。所有新代码先发布并远程SHA核验，再部署/执行；原生产源码不改。执行nice/单线程，与GPU队列无依赖。

私有产物包含pair features、完整cohort outcomes、每项输入SHA、读数/统计执行rc和匿名问题索引。公开只保留聚合关联表、OOF增量、分箱图、每项结论的实际样本与边界、方法源码，不提交题/答案/原trace/token/权重。最终报告明确写出哪个指标相关最强、控制因素后还剩多少、不同组是否一致，以及哪些问题现有数据无法回答。
