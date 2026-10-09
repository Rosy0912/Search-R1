# 固定 heldout 响应的概率／loss 关系分析 v1

此目录是独立事后诊断入口，复用已发布 `opsd_signal_weight_v1` 的只读源码与原305文件科学runtime。它不修改正跑训练namespace，不调用generate、检索HTTP、optimizer或Adam恢复，不增加采样／训练预算；没有执行新GPU作业。大产物只准放HKUST4 study的真实远端 `analysis/` 子目录。Mac只保留源码和小型摘要。

## 顺序与输入

1. 共同zero正式512评估及其真实parent等待／释放通过后，**立即CPU冻结B bank**；四组任一 `evaluations/outer-15` 或 `outer-25` 已出现则拒绝首次冻结，并明确报告时间线。冻结按原严格final-action scorer重核zero逐题native trace，成功响应为B；包括完整检索观测与policy spans。保存512题索引，原错题保留 `available=false`，不新增参考生成。之后所有组／时点只用该同一bank文件SHA。
2. 等四组各25 outer、100条TRAIN汇总、8个post评估全部完成，原study driver真实退出，原训练／评估后代已释放。独立分析manifest先发布、逐文件remote SHA验收。CPU launcher持study-sequence、原共享GPU lease、study GPU lease、四组online锁以及独立replay锁，再核当前实际UUID／PID／直接memory.free／原准入GPU7 oracle。不复用已经自然退出的训练observer。
3. 各保存点重放其**已保存的自主response**。正式EM仍是commonzero+四组15／25合计4608行；replay不新增EM样本。4个独立固定模型leaf分别 `[slot::4]` 取128题，不改变原8逻辑rank生成结果；这里没有生成或RNG采样，所以不是另一个greedy协议。独立run和实际wait0／完整CPU+GPU后代释放单独留证据。
4. 全量replay结束后CPU summary产出全512与逐数据源EM、覆盖／NA、题级概率关系和同题正确／错误翻转。原最终EM主比较的exact McNemar／Holm及分层配对bootstrap仍使用原 `result_analysis.py`；本模块的探索性关联不新增显著性家族或因果结论。

CPU bank命令（CVD空／原Node env／`-B`；远端路径与SHA由实际文件取得）：

```text
bank.py --request BANK_REQUEST --request-sha256 SHA --output REMOTE_ANALYSIS/fixed-common-zero-bank.json
```

`BANK_REQUEST` JSON schema：

```json
{
  "schema": "opsd-signal-bank-freeze-request-v1",
  "configs": ["四个实际 config 的 {path,sha256} 引用"],
  "common_zero": {"path": "实际 shared common-zero wrapper", "sha256": "实际SHA"}
}
```

`configs` 实际为4个对象，不是示例字符串。shared wrapper必须绑定已接受的child、parent actualwait0和GPU释放；不同组config SHA不同，但source100、runtime、512、共同zero adapter和study必须一致。bank只写新路径，禁止覆盖／动态改选。若全部zero失败，空B bank仍合法，全部512 EM及ordinary entropy继续报告。

GPU replay命令，**只在最终资源门禁后调用**：

```text
E1_RUN_ID=独立分析run CUDA_VISIBLE_DEVICES='' replay.py --plan PLAN --plan-sha256 SHA --output REMOTE_ANALYSIS/独立新attempt
```

`PLAN` JSON schema：

```json
{
  "schema": "opsd-signal-heldout-replay-plan-v1",
  "analysis_run_id": "新的私有run ID",
  "helper_root": "独立发布部署的本helper真实目录",
  "source_manifest": {"path": "独立helper manifest", "sha256": "实际SHA"},
  "publication": {"path": "独立helper publication", "sha256": "实际SHA"},
  "configs": ["四个实际 config 引用"],
  "bank": {"path": "已提前冻结的bank", "sha256": "实际SHA"},
  "four_group_terminal": {"path": "实际study_runner four-group-terminal", "sha256": "实际SHA"},
  "endpoints": ["一个commonzero + 八个15/25实际child-complete引用"]
}
```

publication为 `accepted=true, branch=search-e1, source_commit=40字节, remote_readback_sha=source_commit, source_manifest_sha256=manifest SHA`。所有源文件和Python清单鲜活核验；原training helper的published manifest亦鲜活核验。输出／cache不会落Mac。首次attempt任一worker失败保留已有case/token/log、真实退出及fail证据，不自动重跑或拼接partial；恢复需另行只恢复缺失端点／case的独立审查版本，不能假称当前v1支持resume。日志／门禁失败时不发信号，仍持资源lease等待现有自有children自然退出和完整释放。

CPU summary：

```text
summary.py --request SUMMARY_REQUEST --request-sha256 SHA --output SUMMARY.json
```

`SUMMARY_REQUEST` 绑定bank和九个 `replay_receipts`（每个endpoint的真实complete引用）；只有全4608行、同bank、真实worker退出／释放才能完整报告。可把小型summary留本地，完整response/token/encoding/HTTP引用原件均保留远端。

## 数学／覆盖边界

- 对每个模型自己的实际自主response，使用同一个响应历史 `D_<t>` 重新native response_encoding；普通学生和source100普通前缀为相同普通prompt，特权source100 teacher前缀增加固定B。teacher和普通source100始终禁用LoRA，当前student加载实际postouter14／24的未合并FP32 LoRA，BF16 base／SDPA／head、r16/alpha32/dropout0/scaling2鲜活检查。初始zero adapter同样504张量逐项digest核验。
- head按chunk32流式FP32全词表logsoftmax；g为实际答案payload有效token的 signed `logp_student-logp_teacher` 均值，h为全有效policy token完整词表entropy均值。还记录全词表 `KL(student||teacher)`、原signed词表分量只上clip10后的loss、`KL(student||source100 ordinary)`、实际词logp和clip概率质量。没有混合目标、轨迹权重、梯度或额外正确性奖励。
- **无B**／教师联合context>8192只使教师g/KL/logp NA；ordinary学生context合法时，entropy、学生实际词logp、source100普通drift照算。无最终答案payload只使答案指标NA，全policy教师KL仍可算。ordinary也超8192、空response／空policy mask单独记原因，不截断、不删EM、不掩盖非有限。所有有效policy token保留标量；不保存top32或完整词表数组，不声称可重建任意full distribution。
- offset／policy／区域使用原评分函数；think、search、final-answer、tag、other/boundary区域按每轨迹总有效token分母记贡献，所有区域和重构整轨迹均值。区域贡献是loss／entropy标量份额，**不是参数梯度或性能因果贡献**。普通source drift保留完整响应的同前缀定义，不拿teacher特权prefix替代普通source。
- 关联单位为题，按每个模型／时点及NQ、Hotpot分别报告；没有把token当独立观测。EM全512分母不因B／context／答案覆盖而变；各指标自己的可用题数、NA原因、子集EM同时报告。g/teacherKL的可用题是gold选出的特权参考子集，不能代表全512或作为无答案信息的部署成功概率。
- Pearson／Spearman／原始递增方向EM AUC：指标子集无两类EM、少于2题或feature零方差时全部NA（不填0／.5）；commonzero的成功B子集EM恒1，明确NA。另给探索性线性residual partial Pearson控制log1p有效长度、实际search数、dataset；共线／常量控制明确删除，没有宣称逻辑成功概率、调整后无偏或因果消除。
- source→15／25和15→25同题翻转分层展示各标量差；不同模型的响应文本内生，**不会把不同response的第t词当同一事件**，也不由稀疏zero／15／25推断精确退化更新。此次没有新增pass5，因此不把greedy EM叫pass概率。

## CPU验证

Mac不下载Torch／环境。纯Python fixture实际运行：

```text
python -B -m unittest test_analysis -v
```

另外原Node env **必须**运行Torch CPU数值fixture（CVD空、fresh进程、不初始化CUDA）：

```text
OPSD_ANALYSIS_FROZEN_HELPER_ROOT=实际已发布training helper目录 CUDA_VISIBLE_DEVICES='' 原env/python -B -m unittest test_vocabulary_cpu -v
```

真实检查full-vocab entropy、reverse方向、signed分量clip10、source100普通drift及无B覆盖与原native `vocabulary_metrics` 一致。不能把纯CPUfixtures、源码发布或bank创建说成GPU replay／效果分析完成。

实现时已用Mac现有 `python3` 的既有Torch实际运行上述数值fixture，无需下载。仍须在部署后的原Node环境复核。输入编译采用源字节SHA绑定的隔离loader，拒绝外来cached模块并绕过pyc；不会替换正跑namespace。heldout只有一条greedy响应／题，不能虚构五兄弟group权重；实际权重及loss关系仍来自原已提交TRAIN更新票据，分别报告，不能把不同TRAIN题和heldout题硬连成逐题因果。

Node fixture verification revealed a real input-audit boundary: on the remote ext4 filesystem a quick same-size rewrite may retain coarse mtime/ctime. TRAIN analysis terminal stability now streams the complete input SHA and byte count again as well as checking metadata. A deterministic JSON/JSONL fixture preserves metadata while changing content. This is an independent analysis reader repair; the deployed training and its scientific settings are unchanged. Earlier analysis source and failed CPU test evidence remain preserved.

## 最终机制表：独立CPU提取，保持训练定义

`train_signal_analysis.py` 的原输出schema `opsd-signal-train-analysis-summary-v1` 保持，新增以下表；大规模 `private-question-features.jsonl` 和所有输入仍留HKUST，最终只复制小型聚合报告。

- `selected_D_raw_signal_quantile_exposure`：分别对本variant实际训练selected D更新前forward的raw signed gap／raw all-policy entropy作预先固定分位划分。对可测量的n条题级指标取并列平均秩R，`u=(R−.5)/n`；LOW为u≤.25、HIGH为u≥.75，其余MIDDLE。相同值整块入同一桶，不按qid或随机数拆ties；ties可使桶大于或小于25%，报告实际人数，不强凑四分位。全常量只形成MIDDLE tie，排序NA；这不是题内归一r的分位，也不是给更难题加平均权重。
- 各桶同时报full question population、实际K、可测量人数／NA原因、真实selected pair数、CC/CW、self/nonself、Σw、权重集中ESS、未加权／加权loss及采样s/5。all5wrong／无成功B／selected编码skip不被删出总体覆盖，但没有fake w=1或训练曝光。gap无答案有效payload记NA，entropy仅实际训练forward有数值；不把缺失feature填0。Σw只描述目标标量缩放，不等于梯度质量，优化分母仍实际pair数。
- `neutral_and_self_nonself_exposure`：保留原 `normalization.neutral_reason` 原因counts和以真实K为分母的原因率；另独立列derived多轴状态：zero raw_maxabs、0<raw_maxabs<floor的衰减、`normalization.scores[selected_sibling]` 的NA、实际训练forward所选signal的NA（并单列g/h NA）、少于两可评分项、selected恰在组均值、selected w恰为1／约为1、五w全部恰为1／约为1、非恒定组里的偶然selected w=1。权重评分fallback为NA时，真正训练forward的g/h仍可能有值，不能混为同一覆盖；旧 `selected_signal_NA` 仅为 normalization NA 的兼容别名。approx容差预设1e−12，各轴可能重叠，不能合计当互斥人口类别；同时报实际K／full population／该轴可用数据三个分母。**小正dispersion低于floor是衰减，绝不改名中性或人为归零**；reason=None也不等于所有权重正在变化。self/nonself仅对实际K作未加权人数和Σw暴露，CC/CW分别列明。
- `predeclared_training_periods`：0–14（截至正式15）和15–24（16–25）的上述表分开提取，对齐既定保存区间；记录实际observed outer indices。各汇总人数的单位为question×outer曝光，同qid在不同outer重复出现时分别计数，不能把最多12800次观测说成12800道独立题；统计区间仍按question_id整块重采样。不同区间raw score分布不同，其分位阈值也不同，仅机制描述；不能据此估计未保存的退化发生步或联结不同TRAIN题到heldout题。非恒定组偶然w=1同时给exact与1e−12近似两个轴。
- TRAIN的Pearson／AUC以及Pearson bootstrap先以真实eligible distinct values核零方差；常量0.1等浮点值也明确NA，不靠舍入残差凑0／.5／假区间。label只有一类亦NA。原生产 `result_analysis.py` 不改；最终TRAIN关联以此独立分析表为准，避免其旧通用AUC的常量.5约定混入本计划报告。

heldout `summary.py` 保持原schema，额外给同15／25时点四个G↔H交叉方向共8个 `descriptive_cross_signal_pairs`。每项全512（NQ256／Hotpot256）同题gained／lost／preserved_correct／still_wrong，右减左，**不按g/h/KL可用性筛EM**；exact native evaluation／raw checkpoint／adapter tensor digest随模型及pair绑定。此为已有4608输出的CPU描述提取，没有新显著性检验／新Holm家族／采样。正式终态主比较仍只有G_POS−G_NEG、H_POS−H_NEG两项；原same-signal main统计不被替代。

最终报告最小证据组合：100条完整TRAIN及其原模型SHA＋四组独立末CP数值audit/状态链；原 `opsd-signal-weight-final-effects-v1` 的九端点EM和family2统计；本模块TRAIN summary／heldout response relations及固定B bank；所有完整报告和raw原件远端路径、backup验收、publication文件SHA。原study terminal明确 `analysis_still_required=true`，不能独自作为goal complete。结果以正式25为主；没有E25／多seed／新增pass5，不宣称优于未加权E25、跨seed收益或校准成功概率。
