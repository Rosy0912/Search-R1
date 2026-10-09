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
