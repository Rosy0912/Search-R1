# Prep3训练／选择验证／固定七基准题目交集

结论：实际train169103、validation512和固定test51713按(data_source, SHA256(问题原文UTF8))两两交集均0。512选择验证题全部不在test文件内，不是其子集。NFKC＋casefold＋空白折叠、保留标点的归一化后，两两交集仍0；作为补充，忽略data_source，仅比较问题文本哈希也均0。因此这次没有发现用512选择题直接计入最终七基准的重复题泄漏。检查不能排除语义改写、答案事实重用或模型预训练污染。

来源根：/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007，实际receipts/preparation-3-config.json；三文件真实路径都指向/data/shaoshitong/search-e1-config-study-8gpu-20261004/data。evidence.json完整保存path/realpath/bytes/rows/SHA，并逐一断言与prep3配置匹配。

| 文件 | 行数 | SHA256 |
|---|---:|---|
| train | 169103 | `a334d768230fd85d3465e251aee906060150628775b78ab7ce15e5e789f7489f` |
| validation | 512 | `75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76` |
| test | 51713 | `f1ade9239143d53636ae66b9c90ac083a195a1bc2492e6a8bd9dd56111cfdcab` |

| 固定基准 | 题数 | 与512选择题原文／归一化交集 |
|---|---:|---:|
| nq | 3610 | 0 / 0 |
| triviaqa | 11313 | 0 / 0 |
| popqa | 14267 | 0 / 0 |
| hotpotqa | 7405 | 0 / 0 |
| 2wikimultihopqa | 12576 | 0 / 0 |
| musique | 2417 | 0 / 0 |
| bamboogle | 125 | 0 / 0 |

train仅NQ78912＋HotpotQA90191；validation各256。三对交集统计同时记录unique question keys及左右匹配行数，全部0，不以synthetic question_id代替内容比较。

## ID与划分证据

实际冻结frozen-prepare.py:24–26用f'{source}:{split}:{i}'重新构造question_id，并对question执行strip。这是本地split内枚举ID，不是原始数据集全局真ID。原内容SHA针对当前实际文件的question字符串，而非未加工上游原始空白。27–33将NQ/Hotpot train按seed42 shuffle后各留256给validation，其余给train；34–36将NQ/TriviaQA/PopQA/Bamboogle test、HotpotQA/2Wiki/MuSiQue dev拼成最终固定test。数据来自固定本地JSON加载（19–23）。

当前IDs三个文件内部各自唯一，全ID两两交集均0，但这只是辅助，不作为无重叠证明；train和validation ID前缀仍是source:train，test则source:test或source:dev。若只取尾部i或丢掉split会出现无意义的数字碰撞，不能算题目泄漏。本审计使用源名＋内容SHA，并额外做源名无关文本检查。

## 归一化边界与内部重复

归一化定义：Unicode NFKC、casefold、把连续Unicode空白折为单空格，保留标点。它捕捉大小写、全半角及空白差异，是规范化后等值检查，不是语义近似搜索或编辑距离检索。

文件内部按源名＋原文SHA统计，train有7条重复题行、validation0、test1204；归一化口径分别8、0、1207。这是同一split内部重复，不是train/val/test交叉泄漏，也没有据此删题或改变固定benchmark分母。ID唯一并不意味着问题文本唯一。

## 结果表达建议

继续按原固定七基准51713题报告每基准与既定总体指标，同时明确“512题用于adapter选择；实际文件按原文与指定归一化检查均与训练/固定test不重叠”。无需因这个concern另排除512：当前匹配题数为0，附加排除选择题报告将与原固定题集完全相同。不要将同条件native512被选中的validation增益当作独立test收益；最终test仍须实测，同模型基线和训练后使用同评测器。若未来另一个数据版本发现selection重叠，应保留完整固定基准主表，另列明确规则排除重叠题的敏感性附表，不能静默改变固定基准。

当前0OPSD更新、没有新增收益结论。仅CPU文件读取；未输出或发布问题/答案全文，未发HTTP/GPU负载，未改数据、source、配置、作业、publisher文件，未commit。CPU脚本通过SSH标准输入执行，远端未写审计脚本；结果仅保存在本地analysis-output。
