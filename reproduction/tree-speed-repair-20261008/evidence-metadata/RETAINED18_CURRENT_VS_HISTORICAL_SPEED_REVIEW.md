# 当前与历史真实步骤耗时核对

**以前确实没有现在这么慢。** 当前3335第5步扣开发集测试和HF保存是 **47.29分钟**；同模型/同数据/同micro16的Instruct3154早期真实步骤约 **4分钟**，Recovery3234在healthy110附近普通步约 **7.7分钟**。同一fresh200父链已约 **46–48分钟**，这次迁入retained18并非独自引入整个差距。

仅后台读取已有本地证据与冻结源码；未SSH、测试、模型调用、修改代码、停止或重启，当前3335继续第6步。下表只用真实完成outer的logger计时，普通步扣testing/stage_testing/save；fresh父链的stage_testing含滚动状态保存。

| 作业/真实完成步 | 完整步分钟 | 扣测试/保存分钟 | 预算与证据差异 |
|---|---:|---:|---|
| 3335：5 | 65.12 | 47.29 | 512×5叶；mini128/micro16；20 Adam/rank；新增raw/UTF8/μ与优化前后检查 |
| 3328：1/2 | 46.46/48.75 | 45.85/48.12 | 同fresh200预算；1的LR0；真实8rank完成/rolling保存 |
| 3331：3/4 | 47.53/48.33 | 46.94/47.76 | 同fresh200预算；真实父链恢复后3–4完成 |
| 2971：1/2 | 4.72/4.36 | 4.72/4.36 | 512×6；mini256/micro64；12 Adam；旧Base/数据/奖励/warm142 |
| 3154：1/2 | 4.15/3.95 | 4.15/3.95 | 同Instruct/common0b/128/16/warm57；512×6；24 Adam，无raw/μ |
| 3103：1/2/3 | 4.83/4.40/4.10 | 4.83/4.40/4.10 | 512×6；256/64；12 Adam；旧Base与数值诊断，无raw/μ |
| 3234：108/109/110 | 7.69/7.68/17.87 | 7.69/7.68/7.72 | 同Instruct/common0b/micro16；640×6/mini384；10 Adam；trained parent100恢复 |
| 2879：1/2 | 14.74/15.07 | 14.74/14.73 | 512×6/mini256/micro8；12 Adam；旧CPU E5+跨集群A800索引，仅历史 |

Instruct3154 **不是512×5**：冻结resolved和实际rollout均为512×6=3072叶，24优化/rank。它与当前共享Instruct模型、common0b数据、mini128、micro16和warm57，仍因叶数和检查层级不同而不能作严格公平倍数。2971/3103差异更大。

healthy110校准3268：**新完成outer=0、3B Adam=0、scheduler未前进**。1179.98秒是整作业墙钟，不是训练一步。上表采用其真实parent3234的108/109/110；110的1072.25秒扣537.99秒测试和70.85秒保存后为463.41秒。

旧14.9分钟可直接重建：2879两步884.147/904.496秒，平均894.321秒=14.91分钟；第二步含20.425秒保存。本地已有请求日志去掉1条启动probe后，平均CPU E5墙钟412.73秒、旧跨集群索引117.08秒、生成其余180.35秒、actor145.35秒、步内其余38.81秒。两个“其余”来自减法。旧跨集群记录仅解释历史速度，不能用于当前执行或公平效果对照。

| 当前outer5实测阶段 | 秒 |
|---|---:|
| 生成/Tree处理 | 1407.246 |
| 参考模型 | 10.620 |
| 奖励与优势 | 224.979 |
| actor更新含检查 | 742.348 |
| 原有dev512测试含滚动边界保存 | 965.835 |
| HF保存 | 104.174 |

完整步3907.156秒；扣测试/HF保存2837.148秒，另有451.954秒在训练分项之外尚未归因。不能把它全部称为某个guard。

**可证新增开销：** 原始token前缀路径已有wall **621.835秒（10.36分钟）**、CPU623.995秒，占gen44.19%；这是gen内子区间，不能再相加。μ monitor已有656次native forward/rank，host调用区间69.588秒；16次initial repeat包含在656内，不是656+16，也不是全部CUDA/collective/安全检查成本。

**真实CPU位置：** 第6步三次非暂停MainThread栈均在`_encoder_binding:276`→当前tokenizer身份→验证上下文→`sampled_action`，30.21秒窗口耗30.31秒CPU，约满单核并持有GIL。冻结源码显示这里读取完整原生encoder状态`backend.to_str()`。三次实际栈能确认当前热点位置，不能给出整步占比或把全部前缀耗时都归给这一函数。

**micro并不足以解释：** 全局micro16经8rank归一为每rank micro2；当前160训练micro/rank，3154同micro16有192，actor却仅86–89秒，当前742.35秒。源码确有多层raw/UTF8/context/μ绑定重建、优化前后监测、postclip范数和每次写入后的全本地参数/Adam finite扫描，但各层独立秒数尚无证据。update_actor742秒、adv225秒和未分解452秒不能静态猜成某个单一根因。

原始日志出处：

- [RETAINED18_FORMAL3335_OUTER5_SCORE_AND_OUTER6_START.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-fresh200/RETAINED18_FORMAL3335_OUTER5_SCORE_AND_OUTER6_START.json)
- [FOUR_OUTER_OBSERVED_COSTS_CURRENT_CHAIN.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-fresh200/FOUR_OUTER_OBSERVED_COSTS_CURRENT_CHAIN.json)
- [stage.log](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-instruct300-3154-review/actual3154-files/runs/tree-instruct300-3154/stage.log)
- [stage.log](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-instruct-300/failure3234/stage.log)
- [shared-formal-2971-two-prefix.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/evidence/shared-formal-2971-two-prefix.json)
- [stage.log](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-fixed-five-review/actual3103-files/runs/tree-fixed-five-3103/stage.log)
- [tree-two-2879-completed-summary.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/evidence/tree-two-2879-completed-summary.json)
- [RETAINED18_FORMAL3335_OUTER6_NONBLOCKING_CPU_STACK_DIAG.json](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-fresh200/RETAINED18_FORMAL3335_OUTER6_NONBLOCKING_CPU_STACK_DIAG.json)

每组样本、配置、actor完成证据范围、unknown、源码检查层级和文件SHA见 [JSON报告](/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-fresh200/RETAINED18_CURRENT_VS_HISTORICAL_SPEED_REVIEW.json)。旧评分和历史速度不替代当前200/51713效果验收。
