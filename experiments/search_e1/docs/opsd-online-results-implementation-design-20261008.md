# 在线 OPSD 独立结果入口实现设计

这份 CPU 只读设计审查继续已发布的完整200轮与效果计划，没有启动任何评估。终态捕获入口已经独立发布部署；后续推理入口采用新的 `opsd_online_results_eval_v1` 目录/配置/run，不改实际正跑源码或终态捕获清单。配置先允许最终checkpoint为未绑定/HOLD，只有真实200轮终态后才填入outer199末raw adapter。

| 文件 | 工作与必要边界 |
| --- | --- |
| prepare.py | 固定source100、CP89、最终outer199末raw checkpoint；完整12源文件/适配器/runtime/data/scorer/18004/源码和发布SHA。不能从mutable latest或最后采样cache猜最终模型。 |
| completion.py | 对账outer0–199连续唯一、512TRAIN×5样本及最新采样模型链、原矿选P/K、Σceil(K/32)、ΣK及最终八rank状态。使用已捕获真实终态rc0/release。历史已回收的cache只核留存凭据，不声称重新读取旧cache大权重；历史失败attempt或缺失waitrc另列，不由最终wrapper0抹除。 |
| gates.py | 核原四共享锁加独立results锁，八UUID/directfree/无foreigncompute；GPU7仅允许精确18004 oracle。完整ownership扫描，读取不确定则HOLD，不能抢占当前训练或别人作业。 |
| controller.py | 同一新run顺序执行三模型阶段；记录每阶段八worker的真实rc/UID/tick/argv、GPU释放、配置与证据。失败保留分片，不将重跑冒充续算。 |
| worker.py | 八physical worker各对应一个logical rank，每模型按[rank::8]处理64题。直接普通提示调用冻结SearchAgent.run(question,temperature=0)；BF16base/FP32LoRA，不merge、不训练、不恢复Adam。 |
| hooks.py | 保留原调用参数、原返回对象、HTTP与异常，记录真实generated IDs/prompt/检索结果/processor。各模型各rank首题做同RNG plain/hooked控制，额外控制不进入512分母。 |
| summary.py | 同题三模型各hits/512、NQ/Hotpot分集、paired新增/丢失/保持/仍错；strict finalinvalid与valid-gold-mismatch互斥，搜索与termination独立列。 |

已发布旧 `opsd_native512_eval_v2` 仅作代码参考：可复用透明hooks、严格最后authored span计分、cohort/rowSHA、RNG无效应控制、实际FP32 adapter检查与完整ownership算法。必须替换旧四adapter清单、三物理卡映射、M1/debug release、18005、旧双端点239/203硬门禁及GPU7完全空要求。不调用、不部署、不恢复旧M4入口。

预选三权重形成1536个主slot；每模型512、NQ/Hotpot各256、每logicalrank seed42。独立CP0零B控制与hook控制另列，不计主分母。source100须核252个LoRA B全零并做实际零效果控制；最终outer199的采样cache属于outer198末模型，不能用于最终效果。

所有模型使用原HF SDPA/greedy/generation_config及实际repetition_penalty1.05/EOS/StopTags，8192context、每turn500token、5turn/4search/top3/obs480、同一当前18004。source100的新239复现只作条件核查；不复制旧239/203双门禁，也不修改条件凑旧结果。输出保留原始逐题轨迹及来源；没有这项真实评测前不报告新native512 EM。

CPU fixtures覆盖缺final/缺outer/K0与tail/ΣK和Adam错误、采样绑定错位；COMPLETE却无rc/旧pane或PID复用/失败child/opaque与孤儿；八rank64题重复遗漏/错误权重或dtype；hook返回对象/异常/RNG保真与额外控制分母；未闭合think和information答案的评分边界；oracle例外、foreigncompute/锁争用与partial目录禁止复用。fixtures不等于实际GPU验收，所有入口先发布并核远程字节SHA，之后才在授权资源空闲时运行。
