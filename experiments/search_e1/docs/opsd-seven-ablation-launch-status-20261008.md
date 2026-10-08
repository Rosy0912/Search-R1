# 七组10轮消融的执行状态

最新人工goal创建于2026-10-08 23:18:19 CST：停止当时的八卡在线OPSD200轮作业，按独立执行文档启动C0/C1/M/R/W/P/J，每组10个在线outer后立即做完整512题推理。此要求取代旧200轮及其后续评测安排。10outer不是10次Adam。

旧HKUST1 run `1d1d67b60f0e46a2afc0536a36ab69b1` 的controller720753已按精确UID1005/starttick924526765范围停止。实际终态是授权signal退出143，不能写成训练成功rc0。保全38个完整outer和374次Adam的原生全状态检查点；outer38的采样部分另存，没有计为第39个完整outer。所有原source/runtime/log/checkpoint仍保留。HKUST4 archive、18004检索、CP89与full192本地传输以及他人作业均未停止。

停止证据：

- Node目录：`/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/preserved/user-stop-for-seven-ablations-20261008-2347`。目录名是标签，真实时间以terminal内容为准。
- terminal SHA：`8fd4eb76e9fbbd1bddf69278fccd512a2dc38415322683c90d40bfbdbf37eba9`。
- 原生CP374 manifest：`/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008/runs/checkpoints/update-00000374-outer-00037/manifest.json`，SHA `14278230c81774f271679a5345828213b2d57929887341ffcca9e41e2af28b40`。这是全文件SHA保全，尚不冒称374的独立Adam逐张量实读。
- 停止工具source commit：`0eef4b93afe40b6523ad58527735cce525eb8506`。原science env不支持所需pidfd接口的首次无signal失败已保存；同一冻结工具通过独立system Python完成停止，训练环境没有改动。

共同准备已固定：source100原12文件身份、实际504个FP32初始LoRA张量和252个zero B；共同adapter tensor digest为`836590b798434f21160418c6471cbce3bdfac5bf4a9947c1e3e7723d057baf2f`。同一seed42 TRAIN顺序取10×512题，5120题唯一且验证ID/精确问题文本交叉均为0。七组不会继承旧Adam。

新实现将旧forward方向明确替换为reverse学生→固定目标；student概率系数与logprob均保留梯度。M/J固定本outer的旧student概率参与混合；R/W/J用无mod5偏差的确定性随机D；W/J在分量clip和轨迹平均后施加严格正正确性权重。P/J真实FP32主参数与FP16计算链、同步scaler及独立overflow/RNG恢复均属于新实现，实际GPU精度/长上下文仍待实测。

新物理检查点先核八rank实际Adam张量/groups完整摘要相同，再共用一份完整optimizer-shared.pt；八份独立rank文件保留各自RNG/cursor/scaler和明确引用。恢复重新实读共享完整Adam后走原状态检查。科学更新与逻辑完整状态不变，不删除旧或新唯一资产。此改动减少磁盘重复，避免按八份Adam累积超过节点容量。

执行依赖仍须真实区分：新七组的HKUST本地盘例外已一次询问，未获新scope答复前只推进CPU实现、审查、发布和备份。旧run的例外不能视为新goal批准。新GPU尚未开始；配置、CPUfixtures、Git发布、空闲GPU都不等于已启动或完成消融。

每组10outer的原始末adapter将立即独立native512；共同BF16 zero和FP16 zero/P/J另评，共5632正式预测。教师自主诊断与训练原Dhistory上的q分开。最终报告比较各组真实K/Adam/曝光预算、逐题严格EM与gain/lost/协议/搜索，并明确支持、削弱和未知的归因边界。

## Sequential evaluation entry prepared on 2026-10-09

The independent `opsd_ablation_eval_v1` entry is now implemented. It waits for each actual training child rc0, ten complete outer chains, the raw final checkpoint and full owned GPU release, then evaluates that group immediately. Seven main BF16 raw10 endpoints, common BF16/FP16 zeros and P/J FP16 deployments total 5632 official predictions. Transparent hooks and zero-effect checks are extra controls, not part of that denominator. The driver independently starts a source-bound CPU memory observer before each group; only observer bootstrap can lack a live memory sample, while every actual GPU phase requires one. Exact CPU-role identity separates the study driver/independent observer from phase descendants without relaxing foreign or opaque-process rejection.

The final local candidates passed 52 necessary training CPU fixtures and 25 evaluator CPU fixtures, without skips. These establish CPU contracts only; actual eight-rank updates, FP16 memory/scaler behavior and complete native512 outputs remain unmeasured. No new GPU job has started. The executor explicitly requires the HKUST node-local exception to cover this new seven-group goal; the single pending input question has not received a reply. Preparation proceeds while that input remains pending.

Old CP374 backup is now copying from the preserved canonical checkpoint, with all 18 source members freshly inventoried. It is not yet a complete local backup. The earlier CP89 and full192 transfers failed when SSH was closed by the remote host (actual rc255); originals and partial files remain. Full192 native checksum rsync has resumed under its existing backup authorization. A separate prefix-verified recovery helper is being prepared for CP89; no completed-transfer claim is made.

The independent `opsd_backup_resume_v1` helper has now passed eight necessary synthetic CPU fixtures and root code review. It regenerates and hashes the exact failed CP89 USTAR prefix on the node before sending only the suffix; Mac copies into a new private destination and accepts only actual transport rc0 plus strict full archive/member SHA readback. This helper does not modify the failed partial or change training. Its implementation readiness is separate from actual restored-transfer success.
