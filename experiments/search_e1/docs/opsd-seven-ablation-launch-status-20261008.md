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

Old CP374 backup is copying from the preserved canonical checkpoint, with all 18 source members freshly inventoried. It is not yet a complete local backup. The earlier CP89 and full192 transfers failed when SSH was closed by the remote host (actual rc255); originals and partial files remain. Full192 native checksum rsync continues under its existing backup authorization. CP89 has now completed the separately published, exact-prefix-verified recovery described below.

The independent `opsd_backup_resume_v1` helper has now passed eight necessary synthetic CPU fixtures and root code review. It regenerates and hashes the exact failed CP89 USTAR prefix on the node before sending only the suffix; Mac copies into a new private destination and accepts only actual transport rc0 plus strict full archive/member SHA readback. This helper does not modify the failed partial or change training. Its implementation readiness is separate from actual restored-transfer success.

## Actual CPU deployment accepted

Source commit `871dc46f9ae436e5b4115dd33e51715b858ee57d` was pushed to `search-e1`; 51 implementation/manifest/doc files were read back from FETCH_HEAD and matched the local bytes. The seven configs, common cohorts/initial identity and exact frozen305 runtime were deployed to HKUST1 `/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008`, study ID `reverse-seven-a1dacc9d8c77451c8b04140e8e355777`. The actual Node deployment returned0 and the independent CPU readback returned0: all361 packaged files, frozen305/scorer, shared TRAIN/validation cohorts, initial adapter payload and seven distinct variant/run configs matched. This did not import Torch or start CUDA/model work. The first audit wrapper had an unmatched-parenthesis syntax error before its body executed; that failed call is retained, and the corrected wrapper accepted the unchanged published helpers.

The published training helper is `source/training-871dc46/opsd_ablation_v1`; evaluator/driver is `source/evaluation-871dc46/opsd_ablation_eval_v1`. The staged group configs are `group-C0`, `group-C1`, `group-M`, `group-R`, `group-W`, `group-P`, `group-J` under their `receipts/config-candidate.json`. They explicitly remain readyfalse: scoped storage permission, publication admission, resource transition and current retriever admission must be bound before immutable launch configs are finalized. Only the storage choice remains a required human input; the other admissions are executable checks. The existing single storage question remains pending, without a timeout default or repeat question.

CP89 recovery completed at 2026-10-09 00:56:22 CST using the separately published and fresh-verified node helper `.../search-e1-opsd-online-archive100-8gpu-20261008/source/cp89-resume-871dc46/opsd_backup_resume_v1`. Actual suffix SSH and local helper return codes are0. The exact 1,741,258,752-byte old prefix was verified before suffix emission; the original failed backup remains unchanged. The new archive passed strict USTAR validation, all18 member SHA checks against the fresh Node inventory, safe extraction and independent local disk readback: 2,041,167,791 member bytes. Complete receipt SHA is `a7ac51348ad52a4dad8762b08c93192ee5cbfa96a205512135265bcc35b2d3dc`, archive SHA `6f25944be9ce8a6936f422220fe2fe2179812583212ed5cc4c3fbe1416537cb9`. The local destination is `work/search-e1-analysis-output/opsd-ablation-execution-20261008/cp89-resume-871dc46-20261009`. Transport acceptance does not change the stopped training's143 exit or imply a new numeric/effect result. CP374 and full192 continue independently and are still incomplete.

The updated hourly automation follows this seven-group goal, not the superseded200 run, and remains quiet for routine pending/healthy work.

## Fresh resource evidence and remaining input

At 2026-10-09 00:51:05 CST, the single HKUST1 read-only observation returned0. All seven candidate config SHA values and eight GPU UUIDs matched the accepted deployment. The old controller and all396 distinct historical child identities were actually absent. The GPU inventory contained only the approved18004 oracle, PID2239529/UID1005/starttick914664099, using574MiB on GPU7. Direct free memory was24,208MiB on GPU0–6 and23,628MiB on GPU7. `/data` was local `/dev/sdb1` ext4 with1,539,220,144,128 available bytes; `/mnt/shared` did not exist. Original oracle admission/full gate/source and fresh health/process identity matched; no new numerical scan or GPU workload was performed.

Eight same-UID opaque environment reads remain explicitly uncertain; `foreign_compute_none` applies only to the instantaneous GPU inventory. No blanket CPU-process clearance or future launch acceptance is inferred. The exact successful read-only source is archived as `control/opsd_ablation_readonly/resource_snapshot_v1.py`, SHA `88547e96a69e370c8045ee42e78b8dd3445fc0447be0e379dc87fee3daaad62b`. Actual raw/compact results remain private; compact SHA `3e916395b7014b9a4b1e61b16f4c8cd268e8748fd1c379450b6e9a474b663994`.

Seven new scoped training publication receipts were copied to HKUST1 and accepted by the unchanged published CPU admission functions. Fifteen resource-evidence/transition/same-PID oracle technical metadata files were also actually copied and read back with rc0, retaining the snapshot scope and uncertainty. The seven additional `group-*/receipts/config-admissions-prepared-not-ready.json` inputs were then copied and independently read back with rc0: every configured source/publication/resource/oracle reference matched its real file SHA; no group output existed, storage remainednull and readiness remainedfalse. The original candidate configs are preserved. No independent driver, observer, new model or CUDA work has started. Final immutable launch/evaluation configs will be materialized only after the already-pending new-seven-group storage scope is answered, then actual locks/ownership/GPU/free/oracle/observer admission will be refreshed. The resource snapshot cannot substitute for that input or those startup checks.

## Protected HKUST4 archive observation

The unchanged published read-only helper returned0 at 2026-10-09 01:08:06 CST. Archive had completed182 real GRPO updates and entered183 generation; latest step182 reward was0.468. Steps181/182 reported finite pre-clipping gradient norms448.513/605.217, after179/180 values0.670/0.596. These are two new large-norm observations, not recycled step131 evidence. The live run continued and known fatal log patterns were absent. No new native512 result or current weight tensor audit accompanied this observation, so it does not prove a crash, NaN, weight corruption or an EM change.

The latest existing validation/checkpoint remained step150. All14 known UID1001/run/PID/starttick identities, five frozen669-file trees and both fresh observers matched; shared retrieval PID1749337 used718MiB. Own GPU4–7 direct free memory was at least41,317MiB. GPU0–3 and the archive job were not modified. Actual full/compact evidence is private under `archive-readonly-turn2-20261009`; compact SHA `7f297d7524d4138d8a970cdc6bc25a8531835ef2f9f0cb7f874651ee9a29da4b`. Subsequent monitoring must use new actual updates rather than repeating these values as current.

## Latest direct human authorization resolves storage scope

The human replied “是啊，那你启动啊”, then “又不影响的” and “为什么非要等他” directly to the explanation of the new-group local-storage confirmation. This explicitly resolves the new seven-group HKUST node-local plus Mac backup scope. It supersedes the pending-input status above; no further approval is required. Checkpoint/debug transfers continue in the background and are not a training launch dependency.

The immutable authorized packet contains seven exact training configs/storage receipts, eight evaluation templates/storage receipts, nine independent evaluator/driver source-publication bindings and one READY sequential manifest. Scientific source remains871dc46 and all training/evaluation budgets remain unchanged. The independent study driver run is6566ff8f6da34caf8f2484d24f431a64. The packet is CPU metadata, not evidence of GPU initialization or actual updates; real launch/release/source/oracle/memory gates remain enforced by the existing published driver. The Node manifest is `/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/receipts/authorized-launch-20261009/study-manifest.json`, SHA `ac02041a1d648f5aa2ca25abcf00401710a1e280a0a0b2576628c8b48580a3f8`.
