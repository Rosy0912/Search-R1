# 独立执行前代码审查（增量，尚未通过）

日期：2026-10-11。审查者：数学方案独立子代理。范围：`work/tree-speed-repair-20261008/branpo-source` 本地静态 diff；未执行远端命令、未启动训练。本文最终接受前必须绑定完成版本源码/runtime commit、remote SHA与校验时间。

**当前状态：STATIC_SOURCE_RUNTIME_REVIEW_ACCEPTED。源码与 runtime 静态逻辑已审完，无未修复的启动阻断问题；本审查以正文与文件hash绑定，最终发布commit、remote SHA和runtime文件一致性通过单独发布receipt完成绑定。没有训练结果 PASS 声明。**

## 已核对的增量事实

- `SearchRollout.complete` 在每次真实模型 `advance` 之前深拷贝状态。因此 `snapshots[index][-1]` 是末次模型动作之前的状态，包括此前真实工具观察；不是截断完成答案文本。
- `State.branch` 深拷贝全部状态，只更新 trajectory ID；新增路径以 `state_signature` 核对 prompt/rolling/response/mask/steps/anchors/records/raw_ledger/observation/source/tree/node/turns/budgets。原 compose 再核对共享 sample ID 下 token/logp 相等，behavior μ 绑定路径保持。
- `ray_trainer.fit` 已在进入生成前按 n_agent=2 interleave 重复训练 batch；新奖励 callback 用 state.source 从这个重复后的 qa_batch 选择 metadata。targets 被追加至用于评分的 compose 副本，不会放进模型 rolling prompt。
- 新 `_tail_branches` 从两个原叶增加[2,1] continuation，生成后每个 child 均 append，无 reward discard，无 hidden extra候选。根0仅第二槽可退一状态，根1无额外槽。
- 已看到 `tree_grpo` 原优势实现与 `objectives.py` PPO/leaf policy_weights 没有变更；variance 在 controller.compute_advantages 的全局 batch 路径执行，之后才 actor DP shard。

## 已反馈待完成核对

1. 初版诊断先将 reward.double 后重新计算 Z，标作 actual_adv 时不同于原 float32 优势。增量修正已核对：以原dtype的 estimator_score 算Z，相加后double做moments；raw outcome与空loss行处理后的estimator_score分开。
2. 初版 all_correct/all_wrong 以 reward==1/0计数。增量修正已核对：改名 exact_reward_one/exact_reward_zero，避免连续F1/格式负奖错误解释。
3. `test_tree_branpo_tail_cpu.py` 与 CPU_RECEIPT.md 已静态核对：实施者报告18项相关CPU测试通过两次；测试内容覆盖末状态继承（含raw ledger/logp）、callback source→reward_model身份映射、配额不丢弃、纯intra统计与行重排列不变性。独立审查没有重复运行整套测试。历史raw-emission测试有已在基线重现的旧fixture缺参数问题，未为迁就fixture降低生产检查。
4. 新 config 继承原 fresh200；最终启动runtime `work/tree-branpo-runtime-20261011` 已静态核对：supervisor 使用 `ppo_tree_branpo_fresh200`，严格锁定3/2叶、512题、三次新增continuation、4turn和长度上限；fresh actor/ref指向官方版本，禁止恢复变量/继承checkpoint，启动证据核对driver与8个worker Adam/scheduler/sampler初态。行为μ/native-token协议与既有源码hook保持，包内源码与外部helper均hash锁定。output/checkpoint/cache均落远端任务目录（短/tmp别名目标亦是远端runtime）。一次提交intent使用新BranPO命名、排他lock、SUBMITTING→SUBMITTED/UNKNOWN，UNKNOWN不重试，不取消外部作业。人类授权step5继续在同一实例中进行，无训练已验收虚假声明。最终源码/runtime统一commit/push身份及work→repo字节一致性待绑定。

## 已核验源码身份

首个实现commit：`2df5558b04afd302a08b5d0cd14e1c8c3311eb8a`，branch `codex/tree-branpo-tail-20261011`；本地 `origin/codex/tree-branpo-tail-20261011` 与HEAD相同，隔离checkout无未提交变更。最终runtime追加commit完成后需通过独立发布receipt核验其SHA、实际remote SHA、本审查文件hash和runtime文件hash；不能把首commit当成runtime最新版已推送。避免将包含本审查的commit SHA写回本审查造成自引用循环。

## 最终 runtime 修复复核

- 实施者发现 `build_package_draft.read_regular_bound` 使用 `re.fullmatch` 而未import re，真实 guard入口会NameError；独立审查已核对修复为 `import os,stat,hashlib,re`。本地只以隔离production_contract.require stub加载真实helper，并用临时小文件运行实际read_regular_bound，导入、regex与SHA读取均通过。修复文件SHA256：`b07c9cab8c500dcc202a00eda3af724ae14fb27bad1e5a7fa4c9abf0dc27815a`。未运行GPU/远端，不重复18项测试。
- builder已禁止output为repo本身或repo内部，避免归档后输出反过来污染来源；`tar.extractall`移除了本地Python3.9不支持的filter参数，保留逐member拒绝symlink/hardlink、绝对路径及`..`检查。仅归档committed Git代码。
- 本地work runtime与隔离repo/reproduction runtime逐文件比较：27个非pycache文件完全一致、0差异。最终第二commit/push需包含这些字节与本审查；独立release receipt再绑定实际发布身份。
- 启动owner末次扫描又修复两处遗留run identity阻断：`production_contract.submitted_intent` 与 `validation_cost_hooks_instruct300.install` 已核对使用 `tree-branpo-fresh200-JOB`，与supervisor/oncejournal/slurm_identity/owned_lease/guard路径一致。禁止恢复检查现覆盖 `TREE_CURRENT_CHAIN_RESUME_`；experiment_name统一为 `tree-branpo-officialfresh200`（源码yaml、manifest、supervisor三处一致）。已独立再次扫描runtime全部Python/JSON，未残留tree-current200或tree-fresh200旧run身份。
- 新增 `make_release.py` 只生成小型发布JSON，不网络/dispatch；核包内全部文件hash、manifest/build receipt、remote_source_sha等于包commit与审查接受标记，输出review哈希、remote SHA、验证时间、authority/dispatcher/onceintent绑定。actual_remote_source_sha仍须由root实测输入，不能由字符串相等检查冒充远端观测。最终work/repo再次逐字比较为28文件0差异；采用单独receipt无自引用commit循环。

数学接受边界见同目录 `MATH_REVIEW.md`：保留原 Tree estimator；不能声称完整 BranPO复现、保证五叶混奖或无偏PG；prefix不得叠加额外loss，不可先均值优势再调用一次PPO冒充原目标。
