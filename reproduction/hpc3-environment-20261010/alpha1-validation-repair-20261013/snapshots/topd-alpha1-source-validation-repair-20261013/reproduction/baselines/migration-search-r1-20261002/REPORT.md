# 迁移到 search-r1 基础与 search-rl-algorithm 分支

日期：2026-10-02。仓库：`shaoshitong/Search-R1`。
本地绝对路径：`/Users/shaoshitong/Documents/Search-R1-RL-Baselines`。
目标分支：`search-rl-algorithm`。

## 结果

新分支从用户指定的 `search-r1` 创建，固定基点：
`9ab694499ad3913b67499f7a4ea13da6b544d459`。
该提交是新分支的Git祖先，不是只在文档里改了分支名称。
迁移了旧算法分支从b251基点之后的五个提交，来源截止 `68184965c532fe30fef0d89c3ea19f976ac3bd63`；
没有向 `search-r1`、`search-g1`、旧算法分支或另一个agent的checkout/远端训练写入修改。

实际读取的 `search-g1` 参考版本：`4e7448d66a78996467134441571fda0a6844d180`。
其main_ppo_eco/probe及奖励塑形是独立系统；未把它的probe、grounding bonus或wrong-answer floor带入新分支。
该分支已有独立review指出的边界仍可查原报告；本次查看不等于重新验收Search-G1性能或声称所有配置无缺陷。

## 兼容处理

- cherry-pick冲突仅 `.gitignore` 和 `ray_trainer.py` 的新final-validation方法/算法validation decorator。
  ignore规则取并集，保留search-r1对凭证/资产/缓存的忽略规则；最终验证方法和算法decorator同时保留。
- 保留search-r1新增 `_prepare_final_validation()`、验收步checkpoint保存、检索服务和数据准备/验收更新。
  原基座296个tracked文件逐字保留；八个变化文件为算法接入/共享数值修复、测试venv忽略规则和根README导航。
  没有删除原基座文件。详见 [provenance.json](provenance.json)。
- 十一个算法/数值核心文件与旧已修复提交6818496逐字一致，避免迁移时退回未修复版本。
  adapter/controller/holdout/preflight有下面必要的评测协议兼容改动。
- 新配置对齐新基座 `train_paper_grpo.sh`：周期验证 `validation-64.parquet`、batch64，
  终端通过 `final_val_files=test.parquet` 切到全部七集；验收步5保存、初始验证开启。
- BAPO共同2%/98%划分同时排除周期64题和全量final评测题，两个文件的SHA都进split specification。
  Controller再次检查stage集与完整final集重叠；preflight检查final和stage资产存在。
  六算法和新增GRPO都用共同train划分。不能将旧全训练集GRPO数字当作同划分对照。
- checkpoint算法state记录新的search-r1基点与旧算法来源SHA；Search-R1窗口/postprocess规则按用户决定继续保留。

## 验证

[独立迁移复审](REVIEW.md)覆盖六方法actual terminal控制流、BAPO阶段后loader/reward恢复、
实际RLHFDataset终端65题尾批（64+1）、七来源保留和CoSearch异常时validation flag恢复。
[完整CPU回归输出](cpu-all.txt)：**134 passed、1 skipped**，不重复使用旧分支测试结果。
另通过shell语法检查、Python编译和Git祖先/原文件SHA核对。

旧审计/修复/GPU报告及manifest保持历史基点；它们不是新基座的GPU验收证明。
最新迁移快照与文件SHA见 [manifest.json](manifest.json)。本轮没有远端运行、GPU分配或生产训练。
正式3B/vLLM/offload、真实辅助模型成本/检索吞吐与七集指标仍待验收，精确联合resume仍不支持。
