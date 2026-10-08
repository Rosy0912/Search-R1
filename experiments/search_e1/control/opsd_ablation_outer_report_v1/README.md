# 七组逐外轮已提交事实汇总

这是新的只读 CPU 后处理入口，不替代已有512题 `summary.compare`。只使用标准库，不导入 Torch、实验运行模块或旧 pyc，不执行 SSH/HTTP，不取得训练锁，不创建文件、不发信号。Root 发布并核验源码后才实际使用；本地 fixtures 不表示真实70轮或效果验收。

```sh
python3 -B collector.py --goal /EXACT/goal.json --goal-sha256 EXACT_GOAL_SHA > /PRIVATE/new-facts.json
```

本研究沿用原 goal SHA `c65bea46dbb4ef6451ed56b2c4e26adafde5fd247e6b67fb5bea593b8689c94f`，其中精确绑定七个原 config/run/output_root。工具完整核验这七份配置的声明 SHA；读取各固定 `outers/outer-00000` 至 `outer-00009/training/outer-complete.json`，不存在的轮次标为 `NOT_COMPLETE`。存在而身份、SHA、计数有冲突的文件拒绝汇总，不能用 NA 掩盖冲突。

已完成轮次沿原 completion 的精确 checkpoint 引用读取完整 manifest，再沿其 training_input 读取 outer-manifest、encoding、order。核对 reverse/variant/precision/mixture合同、P/K/skipped、原 encoded 行哈希、order置换、最终cursor、更新计数和累计曝光。**仅**读取该最终 CP 的 `outer_update_receipts`，不扫描 attempt、失败 token 文件或 retry capsules。同 path/SHA 引用去重并报告重复数量；同一个 run/update 被两轮计入则拒绝。更新序列、实际1–32条轨迹、八rank `[rank::8]` 分配、有效token及累计曝光均须匹配。相邻已完成轮次对账；K0可合法报告0更新及loss NA。

真实 producer 字段依据：sampling stats来自已绑定的原 outer-manifest，其中 `diagnostics.strict_reward` 是完整2560条轨迹的原生严格评分计数；P、CC/CW、all-wrong/all-correct/self-pair及eligible权重和是miner原记录。当前 sampler缺少这些可选字段时显示 null/NA，不重读原QA并替代评分。完成记录含K/context/mask skip和有效token。每个被CP引用的 update记录含八rank trajectory metrics、raw/post/weighted_post反向loss、teacher/outer-old/mixture drift、clip计数、gradnorm、scaler和该记录的retries。工具不读 Tensor、不重算logits、mask或梯度，不宣称再次数值验收。

每外轮汇总对实际K条轨迹等权平均；weighted_post仍除K，不能除权重和。每更新保留实际尾批分母，提供raw K曝光及实际权重和曝光。缺少任一轨迹的某scalar时整项为NA并报告字段覆盖数。clip只汇总记录提供的vocabulary count；概率mass/fraction没有receipt字段，保持NA。`recorded_retries`只代表committed receipt本次声明，不等于全部历史溢出总数。loss/drift数值不是梯度贡献或因果证据。

输出为事实JSON，包括70个槽的coverage、outer/update表、每个实际读取文件的fresh SHA/bytes/stat、逐项来源和中文报告口径。SHA绑定的是所读记录；未读取的8rank Tensor、token正文、权重和raw轨迹不声称fresh核验。每轮TRAIN是新512题，TRAIN reward变化不能写成固定heldout曲线。即使 `all70_complete` 为true，也仅说明70个完成记录齐全；不证明controller实际rc0、全进程释放或512题评测效果。中文根因结论由人工结合其他证据撰写。

原生 strict_reward 是 `dict(Counter(...))`；全0/全1可只有一个key。该明确Counter的缺类别计数为0，但diagnostics/reward整个字段缺失仍是NA，不能将未知评分补零。

仅三项必要CPU fixtures：真实producer嵌套shape/七组部分coverage和SHA拒绝；committed引用去重、失败文件不计曝光及尾批1的分母/rank分配；K0/缺字段NA/非有限与token计数拒绝。未重复科学forward或广泛tests。
