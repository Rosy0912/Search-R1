# 六种 Search-R1 算法独立审查汇总

审查日期：2026-10-02。固定实现：`7b93fbd01742715dc14290b0a6e2d3b42e6c19f7`，
基座：`b251ba6b503217a59cd3395c1326d7b66b602989`。
仓库：`/Users/shaoshitong/Documents/Search-R1-RL-Baselines`，分支 `codex/search-r1-rl-baselines`。

**六种算法都有移植实现；六个独立单算法 agent 已完成检查。结论是有条件通过接入审查，
存在未修复问题，尚不能宣称全部复现/生产验收通过。** 当前仍可开展隔离的小规模验证，
不应把已有 tiny CUDA/FSDP 测试当成正式 3B/vLLM 训练或论文指标复现。

本轮为只读审查：没有修改算法、没有启动远端任务，没有碰另一个复现 agent 的代码/训练。
报告对照论文版本和作者固定源码，独立核数学、代码、执行逻辑与infra；此前PASS没有直接继承为本轮结论。
实现报告中允许的任务适配仍有效，但新增发现不能仅靠“这是适配”忽略。

## 1. 六名 agent 和逐算法结论

| 算法 / 独立 agent | 核心核验结果 | 主要未解决事项 | 判定 / 详细报告 |
|---|---|---|---|
| GiGPO / audit_gigpo | 默认mean-only、exact anchor的核心优势与固定作者函数差分吻合；step时钟、mask/uid重排闭合 | 共享ctx错位/空mask NaN；作者search默认std、近似anchor、cross-step统计及惩罚不同；micro/token权重需披露 | CONDITIONAL / [gigpo.md](gigpo.md) |
| Tree-GRPO / audit_tree_grpo | 真分叉、复用prefix和工具结果、继承预算、node去重、叶来源及独立inter+intra zscore成立 | 共享P1；token PPO不同逐轨迹公式；逐root串行padding开销、final轮配方差异 | CONDITIONAL / [tree-grpo.md](tree-grpo.md) |
| BAPO / audit_bapo | 全错合法IDK bonus、early/diversity gate、整组重采样、阶段状态串行接入成立 | **F1作者规则差异改变bonus和重采样**；共享P1；独立阶段holdout仍未选；完整stage生命周期未测 | CONDITIONAL / [bapo.md](bapo.md) |
| OPSD / audit_opsd | 12组作者FKL/RKL/JSD值+梯度差分通过；真实tiny教师位置/左右padding/observation mask、两rank全局tokenmean成立 | 共享ctx错位；global micro可floor成0；全空监督仍AdamW step；CPU教师/full logits预算与精确resume未验 | CONDITIONAL / [opsd.md](opsd.md) |
| TOP-D / audit_topd | proximal logaddexp、exclusive future有效token均值、逐题token pooling、固定old E轮、DP分母成立 | 共享ctx错位；masked exp溢出；跨题token weighting适配；无已确认作者代码，真实训练教师/生产评分资源未验 | CONDITIONAL / [topd.md](topd.md) |
| CoSearch / audit_cosearch | 两optimizer/uid查表、temperature ratio、语义组、fallback/非法主轨迹过滤成立；真实tiny HF generation/logprob独立验证 | **trivial答案及精确文档命中标签与作者不同**；共享P1；CPU大ranker/top50整链未验；同奖step/空F1差异 | 接入CONDITIONAL，生产gate未满足 / [cosearch.md](cosearch.md) |

报告里的“正式验收FAIL”是放行条件未满足，**不是已经启动生产训练并运行失败**。
未发现本轮可证明的P0，不代表证明不存在P0。共有问题由[共享 infra 报告](shared-infra.md)归并。

## 2. 优先修复：已证实的缺陷/作者一致性问题

| 优先级 / 编号 | 条件与实际反例 | 影响 | 处理建议 |
|---|---|---|---|
| P1 / INF-01 | 本地micro全部action mask=0，原compute_policy_loss和梯度NaN | GiGPO/Tree/BAPO/CoSearch主actor及原GRPO；TOP-D/OPSD有限输入零mask不受该分母bug影响 | 共同actor安全处理零mask，并保持各rank collective/backward同步；不要某rank单独跳过 |
| P1 / INF-02 | 初始generation完整输入而scoring尾部max_start；后续rolling尾裁/response头裁不一致 | 六方法的行为策略概率解释与原Search-R1；TOP-D/OPSD尤其依赖同条件评分 | 记录真实action prefix并一致评分，或共同协议保证不越界；report只诊断，error只拒绝；GRPO同修 |
| P1 / BAPO-1 | yes indeed/yes：本地F1=2/3、作者0；the/a：本地1、作者0 | 全错组被当部分正确，IDK bonus和重采样被关闭 | BAPO采用作者特殊类别/空归一化规则，并针对bonus/resampling做差分验证 |
| P1 / CS-01 | 主答案错，无关文档含yes：本地relevance奖励1、作者trivial规则main-only0 | ranker学布尔词命中而非证据；availability和reward分支改变 | 统一trivial答案过滤和availability/relevance标签 |
| P1 / CS-02 | The Hague 对 I live in Hague：本地命中True、作者False | 伪监督和Hit@k/复合reward不一致 | 采用作者精确token matcher，或明确接受独立标签设计并单列实验 |

这些缺陷目前**未修复**；给出的P1是代码反例/一致性证据，不声称远端生产已遇到NaN或效果下降。
BAPO和CoSearch的作者主F1规则并不完全相同：CoSearch作者没有BAPO的yes/no/noanswer特殊分支。
修复共享token_f1时必须保留算法各自语义，不能把BAPO特判盲目施加给CoSearch。

## 3. P2与可接受适配

- TOP-D先exp后mask可让无效token溢出污染loss/gradient（old=[0,-100]、current=[0,0]、mask=[1,0]）；
  root独立复现。应在exp前屏蔽无效delta，对有效token非有限值明确拒绝。
- OPSD early config允许global micro小于rank数，worker floor后为0；应前置检查限定配方global micro=DP ranks及mini整除。
  CUDA_VISIBLE_DEVICES只限制可见卡，不自动将默认8卡Ray配置改成2/4卡。
- OPSD/TOP-D的有限输入空监督loss与梯度可为0；完整空mini仍可能因AdamW衰减/历史动量移动参数。
  CoSearch同奖adv=0也仍step。需声明正则语义并计有效训练事件，不能用delta单独证明有效监督。
- CoSearch主F1双方归一化为空时本地1、作者0（the/a）；未读取当前目标parquet确认发生频率，属条件性差异。
- Tree每root单独扩展导致八卡补行放大，已有探针两root有24 padding rows；成本已计入，没有隐瞒，仍需聚合/实测。
- GiGPO exact/mean-only、Tree六叶/四动作、BAPO连续停滞+实际G的论文语义、
  OPSD固定teacher/答案-only/full FT、TOP-D E2及跨题global-token weighting、CoSearch K3与CPU HF backend，
  都需作为明确配方记录；是否采用作者代码默认、论文描述或共同基座优先，要在实验协议固定。
- OPSD作者脚本本身采用fixed_teacher+LoRA，因此“冻结teacher”不是一概错误；本移植用独立CPU模型替代禁用LoRA。
  若选外部不同权重教师则不应宣称无外部教师的自蒸馏。TOP-D无确认官方源码，不能声称官方代码逐行一致。
- TOP-D pooled whitening sample std的ddof未在论文明确，不能假称官方确认。
  GiGPO/Tree等micro-token reduction与论文逐action/轨迹权重、TOP-D跨题全token mean与逐题期望均有差异。
- raw token decode/标签截断/reencode等价性、SP>1、dynamic micro collective数量仍是未量化的运行边界；
  主配置SP1，不能从静态代数证明推出所有非默认分布式配置已验收。

## 4. 与原 Search-R1 是否一致

基础模型、数据、搜索任务和共享检索topk3保持；基础paper-grpo数据脚本生成全局唯一index。
[逐字校验](evidence/base-unchanged.json)确认原generation、core PPO、QA EM、RLHFDataset、H20启动/资产准备脚本未改。
旧无baselines入口保持原分支；必要hooks在新配置下启用，默认E1保留旧schedule，E>1现在真实迭代。

“原文件没改”不等于六方法与原流程完全相同：新算法需要新rollout/advantage/reward/teacher/ranker。
按算法要求允许改变的训练信号和预算、以及有意不同于作者配方的部分均在六份矩阵逐项列出。
没有改主actor checkpoint/data、最终评测题或原agent训练来解决新算法问题。
如果后续统一修复共享NaN/context机制，原GRPO对照也需相同修复和重新验收，避免不公平比较。

## 5. 已完成证据与仍缺的生产证据

本轮root复跑：24 baseline tests通过；39原复现tests通过，1项Linux procfs检查在Mac跳过。
六种Hydra compose+validate_config和shell语法检查通过，但aux路径是占位，不是资产或模型加载验收。
各agent额外差分/反例/实际tiny模型/Gloo测试见独立报告；不可把各agent重复跑的测试数相加成独立覆盖数量。

原A800 GPU0/1两卡证据仍有效：真实tiny CUDA/NCCL/FSDP actor及ranker optimizer、零decay下非零梯度/更新与权重保存。
本轮未重新占GPU；原测试已释放，未干预他人。其Torch2.9/cu128与正式Torch2.4/vLLM0.6.3不同，
TOP-D GPU teacher logprob和ranker事件是构造的，故不证明真实教师/生成/检索/controller整链。
CoSearch本轮额外的真实HF生成是随机小模型CPU、非法ranking/fallback路径，不是生产合法排序质量验收。

正式实验还需：

1. 处理P1与条件性数值问题，冻结每种适配recipe；BAPO选独立且无训练/终测prompt重叠的holdout。
2. 固定兼容且来源可追踪的TOP-D教师、OPSD初始化教师、CoSearch ranker，登记checkpoint hash和辅助成本。
3. 目标节点先独立预约GPU、实时检查双阈值/已有进程，隔离输出/Ray缓存，验证模型/数据/环境与真实retriever。
4. 在正式3B、remove-padding/offload、实际vLLM/worker/Ray设置跑受控完整生命周期，
   包括真实搜索→old/teacher评分→actor/ranker更新→验证→checkpoint，测context mismatch及CPU/GPU/耗时预算。
5. 目前resume_supported=false，只保存权重和部分状态；若运行长任务需实现/验收完整恢复，或明示不能精确续训。
6. 然后才做全量训练和七集指标，对不同rollout/teacher/ranker预算单独报告。现阶段没有论文指标复现结论。

## 6. 报告绝对路径

汇总报告：
`/Users/shaoshitong/Documents/Search-R1-RL-Baselines/reproduction/baselines/audit-20261002/SUMMARY.md`

逐算法和infra：

```text
/Users/shaoshitong/Documents/Search-R1-RL-Baselines/reproduction/baselines/audit-20261002/gigpo.md
/Users/shaoshitong/Documents/Search-R1-RL-Baselines/reproduction/baselines/audit-20261002/tree-grpo.md
/Users/shaoshitong/Documents/Search-R1-RL-Baselines/reproduction/baselines/audit-20261002/bapo.md
/Users/shaoshitong/Documents/Search-R1-RL-Baselines/reproduction/baselines/audit-20261002/opsd.md
/Users/shaoshitong/Documents/Search-R1-RL-Baselines/reproduction/baselines/audit-20261002/topd.md
/Users/shaoshitong/Documents/Search-R1-RL-Baselines/reproduction/baselines/audit-20261002/cosearch.md
/Users/shaoshitong/Documents/Search-R1-RL-Baselines/reproduction/baselines/audit-20261002/shared-infra.md
```

临时探针已复制到evidence供审计，其原始绝对/tmp源码缓存路径保留；重跑作者差分需按报告下载相同固定源码，
不能把临时缓存当永久安装依赖。所有实现文件、报告和证据hash见[manifest.json](manifest.json)。
