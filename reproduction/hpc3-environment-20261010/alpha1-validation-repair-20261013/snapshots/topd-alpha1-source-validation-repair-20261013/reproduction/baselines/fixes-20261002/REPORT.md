# 六算法审计问题修复报告

日期：2026-10-02。独立仓库绝对路径：
`/Users/shaoshitong/Documents/Search-R1-RL-Baselines`。
分支：`codex/search-r1-rl-baselines`，修复起点 `afc0a3b17e9f872c792af8e70a5290a0de7ea9a9`。
原 Search-R1 基点：`b251ba6b503217a59cd3395c1326d7b66b602989`。

**结论：原审计确认的代码缺陷已修复，并通过本地测试和独立复审；不能据此称正式训练已经全部验收。**
本轮没有修改原复现 agent 的 checkout、H20 训练或其他用户的任务。原始审计报告保留历史状态，
不回写成当时已经修复；最新源码与报告SHA见 [manifest.json](manifest.json)。

## 用户决定与共同控制变量

- 保留 Search-R1 原 rolling生成、整轨迹重构评分，以及decode/postprocess/reencode流程。
  不要求生成与评分上下文一致。六算法及新增GRPO对照共用这些规则；context_mismatches只记录。
  因此不宣称严格on-policy或原论文理论条件在这个搜索环境无条件成立。
- 从原训练集按问题固定划分2%阶段验证、98%训练，seed42，六算法及新GRPO共用。
  这是对旧全训练集对照的有意改变，应重跑新GRPO比较，不把历史全训练集结果直接当同条件对照。
- 主模型Qwen2.5-3B base、原prompt、搜索任务、四动作/topk3、七个最终评测集不变。
  教师/ranker为用户已经允许的辅助组件，其成本单列。

## 已修复内容

| 范围/原finding | 修复 | 验证 |
|---|---|---|
| INF-01，GiGPO/Tree/BAPO/CoSearch共享空mask NaN | masked_mean空分母返回graph-connected0；全局空mini所有rank共同跳过optimizer/scheduler；局部空rank仍backward | 原NaN反例、真实两rank Gloo actor、已有AdamW动量保持、OPSD空mini |
| INF-04，TOP-D/共享 masked exp 溢出 | exp前屏蔽无效logp/adv；有效非有限值/ratio overflow明确拒绝；KL输入同样先mask | masked old=-100与NaN/Inf反例，梯度有限且无效位置0 |
| 熵/logits非法Jacobian | raw logits非有限在softmax前拒绝；非有限梯度norm拒绝optimizer step | 独立inf/-inf/nan反例与编译行为探针 |
| GiGPO G4 | 按论文Eq9整段文本action概率比，再action层clip，按episode/有效action等权；不把token平均冒称action ratio | 两token联合ratio=1.21跨clip、长短action、微批1/2、跨rank参考梯度 |
| Tree F2/F3 | 逐叶token mean再叶mean；所有root的children合批续写，保留树身份/预算/节点去重/来源 | 长短叶梯度抵消；两root八卡padding反例24→16，三root多轮；不重放共享检索 |
| TOP-D T02 | 完整rollout预计算每题token数，token权重B/(Q×Nq)，通过DataProto重排携带，使用全DP mini行分母 | 两题长度2/4目标梯度2而非7/3；同题跨rank、不同micro与参考一致 |
| BAPO scorer | 作者yes/no/noanswer类别判等、空归一化F1=0；普通CoSearch F1独立 | 固定作者scorer差分8/8、IDKbonus和重采样反例 |
| BAPO holdout/阶段 | 固定共同split、重复prompt同侧、阶段题不重叠训练/最终测试；阶段验证独立于test_freq，仅真实actor step后observe | 临时parquet 98/2、顺序无关、SHA防篡改、缺UID/碰撞拒绝、warmup/空update不推进stage |
| OPSD/INF-03 | global micro=DP ranks，提前DP/SP/mini/micro整除非零检查；纯蒸馏全空mini无更新，保留原full-vocab tokenmean | 配置反例及真实tiny teacher/student，非零动量AdamW空更新保持 |
| CoSearch labels/group/loss | 作者NFD精确token匹配；trivial布尔答案不作伪监督；query独立标点分隔；全同奖组跳过AdamW，informative组零adv成员保留；ranker与主actor序列等权 | 固定作者matcher12/12、真实HF更新、同奖无参数漂移、非法main过滤、序列长短权重 |
| 部署入口/INF-03 | GPU双15%阈值且拒已有compute process；真实持锁FD检查；最终GPU计数匹配；单节点门禁；独立tmp/Ray/HF/output；资产与retriever预检查 | 真fcntl租约、绕过/超配/多节点拒绝、GPU边界/重复设备反例、Hydra与shell检查 |

## 数学与原代码的区别

GiGPO作者实际使用token PPO；本轮按用户要求对齐论文Eq9的文本action比率，明确是有意不同。
变长搜索episode用有效非空action数Ti作均值；零token动作仍参与advantage时间轴/anchor baseline，
空轨迹保留行分母、贡献0。KL/entropy额外项沿用Search-R1但使用同一行/action token权重，不称整段action KL。
mean-only/exact anchor/per-episode统计仍是文档披露的论文允许适配，不强行照搬作者近似anchor或跨step std。

Tree保留token ratio及shared prefix在每叶loss中的重复；按论文逐轨迹归一而非作者旧micro tokenmean。
续写合批改变随机数消耗顺序，同seed不保证旧串行采样逐叶相同；树分布、预算、来源和节点去重语义保持。

TOP-D权重在完整rollout上定义。完整rollout loss准确题等权；mini用固定rollout权重优化这个目标，
**不是每个mini都包含完整题组、也不是每个mini重新题等权**。不根据局部rank题数重算Nq。
OPSD保持作者full-vocab/tokenmean。BAPO保持作者standard token GRPO（论文Eq2的整轨迹比率简写不被冒称实际实现）。
CoSearch保留既有受控适配：观察top3、continuous主F1、threshold=.5、默认不额外按availability分组。

## 测试和复审证据

全套命令：

```bash
.venv-cpu/bin/python -m pytest tests/baselines tests/reproduction -q
bash -n reproduction/baselines/run.sh
```

结果 **102 passed、1 skipped**（baseline 63通过，原回归39通过）；
原始输出见 [cpu-all.txt](evidence/cpu-all.txt)。包含真实离线tiny Qwen/HF、
actual fit但GPU transport mock、actual actor/optimizer但CPU forward适配，以及真实两进程Gloo/DDP。
Linux procfs检查在Mac跳过。测试不代表3B收敛、论文指标或生产CUDA整链。

独立post-fix复审由三组agent覆盖全部六算法与共享infra：

- [目标/数值/GiGPO/Tree/TOP-D](review-objectives-numerics.md)：找到权重接错函数的P1，已纠正；
  非空actual actor集成和独立Gloo梯度复核通过。另找出非法entropy Jacobian并补fail-fast。
- [infra/OPSD/共同holdout](review-infra-opsd.md)：找到直接入口、最终GPU override和多节点绕过，
  已修复，真fcntl锁回归通过。
- [BAPO/CoSearch最终整合](review-rewards-final.md)：此前交叉复审找出的UID缺列、无更新阶段切换已修复并补实际fit回归。

原审计是六名单算法reviewer；本轮三组增量复审，不将其计作新的六名单算法reviewer。
辅助报告：[BAPO](bapo.md)、[CoSearch](cosearch.md)、[Tree batching](tree-batching-report.md)、
[数学设计](objective-design.md)。早期交叉复审报告保留发现时版本，最终状态以这三份增量复审为准。

## 本轮尚未验收/不支持的范围

1. A800两次SSH探测都返回exit255，入口关闭连接；未传文件、未分配或启动GPU作业。
   [连接记录](evidence/gpu-attempt.json)。历史tinyFSDP结果属于旧提交，不能代表本轮新目标通过GPU。
2. 真正Qwen2.5-3B + Torch2.4/vLLM0.6.3 + remove-padding/offload + Wiki18检索短训练、
   真实teacher/ranker、50候选检索、RAM/显存/吞吐、七集指标尚未验收。实际原数据split产物也未在远端生成；
   本地已用临时真实parquet验证划分工具，没有伪称读取了169615题远端资产。
3. TOP-D需要明确已训练且tokenizer兼容的教师，路径不同本身不证明权重不同/教师更强。
   CPU教师/CPU ranker成本风险仍需测量，不是修改配置就已经证明资源够用。
4. checkpoint仍是权重和部分算法/ranker状态保存，**不支持完整actor optimizer/dataloader/RNG联合精确resume**，
   manifest保持resume_supported=false。这是明确不支持的能力，不能把权重warm start叫精确恢复。
5. 非有限raw logits扫描会同步设备，并可能产生torch.compile graph break；已有eager backend语义验证，
   未测CUDA Inductor/3B性能。单节点启动器租约只能协调同用户采用本流程的作业，不能代替集群调度器。

上述是保留的能力/输入/验收边界，不表示当前已证实训练发散或3B OOM；也不能反向宣称所有环境都没有问题。
