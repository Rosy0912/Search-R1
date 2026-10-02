# Tree-GRPO 独立复核（2026-10-02）

结论：**当前 Tree 算法代码/数学/CPU 数据流通过；允许进入计划规定的真实 GPU smoke，尚未完成效果复现。** 初次固定审查 HEAD `1620c2edb13809d19cc3f5af4a298ccd75488104`，算法差量复核 SHA `56e626aceaceec961d90d427dacf3bde3a7bbbb9`（含实现fix `165b80913f72ae48d9595ea0fe61814b453681be`），2026-10-02集成复核 SHA `6dcb667014aa275b982112f600af3d9b03298838`，指标修复复核 SHA `3f6a7307f997d9180a88cf87f7006e5961aa43cc`，实际预算复核 SHA `de9ef3d53b9fb5a3c6b66e24fce1d81304db5c5b`，最新最终复核 SHA `c7091e792d930c6e2488c395a1dbd324ce0eec73`（2026-10-03），分支 `reproduce/tree-grpo`。本审查未修改实现、未启动远端训练、未更改其他 checkout。不存在本次查明的新 P0/P1 算法阻碍。公共部署冻结、资产/资源验收及 3B GPU 结果仍是正式运行前提。

## 来源及复核方法

重新浏览 [论文 v1](https://arxiv.org/html/2509.21240v1) 的 §3.1、§3.2 Eq.6–8 及 §4.1；重新浏览并下载 [作者固定 19bf3fa](https://github.com/AMAP-ML/Tree-GRPO/tree/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59) 的五个文件。GitHub tree 页面浏览器返回 restricted URL，但同 SHA raw 源码成功读取，不以旧 audit 结论代替本轮来源检查。源码副本与 SHA256 见 `evidence/author-*` 和 `author-source-hashes.json`。初次五个核心实现文件 SHA256 记录于 `evidence/independent-probe.json`；最终 SHA 的重复探针及核心文件 hash 见 `evidence/independent-probe-rngfix.json`，保留了旧证据。

独立运行 31 项 CPU 测试，包含树续写、跨 root batching、估计器、目标权重、2-rank CPU DDP 与真实 trainer fit 的 mock rollout 数据流；全部通过。另运行独立探针，结果与脚本保存在 `evidence/independent-probe.*`。字符 tokenizer/mock worker、CPU torch 与生产 CUDA/vLLM 有明确边界，不能代替真实 Qwen2.5-3B 验收。

## 逐项对照

| 项目 | 当前实现与独立判断 |
|---|---|
| 初始化 | trainer 对每问题重复两次；`initial_states` 为每行独立建立 tree UID。两条初始链对应 M=2。 |
| 真实分叉 | `complete` 保存动作前完整 state；`branch` 深复制共享 action/observation/prefix 及剩余 turns，仅换 trajectory UID。已执行的检索结果直接复用。 |
| 候选与续写 | 随机有放回选择 root/非终止节点，候选按 node UID 去重；与 [作者 tree_node.py](https://raw.githubusercontent.com/AMAP-ML/Tree-GRPO/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59/search_r1/llm_agent/tree_node.py) 的 root+nonleaf 选择一致。每个续写继承剩余预算，未变成独立完整采样。 |
| 并行执行 | 当前 `run` 汇总本轮所有 root 的 children 后调用一次 `complete`，与 [作者 generation_ts.py](https://raw.githubusercontent.com/AMAP-ML/Tree-GRPO/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59/search_r1/llm_agent/generation_ts.py) 的跨 root 批处理相符。旧逐 root 串行缺陷已不存在。 |
| 叶与来源 | L=1,N=2,K=3，每 root 恰有三叶，总六叶。选择叶后 sources 显式映射 QA metadata；tree/trajectory UID 随 DataProto permutation 保存；共享 prefix 在各叶路径重复参与训练，未去掉必要叶权重。 |
| 优势 | `tree_grpo` 对原始 outcome 分别计算 question 与 (question,tree) sample-zscore，epsilon=1e-6，然后相加，再乘 action mask；符合论文 Eq.6–7。对应 [作者 ray_trainer_ts.py 默认 tree 分支](https://raw.githubusercontent.com/AMAP-ML/Tree-GRPO/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59/verl/trainer/ppo/ray_trainer_ts.py)；作者 core_algos 同名 tree 函数属于另一个 tree_2norm 分支，不能混淆。 |
| token PPO | `attach_policy_weights` 每叶有效 actor token 权重为 1/length，`weighted_policy_loss` token ratio/clipping 后按叶等权。微批共享 minibatch 行分母，梯度累积无需重复除 accumulation；SP=1 下 2-rank DDP 测试通过。 |
| 观察及空样本 | observation mask=0，terminal scalar 重落到最后保留 actor token，outcome 不变；空叶权重为0但保留外层样本分母。数值 helper 在 masked input 为 infinity 时仍能安全归零；全局无有效 token minibatch同步跳过。 |
| KL/熵 | actor policy objective、reference KL及entropy使用同一逐叶 token权重。默认 actor KL独立于outcome advantage，不暗中改变tree奖励；需生产日志验证实际参数。 |
| offload/同步/保存 | 调用沿用 Search-R1 FSDP/vLLM worker；CPU gate覆盖入口和控制数据流，不证明生产权重同步、显存或checkpoint质量。无精确resume保证。 |

## 本轮实测证据

- 两 root 强制分叉于搜索后的共享状态：sources=`[0,0,0,1,1,1]`，tree UID=2、trajectory UID=6；仅两次检索，prefix 未重搜；worker batch sizes=`[8,8,8]`，padding rows=16，generated tokens=544，prefill tokens=1392。旧 audit 的 `[8,8,8,8]`/24 padding不能沿用。这是小批探针；正式1024 root 批处理开销需实际记录。
- 六叶 reward `[1,0,0,0,1,1]`：两组 zscore 的独立计算及置换等变通过，输出见 JSON。
- 一个1-token正优势叶、一个3-token负优势叶、一个空叶：逐叶权重总和 `[1,1,0]`，loss≈2e-8，梯度≈1.5e-8；当前目标达到预期等叶抵消，不再是旧 microbatch token均值的0.5偏差。
- 四轮搜索、人工 cap80探针：context mismatches=2，rolling=80、scoring=98；该现象真实存在。用户已授权统一保留 Search-R1窗口差异并report，故此审查不要求改动该协议；正式报告仍必须给出真实触发率。

## 差量复核：采样 P2 已解决

初次发现 `Controller.generate` 每次新建 `SearchRollout(seed)` 导致跨outer update重新开始同一个分叉随机序列。主agent在独立实现提交 `165b80913f72ae48d9595ea0fe61814b453681be` 中只为Tree创建持久 `tree_rng`，每次新engine复用该实例。审查diff确认修改仅涉及controller的RNG持久化和一项controller连续调用测试，其他算法仍使用原engine逻辑，未改树叶数、优势、mask或成本计数。原P2已解决，不再作为最终未解决发现。

独立重新运行 `test_tree_batching.py` 与 `test_models_and_driver.py` 的测试（结果见 `evidence/rng-fix-tests.txt`），确认真实controller的两批采样RNG状态连续演化、同seed整段可重放，其中tree连续采样三项通过，但真实fit-driver测试fixture采用 `Controller.__new__`、未初始化新 `tree_rng`，触发AttributeError；本次合计6 passed/1 failed。生产正常 `__init__` 初始化路径未发现此缺陷，fixture需同步更新后重跑，此结果保留为历史失败证据；主agent随后在 `56e626aceaceec961d90d427dacf3bde3a7bbbb9` 修复fixture，独立审查确认仅增加random import与tree/非tree初始化。最终重复相关完整32项测试全部通过（`evidence/final-independent-tests.txt`）。最终CPU gate通过，真实生产GPU仍未验收。重新运行独立probe核对其余树行为与数值不变。仍未提供精确checkpoint+RNG恢复，因此不宣称中断后等价resume。

已授权的偏离还包括四轮且不强制final、统一窗口差异report、相较五链GRPO的六叶预算，以及遵循论文逐叶token均值而非作者源码microbatch token均值。默认group至少三叶，singleton零优势保护与作者singleton返回score不同，但默认实验不触发。不能宣称逐参数复现作者训练脚本。

## 真实 3B GPU 待验收

1. 公共部署commit冻结并一致cherry-pick；h100-9 Slurm独占allocation与8张实际UUID核对；双15%阈值、既有小显存进程记录、启动前复查。未获得资源不得绕过调度。
2. 固定Qwen2.5-3B base/tokenizer、98/2 split的跨路径content_identity/产物SHA、E5/语料/index/hash、CPU检索top3及真实schema与排序一致性；周期64/最终七集与holdout隔离。
3. BF16/NCCL/flash/vLLM 8卡通信及最小推理；生产tree rollout六叶来源、shared-prefix复用、remaining turns、mask、真实generation/scoring mismatch计数。
4. 至少两次actual outer update：old/ref/current logprob、loss/grad finite，非零实际参数delta；offload、vLLM权重同步、8卡每rank1微批、optimizer step计数一致。
5. 五步保存/验证acceptance，完整HF导出与重新加载；确保正式500 actual update对应501总步入口，100步保存/50步周期评估，不宣称等价resume。
6. 正式训练及51,713题七集最终评估，与同split新GRPO比较EM/F1/格式率。单列实际生成/prefill token、工具请求、padding、六叶训练样本、GPU小时、CPU检索/RAM/墙钟/显存峰值；不能按相同步数直接推论同计算成本或论文节省率。

严格判断：本报告完成的是实现复核和CPU可验证性；没有真实3B训练指标，因此目前不能认定Tree-GRPO在search agentic RL有效果优势或完成复现。

## 公共部署最终集成差量（2026-10-02）

最终集成 HEAD为 `6dcb667014aa275b982112f600af3d9b03298838`。公共源码commit `297e1c140677dc621302f1d49fefdb31023fd2b7`，本Tree分支cherry-pick为 `b5bd22605bdc1cbb5853efbb0a7a127e4d0c8334`；独立核对两提交stable patch-id均为 `906aa84736d434efb638790b2c4bab0f6f9fbf7e`。相较算法审查SHA56e626，rollout/advantages/controller/objectives/dp_actor五核心文件diff为空，核心树数学和RNG修复保持不变。证据见 `evidence/integration-delta.json`。

公共变更对Tree的实际影响已审：路径参数化用于独立资产根；双15%GPU占用门槛接受既有低占用进程但保留inventory/UUID lease/启动复查；NCCL rank和使用实际world；CPU检索明确移动encoder及inputs并禁用GPU FAISS，host/port/thread独立可配，默认CUDA行为保持。split新增路径无关content_identity，未改变question分组及98/2数学。验证增加actor生成answer mask范围内的F1/EM/格式/token/search-tag统计，未把观察或prompt答案混入评分，未改变训练reward。Tree不会进入GRPO专用holdout teacher选择分支。smoke_only默认false，正式最终七集评估保留；只在限定最多五次update的smoke跳过末尾全测试集。CPU/GPU检索数值及排序一致性仍需真实资产测试，不能由代码分支推出等价。

最终6dcb667仅给BAPO阶段验证fixture设置tree_rng=None，检查diff确认没有生产代码变动。主agent提供的集成日志 `cpu-all-after-common.txt`为 **140 passed / 1 skipped**；本复核读取该日志，不把它误标为审查者重跑。此前审查者独立32项通过及初始失败/修复日志完整保留，未重复全suite。

**集成结论：未发现新增Tree算法或公共集成代码阻碍。** 实际部署仍未就绪：主agent报告远端SSH反复超时及大文件传输stall；资产与split identity尚未取得，Linux依赖/E5检索/3B训练及GPU集成gate未完成。既有Slurm短预约探针/低占用GPU快照不构成当前持有allocation或GPU smoke验收。不得将140项CPU gate或公共代码冻结解读为Tree-GRPO效果验证成功。

## 2026-10-03 指标归约修复差量审查

最终复核 SHA `3f6a7307f997d9180a88cf87f7006e5961aa43cc` 复用Agent1源commit `31a9578465a9a0faee98ff2b6bdf2bcb6834bd92`；两提交patch-id相同 `cf688eade2fb5ec51730582730c3e6c38261ada8`。公共部署来源仍为297e1c140677dc621302f1d49fefdb31023fd2b7，Tree cherry-pick b5bd22605bdc1cbb5853efbb0a7a127e4d0c8334。核对diff，rollout/advantages/controller/objectives文件未变化，dp_actor仅变更paper-objective日志归约与额外指标质量all-reduce，loss/backward/优化器公式未变。

Tree每叶token权重和为1（空叶为0），PG/entropy/reference KL已使用完整minibatch行分母归一化，日志应先求各micro贡献之和，再跨DP ranks平均。旧代码把micro贡献再次等权平均、reference KL只留最后micro，导致改变微批大小即可改变报告loss。本次按minibatch累加解决此问题，不重新缩放梯度。clipfrac与PPO KL本身是micro权重平均，现乘micro的policy_weights mass再除全局mass/DP world，跨rank平均后得到整个mini的有效叶质量平均。空叶仍在PG/entropy/KL loss外层行分母中，但不计入clipfrac/PPO KL的有效质量分母；两者语义不同且正确。SP=1为本计划配置，实际GPU通信仍待验收。

审查者独立运行Tree特定probe `evidence/tree-metric-probe.py`，复用真实actor测试transport但将baseline_method/weights改为tree_grpo：micro1与micro2的PG=-1、entropy=3、reference KL=.1以及clipfrac/PPO KL日志一致，参数更新/梯度范数一致。另对1-token叶、3-token叶、空叶、2-token叶及不等长micro分组核对PG/clipfrac/PPO KL聚合，结果与完整batch一致（有效叶质量3；clipfrac=2/3）。输出及warning记录在 `evidence/tree-metric-probe.json`、`tree-metric-probe-stderr.txt`；未将单rank CPU probe误称多卡实测。

主agent最新完整CPU日志 `cpu-all-20261003.txt` 为 **141 passed / 1 skipped**；此数字是提供日志，不是审查者重复全suite。原独立32项及140项集成证据继续保留。owner审查见 `reproduction/baselines/gigpo-20261002/REVIEW.md`，本Tree复核不以其结论代替上述独立检查。

结论：指标修复适用于Tree逐叶目标，未发现新增算法/日志归约阻碍。主agent报告远端同SHA已部署、固定资产正在CPU tmux下载；下载完成/hash/检索一致性/环境尚未构成验收，真实GPU训练仍未启动。没有生产F1/EM或效果优势结论。

## 2026-10-03 实际outer update守卫独立审查

最终代码SHA `de9ef3d53b9fb5a3c6b66e24fce1d81304db5c5b`：复用源767cc9b在本Tree分支cherry-pick7217a598b80e3215f65c108fed6de81e7025c6e6，随后de9ef3d将GRPO/GiGPO守卫扩展至tree_grpo。静态diff确认只增加trainer预算日志/拒绝零更新及相应测试，不改Tree目标或梯度。守卫位于actor RPC完成与指标收集之后、周期验证/保存/日志/预算推进之前。

- actor对global_token_count==0的minibatch同步跳过；若整个outer没有任何optimizer step，返回0并被driver拒绝。若仅部分minibatch为空而其他mini正常更新，则计一次outer是正确行为；不能将多个mini/inner epochs误计为多个outer。
- `_optimizer_step`先检查finite grad norm、再调用optimizer.step；异常使RPC失败，不会返回成功步数。完成的每个mini使updates加1，driver只要求整体步数>0。全部warmup跳过也无成功步数，拒绝前进及保存；正常Tree配置critic-free/warmup0。
- 新actual_outer_updates初始0，成功outer每次+1。原global_steps从1开始且每次完成+1，total_training_steps501对应500次outer。守卫通过后两计数保持同速；没有增加可精确恢复的RNG/optimizer/dataloader机制。

独立运行相关实际trainer尾部/最终保存/CPU DDP测试 **15 passed**，见 `evidence/actual-update-tests.txt`。额外独立probe直接编译执行真实fit末段500次，每outer假设12次optimizer steps，验证actual_outer_updates=500、global_steps501，逐次日志1..500连续；保存5/100/200/300/400/500，最后导出与最终验证都使用第500次更新的policy。probe日志/脚本见 `evidence/actual-update-probe.*`。worker是CPU double，此证据仅证明预算控制流和边界，不宣称500次实际模型训练。

### 多rank核对与此前指标结论修正

本次沿实际collect路径检查发现：`collect_dp_compute`返回所有rank输出，而 `DataProto.concat`明确只保留 `data[0].meta_info`；trainer的 `reduce_metrics=np.mean`只处理该meta中的各指标，并不跨rank合并字典。因此actor/optimizer_steps实际上来自首rank，而非八rank求均值。计划SP1/固定rank批数下，每mini的global_token_count all-reduce及同步空batch决定保证各rankoptimizer step计数一致，FSDP RPC所有rank返回才继续；该机制支持当前实际预算守卫。probe核对八rank相同步数的meta归并得到12，未把12*8误计为outer或optim步数。

**P2，纠正2026-10-03指标章节的过强结论：** 该章节描述“跨rank平均后得到全局mini指标”只是数学上的条件，不是已实现的日志归约路径。3f6a730修复了单rank微批归约，但当前paper PG/entropy/reference KL及clipfrac/PPO KL日志仅取首rank的局部贡献，不保证代表全局minibatch；单rank或各rank恰好相同的probe不能证明全局日志正确。FSDP训练梯度确有平均，不受该日志问题影响，actual step计数因各rank同步也不受影响。建议后续只针对数值日志做显式DP汇总，并独立测试不同rank不同叶长/优势/空叶；未修改生产实现。本次审查报告覆盖并纠正旧结论，不能把owner历史审查或此前单rank证据当成已完成全局日志验收。

结论：实际outer预算守卫与500边界通过CPU/静态审查，无新的Tree训练数学阻碍；全局paper指标日志尚有P2，真实3B/GPU训练仍未启动。

## 2026-10-03 P2修复严格复审：真实Gloo/DDP日志归约

本次复核在de9ef3d基础上，审查 `objectives.reduce_paper_metrics` 与actor返回前调用的生产diff。函数只把五项detached列表日志打包为float64进行SUM/world并还原列表，不读取或修改模型/loss/gradient/optimizer；因此DataProto保留首rank meta现在可以得到相同的全局指标。默认计划SP1、各DP rank固定相同行数及mini次数，global empty mini同步跳过，故被归约的key/列表长度在各rank一致；动态micro切分数量不同也不改变最终mini列表长度。全局所有mini为空时五项日志全部缺席，函数同步直接返回，不启动多余collective。CUDA生产路径batch先由FSDP worker移GPU，归约device继承该batch；Gloo CPU仅证明本测试环境。

审查者新增唯一测试文件 `tests/baselines/test_distributed_paper_metrics.py`，实际spawn两个进程、初始化Gloo、使用PyTorch DistributedDataParallel及真实DataParallelPPOActor.update_policy；只替换CUDA transport与toy前向，不替换日志计算、loss、optimizer、DDP backward或DataProto.concat。

独立测试覆盖3种场景（不等长叶；一个空叶；rank0全部为空而rank1有效）×每rank微批1/2×日志归约开/关，每次使用不同叶的advantages、初始current/old/ref logprob和feature，开启reference KL及entropy。完整四叶非分布式真实actor作为reference。所有五项日志经过真实DataProto.concat后与完整batch一致，两个rank返回日志bitwise相同；每rank开启/关闭此次日志归约的梯度及最终参数 **bitwise一致**，分布式梯度/参数与完整batchreference在1e-6容差内一致。空rank关归约时PG日志为0，开归约时得到全局非零PG，直接检验原缺陷已解决。

执行日志 `evidence/distributed-paper-metrics.txt`：**1 passed**，一项测试包含上述全部组合。代码内容hash及运行条件在 `evidence/distributed-metric-fix-hashes.json`。已检查生产diff，而不以其他owner的审查代替本次实测。

P2状态：**已解决**（针对这次已审代码内容；此前未归约的历史日志仍不能当作全局指标）。局限：CPU两rank/toy前向、SP1、单mini/一epoch；不等价于H100八卡NCCL/FSDP、实际3B vLLM/offload验收，也不证明真实效果。500 actual更新守卫和Tree逐叶目标未改变。


最终P2修复提交 `c7091e792d930c6e2488c395a1dbd324ce0eec73`；独立核对提交后objectives/dp_actor/新Gloo测试三个文件hash与审查内容完全相同。追加相关回归 `test_paper_ddp.py`（含多算法、empty rank、全局空监督AdamW不推进）与 `test_paper_metric_logging.py`：**2 passed**，见 `evidence/distributed-paper-metrics-regression.txt`。没有扩展或重复完整suite。远端资产直接下载进展为主agent提供信息，真实GPU尚未运行；不据此声明环境/资产验收完成。
