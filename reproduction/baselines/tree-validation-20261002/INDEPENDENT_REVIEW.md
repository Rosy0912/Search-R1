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

## 2026-10-03 短Slurm硬件Gate脚本独立复核

复核本chat `work/gpu_hardware_gate.sh`、`work/gpu_hardware_probe.py`、`work/expected-gpu-uuids.json`（脚本不是训练生产代码；内容SHA256见 `evidence/hardware-gate-script-review.json`）。本审查仅静态核对、bash语法检查、Python compile与8项UUID唯一性检查，未另行启动远端作业。

结论：**未发现阻碍按计划执行短硬件Gate的问题。** 脚本要求environment.ready、非空SLURM_JOB_ID以及SLURM_JOB_NODELIST=h100-9；进入common preflight获取同用户共享UUID lease，锁后复查双15%阈值；内层再次validate_gpu_lease核对8个可见设备、继承FD/inode/实际锁，并再次读取占用与进程，要求实际UUID集合完全匹配固定h100-9八卡。低占用既有compute process仅记录，不结束。脚本中没有Ray、模型、检索启动，也无kill/清理其他进程指令。Slurm独占8卡的实际申请/分配仍由主agent调度入口负责，脚本env与UUID检查不替代scheduler资源证明。

NCCL/BF16阶段使用torchrun单节点8rank/LOCAL_RANK逐卡，all_reduce rank之和期望28，8rank各执行BF16矩阵乘并检验finite。collective通信本身为FP32 rank tensor，BF16部分为逐卡matmul；不应将此描述成BF16 NCCL通信带宽或长时稳定性测量。随机空闲localhost端口可能被竞态占用，若失败应以本轮退出和日志为准，不跳过gate。

Flash阶段只在CUDA device0执行BF16 attention，显式转FP32构造scaled QK、softmax及V计算reference，检查finite与max absolute error<.03；覆盖B2/S128/H2/D64、非causal、dropout0的一个shape。不是八卡Flash逐卡验收，也不覆盖causal/长序列/反向；这符合当前短基础硬件Gate范围。全8卡torch可见性和名称会记录，但不能把设备名称列表解释为每卡Flash已测试。

shell set-e确保NCCL步骤失败不继续写Flash成功record；hardware-acceptance.json标明production_model_inference=false，硬件结果对应Slurm job ID且引用hardware-cuda.log。后续必须保留本轮完整stdout/GPU占用快照、两个日志与record，并与实际job资源/UUID共同验收，不只看一行成功标记。

主agent提供信息：h100-9固定生产Python依赖已准备（已有cuBLAS覆盖例外）、共同split holdout3393/train166222，final51713。此次硬件脚本审查不独立背书全部资产hash/数据身份/CPU-GPU检索排序一致性。即使本Gate成功，也只证明限定NCCL/BF16/Flash基础运行，不等价于Qwen2.5-3B/vLLM/FSDP/offload推理、两actual update smoke、五步保存加载或Tree-GRPO效果复现。

## 2026-10-03 真实模型与CPU/GPU FP16检索脚本只读审查

复核chat `work/real_model_gate.sh`、`work/real_model_probe.py`、Agent3结果目录的 `bapo-retrieval-parity.py`。脚本hash及语法检查/比较器合成探针见 `evidence/model-parity-script-review.json`；本审查未执行GPU/资产读取、未改脚本或其他生产代码。主agent提供真实environment/split/hardware/model Gate通过，生产训练仍0；本次只审这些Gate的代码范围。

真实模型wrapper限制Slurm h100-9、8个固定UUID和common lease/占用复查，继承offline设置，不启动Ray训练/检索、不结束原小进程。probe加载固定Qwen模型前逐条校验model manifest文件，vLLM TP1、maxlen1024/eager、GPU memory fraction .12，单prompt greedy最多16token。结果只声明token生成且production_search_training=false。**模型实例实际只使用GPU0**，8卡为分配和可见性条件；已通过也不能表示8卡vLLM replica、3B FSDP/offload、搜索交互或训练梯度已验收。

检索脚本沿真实DenseRetriever/encoder/index/load_docs，CPU和GPU均E5 FP16、mean pooling/query cap256、top3/top50；GPU FAISS原路径固定useFloat16/shard，CPU FAISS仍原CPU index，二者index数值路径不同。CPU模式在torch导入前清空可见GPU；GPU模式要求Slurm与8卡common leases、双15%复查。输出完整query、doc ID/rank/score/内容hash/来源/资源及实际faiss_import_path，重复检索不同会拒绝；大型index/corpus只有指定hash-large-assets才标asset_identity_verified=true。compare报告相同资产与否、matched-ID score delta/rank偏移及only-A/B，**永远DIFFERENCE_REPORT_ONLY**；相同/不同合成输入均独立验证没有自动PASS。不以summary平均overlap掩盖逐query top3变化。

**部署前提：** 原样parity脚本本身不固定h100-9/Tree预期UUID，且不创建或约束GPU FAISS overlay，仅记录加载路径。Tree必须用专用wrapper额外核节点/固定八UUID，并以该Tree独立root的只读GPU-FAISS overlay在单个检索子进程注入PYTHONPATH及必要动态库路径；不得pip替换生产环境CPU FAISS或把overlay全局export至训练env。需要记录overlay包/库hash与实际faiss_import_path，CPU与GPU recipe/model/index/corpus内容身份均核对。没有这些caller证据，本脚本单独不足以证明overlay隔离或正确Tree节点；此为复用条件，不要求改生产检索数学。

**差异须人工审查：** 不同top3 doc ID或排序、共享doc内容hash不符、资产身份不一致、重复稳定性失败均需要逐query记录；top50差异单列，其score阈值/排序边界与top3输出不能混为一谈。可以利用保存embeddings的GPU-encoder/CPU-index模式与embeddings-in ablation区分encoder FP16差异和GPU FAISS FP16量化差异，ablation也只产生差异报告，不构成训练协议替换。默认14条固定query是有限sanity coverage，零差异也不证明所有搜索query全局等价。原H20历史CPU/GPU FP32 top3 overlap .976既是不同数值recipe，又存在真实差异，不能移用为当前CPU FP16通过证据。当前未给Tree真实FP16逐query结果，所以尚不能判断具体差异是否可接受；若改变检索观察内容，需按PLAN对差异进行真实解释及决定，不能静默用旧经验放行。

## 2026-10-03 检索Gate/等待派发/资产派生只读审查

复核本chat `work/real_retrieval_gate.sh`、`wait_and_run_retrieval.sh`、`supervise_asset_copy.sh`、`prepare_derived_assets.py`及原样复制的 `real_retrieval_probe.py`；同时只读核对其download status写入方copy_direct_assets.py。语法与compile通过，parity副本与Agent3来源逐byte一致。当前审查文件hash见 `evidence/retrieval-dispatch-script-review.json`；未改scripts、未启动远端或训练。

资源与测量范围符合计划：Tree wrapper确实固定h100-9/Slurm八卡UUID/common lease启动复查，CPU child清空CUDA且PYTHONPATH只指repo；GPU reference的FAISS overlay只通过各子进程的PYTHONPATH注入，不覆盖训练env。三个FP16模式加CPU embeddings复用的GPU-index测量，hash-large-assets、repeats2、固定query与recipe相同；结果只写measurements.ready，不运行训练、不自动声明一致。run目录按Slurm job ID且mkdir拒绝覆盖；gate退出保存job/time/exit code，整个CPU/GPU测量期间持有八卡，不能只计算GPU CUDA动作时间。

等待派发在head先flock锁与dispatched.marker，未ready前没有srun/GPU预约；资产ready后仅一次srun h100-9 8H100/exclusive，32CPU/256G/2h，失败保留exit和marker，需要明确人工重试而非自动重复申请。GPU条件由common preflight实际验UUID与双阈值，并保留既有低占用进程。监督下载最多20次，既有hash正确文件复用、partial保留，未见结束其它进程或共享目录改写。

发现两个P2可靠性问题（已反馈主agent，未自行改scripts）：

1. copy_direct_assets.py在目标status上直接write_text，写入时存在短暂空/不完整JSON；prepare_derived_assets.py轮询直接json.loads且无容错，可因这次正常写入竞态终止。等待dispatcher只等derived.ready，对派生异常未直接检测，会持续等待。建议status原子写替换，或reader对临时解析失败等待，并将派生终止写exit供派发读取。此问题使自动执行链可靠性有条件，不能仅凭“后台已启动”认定会完成。
2. 2880轮sleep30秒不构成严格24h wall-clock截止，因每轮SSH检查还可用几十秒（两次检查/连接超时）。建议absolute/monotonic deadline或修正报告为“最多2880轮，至少24h睡眠再加SSH”；srun排队阶段也无该循环deadline。等待本身没有持有GPU，但报告不能声称严格24h后自动退出。

资产派生使用status.complete信任下载方已hash所有源，拼接part_aa/part_ab并gzip解码到临时文件后replace，记录真实派生输出hash与size；旧dest移至previous保留。source_sha256当前从manifest抄入，并未在这次拼接/gzip读取中独立核对源byte hash。只要上游verified assets真正保持不变则数学正确，仍应标为信任上游的来源身份；更强ready证据可在派生过程中验证实际source hash。ready marker不代表FAISS/corpus成功加载，该边界文本已正确说明。未见源corpus/index内容改变或非真实替代检索。

GPU-encoder/CPU-index测量也使用overlay FAISS构建，可能不同于生产CPU FAISS版本；保存CPU embeddings后GPU-index复用可排除encoder差异，但剩余差异包含index/GPU backend/FAISS build/tie ordering，不应全部断言只来自FP16量化。报告已记录faiss_version/import_path，应联合分析top3/top50逐query内容/rank/score及同内容身份，不自动放行。主agent解决上述可靠性项后可继续自动测量；真实检索差异仍待人工判断。


上述资产/等待可靠性P2已在后续脚本内容中修复并独立复核：status由同目录pending.PID写入再原子replace，旧已运行downloader可能短暂发布不完整JSON的reader增加JSONDecodeError兼容；等待改为epoch绝对deadline 86400秒，因此SSH耗时不再每轮额外扩增总体24h。存在至多当前SSH检查与sleep30的越界余量，进入srun后的scheduler排队不属于资产等待deadline。单引号remote命令保留$(cat...)在目标端执行，未误在head读取目标路径。

派生函数现在对concat各source读取时同步SHA并与manifest比对，压缩源先核compressed byte SHA再gzip解码，成功才replace派生成品与最终ready。真实上游asset仍须保持不可变；任何校验失败都会保留partial/building与旧dest，并拒绝ready，而非继续申请GPU。

独立toy验证执行现有派生脚本（只把root替换为临时目录；旧JSON竞态case只把sleep替换为下一次合法status发布）：合法拼接/gzip内容及派生hash正确、空JSON恢复、损坏part拒绝且旧index不变、损坏compressed源拒绝ready，**4场景通过**。证据 `evidence/derived-assets-toy-review.py/.json`；最终脚本hash及deadline边界 `evidence/retrieval-dispatch-fix-review.json`。原初次审查hash/问题记录保留为历史，不覆盖为“从未有问题”。

最终判断：已解决所报两个P2并强化源hash证据，未发现新的资源隔离或真实测量阻碍，可继续按该wrapper派发检索差异测量。仍只是测量准备审查；FAISS加载后的真实doc/rank/score报告未产出，CPU/GPU FP16一致性、训练smoke和效果均不能自动判PASS。

## 2026-10-03 检查点CPU验收器与只读peer资产复用审查

只读审查chat `work/check_real_checkpoint.py`及 `copy_verified_peer_assets.py`，最终内容hash记录 `evidence/checkpoint-peer-script-review-hashes.json`。本次未改生产脚本、未读取或接受真实训练checkpoint、未启动GPU/远端copy。

验收器最初存在P1：默认HF from_pretrained可自动初始化missing weights，而后续完整state_dict/finite/非零delta检查可能错误接纳不完整checkpoint。主agent已修为output_loading_info并拒绝missing/unexpected/mismatched/error；初始直接比较state keys与safetensors keys也会拒绝合法tied alias，现只容许与已保存tensor同storage pointer的别名，其他额外key拒绝。移除low_cpu_mem_usage附带的accelerate依赖，保留CPU float32完整重载、逐base shard同shape/finite/delta、tokenizer词表/BOS/EOS、CPU前向finite与checkpoint各文件hash及成本记录。nonzero delta不代表效果改善或每个tensor都必须更新；不宣称optimizer/RNG/dataloader精确resume。

审查者用Mac现有CPU环境构造真正tiny Qwen2 HF safetensors及WordLevel tokenizer，执行最终未修改的验收器源码副本（只为隔离临时目录复制脚本，无运行逻辑替换）：更新权重正例通过、与base完全相同负例拒绝、删除q_proj.weight的缺权重负例拒绝、tied embedding合法更新正例通过。**4个验收器fixture case满足预期**，脚本与输出 `evidence/checkpoint-validator-toy-review.py/.json`。首次尝试旧low_cpu版因Mac缺兼容accelerate无法进入重载，该环境问题未当成负例通过；最终测试无此替换/依赖，生产脚本旧P1已解决。production Linux RSS计量使用KiB×1024；Mac ru_maxrss本为bytes，此处toy只检查验收逻辑、不使用其RAM结果作为生产资源数据。

只读peer复用脚本在head通过已有h100-11 alias运行cat固定TOP-D资产路径，目标h100-9 receiver写Tree ownroot的unique peer-copy文件，同时统计实际size/streaming SHA，与固定原manifest相同才保留旧dest至previous并原子replace。源仅cat，无源目录、训练环境、凭证或进程改动。源端或目标SSH非零退出、size/SHA不符均拒绝本轮成功并保留自身partial；父进程关闭重复管道FD避免悬挂。三份fixed filename/size/SHA已静态核对路径边界，主agent报告来源三SHA全部实际匹配，此信息不代替目标streaming核验。

复制完成新增调用Tree ownnode的原downloadhelper逐项全manifest再次核验并原子发布status，再由已审派生脚本核source SHA；若有其它未完整资产，helper只在Tree ownroot补下载，未写peer。脚本本身无重复copy锁，主agent应像本轮已说明的那样先只读核已有自有head session再重试，不凭SSH未知结果启动第二个copy。该复用为授权的只读公共资产方式，不需要训练/教师数据混用，也不把peer的环境或checkpoint视为Tree实验产物。

结论：修复后的检查点验收器及固定peer-copy路径未发现新增阻碍。当前只是验收器测试和资产复制安全性检查；真实Tree生产checkpoint与参数delta验收尚未发生。

## 2026-10-03 运行期资源守卫与分阶段训练launcher审查

只读审查chat `resource_guard.py`与 `train_tree_stage.sh`，脚本hash及来源记录 `evidence/resource-guard-launcher-review-hashes.json`。本轮未实际调用NVML/GPU/scancel、未派发训练。浏览独立核对 [NVIDIA官方NVML文档](https://docs.nvidia.com/deploy/archive/R550/nvml-api/group__nvmlDeviceQueries.html)：process utilization的NotFound指游标之后没有sample entries，NotSupported表示不支持；sample timestamp为CPU微秒，游标返回严格更新样本。脚本只将NotFound当空样本，其他NVML错误终止自身作业，符合该语义。

初次静态反馈发现self/other cgroup模糊job substring及compute+graphics同PID memory覆盖问题；主agent修复为完整job_ID路径segment，并对每条foreign memory先验合法范围、同PID取max后按不同PID求和。守卫先证明自身同job cgroup与节点，再取得固定UUID handles；scancel目标只能此数字job_ID，未操作外国PID或global Ray服务。Ray等同cgroup进程为own，foreign memory或SM上界超15%即记录STOP_OWN_JOB并scancel自身。当前游标从启动epoch微秒开始、后续max sample推进，避免误判启动前历史负载。

各foreign PID采样期SM peak之和是保守上界：不同PID不同时间的peak也相加，可超过真实同一时刻利用率；可能提前停止自身，不能将其称精确设备SM利用率。已退出PID仅在fresh NVML名单也消失时忽略memory；sample未知则保守计foreign，较高活动取消。已识别own PID历史时间缓存只豁免不晚于最后known-owner观测时间的旧sample；更晚未知sample不盲目豁免，仍可能导致false-positive自身停止。这是有限5秒polling观测，不能宣称连续即时完整NVML trace。

独立执行真实守卫源码配合fake NVML/procfs/clock/scancel，10场景输出 `evidence/resource-guard-probe-final.json`：NotFound基线完成poll；foreign memory/SM超阈值、未知memory、NotSupported、同PID graphics覆盖、unknown compute memory再known graphics均拒绝；job1234与job12345拒绝且不取消未验证job；departed known-own历史sample通过，更新的unknown departed sample拒绝。脚本为 `evidence/resource-guard-probe.py`。正常case仅用注入EndPoll结束无限loop，此人工异常也进入mock取消流程，不能误读为真实正常轮询会自行取消。该测试不等于真实NVML硬件guard验收。

launcher分two/five/formal为3/6/501总步，对应2/5/500actual outer；要求已人工审真实retrieval-acceptance状态及所引证据SHA，five/formal要求前一stage accepted。Slurm h100-9及八固定UUID门槛、commonlease训练入口、production模块与c7091e7差量为零、独立run/runtime/cache/端口及CPU-only检索、局部repo PYTHONPATH/no GPU FAISS overlay、offline原WandB logger均可见。run输出completed标记只表示进程成功，不自动写stage accepted，真实参数delta/模型重载/finite更新仍需后续审查。

初次launcher只在retriever启动期检查guard存活，存在守卫死亡而scancel失败时训练继续的P1；主agent已补父launcher在训练前及训练期间每5秒监控guard/retriever，任一死亡即scancel本job，scancel失败也TERM自身driver并退出，由Slurm清理自身Ray cgroup。TERM/INT/EXIT记录cost并清理自有retriever/guard，不执行foreign PID或全局ray stop。shell语法检查通过；尚未通过真实训练动态触发stop/retriever failure演练。

最终判断：已审资源守卫/launcher所报问题已解决，未发现新的静态阻碍；只有在真实检索差异接受、上一stage验收和Slurm资源条件实际满足时才可启动相应阶段。本审查不执行或授权自动dispatch，生产训练仍0。主agent提供retrieval job2854的14固定query/两轮性能是有限测量，不能推出真实512题搜索吞吐或CPU/GPU等价。

## 2026-10-03 Slurm2854真实检索测量独立结果审查

独立重算三原始measurement与两个comparison JSON，脚本 `evidence/review-retrieval-results.py` 断言通过；保存逐query两模式集合、顺序、内容hash与matched-ID score delta到 `evidence/retrieval-independent-result-review.json`（含五输入文件SHA）。自包含结果报告在本chat `outputs/TREE_RETRIEVAL_REVIEW_2854.md`，14条query逐条列top3 ID/最大得分差及两GPU路径top50变动位置。

三模式固定source a545a0d2349be9f4d591576f56f62b7d35d1bed8、同retrieval_server SHA、Slurm2854，recipe与verified资产身份完全相同。生产CPU FP16 E5 / FP32 IndexFlatIP、GPU FP16 E5 / CPU IndexFlatIP、GPU FP16 E5 / 八卡sharded float16 GPU index；仅GPU child导入ownroot GPUFAISS overlay。14query两repeat稳定性由probe实际rows等值检查强制，资源记录有两轮；本次独立review重算已保存第一轮结果并核对summary，不声称能从JSON重建未另存的第二轮文档。

CPU对两GPU路径top3共14/14 ID、顺序、内容hash完全相同，最大matched score delta3.975033760070801e-4。top50集合/内容hash共14/14完全相同，但仅5/14排序相同，9/14重排，最大delta4.2307376861572266e-4。两GPU路径embedding hash相同、CPU不同，encoder变化已经足以出现这9条重排；Apollo query8两GPU路径具体排名却有额外11/12互换，因此不能说所有变动位置完全相同，也不能把变化只归因GPU index quantization。CPU embeddings / GPU index ablation尚待证据。

结论：这些真实测量足以支持Tree top3 CPU部署有限sanity gate的下一阶段两次实际更新smoke审查。当前observation formatter只使用contents按rank编号、不使用scores，这14条top3观测未变化。没有观察到需要阻止此有限阶段的top3差异；须披露并接受top50重排，不能推广CoSearch top50排序等价或全部生产query等价。14query（batch upper512）CPU两轮search36.746/22.541秒含top3和top50、load73.952秒/self peak67.48GB不能证明512吞吐，真实smoke仍须记录latency/RAM/timeout、六叶、finite loss/gradient、非零参数delta、actual outer updates、GPU峰值与checkpoint重载。本review不写acceptance记录、不派发训练；生产训练完成状态仍未取得证据。

## 2026-10-03 CPU embedding / GPU index ablation与console精度差量复核

补充独立脚本 `evidence/review-ablation-logger.py` 执行通过，证据 `evidence/ablation-and-logger-independent.json` 包含逐query变动、输入SHA和日志roundtrip。真实Slurm2854 GPU index复用CPU embedding hash完全相同，同verified资产/recipe/source；top3共14/14 IDs/order/content同，max score delta1.1920928955078125e-7。top50共13/14排序同，集合/content全同，max delta1.7881393432617188e-7；唯一Apollo query8排名11/12（11729878/11729861）交换，CPU两条同分0.8659123182296753，GPU也同分0.8659123778343201。支持这14条中索引路径只见微量误差与并列排列，不能外推全部query。完整report已补到chat outputs/TREE_RETRIEVAL_REVIEW_2854.md，原报告的pending说明由此更新。

只读核diff：aggregate_logger.py唯一.3f改.17g，不修改dict、metrics生产、actor梯度/目标、backend。原函数真实普通float/np.float32/np.float64/np.int64/500/1e-12/负小数回读与float(value)逐项完全相同，输入不变。小梯度不再console四舍五入到0，未发现阻碍。此结论限定普通real-valued训练metrics；不以此宣称任意大整数/complex均可无损解析。W&B0.30旧datastore API限制与已成功offline logger区分，尚未复核真实训练日志。新生产SHA需由主agent冻结/部署，当前独立复核以已核logger文件SHA为准；生产训练仍未启动。

## 2026-10-03 logger冻结SHA与独立真实stage验收器准备

重读launcher已将production freeze改为aed28e176da946dc6cff73cfb23fac5cf2a879a8；shell语法通过，其余入口/lease/ownPID守卫边界保持此前审查结论。本轮不派发训练。源文件hash见evidence/stage-validator-source-hashes.json。

独立CPU-only工具chat work/validate_tree_stage.py要求指定two/five/formal（分别2/5/500真实更新），解析console精确.17g数值（去ANSI/Ray前缀），要求完整有序1..N记录、无重复、每步actual count=step、optimizer_steps正整数、grad_norm正有限，七必需actor指标及全部打印指标finite；两步产物不能通过五步验收。核cost成功/8卡/stage、resolved预算N+1与smoke模式、tree方法、checkpoint在自身run、actor和algorithm均final global_step_N、algorithm_state来源与resolved一致且不声称resume支持。要求调用base等于resolved actor model路径，随后实际CPU调用已审check_real_checkpoint.py，完整HF重载/加载缺失拒绝/tokenizer/finite全部权重与forward/相对base非零参数delta/全HF hash；该步骤任何失败无最终report。输出只REAL_STAGE_EVIDENCE_READY_FOR_INDEPENDENT_REVIEW、not_stage_acceptance=true，绝不写*-stage-acceptance.json。

12项fixture检查通过（evidence/stage-validator-probe.py/json）：two/five精确通过，two当five/five当two/重复/非finite/零optimizer/零grad/缺metric拒绝，结构检查、wrong algorithm globalstep拒绝、无真实产物拒绝。结构fixture故意非真实weights，未调用HF路径，不能冒充真实checkpoint。HF验收器本身此前tiny Qwen正/未更新/缺权重反例已审；本轮没有真实训练产物，因此没有完整stage成功、GPU训练或效果验收结论。

局限：console信息是记录证据而非可证明每rank完整trace；正grad_norm为保守sanity条件，不证明每张卡每叶都有非零梯度；algorithm_state只存globalstep/来源，未存actual_update/optimizer/RNG/dataloader，不能宣称精确恢复。本工具不能代替资源守卫NVML trace、六叶rollout来源/观测mask检查或后续七集效果验证，需要人工审查相应证据；checkpoint参数delta为相对immutable base的最终差异，不证明第二步相对第一步独立delta。报告输出是review-ready，最终stage接受仍由主agent在实际产物审查后决定。

## 2026-10-03 Slurm2866未进入训练的Ray/Click兼容性失败与修复范围

真实outputs/evidence/two-2866-infra-failure.log含cost exit1，Ray CLI import时add_command_alias/copy.deepcopy抛Sentinel ValueError，launcher尚未执行实际run.sh训练。该作业不能计入actual updates、checkpoint或smoke成功。独立浏览[Ray官方repo issue56747](https://github.com/ray-project/ray/issues/56747)：正文实报Click8.3.0及yanked8.2.2，8.2.1可用，提出<=8.2.1；将现场8.5.0相同栈归因该兼容性问题是证据支持的推断，需ownenv真实CLI探针验证，而非官方正文明确测试8.5。

[PyPI8.2.1元数据](https://pypi.org/pypi/click/8.2.1/json)独立确认wheel SHA61a3265b914e850b85317d0b3109c7f8cd35a670f963866005d6ef1d5175a12b，102215bytes，Python>=3.10，未yank，匹配目标Python3.10。仅own Tree env把Click固定8.2.1且不升级依赖是有限CLI infra修复，无需变动核心算法/model/batch。核查时本地production HEAD与launcher freeze均aed28e176da946dc6cff73cfb23fac5cf2a879a8，search_r1/verl相对该commit diff空；linux-requirements.txt当前仍8.5.0，后续真实新environment freeze应记录8.2.1，不能继续把旧freeze当运行版本。

证据evidence/click-cli-compatibility-review.json保留故障log/launcher SHA、官方来源、旧/new Click。实际ownenv wheel安装SHA、ray --version及pip check、新freeze尚待主agent结果；本轮独立审查只判定修复范围合理，不宣称CLI修复真实通过，更不宣称两更新smoke通过。

### Click兼容性修复真实证据最终复核

真实click-compatibility-fix.log显示own Tree env Python3.10.12只卸载Click8.5.0/安装8.2.1 wheel，ray --version返回2.40.0。复读fix_click.py，先SHA核对官方wheel再写own wheelhouse，uv pip --python ownenv --no-deps仅改变Click。完整freeze独立集合diff确认唯一删除click==8.5.0、增加ownwheelhouse的Click8.2.1 direct URL，其余163包行相同；差量保存evidence/click-freeze-independent.diff，源hash追加click-cli-compatibility-review.json。

新dependency check与旧check的依赖检查内容相同，仅末尾增加ray, version 2.40.0。仍只有原允许torch/cublas12.4.5.8对12.1.3.1的不匹配，未新增依赖错误；不能称pip check完全零错误。production search_r1/verl相对aed28e1无diff。修复已取得真实CLI通过证据，未发现范围阻碍。Ray start/actor实际更新仍需新作业证据；2866仍是训练前失败，不计更新或smoke成功。

## 2026-10-03 Ray短socket目录与诊断设计（尚非真实更新验收）

真实2867日志AF_UNIX socket path超107bytes，训练前失败。只读复核新launcher：新RUN/runtime为持久目标，/tmp/sst-tree-JOBID经不存在/非悬空symlink双检查再ln-s，不复用现有路径，head temp为SHORT/ray，run.sh直接拼RAY_TMPDIR不realpath，故保持短socket拼写及持久数据。shell语法通过；不改production数学/配置冻结aed。建议GNU ln-sT进一步避免检查后目录竞争时默认往目录内建link的边界，常规唯一job路径未发现静态阻碍；尚需实际新job启动证据，不能将路径检查当smoke完成。

已提供只读设计chat outputs/TREE_DIAGNOSTIC_DESIGN.md：env opt-in、branch即时hash/source/tree/trajectory继承、每问题2roots*3leaf分组、retained observation mask/steps零与-1、compose/padding/后续lossmask长度及来源、context差异单独report、无文本和无RNG副作用、有意义mutation/开关等值测试。尚未实现或修改运行source，无此真实训练诊断产物，故当前stage验收仍缺六叶/sharedprefix/mask原始证据。

## 2026-10-03 独立副本Tree诊断实现与训练前infra范围补充

按主agent新增授权只在work/tree-diagnostic-development复制代码中实现，原production search_r1/verl相对aed28e1仍diff空。outputs/tree-diagnostic.patch（四文件）git apply --check通过，未应用；副本REVIEW.md/CPU_TESTS.txt及evidence/tree-diagnostic-implementation.json保存范围/hash。30 CPU tests通过（8新诊断+3原batching+19算法），实际小CPU DataParallelPPOActor控制路径证明开关梯度/参数bitwise相等、前向stubbed；不等于GPU真实模型。diagnostic目录own-output containment、step/index绑定、branch即时状态hash、2roots*3leaf按问题分组、retained record mask/steps/padding/compose UID/reorder、context差异记录、零RNG/默认零IO和污染/覆盖反例已测。未加入trainer最终loss_mask hook，无真实运行诊断文件，不能验收stage；unexpected异常/IO失败可能无FAILED JSON或遗留pending，缺少completion必须拒绝。仅hash/count/metadata，不含原文本/token ID，hash不保证抗字典攻击隐私。

2869真实日志Ray默认worker10002..19999撞固定GCS16382，训练前失败。独立浏览Ray2.40.0官方tag scripts.py/parameter.py，min/max-worker CLI有效且inclusive；launcher22082..23081共1000端口不覆盖GCS16382/retriever18082/失败log其它已知组件端口，保留计划GCS，bash语法通过，ln-sT修复确认。尚不能证明随机组件或foreign端口永不冲突，实际Ray bind仍需通过，未计smoke成功。来源https://raw.githubusercontent.com/ray-project/ray/ray-2.40.0/python/ray/scripts/scripts.py 与 https://raw.githubusercontent.com/ray-project/ray/ray-2.40.0/python/ray/_private/parameter.py。

2870主agent报告actor entropy torch.compile缺Python.h，actual0。本轮核outputs/evidence/python-headers-cpu-compile.txt真实CPU gcc成功与包SHAverified，runtime/deb版本均3.10.12-1~22.04.18；仅ownroot dpkg-deb-x提取/C_INCLUDE_PATH子进程，未系统安装，范围合理。check_actor_compile.py使用真实entropy函数、GPU0 FP32/BF16值对照+finite/nonzero反传，是小shape compile能力gate，不是8rank/真实policy更新或smoke通过；当前未取得GPUprobe产物。

### 独立副本actor前批次证据扩展

完整patch扩至五文件（新增ray_trainer hook），原production未改。Controller.audit_actor_batch只Tree/env开启，真实_create_loss_mask后/update_actor前执行；step*_policy_batch.json独立发布并引用原rollout JSON SHA。核原始loss_mask==inforesponse，不误要求attention product；有效mask=attention*lossmask。trajectory set与每行response/info/attention/steps/tree/question hashes随balance/reorder一致；权重finite且effective/rowcount、空叶0/其它sum1，有效位置oldlogprob/advantages finite。数据/metrics只读。成本记录branch/finish检查本身wall/processCPU，排除生成等待/publication；policy包含原rollout hash读取，排除actor/publication，不称独占CPU profiler。

最终33CPU tests通过，含10种policy污染拒绝、真实rowreorder、原始mask1但attention0空叶权重0（无效位置非finite值不误拒绝）、hook位置及真实小CPU actor梯度/参数开关bitwise相等。先前“未加入最终lossmask hook”局限由此实现解决，但没有真实GPU诊断产物/效果验收，仍不接受stage。副本REVIEW.md/CPU_TESTS.txt/MANIFEST.json与evidence/tree-diagnostic-implementation.json已更新，root待独立review/完整gate后决定冻结。

### 最终trainer hook兼容修复与完整CPU gate

root本地应用patch后完整gate发现14原fixture失败（trainer无baseline_controller属性/AST片段无os）；初次字符串replace未命中原缩进、旧失败log保留，独立review实际复读指出未落盘，root已精确修复。最终hook为audit_controller=getattr(self,'baseline_controller',None); hasattr(...,'audit_actor_batch')后调用，Controller内部method Tree/envguard保持。检查仍_create_loss_mask后/update_actor前，无metrics/math改动，旧trainer/AST fixtures兼容。

只在独立副本同步新hook，44相关fixture+diagnostic tests真实通过，log work/tree-diagnostic-development/HOOK_COMPATIBILITY_TESTS.txt。独立读取root完整cpu-all-tree-audit-final.txt为162passed/1skipped/4warnings/26.25sec，与旧fixed log14failed明确区分。五patch文件副本与root本地逐字节相同；root改原source、reviewer未改，远端仍aed据root报告。完整outputs/tree-diagnostic.patch重新以git aed28e1为baseline生成，MANIFEST与evidence/tree-diagnostic-implementation.json含最新patch/fullgate/log hashes。此证据解决兼容性测试阻碍；无真实Tree GPU rollout/policy诊断产物或stage接受，不能宣称smoke成功。

### dc92f23最终commit复核与stage强制诊断支持

独立读取已提交dc92f237e4007404b9855fdc57f5547cd11fd49a：仅五已审文件，trainer hook新增getattr/hasattr三行，其余Tree目标/optimizer/metrics不改；代码与已测副本五文件字节完全相同。最终CPU gate162passed1skipped、独立兼容子集44passed证据保持。

chat work/validate_tree_stage.py新增--require-tree-audit（可指定--tree-audit-dir，默认ownrun/output/tree-audit）。开启时在CPU HF重载前要求每实际step严格一对rollout与policy完成文件，目录在ownrun内且无pending/多余/重复；核step/sequence、algorithm来源、policy引用rollout文件SHA、trajectory/tree/question逐行绑定、三叶每source与两树/六叶每问题、branch同字段hash继承、有效token范围/空叶/权重质量及计数。返回artifact SHA列表；缺失、FAILED、数量错、绑定污染均拒绝，无最终report。legacy无flag仍可生成不包含audit的review-ready报告，不能用作今后完整stage接受；主agent必须调用--require-tree-audit，不得把legacy报告误作stage通过。

14项CPU验收器fixture通过（evidence/stage-audit-validator-probe.py/json/txt）：真实fake-manager CPU Tree Controller两步生成+policy/reorder文件检查正例；缺失/FAILED/额外/rolloutSHA/trajectory/mass/group/branch/step/algorithm/pending/five-from-two/missing-directory反例拒绝。无真实checkpoint或GPU训练产物，此测试仅验收器逻辑。HF reload仍由真实checker强制；stage接受还需真实checkpoints、GPU资源和log等证据，工具永不自动写acceptance。

## 2026-10-03 CPU BLAS隔离小benchmark与辅助GPU4候选复核

按授权通过既有work/remote.py在h1009独立CUDA不可见子进程，仅50k*768 synthetic index，threads8/nice19/timeout120，不碰检索服务/训练env/global变量。默认FAISS嵌入BLAS实际SGEMM绑定自身so、thread API1；ownroot pthread OpenBLAS preload实际SGEMM relocation由LD_DEBUG确认切换ownlibrary，global线程API8、OpenBLAS0.3.20/CorePrescott，FAISS direct-handle API仍1（只是符号查找，不代表真实调用）。默认14/128/512墙钟.016035/.326806/1.247163s，preload.016180/.080615/.297909s，约1/4.05/4.19x；全部合成top3 IDs/scores逐值相同。512CPU时间2.789582→3.896805s增加，非成本全面改善。单次顺序共享host小index，仅候选证据。

第一次preload在LD_DEBUG巨大输出时SSH broken pipe255，没有成功结果；保留失败log并确认ownPID退出后，仅preload用远端过滤重试0成功。work/cpu-blas-review/含原始测量、loader binding/thread identity、SUMMARY/README/MANIFEST与future full-index-parity.py；完整64GB未执行，工具显式--allow-large-index、固定CPUemb/indexSHA、14及重复512、独立k3/k50两repeat和DIFFERENCE_REPORT_ONLY，重复512不是512真实unique吞吐。建议值得下一独立CPU/RAM受控full-index parity gate，不能直接切生产后端。

辅助A800 GPU4 FP32 index-only候选首次审查指出watchdog初始化不在try导致静默死、仅k50前3冒充实际k3、embeddings未bytesSHA绑定三阻断。最终probe已修ready/fresh/liveness/init failclosed+graphics合并、真实k3/k50分别query、CPUbyteshash与index/corpus基准SHA及前后stat、searchCUDA同步；无encoder/量化/自动接受，单次gate未见剩余数学阻碍。ownenv import/prefix/.pth和sharedUUIDlease/platform cgroup授权须实际artifact证明；单纯cri-containerd字符串不足独立证明GPU4分配。极快thread.start-before-try可能缺失败cost但failclosed，不会继续GPU；transfer建议前后CUDA同步，foreign同PID记录覆盖只影响记录不影响任意foreign自停。未启动此GPU gate或服务。

## 2026-10-03 2873真实actor TensorDict缺step_ids修复独立复核

真实outputs/evidence/two-2873-step-ids-failure.log为TensorDict.get无默认在Tree actor选中keys无step_ids时报KeyError，actor未有效更新，actual0，不计smoke成功。仅data.get('step_ids')→data.get('step_ids',None)合理：Tree选择keys不传GiGPO steps，weighted_policy_loss None走token ratio；GiGPO实际step字段仍进入joint action分支，不改数学。新增真实DataParallelPPOActor/TensorDict Tree删step字段SGD非零/finite/microbatch等值回归，独立执行4tests（paper_metric_logging/distributed_paper_metrics/paper_ddp）通过15.31秒，含CPU2rankGloo不等于NCCL。另独立读root完整gate163passed1skipped/25.70sec。evidence/optional-step-ids-independent-review.json保存source/fullgateSHA。修复静态/CPU无阻碍，新真实stage仍待新的训练产物。

## 2026-10-03 A800 GPU4真实FP32 full-index结果与候选HTTPgate审查

独立重算outputs/evidence/a800-fp32-index-difference.json四组共1052行，assetSHA同2854，实际k3 14/14及重复512/512全同；k50集合全同，14排序11/14同、重复512402/512同。matched score max分别5.96e-7/5.96e-7/8.94e-7/1.192e-6，不是逐值exact。q4 Everest7/8、q8Apollo42/43、q12最高建筑1/2为同分swap，512每原query重复只有一种顺序。关键q12 GPUk50前3与真实GPUk3不同，Tree必须实际search(k3)不能切片k50；不满足CoSearch top50排序等价。

原始cost与44条samples独立核：GPU4 UUID一致、无foreign、peak67521019904bytes（约62.884GiB），80GiB容量在观测峰尚余17.116GiB，间隔最大2.208623秒，不是连续瞬时allocator峰。probe44.758524秒/选定单GPU.0124329GPUhours；只含probe body，不含原8GPU平台分配/准备/空闲账单。CPU自身RSS65.376GB、load32.6695秒/transfer6.7404秒；index-only不含CPUencoder/corpus raw内容加载，该内容相同仅按同corpusSHA+ID推断。完整outputs/TREE_A800_INDEX_REVIEW.md与work/cpu-blas-review/a800-independent-differences.json，evidence/a800-index-independent-review.json存报告/sourceSHA。

判断支持下一独立CPUencoder+HTTP候选gate，不production接受。首次service脚本复核仍缺verified assetSHA明确对baseline、原CPU npz bytes/querySHA绑定、platform/cgroup实际条件验证，已反馈root；只有修复/实际取得条件后可启动候选。原rs endpoint async在eventloop同步batch_search串行，当前singleworker资源线程安全；不应换threadpool/multiworker。CPU E5原recipe/k3+k50内容hash/noAUTOpass/loopback+4h寿命/foreign仅自停合理。该gate还须隧道实发HTTP、encoder parity/latency/RAM/驻留GPU/guard成本检查，不能从固定CPUemb indexgate推出A800 CPUencoder parity或512unique吞吐。reviewer未启动HTTP或训练。


## A800 auxiliary server final static review (2026-10-03)

The three initial identity/ownership blockers are resolved in `work/aux_retrieval_server.py` (SHA256 `0a3642348444b5df6bd8b65f1cd83ffd849c8ef046b3970c96ce4c22c27d2be8`). The service asserts the full platform cgroup equals the real successful index-probe dispatch allocation, validates fixed GPU4 UUID and inherited common lease, binds verified index/corpus SHA and original query/vector SHA to 2854, and checks asset stat and full E5 file hashes again after loading and the gate. Clone and direct gate search use explicit CUDA synchronization. The original async `/retrieve` endpoint remains synchronous on the initialization event-loop thread and uvicorn has one worker; the wrapper adds costs without changing results. A ready/live/fresh independent NVML thread rejects any foreign compute/graphics process and cancels only this server PID; its bounded lifetime includes loading and idle.

No remaining static blocker to entering a **candidate real CPU encoder + tunnel HTTP gate** was found. This is not production acceptance or training permission. Its direct local 14-query result remains DIFFERENCE_REPORT_ONLY and still needs independent review plus actual HTTP POST protocol/content/score checks. The loopback provenance route can wait behind CPU encoding; a proposed 180s stage check must be independently reviewed with finite request bounds and cancel only the own training allocation. No checker or new launcher was accepted here. The reporter records sampled GPU peak and process RSS, not whole-platform billed allocation or simultaneous dataset-worker RSS. Evidence: `evidence/aux-retrieval-server-static-review.json`.
