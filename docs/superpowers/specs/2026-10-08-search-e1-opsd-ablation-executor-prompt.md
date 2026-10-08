# 可直接交给执行 Agent 的 Prompt（Reverse KL修订版）

以下全文可以作为执行线程的初始任务。设计线程只交付文档，没有为本消融实现、运行新测试、启动训练或执行新GPU评估。你收到用户直接交付此prompt后负责实施；不要把旧设计线程goal的“未执行”状态误认为实验已开始。

---

请建立并持续完成一个Search-E1 OPSD消融执行goal。目标是从同一Archive GRPO100起点，完成下面七组各10个在线outer及同一512题验证，判断KL目标、随机配对、正确性权重、学习率和精度执行链的影响。实现、实际更新、完整评估、恢复资产、debug和中文结果报告都完成后才验收。不要仅交计划、后台初始化、前五步门禁或loss下降。

请先阅读随本prompt交付的完整需求文档。仓库路径为 `docs/superpowers/specs/2026-10-08-search-e1-opsd-ablation-requirements.md`，Mac成品路径为 `/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/search-e1-opsd-ablation-requirements-20261008.md`。以下prompt已重复核心算法与验收，不依赖原聊天。若其内容与更晚人工指令冲突，按更晚人工指令处理并记录范围。

## 一、不能误解的最新决定

1. teacher与student均沿同一条D的历史预测D的下一token；**不换成B的prefix**。B只作为teacher额外的完整成功参考，student没有该参考。
2. **所有新组用reverse KL：学生→目标，即KL(p_student||q_teacher)或KL(p_student||m_mix)。旧E1源码计算的是forward KL，不能直接沿用。**
3. 混合方式是概率混合后做KL，alpha0.1乘teacher、0.9乘本outer冻结旧student。不移植完整TOPD的reward/future mean/whitening/PPO，不称为完整TOPD复现。
4. 正确B与五个sibling内均匀随机D配对，**允许D就是B**；D的正确性权重非负，正确高、错误低但非零。
5. 新改进组LR1e-6，另设LR5e-6的reverse基础对照。FP32base/FP32LoRA主参数＋FP16主要compute/activation及FP16推理是独立精度组。
6. 本设计将10iter定义为**10个在线outer**，每轮512 TRAIN题×5条新轨迹，再矿选、优化一次。不是10次Adam。每组全程直接继续，不做2/5轮测试后从0重跑。

## 二、实验矩阵与共同条件

| ID | LR | 配对 | 目标 | 精度 |
|---|---:|---|---|---|
| C0 | 5e-6 | 原B＋最远错误优先D | reverse KL，分量上裁剪10 | 原BF16 base/compute＋FP32 LoRA |
| C1 | 1e-6 | 同C0 | 同C0 | 同C0 |
| M | 1e-6 | 同C0 | reverse KL到0.1q＋0.9p_old | 同C0 |
| R | 1e-6 | 原B＋五条均匀随机D | 同C1，不加权 | 同C0 |
| W | 1e-6 | 同R | 同R，乘D正确性权重 | 同C0 |
| P | 1e-6 | 同C0 | 同C1 | FP32 master／FP16主要compute及采样权重 |
| J | 1e-6 | 同R | 混合reverse KL＋正确性权重 | 同P |

七组全部fresh reverse训练，共70outer、179200条训练采样轨迹上限。每组最多5120pair曝光；实际可用pairs和Adam独立计数。可按资源顺序执行，不能因无空卡占用或停止其他任务。

可附加L历史forward对照：只核验和评估已经存在的原在线前10outer，不新跑第八个forward训练。原历史末点预期Adam99／outer index9、3021pair，**不是Adam10**。必须实读末checkpoint及完整题序/配置/初始身份后才补评。L不可用就如实记缺失，不能替代C0，不能恢复或修改原在线作业。

全部新组同source100原始12文件、同一504 FP32 A/B初始LoRA模板（B全0、逐张量同SHA），fresh Adam，不恢复GRPO Adam或旧退化OPSD Adam。TRAIN169103题按原seed42确定顺序，预先冻结共同10个512题cohort，排除validation ID及精确问题文本交叉。每outer由本组最新student生成5条，不复用其他组后续采样或旧离线845515条冒充online。

固定：LoRA r16/alpha32/dropout0，q/k/v/o_proj、gate/up/down_proj；AdamW betas(.9,.999)、eps1e-8、weight_decay.01；global trajectory batch32、micro1、8rank默认replica、每rank满批累积4条；grad norm clip1、chunk32、无scheduler/warmup、seed42。K条可用pairs做一遍，Adam=ceil(K/32)，尾批实际人数；K0记录零更新且完成该outer。轨迹先有效token平均，再实际global轨迹平均，rank梯度SUM后不再除8。

生成：8worker每worker64题，原seed42–49；temperature1/top_p1，context8192、500tokens/turn、5turn、最多4search、top3、observation480tokens。保留原stop/forced/executor逻辑和获准同集群检索。C0/C1/M/R/W为原BF16 vLLM CPU-merge方式；P/J使用本组准确FP32base＋FP32LoRA CPU merge后一次转FP16，vLLM实际加载FP16。固定物理attention backend（参考原XFORMERS）、eager及明确utilization/cache设置；内存可改物理实现，但不改上述科学预算。若必须更换backend，先声明与记录数值差异，不能声称bitwise相同。

## 三、目标公式与梯度合同

令q_t=P_source100(v|问题,完整B,D_<t)，p_old,t=P_outer起始student(v|问题,D_<t)，p_theta,t=P当前student(v|问题,D_<t)。相同target token D_t，不按B/D两条轨迹的第t个位置做错位对齐。保留完整B；D=B时B含未来D属于允许的teacher特权参考，不删除B，但D continuation仍严格causal。检索文本保留为上下文，不直接参与policy loss。

非混合组a=q；M/J：

```text
logm = logaddexp(log(0.1)+logq, log(0.9)+logp_old)
a = stopgrad(m)
component_v = p_theta(v) * [logp_theta(v)-loga(v)]
raw_reverse_KL_t = sum_v component_v
post_loss_t = sum_v min(component_v, 10)
trajectory_loss_i = mean_{valid policy token t}(post_loss_t)
global_loss = sum_i(w_i * trajectory_loss_i) / actual_global_pair_count
```

无加权组w=1。全部词表，不只target token或top32。log-softmax／logmix／KL用FP32稳定计算，检查概率和与有限性。只upperclip每个词表分量、保留负分量，不floor0、不clip总KL、不做PPO概率比clip。权重在clip和轨迹平均之外，不能先乘分量再clip。

**p_theta系数与logp_theta均必须保留梯度；q、p_old、m全部冻结无梯度。** 若把exp(student_logp)detach，只让logp反传，未裁剪完整reverse KL在本forward的梯度会抵消为0。这必须有非对称分布测试抓出。未裁剪logit梯度应为：

```text
dL/dz_v = p_v * (log(p_v/a_v) - raw_reverse_KL)
```

它不是forward的p−a。用FP64参考、有限差分和autograd校验至少三组非对称分布、p=a、alpha0/1、clip触发及负分量。p=p_old时混合初始raw KL<=−log(.9)，但不保证后续p_theta仍满足该界。不要把旧forward“动态混入当前student只缩放0.1梯度”结论套到reverse。

p_old固定整个outer，在本组训练精度下未合并replay，身份绑定outer起始adapter；不是宣称vLLM采样时精确processed行为概率。恢复mid-outer时也保持原p_old，下一outer才刷新。不能使checkpoint重算时active adapter变成old/teacher；current forward到backward完成期间保持current权重及requires_grad稳定。可共享一份冻结base＋old/current adapter，不能24G上复制三份FP32base。

这是固定D历史上的全词表reverse-KL局部蒸馏，历史在每outer在线刷新。优化时没有对离散history分布反传，加上矿选/权重/token平均/clip，不能宣称完整sequence OPD或TOPD定理保证。原旧forward训练保持真实标签，不能倒写历史。

## 四、矿选及正确性权重

B仍按原core选：五条内reward1者中search_calls最少，其次token_count（缺失才用字符长度）；原first-tie规则。无成功B则整题不配对，全对仍保留。C0/C1/M/P的D仍失败优先、与B完整response normalized Levenshtein距离最大；全对用原fallback。

R/W/J的D从全部五个sibling均匀选，包含B，用seed42＋outer＋questionID的独立确定性哈希流；必须采用确定性rejection sampling避免mod5偏差，并保存哈希算法、编码、版本和拒绝次数，避免rank或调用顺序影响。记录sibling身份self-pair及文本相等分别判断。采样、controller resume和trainer三处使用同一versioned新miner，不能只改一处而其余仍强行等于旧core.mine_pairs。

reward只用冻结strict authored-final-action scorer：合法最终answer且EM正确为1，其余0。termination=answer或information/未闭合think中的答案不等于正确。不要新增评分漏洞或重复实现一份简化scorer。

W/J设计权重：

```text
mu=mean(r[5]); sigma=sample_std(r[5], correction=1)
A_j=(r_j-mu)/(sigma+1e-6) if sigma>0 else 0
w_j=5*softmax(A)_j  # temperature1
pair_weight=w_selected_D
```

五条平均权重1，strict positive；全对A0/w1继续训练，全错无B跳过。只有一条正确时正确约3.503、错误约.374。禁止负advantage乘KL、按minibatch二次归一化、用B的reward代替D权重，或把全对组丢弃。正确性通过曝光权重影响蒸馏，不是答案SFT或EM策略梯度；仍可能减少错误D纠错，需实测。

## 五、精度及恢复

P/J直接加载真实原FP32source100，base冻结、LoRA master/grad/Adam moments FP32。主要matmul/attention/activation FP16；norm/softmax/logsoftmax/KL/mix/必要累加FP32例外明确记录。teacher、oldstudent、currentstudent采用本组相同计算合同，不能暗中全FP32 teacher前向。autocast要覆盖head与checkpoint重算，实际hook embedding/residual/QKV/MLP/head及重算dtype，不能仅修改torch_dtype便宣称满足。

八rank同步GradScaler：scale→gradient SUM→unscale→全rankfinite→clip1→实际Adam。任意rank梯度溢出全体跳过，不计Adam、不推进committed pair cursor；降低后的scale和完整RNG存独立overflow/retry capsule，不覆盖同编号accepted CP。前向FP16非有限单独记录，GradScaler不能修复它。保持same run科学配置恢复，不暗中回退BF16、缩batch/context/clip或重置Adam。

完整CP包含初始/current LoRA、8rank Adam、Python/NumPy/Torch CPU-CUDA RNG、scaler、data/pair/order/cursor SHA、outer-old adapter、direction/precision/target version、实际update/exposure及日志清单。保护initial、update1/5、全部outer末、latest complete。新retention未获授权前不删除唯一资产或旧实验。真实更新要求参数finite且确实改变、八rank实际Adam一致；正常前1/5更新后继续原10outer，不从0重跑。

24G可行性未验证，八replica不分摊base内存。实测长context峰值、直接memory.free/UUID/PID，不以total−used代替actualfree。可在授权范围采用物理内存优化/已获准空闲48G卡，科学参数与八rank全局目标保持；资源不足先说明并排队，不能抢他人卡。

## 六、512评估与报告

主要评估：共同原BF16base＋FP32 unmerged各组raw第10outer adapter，冻结普通SearchAgent/HF SDPA greedy、原generation_config、repetition_penalty1.05/EOS[151645,151643]、相同8192/500/5turn4search/top3/obs480、同检索/scorer。512=NQ256+Hotpot256；logical rank r处理validation[r::8]64题、seed42，完整唯一覆盖。另有共同BF16 zero-update512。

FP16部署评估：同512、同greedy条件，FP16零更新source100、P10、J10，绑定实际FP16权重；必须与其FP16zero比较，不能减历史BF16239/512归因。P/J共同BF16评估与FP16部署结果分别报告。七组＋commonzero＋3项FP16共5632预测；若L完整可用再补512，上限6144。没有授权恢复旧七基准作业。

输出每outer TRAIN reward、CC/CW/allwrong/selfpair、P/K/skip、raw/post reverse loss、teacher/old/current/mix分歧、clip命中、tokens/exposure/weighted exposure、真实Adam/数值/显存。TRAIN reward来自不同题及更新前policy，不当固定heldout曲线。固定512输出逐题strict EM、NQ/Hotpot、协议invalid/合法错答案、搜索/零搜索、paired gained/lost和按题bootstrap区间，不把5sibling当独立样本。

分别分析C0→C1、C1→M、C1→R、R→W、C1→P及J组合；L→C0仅当身份与旧链证明充分。报告实际K/Adam差异，10outer不是相同optimizer预算。按“支持/削弱/未知”给结论，允许下降，不虚构未测根因。最终用raw末模型，不能best回退掩盖训练退化。

## 七、debug、身份、资源和发布

保留每组每outer全部五siblings、B/D、实际ordinary生成的上下文/rawIDs/stop/executor状态、检索query和完整HTTP文档ID/score/content/截断记录。全部有效token记录真实训练前向raw/post reverse loss、mask/offset、valid分母、w/global人数及最终贡献；checkpoint重算不能重复写。保存q/p_old/m/p_theta独立observed logprob/entropy/top32与target checkpoint。reverse裁剪命中集合分别记student mass、teacher mass、mixed mass，不能复用teacher字段装student质量。teacher-forcing与自主teacher generation分开。

top32 cap默认每组每outer8192有效位置，全部标量token loss不截断；少量全词表快照全study合计<=24GiB，预选规则、finite/shape/SHA和cap_skip保存。top32不能重建完整KL。JSON/JSONL另计空间。原题、完整trace/token/weights/env/凭据不Git；源码、入口、部署/恢复/评分脚本及必要config必须持久本地、提交、推送并远程文件SHA读回。

每组每outer另从有B的pair固定seed/hash分层选最多8题，覆盖CC/CW/self-pair；冻结teacher＋完整B自主greedy生成，保留其实际完整推理/HTTP/IDs/strict EM并关联原B/D。诊断RNG隔离，不改变训练采样或pair；最多560条额外teacher诊断，不计训练或512分母。标签为diagnostic_teacher_autoregressive=true、original_training_objective=false；teacher自己历史不等于训练原D history的q。无成功B的全错题不补造teacher参考。

仅HKUST集群，禁止跨集群data/index/model/存储通信。用户指定remote为shaoshitong/Search-R1，既有源码目标search-e1；独立worktree/branch准备，不切换或覆盖其他会话checkout，不force-push，不通知原七算法线程。

启动前实查当前获准node/UID/UUID/PID/starttick、所有资源锁、完整子进程释放、检索准入/actualfree、输出持久条件/可写空间/释放日期。原K1 online、K4 archive及他人进程不能自动停止或热改。旧的“停掉某次作业”不能扩为新对象授权。共享存储按用户memory；新实验使用本地HKUST例外须确认明确覆盖本goal，不能沿用原run许可或以超时视为批准。缺输入期间推进独立CPU实现/审查/备份，不启动依赖作业。

定位线索（重新SHA核验，不当当前准入）：

```text
K1 source/data/env root:
/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007
model: assets/archive100
train: data/train.jsonl  SHA a334d768230fd85d3465e251aee906060150628775b78ab7ce15e5e789f7489f
validation: data/validation.jsonl SHA 75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76
python: env/bin/python
305manifest: receipts/preparation-3-runtime.sha256.json
SHA 3c05be8e2ae52278eaa7f28f40f4fc07a87c847fd67a7bb2217777f4f78467bc

K4 source:
/weka_3/shaoshitong/search-e1-config-study-20261004/runs/archive/round-1-grpo/actor/global_step_100

Old online (只读，不接管):
/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008
config SHA d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17
legacy end candidate: runs/checkpoints/update-00000099-outer-00009

Repo reference interfaces:
experiments/search_e1/control/opsd_online
experiments/search_e1/control/opsd_debug_frozen_runtime_20261008/runtime
experiments/search_e1/control/opsd_online_results_eval_v1
experiments/search_e1/control/opsd_native512_eval_v2
```

旧18004/18005端口只作服务定位，当前PID/路线/准入必须实际重新读取。保持完整FP32索引21015324×768、原CPU FP32 E5 batch128/query64/vector32768/TF32false合同；服务变更需真实完整score/ID/shape、HTTP、并发、显存gate，不把端口健康当全部通过。

## 八、执行顺序及交付

1. 读完整需求与实际源码/manifest，列需求追踪、冻结config/variant表、共同初始adapter/cohort和所有待输入依赖。
2. 在独立副本实现并审查reverse方向与梯度、矿选/weight、old-target冻结、mask/尾批/rank缩放、精度/scaler/恢复和不改变原返回对象的日志hook。不要直接跑旧helper：它目前硬编码LR5e-6、BF16及forward KL。
3. CPU必要fixtures通过后，源码本地持久/Git推送/远程SHA读回；资源和存储实查通过才启动新GPU。
4. 七组独立run完成10outer；前1/5实际更新验收后同run继续；故障保留原日志/源/CP，在授权内恢复完整状态，不能重置Adam、重采样已提交prefix或从0冒充续算。
5. 完成全部主要512及FP16zero/P/J；可用L补评；私有资产同步并逐清单SHA验证。
6. 交付中文消融报告、逐轮曲线与真实计数、逐题512和paired变化、debug索引、准确config/源码remoteSHA、node/run及所有CP/日志/报告绝对路径、完整备份/未覆盖项。上述完整后才把执行goal标记complete。

不要为“快速过验收”缩为小样本、不报告原始末权重、或用旧结果填新reverse组。没有提升也要交完整真实结果；未完成或未核验必须明确标注。

方向依据：[TOP-D §2.1式(2)](https://arxiv.org/html/2607.04751v1#S2.SS1)定义OPD为reverse KL。这里明确保留逐token full-vocab、筛选/权重/clip/trajectory mean的实验差异；不直接套用完整序列策略梯度的等价或保证。
