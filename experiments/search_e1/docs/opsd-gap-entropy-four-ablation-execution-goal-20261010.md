# 待发送执行goal：四卡48G、四消融各25轮

2026-10-10，Asia/Shanghai。本文件是用户可重新发送的执行指令，制定时不创建活动goal、不启动训练。收到用户发送本指令并要求执行后，按以下范围推进到验收完成。

## 可直接发送的执行指令

创建并执行以下goal：在HKUST4原Search-E1使用的四张48G GPU4–7上，完成原计划的四个在线OPSD权重消融，每组25个outer iteration，共100轮；每轮记录完整训练集EM，并在每组第15和25轮训练结束后计算固定512题测试集EM。持续推进直到四组及评估、记录、源码发布和远端产物保留均完成，不以初始化或后台启动为完成。

详细计划：`/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/opsd-gap-entropy-four-weight-ablation-plan-20261009.md`；数学审核：同目录 `opsd-gap-entropy-four-weight-ablation-plan-review-20261010.md`。本goal的15/25评测时点优先于旧每5轮安排。直接使用当前会话权限执行，不走旧账号中转链。

### 1. 范围与共同基线

- 仅使用HKUST资源；四卡共同训练、四组依次运行，顺序G_POS、G_NEG、H_POS、H_NEG。实际启动前重新核GPU UUID/UID、ownership、共享检索和显存；旧资源快照不替代本次验收。不碰GPU0–3他人作业或HKUST1现有七组。
- 共同起点为已验收archive Search-R1 GRPO checkpoint100原12文件权重及共同零B LoRA，各组fresh Adam；不恢复GRPO Adam，也不接旧退化OPSD/其他组末权重。旧archive已经按人工停止，禁止自动重启。
- 执行基线合同明确采用原计划的R式候选：固定source100特权教师，原成功B规则，五兄弟均匀选D（含B），teacher reverse KL分量上限10，不加正确性权重、不混合old目标；LR1e−6 constant、LoRA16/32/dropout0、BF16 base/compute及FP32 LoRA、global trajectory batch32/micro1、normclip1/chunk32、seed42及原AdamW设置。此前E名称不作为额外算法参数。若实际原始源合同与这些字段冲突，报告具体差异，不猜测或修改科学设置。
- 每outer固定512道TRAIN题×5条最新raw学生fresh轨迹，seed42完整排列前12800题、首10批与原cohort一致；完整生成/检索预算按计划不变。八逻辑worker、seed42–49映射四物理卡两波，训练四rank满批各累积8条、尾批实际分母，梯度SUM不额外除world。

### 2. 四种消融保持原计划

| 组 | 信号x | 方向 |
| --- | --- | --- |
| G_POS | 答案有效token平均signed logp_student−logp_teacher | 较大者权重高 |
| G_NEG | 同一signed gap | 较小者权重高 |
| H_POS | 全有效policy token平均full-vocab student entropy | 高熵者权重高 |
| H_NEG | 同一entropy | 低熵者权重高 |

每次真实Adam前以当前学生、同一个成功B对本batch每题全部五兄弟无梯度评分。对可评分项减均值μ，再除共同 `max(max|x−μ|,1e−6)` 得r；NA令r=0。权重为 `w=1±0.9r`，范围.1–1.9、整组均值1；保留原NA/整题编码中性fallback及非有限故障规则。

权重detach，放在原词表分量clip及轨迹内有效token均值之外，最后除实际global pair数；仅训练原selected D，不重选B/D，不训练全部五条，不除Σw/5/world，不改η为1或秩归一化。allwrong保持原skip，实际K=0则记录0 Adam，不凑更新。

### 3. 每轮TRAIN EM：必须完整、可复核

每组每轮按该轮最新raw学生实际生成的全部2560条轨迹计算原严格final-action EM：

`train_em = 严格正确轨迹数 / 2560 = mean_512_questions(s/5)`。

包括all5_wrong、未被选中兄弟、协议无效和无答案轨迹；后两者按冻结scorer结果计0。不能按成功参考资格、mined pairs或usable K过滤。必须核512题×5槽唯一完整，partial不能当一轮完成。它是本轮采样起点权重的更新前TRAIN表现，不冒充该轮训练后测试EM；每轮题目不同，不能只凭曲线起伏认定退化。

将每轮的variant、显示iter1–25/outer index0–24、采样模型SHA、正确数/2560及EM%、各数据源实际分母、allwrong、P/K、实际本轮/累计Adam、未加权/加权loss、selected-weight/ESS、clip/grad、训练后raw checkpoint绝对路径与SHA、阶段/时间写成小型汇总。训练原件与逐token证据保留远端；每组最终必须有25条完整、连续、唯一的轮次记录。

### 4. 第15/25轮512测试EM

- 第15和25轮指各组真实完成15/25个训练outer，对应索引14/24的末raw checkpoint；不是累计Adam15/25，也不是采样前cache。原子提交完整四rank状态、训练child真实exit/释放后，隔离评估child再运行，完成后核训练状态未变并继续第16轮。
- 使用同一512题（NQ256/Hotpot256）、普通提示、原HF SDPA/BF16 base＋未合并FP32 LoRA、greedy/repetition1.05、原EOS及8192/500/5turn/4search/top3/obs480、冻结严格scorer和原八逻辑分片。保留逐题响应/生成IDs/检索证据及完整唯一覆盖，不用best adapter。
- 每组产出15/25各一次完整512结果：hits/512、EM%、NQ/Hotpot、相对共同起点及15→25的gained/lost/preserved/stillwrong；协议无效与协议有效但gold不匹配分开。第25轮为主终点，不挑较高点替代它。
- 共同source100零更新基线按同四卡协议评一次；已有完全同条件且身份/逐题输出验收者可复用。四组正式测试共4096条，加一次共同基线为4608条；不执行旧5/10/20的正式512测试，不自动增加pass@5、E25控制或其他seed训练。
- 两个终态主比较G_POS−G_NEG、H_POS−H_NEG按原计划题级配对/分层区间及Holm规则报告；512是复用的TRAIN外验证集，不把无E25的一次结果解释为加权必优于不加权E25或跨seed泛化。

### 5. 实现、debug、恢复与存储

- 先在独立新源码/目录完成四rank、25outer、全五兄弟scorer、HKUST4 UID/路径/检索准入/锁/两波worker和15/25评估恢复的适配及数值检查；先Git提交/push search-e1并逐文件远程回读后部署，不热改其他任务。不得把现有14项标量数学测试当生产验证。
- 同一正式run首outer与第1/5真实Adam核有限loss/adapter、实际权重变化、四rank参数/Adam逐项一致，各rank RNG完整保存且恢复与本rank检查点一致（不同rank RNG不要求逐字节相同），以及显存直接余量与已获准检索共存；通过即连续跑，不做测试后从0重跑。25 iter是25个采样训练outer，真实Adam为Σceil(K/32)，分别记账。
- 保留实际训练forward的所有有效token raw/post KL、双方logprob/entropy、clip、mask/offset/区域、权重/模型/更新身份，以及五兄弟score/μ/d/r/w、轨迹/HTTP/生成IDs。沿原top32/抽样full-vocab cap，明确完整标量记录与抽样分布区别；不把posthoc重算伪称在线loss。
- 每outer完整raw adapter/四rank Adam/Python/NumPy/TorchCPU-CUDA RNG/order/cursor及score manifest按原计划保存。故障先核真实身份、已提交更新和现场，在独立修复副本恢复未完成部分，保留失败日志/原码，不停他人、不重复已提交Adam、不回滚到初始模型。
- Mac仅保留代码、配置、文档和小型审计/聚合结果；weights/checkpoint/tensor/optimizer、raw rollout/token/HTTP、完整测试输出、环境/cache全部在HKUST真实远端路径。启动前核realpath/挂载/权限/容量及保留/远端备份，不把大产物回Mac，不跨集群，不释放唯一未持久保留资产。
- 每次修复同步持久源码、commit/push及远程文件SHA，最终交付配置/源commit/远程SHA、真实run/日志、所有checkpoint和评测报告绝对路径、四组训练EM曲线及15/25测试表。

### 6. 完成标准

四组各25个完整outer（共100）、各25条TRAIN EM、真实Adam与P/K/cursor链对账、最终四rank参数/Adam逐项一致且有限、各rank完整状态/RNG可恢复；八个完整512测试及共同基线身份验收；逐题配对结果与权重/loss/概率关系分析、必要debug证据远端保留；最新实现已推送且逐文件SHA通过。到此才标goal complete。

后台初始化、首五Adam健康通过、只跑15轮、仅有TRAIN EM或缺512分片均不算完成。正常进展简洁汇总，重大故障/恢复及新测试结果及时报告；遇到必要输入依赖明确说明，同时推进不依赖该输入的工作，不以一次巡检结束目标。
