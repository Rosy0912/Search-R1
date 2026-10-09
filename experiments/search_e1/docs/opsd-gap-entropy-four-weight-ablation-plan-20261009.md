# OPSD 答案分歧／学生熵加权：四组各25轮消融计划

2026-10-09。任务来源：用户要求查看 `01a1028b-b626-7f91-93db-0099ff1623c5`，在其“E方法”上增加答案 student−teacher logprob 差及 student entropy 的正／负相关权重，四种形式各跑25 iter；允许停止HKUST4旧的性能崩塌Search-E1，使用其四张48G卡。**本文件是数学与代码执行计划；新训练尚未启动。** 没有使用 brainstorming。

**2026-10-10数学审核修订。** 用户最新明确将基线/关联等作为成立的假设，本次核心审核是权重、0–2范围、平均1/无偏、梯度与归一化顺序。结论：当前中心化公式数学自洽；它对任意有限group已经严格平均1，不需要等待group趋于无穷。数学通过不代替生产实现或GPU验收。本次只读本地代码和规则，不判断当前远端作业状态。完整证明与边界见 [数学审核](opsd-gap-entropy-four-weight-ablation-plan-review-20261010.md)。原始版本SHA `c8fa402de0a128417e1ae027abb7f4af67658ee19702026e091b14e42bc027ed` 已另存。

## 1. 先绑定基线，避免做错实验

实读指定线程、已发布矩阵与 `opsd_ablation_v1/miner_v1.py`，实际组名是 C0/C1/M/R/W/P/J，没有E。已完整评估的BF16组为：共同零更新239/512、C0=227、C1=236、M=237、R=240。R是用户所述“唯一没有下降”的候选，但其净增1题的配对区间跨0，不能称可靠提升。已向用户一次性核对E是否指R；回复前不把E强行映射，也不启动依赖此身份的新训练。

正负权重的数学可复用，但具体训练目标须绑定E；**具体配置以R作为条件候选**：LR1e−6、固定source100教师、teacher reverse KL、词表分量上限10、BF16 base/compute＋FP32 LoRA、每题从全部五兄弟均匀选D（包含B），不加正确性权重、不混合old目标。若E另有混合／既有轨迹权重，先重写精确基线及“新增权重是替换还是相乘”，不能假称乘积仍满足本计划的均值与范围。按本次数学审核，假定基线合同和上述条件已成立；这不构成生产代码已落实的证据。

证据：[R完整512同题结果](opsd-ablation-R-native512-paired-results-20261009.md)、[前五轮EM/loss关系](opsd-first5-em-loss-relationship-findings-20261009.md)。R的10轮实际3030 pair／99 Adam，不是10次Adam。

## 2. 要回答的问题和实验矩阵

已有关系来自固定学生历史的**更新前forward**：答案signed gap越大，成功比例越低；全policy学生熵越大，成功比例通常越低。但“能预测失败”不等于“给它更大梯度有益”。分歧大可能值得纠错，也可能意味着教师不适用；低熵可能是可靠行为，也可能是自信地错误。因此四个方向都测，预先固定，不根据中途EM选方向／幅度。

原关联还有矿选边界：C0/C1/M失败优先，4,529个有B pair实测 `D正确⇔五兄弟全正确`；它不是随机D的无偏判别。按用户最新指令可将新分布中的方向关系作为实验假设，但不能把旧AUC写成对随机D的已证事实。

另一个核心边界是：已有表主要反映跨题关系，而本计划题内中心化会去掉题间平均分差。**跨题相关不保证同一题五兄弟内也同方向相关**，也不能称本方案直接给“整体更难的题”更大平均权重。首个正式outer起就记录全部五兄弟分数与结果，用题内比较／题级block区间检查方向是否成立；它是机制诊断，不据此临时改四组定义或只保留看起来好的方向。

| 新ID | 评分 | 权重方向 | 要区分的机制 | 训练预算 |
| --- | --- | --- | --- | --- |
| G_POS | 答案平均logp_student−logp_teacher | signed gap较大者权重高 | 更强纠正学生与教师不一致的行为，还是放大不适用监督 | 25在线outer |
| G_NEG | 同上，同一signed定义 | signed gap较小者权重高 | 强化教师较认可的行为能否保留EM，还是只重塑已会题 | 25在线outer |
| H_POS | 全有效policy token平均学生熵 | 高熵者权重高 | 给不确定轨迹更多学习机会，还是强化噪声 | 25在线outer |
| H_NEG | 同上 | 低熵者权重高 | 强化较确定轨迹能否稳定策略，还是放大自信错误 | 25在线outer |

四组共100 outer、256000条训练生成轨迹上限、最多51200条selected-D pair曝光。**每组25 outer=25次512题×5的刷新，不是25次Adam。** 实际Adam为每轮 `ceil(K/32)` 的和；不为凑次数补题／删尾批。

四组从同一个archive GRPO100原始权重＋共同零B LoRA fresh开始，各自fresh Adam；不是从R10／旧退化OPSD继续。这样检验的是“在E的训练方法上加入权重”，不是把权重和不同起点混在一起。每组各自连续25轮；不做五轮测试后从零重跑。

建议有同机同预算的无新增权重 `E_25` 控制组。若已有完全匹配的E25可直接复用；否则另需25 outer，**不自动增加到本次四组100轮预算**。旧E/R10只作10轮时点参考，且注明八卡／四卡数值执行差异；没有E25时可以比较四个方向及相对共同起点的EM，不能声称证明“加权优于不加权E25”。

## 3. 两个评分的精确定义

设题q有五条本轮真实采样轨迹 `D_qj, j=0..4`，B仍按E的原规则选取成功参考，D仍按E的原规则选取。新增评分**不改变B、不重选D、不把五条都变成训练样本**。

每个Adam minibatch开始、参数为θ_k时，对该批题的全部五条兄弟无梯度teacher forcing。普通学生是当前 `p_θk(v | question, D_qj,<t)`，教师是固定source100的 `q(v | question, complete B_q, D_qj,<t)`。两者预测同一D的同一实际token，各自使用同一D历史，不做B/D位置配对；B是教师的完整特权参考，不截去其答案。q若不是E原teacher则必须先修订基线合同。

### 3.1 答案signed gap

\[
g_{qj}=\frac1{|A_{qj}|}\sum_{t\in A_{qj}}
\left[\log p_{\theta_k,t}(y_t)-\log q_t(y_t)\right].
\]

自然对数，按答案有效token平均，不取绝对值，不替换成全轨迹KL，也不把负值裁成0。大gap表示学生比教师更相信**这条轨迹实际写出的答案词**；不是答案正确概率。

复用既有分析的 `character_regions/category`：最后一个authored policy span内最后闭合 `<answer>…</answer>` 的纯文本payload；只取属于final_answer且policy mask有效的token，跨tag/payload边界token仍属boundary。**这一定义不要求strict final action有效，也不使用gold来筛答案**；协议无效但payload存在者仍可评分。无payload或无有效答案token记NA，不能填logp=0。旧数据已有strict正确但答案区域无有效token的个例，二者不可混用。

### 3.2 学生entropy

\[
h_{qj}=\frac1{T_{qj}}\sum_{t\in\mathrm{policy}_{qj}}
\left[-\sum_v p_{\theta_k,t}(v)\log p_{\theta_k,t}(v)\right].
\]

沿用原关联分析的全有效policy范围；观测／外部forced内容不进policy mask。使用FP32全词表归一化，单位nats；不是top32 entropy，也不是实际token的 `−logp`。本次不同时试“仅答案entropy”或额外熵奖励。

## 4. 五兄弟内归一到0–2，并保持均值1

对一个指标x，令V为该题可评分的兄弟；缺失者保持中性权重1。固定：

\[
\mu=\frac1{|V|}\sum_{j\in V}x_j,\qquad
d=\max_{j\in V}|x_j-\mu|,\qquad
r_j=\begin{cases}(x_j-\mu)/\max(d,10^{-6}),&j\in V\\0,&j\notin V.\end{cases}
\]

\[
w_j^{+}=1+0.9r_j,\qquad w_j^{-}=1-0.9r_j.
\]

可评分者少于两条时整组w=1；相同分数自然w=1。数值floor固定1e−6，避免极小FP32差异放大，四组不扫描。实现先对一个anchor作差再中心化，减少共同大偏置的浮点消减误差。

由此五兄弟平均权重为1，`0.1≤w≤1.9⊂[0,2]`，同一模型、同一五个分数下正负方向逐项相加为2。固定幅度0.9是本计划的设计选择，不是用户逐字指定；保持严格正权重并兼容现有objective，避免w=0时纯momentum/weight-decay更新的额外语义。最大相对权重可达19倍，是明确的较强干预，不能称微小扰动。

### 4.1 任意有限group的平均1与均匀选D的无偏性

推广到group大小G≥1，有效评分数m=|V|；m<2时全为1。否则令 `c=max(d,ε)`、`r_j=(x_j−μ)/c`，NA的r为0。对任意有限分数集合，`Σ_j r_j=(Σ_{j∈V}x_j−mμ)/c=0`，所以对正负任意方向与固定 `0≤η≤1`：

\[
\frac1G\sum_{j=1}^{G}(1\pm\eta r_j)=1,\qquad
0\le1-\eta\le w_j\le1+\eta\le2.
\]

这是有限group的确定性恒等式，不依赖分数正态、对称、独立或G足够大。故G→∞时group平均仍是1。在同一固定模型/评分下，若原D独立均匀取自G个兄弟，则 `E[w_D | 当前模型及全部分数]=Σw/G=1`，任意有限G就无偏到1。若D是失败优先/最远/依分数选择，则这个selected-D期望结论不适用；题内平均1仍成立。

这里“0–2”指落在区间内，不是要求每个group必同时取到0和2。主计划保留η=.9，因此范围.1–1.9；设η=1可得到0–2范围且仍平均1，但不在本次审核中悄改四组幅度。零权重数学上合法；若将来采用η=1，原objective的严格正权重检查需调整，零权重pair仍计实际分母、不得删样；Adam历史/decay不因此自动停止。

直接min-max虽把最小/最大映射0/2，其平均为 `2(μ−min)/(max−min)`，只有μ恰好为极值中点才平均1。对非对称分布，G→∞一般也不趋于1，不能替代当前公式。普通均值归一化虽可平均1，却不保证上界2。当前形式同时保证方向、范围和平均。

### 4.2 大group的极限边界及可选秩形式

平均1不等于大group时权重仍有明显差异：gap可无界，若G增大使maxabs偏差趋于无穷，普通样本的r可能趋近0，权重接近1；entropy有上界log(vocab_size)，情形不同。本实验固定G=5，所以不把“大G无偏”解释为“大G始终有同等强的干预”。floor1e−6仅防分母过小，不保证BF16前向排序没有噪声。

如果后续更强调group无限增大时保持非平庸的排序权重，可用有效兄弟的**平均秩**R（并列采用midrank），m≥2时 `w⁺=2(R−1)/(m−1)`、`w⁻=2−w⁺`，NA=1、m<2全1。因为平均秩和始终m(m+1)/2，该形式任意有限G也平均1，且范围0–2；若m→∞且有效分数的经验分布收敛到固定F（例如固定分布iid），连续分布极限为2F(X)，有原子分布定义 `F_mid(x)=F(x−)+Pr(X=x)/2`，则 `E[2F_mid(X)]=1`。任意变化/相关的group不自动有这个指定CDF极限，但不影响有限均值恒等式。它仅保排序、不保原分差大小，属于**另一归一化定义**；本次只审核说明，不加入新实验组或替换主计划。两端均达到0/2、均值1且线性保分差三要求一般不能同时满足。

例如分数为 `[0,1,2,3,4]`，正向权重 `[0.1,0.55,1,1.45,1.9]`，负向反序。不采用直接min-max：它一般不保均值1；不采用 `5*softmax`：它可能超过2。

**边界规则：** 附加评分若有任一兄弟的student／teacher联合上下文超过原8192、空response／空policy mask，整题新增权重全部中性，保存原因；不截断、不改选、不改变原selected-D的K资格。答案NA只在其余有payload的兄弟内中心化，NA自身w=1。非有限logprob／entropy是数值故障，保存现场并恢复独立副本，不能用中性fallback掩盖。无成功B者沿用原allwrong skip，不虚造参考；因此本干预没有解决全五次失败题无监督的问题。

## 5. 训练loss、detach和真正的动态时点

若基线确认为R，原每token目标为：

\[
\ell^E_{qt}=\sum_v\min\{p_{\theta,t}(v)[\log p_{\theta,t}(v)-\log q_t(v)],10\}.
\]

保留signed词表分量，只作上裁剪10；不把总KL clip到10，不加floor0。teacher冻结；学生p及logp均有梯度。

新的唯一优化改动为：

\[
L_k=\frac1{B_{\mathrm{actual}}}\sum_{q\in\mathcal B_k}
\operatorname{stopgrad}(w_{q,D_q})
\left[\frac1{T_{q,D_q}}\sum_{t\in\mathrm{policy}_{q,D_q}}\ell^E_{qt}\right].
\]

即词表分量裁剪→轨迹内有效token均值→乘原selected-D的权重→除实际global pair数。**不除以Σw、不额外除5或world_size。** 这是整条轨迹权重；即使评分只来自答案，也没有单独增加答案token在轨迹内部的相对权重。

权重以Python finite scalar／detached tensor送入目标，不对w反传；否则出现 `L∇w`，已经变成不同的entropy/logprob优化目标。每个batch的五兄弟评分全部结束后才建立selected-D训练图；不能在backward之前切PEFT adapter。

具体时序：

```text
outer r: 最新raw学生生成512×5 → 原规则选B/D → 原编码确定K/order
for 原顺序每个实际global minibatch(≤32):
    固定当前θ_k；核四rank参数身份一致
    no_grad + 原精度合同：全部本批题×五兄弟，chunk32流式评分
    评分期间不得optimizer.step；还原current adapter/mode/flags/RNG
    计算五项w，绑定θ_k、B/五条原row/mask/score SHA并持久化
    原selected-D forward，保存实际更新前全部token指标
    Σ detached(w_D) * 原trajectory_loss / B_actual
    四rank梯度SUM → 原normclip1 → 一次真实Adam → 完整状态提交
下一batch重新评分；下一outer使用本组最新raw学生重新采样
```

本计划选择**每次Adam前的当前学生**，以对应已分析的更新前分布；不是冻结outer-old来打权重，也不是vLLM生成时经过processor的概率。batch顺序因此属于定义的一部分，必须固定并记录；四组均采用这一时点，不同时比较两种时点。teacher forcing分布与自主生成processed scores分开标注。

评分无采样、dropout保持原0、不消费生产RNG；前后核Python／NumPy／TorchCPU-CUDA RNG及模式。恢复时保存本batch评分产物与preupdate CP/cursor，不能让θ变化后沿用旧评分。失败attempt保留，新attempt与已提交更新分清，不从0重跑。

### 5.1 均值1的边界：仍须测梯度尺度

题内五兄弟均值1，**不保证实际选中D的batch均值1**。即使D叫“uniform”，只有将模型和五分数视为固定且D独立均匀时，才有条件期望1；不能据此宣称当前在线训练完全无偏或学习率混杂已消除。不同权重也会改变梯度方向、裁剪及Adam历史。

更精确地，设未加权每轨迹梯度a_j=∇L_j，`w_j=1+sηr_j`（s=±1，训练时detach）。独立均匀D时：

\[
E[w_Da_D]=\bar a+s\eta\frac1G\sum_jr_j a_j
=\bar a+s\eta\operatorname{Cov}_{\rm group}(r,a).
\]

因此无偏到1指**权重均值/质量**；加权梯度有意加入分数–梯度协方差，一般不是原未加权梯度的无偏估计。loss同样有 `E[w_DL_D]=平均L+sηCov(r,L)`。梯度协方差为0只恢复该step未裁剪半梯度的条件均值，loss协方差为0只恢复该step条件平均loss；二者不等价，也不能保证梯度方差、normclip或Adam历史/更新量不变。不能将“权重均值1”解释为恢复整个原训练过程。

不另做batch归一化来掩盖这一现象；每batch保存selected-weight mean/sum/sumsq/min/max、`ESS=(Σw)^2/Σw²`、clip前后grad norm／clip factor、实际Adam参数变化，以及pair／token／加权曝光。若收益伴随持续不同的权重尺度，只能把它视为该加权方案整体效果；分离尺度因果需要后续单独控制，不在四组中悄加第五种变换。

## 6. 固定设置、四卡适配和算力安排

候选R合同：source100原12文件、共同504个FP32 LoRA张量、LoRA r16/alpha32/dropout0；AdamW LR1e−6、betas(.9,.999)、eps1e−8、decay.01；global trajectory batch32/micro1、normclip1/chunk32、无warmup/scheduler；trajectory mean。四rank满批每rank累积8条，尾批仍实际人数，梯度SUM后不再除4。

每组固定seed42的TRAIN完整原始排列前12800题，25批×512；先核与512 heldout ID及题文无重叠；不偷偷跳题／替换。首10批应与已有R/C1等cohort逐SHA相同。所有组同outer题与顺序一致，但随模型更新各自生成五条；不同后期文本不能强行共用或按token位置硬配对。

采样与原合同一致：8个**逻辑**worker，每个64题、seed42–49、temperature1/top_p1、context8192、500token/turn、5turn/4search/top3/obs480、原stop/forced/executor、BF16 merge／XFORMERS/eager。四物理卡分两波逻辑worker，避免一张卡同时载入两套vLLM；逻辑分片及seed不改。物理排队与8卡并发可能改变批处理数值，不声称bitwise复现；正式无权重控制若补跑也使用四卡同一执行链。

五兄弟评分只对有B的题进行：约需5K条附加无梯度replay，另有原K条训练。gap与entropy的评分路径统一流式算全policy head，均只保存标量；不让两类组因不同优化实现引入额外精度差。成本不能称“五倍训练”；先在同run首outer实测生成／检索／评分／backward／保存秒数与峰值，不编造完成时刻。

### 6.1 资源实查及停止范围

2026-10-09 16:22–16:25 CST，本机无代理实连HKUST4=`xielab5`、UID1001。物理GPU4–7每卡49140MiB，旧archive worker各占约24822MiB，GPU7另有检索1749337约738MiB；并非已空。旧组已完成GRPO199、进入200，最新TRAIN reward/strict EM/finish ratio为0，是性能崩塌，非已观察到进程退出。最新持久actor checkpoint当时仍150；不要把199日志冒充199权重或Adam。

用户本次最新人工“那个search-e1已经崩溃可以停掉了”覆盖旧heartbeat的K4继续安排，仅限archive run `f04fe593830c4831918ede872812c016` 及其已核验supervisor／子进程。原pipeline会自动进入后续OPSD，故停止须覆盖supervisor；按精确PID/UID/starttick/run与pidfd执行，保存真实停止票据。原目录／日志／checkpoint／冻结runtime保留；**检索1749337、两个observer及GPU0–3他人作业不在停止范围**。停止后的实际GPU与checkpoint状态另附receipt；本段是此前只读快照。

**17:05:42 CST 实际释放完成：** 原run188个历史身份已无活跃进程（ABSENT_OR_ZOMBIE），GPU4–6直接free各48749MiB，GPU7 free47995MiB／仅保留738MiB原检索；两个原observer也保留。4449个证据副本共91891952bytes独立fresh SHA通过。最新完整actor仍150；50/100/150原36文件保留，不声称199的Adam/RNG已捕获。原停止脚本因进程退出瞬间argv变化留下rc2/HOLD；独立只读终态已真实rc0/accepted，旧HOLD未改写。详见 [实际停止与证据](opsd-signal-weight-archive-stop-20261009.md)。本次人工停止取代该旧run的“继续K4”历史安排，禁止自动重启它；K1七组及检索不受此停止范围影响。

新任务拟用 `/weka_3/shaoshitong/search-e1-opsd-signal-weight-4gpu-20261009` 独立目录、四组独立run/config/tmux/log。四卡合用顺序跑四组，不分成四个单卡组。资源租约绑定实际GPU UUID，启动前确认旧run已停止、自有GPU4–7没有外来compute、真实memory.free及retriever准入。公共检索按已验收全FP32索引／CPU FP32 E5 batch128／query64／TF32false，路由不改；若沿用K4 GPU7共享检索，保持进程≤1GiB、实际运行余量≥6GiB，新的48G worker utilization/cache仅按物理内存验收修改，不能改上下文/生成科学预算。HKUST1的既有七组消融继续，不等待其备份或抢占其GPU。

2026-10-09最新存储规则覆盖旧备份安排：Mac仅留代码、配置、必要文档和小型审计摘要；checkpoint/LoRA/tensor/optimizer、原始rollout/HTTP/token、完整评测、环境/cache和全词表NPY均保存在HKUST远端，压缩不构成回Mac长期保留的例外。启动前核output/checkpoint/cache真实远端路径、realpath/挂载/权限/容量及远端保留/备份；不虚称同节点两个目录是独立故障域。`/weka_3`约8.2TiB只是旧快照，须重读；本次不操作或删除其他任务资产。不得释放无远端持久副本的唯一资产。HKUST节点本地盘例外保持，禁止跨集群。

## 7. 代码实施清单：独立版本，不热改七组

纯CPU权重参考与测试已随本计划提供：`control/opsd_signal_weight_plan_v1/weights_reference.py`；它**不是已经可启动的训练controller**。生产实现应进入独立 `opsd_signal_weight_v1`，冻结现有source/runtime，发布后再部署。

部署必须绑定已确认E的原始已发布helper、完整冻结runtime及环境清单；若基线是当前R，使用其原305文件runtime，不凭HKUST4已存在的 `opsd-runtime/archive` 目录名就认定同版本。实读Torch／PEFT／Transformers／vLLM／tokenizer与评分模块版本、精度、源commit和文件SHA；环境差异在独立副本做兼容与数值检查，不升级共享环境，也不把四卡迁移和权重干预混作同一项科学变化。

| 模块 | 必须落实的改动与核验 |
| --- | --- |
| prepare/config | 新schema，4个variant及显式base-E身份、outer25、score timing current_preupdate、region/dtype/version、η.9/floor1e−6、四UUID/UID；拒绝借旧八卡10轮配置启动 |
| cohort/miner | 原cohort完整shuffle扩25轮；miner去除outer<10硬编码但保持B和D算法／SHA随机流首10轮一致；score不得改D。新增版本与末轮24验收 |
| score_group | 原encoded_pair/offset/mask复用，teacher关闭adapter、student当前adapter；先评分全部5、chunk32流式全词表FP32指标；输出原row/θ/B/offset/NA/fallback/score/w身份与RNG审计 |
| objective/trainer | 复用原 `pair_objective(pair_weight=…)` 正权重入口，权重置于clip与trajectory mean之外；actual global denominator及SUM不变；score lease在current_graph之前，不能图内切adapter |
| checkpoint/retry | trainer/checkpoint/state_storage/retry_state内所有八rank硬编码改新四rank合同；四份Adam/RNG/order/cursor/θk-score/inputhash完整保存与实际CPU实读。不能用八rankreceipt冒充四rank |
| sampling/controller | 原8逻辑worker映射4物理卡两波；nproc_per_node4、worldsize4、outer24自然结束；sampling/teacher_diag等outer<10限制统一新版本；child真实exit与释放对账 |
| memory/ownership | 新四卡observer，核实际UID/UUID/PID/starttick/run；只允许已获准GPU7检索共享；其他卡现有作业不构成错误，也不能误停 |
| evaluation | 普通greedy原native512、8逻辑rank `[rank::8]` 各64映射四物理卡；新25轮及各保存时点身份，不改成rank::4后假称原协议 |
| debug/analysis | 每有效训练token及所有五兄弟score记录可重构w与实际weighted loss；完整实际轨迹/HTTP/IDs保留；分析按题block、dataset/outer分层，不拿TRAIN loss预测总体EM |

已实读硬编码示例：`miner_v1.py` outer0..9；`cohort_v1.py` first5120/range10；`prepare.py` owner八UUID与授权／准入；`trainer.py` world8、rank状态和retry capsule；`controller.py` nproc8／8worker；`checkpoint.py/state_storage.py/retry_state.py` complete_rank_count8。不是修改CUDA_VISIBLE_DEVICES即可启动。

### 7.1 开跑前有意义的数值测试

1. 五兄弟均值1／范围／正负互补／排序／共同偏置／缺失中性／微小差异floor／全等／全NA／非有限拒绝；已提供CPU参考fixtures。
2. 用小词表独立autograd及有限差分核：w是常数时梯度恰为原梯度乘w，没有 `L∇w`；clip前后次序不可交换；η0仅作CPU null control，应精确回原E loss／梯度。
3. 四rankSUM与单进程global32等价；尾批1/7/31实际人数；无额外/world4或/5；共同R第一轮同模型／B/D/mask下w1分支对原结果数值核验。
4. scorer与实际selected-D更新前forward的gap/entropy逐项匹配，模型身份／编码／精度相同；不得拿采样时p_old日志填当前θk。score前后RNG／mode／adapter／requires_grad完全恢复。
5. 四rank完整状态、失败重试不重复已提交Adam；生成八逻辑slot完整覆盖；no-answer／malformed-payload／boundary／context fallback场景均有fixture。

新源码、入口、评分与部署配置先持久本地＋Git push search-e1＋FETCH_HEAD逐文件读回。所有fixture通过后，同一正式run首个outer与第1/5实际Adam有限／权重改变／四rank一致／显存验收；通过即连续跑到25，不为测试从0再来。

## 8. 核心验收：EM／pass与loss、概率到底有什么关系

### 8.1 固定验证曲线与主终点

每组保留raw checkpoint outer5/10/15/20/25；分别评估同一512题（NQ256/Hotpot256），共同source100零效果基线同四卡协议评一次。普通提示、原HF SDPA、BF16 base＋未合并FP32 LoRA、greedy／repetition1.05／原EOS／8192／500／5turn4search/top3/480、原严格final-action scorer不变。不能使用best回退，也不能用采样cache代替实际outer末raw权重。

主终点预先指定为**outer25的heldout strict EM**；中间点用于退化／保护曲线，不能挑最高点当最终结果。四组5时点＋共同基线共10752条正式greedy预测；score hook、teacher独立生成及控制样本不计入512。

按同题配对报告：hits/512、NQ/Hotpot、相对共同起点及四方向间的gained/lost/preserved/stillwrong；lost分为最终协议无效和协议有效但gold mismatch两互斥类，搜索次数/终止另轴。题级bootstrap区间与多重比较范围明确；预先主比较为 G_POS↔G_NEG、H_POS↔H_NEG，另报四组对起点的描述比较。有单seed只能说明本seed／题集，不能假称已估算法跨seed不确定性。

统计执行补充：两个终态主比较做双侧exact McNemar（无discordant时p=1），Holm家族α=.05；NQ/Hotpot分层、seed42、10000次question block bootstrap，所有组/时点/同题siblings一块抽，百分点差95% percentile CI为单比较区间。曲线、跨指标及对起点比较只作探索性描述。该512是反复使用的训练外验证集，不是完全未见最终测试，不能把一次正结果称为未知题集/训练seed的可靠泛化。

heldout回放参考补充：EM始终统计全512。默认复用共同source100本次greedy正确响应作为固定B bank，在看四组输出前冻结并逐SHA绑定，不新增参考生成预算；所有组/时点共用同一B，不让各组自己挑成功参考。无B或teacher联合context超8192时g/teacher KL记NA；无答案payload只使答案gap NA，全policy KL若其余编码/mask合法仍可算；ordinary entropy不因无B而删题。报告全512 EM、bank及各指标覆盖、参考子集EM；g/KL关联不代表全512。baseline在该参考子集EM恒1，零方差/无两类时Pearson、Spearman及EM AUC必须NA，不能填0/.5；后续各指标子集也核此条件。该成功资格已经使用gold，是特权事后诊断，不是普通推理可部署的正确概率。默认bank不能覆盖原错→新对题的teacher信号；额外参考生成另列预算。

评估边界补充：完成5/10/15/20/25轮对应索引4/9/14/19/24；原子提交完整raw adapter/四rank Adam/RNG/order/cursor→训练child真实exit/release→隔离eval child→退出释放→核原checkpoint状态未变→继续下一outer。评估失败只恢复评估，不重做已提交训练Adam；不让评估消耗生产RNG或切换训练adapter。

### 8.2 Pass评估：与greedy EM分开

首要交付是上述EM。若需要真正的heldout成功概率关系，预设第二优先级：共同起点＋四组outer25，各在同512题独立采样5次（另12800条轨迹），固定同一HF SearchAgent表示、temperature1/top_p1、原processor及预算、独立种子表；报告s/5、经验pass@5及同题差值。s/5只是五次试验的粗估计，不是精确概率；在“至少成功一次”的训练配对子集上pass@5恒1，不能关联loss。

不能把TRAIN某轮s/5冒充heldout pass：TRAIN每outer题不同、经过本轮选样，且量发生在更新前。附加pass评测不改变训练25轮，分开登记推理预算／执行协议；不以温度采样指标替换原greedy结果。

### 8.3 证明权重作用到哪里，而非只报loss变小

- TRAIN端按同题五兄弟保存g/h/w、strict结果及s/5、每类pair曝光。比较加权前后CC/CW、自配对／非自配对、低／高gap与entropy分位桶、NA/neutral比例；特别检查G_NEG/H_NEG是否仅减少错误轨迹训练曝光。
- 每个正式heldout时点保留实际自主轨迹；随后独立固定模型replay，可对同题同response核g/h、teacher KL、old drift与区域贡献，标签为posthoc，不冒充历史在线loss。新paired分析以题为单位，不能将不同文本的第t词视为同事件。
- 将上一次评估正确／错误转换与此前训练中权重分布、weighted/unweighted KL、更新幅度和clip变化对齐到保存时段；仅5/10/15/20/25点不能插值精确退化步。
- 评估可报告题级EM与g/h的Pearson／Spearman、AUC及控制长度/搜索/dataset后的关系；对筛选规则、训练／验证不同题和模型生成文本内生性明确限制。词logprob高不等于题正确概率高。
- 若有E25，检验加权是否提高heldout EM及减少lost；没有E25，不从“本组loss更低／gap更小／entropy更小”得出加权有效。答案信号只决定整轨迹w，区域loss贡献不是参数梯度贡献。

## 9. Debug输出和交付

每个有效训练policy token保留原全词表reverse KL／postclip loss、student→teacher／old／target的区分、实际词双方logp、full-vocab entropy、clip触发及概率质量、offset/mask/区域、preupdate参数身份、w和实际batch贡献。它们来自真实训练forward；不同于评分阶段的无梯度forward。

每个本批题的五条兄弟另保存无梯度score标量、答案区token IDs/offset、NA理由、μ/d/floor/r/w、原row与B SHA、selected sibling/weight、θk身份。可重建所用权重；所有兄弟全policy的entropy计算使用完整词表，但不承诺把所有完整词表数组dump下来。top32维持每组每outer8192位置cap；少量full FP32快照另设24GiB总cap并登记cap_skip。

保留全部student轨迹／检索请求文档／生成IDs、配对参考；独立teacher自主生成每outer最多8例，保持原诊断用途、与训练q同前缀评分分开。梯度主证据为batch norm／clip／参数变化，不宣称保存了逐token参数梯度。原题、完整HTTP/token/权重/凭据不进Git。

里程碑顺序：E身份绑定与旧archive窄停止→独立四rank/25outer/评分实现及CPU核验→源码发布／资源验收→四组同run首outer与五真实更新→每5outer原生512、连续到25→配对EM及权重／loss／概率解释→源码／raw权重／配置／报告绝对路径与备份验收。四组顺序预先固定G_POS、G_NEG、H_POS、H_NEG；不因中间优劣换预算。没有新训练或自动调lr／clip／η的授权扩展。

## 10. 当前完成与待执行

已完成：指定线程／已发布R结果及数学代码实读；HKUST4实时资源和旧run身份检查；四种定义、异常／归一／detach／恢复／评测计划；独立纯CPU权重参考。旧archive的停止状态以另存实际receipt为准。

尚未完成：E名称的人工映射；生产四rank训练／score/evaluation/controller实现与GPU数值核验；四组25轮与新评测。不得将本计划、数学fixtures、GPU空闲或后台启动称为这些实验已完成。
