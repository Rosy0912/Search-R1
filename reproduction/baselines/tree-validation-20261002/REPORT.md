# 奖励协议修复状态（2026-10-04）

已在本地合入最终独立answer/EM、共用F1/action legality、optional格式reward的info_mask来源修复，并保留远端修复历史。修复后的source集成提交1796f54，Tree训练数学与rollout/controller/actor/rank audit字节保持不变；针对性38pass、全CPU192pass/1skip、独立协议审查PASS。详情见REWARD_FIX_INTEGRATION.md和REWARD_FIX_REVIEW.md。

**仍运行的2971源码冻结4d133ef，使用修复前奖励。以下奖励/F1/EM及其历史周期审查仅说明旧协议当时的计算一致性，不能作为修复后协议效果验收；旧rollout/checkpoint不会自动重算。** 本次未热改、停止或重启训练。修复后实验需要独立固定提交的新副本及新的实际训练/评估验收。

# Tree-GRPO Search Agentic RL 验证

当前未完成效果验证。新同机协议2934已真实完成两个outer并保存完整HF2；CPU完整权重重载2945通过。训练body成功，但后续清理在约4.75秒等待后仍枚举到自己的后代进程（原异常未记录PID/state，不能确认是僵尸或GPU退出延迟），原父作业仍FAILED1:0、原成本保留。两步科学证据及实际Linux清理修复已独立通过，准许fresh五步；五更新已正常完成并独立通过，正在推进正式500更新及完整51,713题测试。

生产冻结4d133efeb1f675139d2558640f1e21d50780f133，公共冻结4bda3078b255b5a4d9cf90bf9a443ba3102c1e35。训练、Ray、GPU E5和完整8卡FP16分片FAISS全部在h100-9；检索仅127.0.0.1:18082，Ray强制回环。禁止跨集群数据、检索、存储、权重及中转。保留原8个小进程，并执行正常Slurm与GPU leases及双15%门槛。

固定Qwen2.5-3B base、train512/mini256/globalmicro64（rank8）、rollout/ref micro128、四轮搜索、不强制最终回答。Tree为2roots×3leaves=6，每question共享前缀续写，保留额外分支成本。阶段/周期验证64题；完整holdout3,393仅供约定teacher选择，最终测试51,713未裁剪。每阶段从base新启，完整optimizer/RNG/dataloader精确恢复不受支持。

| 阶段 | 实际状态 | 是否代表效果完成 |
|---|---|---|
| 全部CPU回归 | 187pass/1skip/4warnings；独立Tree rank审计11tests与2rank Gloo开关等值 | 否 |
| 同机GPU检索2919 | Slurm140s/8GPU/实际192CPU/request32，exit0；独立有限门禁PASS | 否 |
| Ray回环2930 | 真实Ray2.40 CLI head+remote worker同nodeID/127.0.0.1，0GPU/4CPU/6s，exit0 | 否 |
| 新两更新2934 | 真实两个更新/16rank记录；HF完整重载有限且非零delta；原父作业cleanup FAILED；科学two及独立运行修复2953组合通过，准许five only | 否 |
| 新五更新2956 | 实际5步/40rank/Slurm COMPLETED0:0/正常cleanup；CPU2957完整HF重载PASS；独立验收PASS，准许freshformal500 | 否 |
| 正式500更新与完整测试2971 | 正式freshbase实际已完成224个outer/1792rank；initial64 F1 0.173780/EM 0.125，耗时45.793s；periodic64/final51713尚待实际结果 | 否 |

2919真实固定14题的k3/k50 ID、顺序、内容SHA和score逐值与原GPU8分片参考一致。512重复题两次k3 HTTP约0.197/0.192秒；ID与内容排序一致，batch score相对14题最大差约1.19e-6，两个repeat一致。容量重复、空请求和513题原内部拆批正常。这些重复/合成题不代表512个自然独立查询吞吐。含原背景进程的检索常驻显存GPU0 7765MiB、其余7133MiB；idle SM0。

当前审计记录每个rank成功返回的真实optimizer calls与已有FSDP全局norm，以及至多8192个固定本地坐标的参数差值；不增加collective/gather/RNG，不改变数学或原metric归约。driver的optimizer_steps只保留rank0，不能拿它替代直接8rank证据。样本delta为零也不能证明整片未更新，完整HF差异及forward重载另行验收。已记录包含审计的训练墙钟；现有rank hook计时覆盖原actor调用，未单独分离审计增量成本，不能据此声称零开销。

失败记录：2923在真实训练前被严格local boundary拒绝，actual0，原背景进程保留，自己的GPU/Ray进程已清理。Ray2.40默认将127.0.0.1转为本机另一NIC；使用既有Ray开关强制回环后，2930真实CPU证明通过。CPU依赖任务2924取消，没有HF验收结果。

历史2879跨集群部署、2883停止的五步候选、2896/2905独立H100检索节点及micro8诊断均已废弃，不计入本次验收或新协议性能。历史详细记录在REPORT-historical-before-new-protocol.md，原始成本和证据保留。最终F1/EM/格式/工具调用、生成/prefill/额外扩展/重复前缀成本及同协议新GRPO对照尚未产出。

2934两步分别289.809/296.905秒（第二步含20.305秒保存），尚无同协议GRPO对照，不能据此宣称算法效率优势。各rank每outer真实12次optimizer返回；第一outer保留原LambdaLR初始化lr0，sample delta0；第二outer及完整HF核验更新。HF2完整3,085,938,688参数值中3,085,195,414变化，434tensor变化，最大差1.248e-5，CPU forward有限；自进程CPU53.408s/wall18.691s/峰值RSS15.90GB，CPU资源不算GPU训练成本。原2934 Slurm792s/8GPU/192CPU（request32）FAILED1:0，约1.76GPUh/42.24allocatedCPUh，supervisor783.849s。调度回收后原8GPU1131MiB/SM0及8PID完整，未将失败改成成功。

2956五步正常完成：Slurm1529s/8GPU/192CPU(request32)/256G，约3.398GPUh及81.547allocatedCPUh；supervisor1517.341s。各步283.835/271.104/243.108/232.972/235.705s，最后含模型保存；生成量下降时墙钟缩短，不能将其全归因系统加速。完整HF5的434tensor、3,085,628,392参数值变化，最大差4.746e-5；CPU2957完整加载/finite forward/Tokenizer一致，Slurm22s/8CPU/0GPU退出0，模型重载wall18.602/selfCPU53.114s/峰值RSS15.703GB。原8GPU小进程保留，任务cleanup_residual为空。五步不触发周期64题验证；正式启动必须实际initial64、每50周期64以及最终51713。

正式2971使用未来versioned助手/身份guard，原source4d133、Tree数学、训练配置和hook不变。新guard只对原heldpidfd+starttime同一且Z状态接受退出残留，living/reuse/unknown拒；保持双15%及mandatory finally plain scancel自身作业。Linux2964 proper0GPU4CPU实际身份测试通过（outside cgroup模拟，未证明真实NVML退出时延），softFD1024/hard131072/open5，正式实际FD增长仍可观察。正式独立Slurm8GPU/32CPU请求/256G，72h只作时间上限；完成自动释放。最终同协议新GRPO比较尚未取得。

正式2971初始64题已实际完成，真实Ray主任务安装计量observer，F1 17.3780%、EM 12.5%。首outer实际282.930s，其中生成185.726s、actor68.447s，生成2,502,430token/2,173检索请求；8rank记录及12次optimizer返回已产生。原LR第一outer为0，不能称首outer已经改变权重；后续更新及完整HF另核验。该64题小样本仅作基线/周期日志，不能代替51,713题完整效果结论。

正式第2步实际261.543s（生成178.303s/actor54.979s），生成2,544,358token及2,275检索请求。原warmup142阶段，第2步八rank每8192固定本地坐标分别7738/7666/7627/7567/7563/7580/7572/7523发生有限非零变化，证明真实权重更新。未来0GPU同节点依赖作业2972核验fullHF500；2975保存parent终态、2976在HF通过后提取500/51713汇总、2977保存CPU终态。以上均待真实执行，不算已经完成。

正式前两outer原件已独立重算：resolved/actualentry/rank配置SHA一致；2×512题的树结构、16字段继承、policy等叶质量及8rank非零参数变化通过，见FORMAL_TWO_PREFIX_REVIEW.md。仅接受当前前缀证据，不提前接受500步、完整测试、最终HF或公平GRPO效果比较。

截至正式2971实际第5步：五步墙钟282.930/261.543/265.453/263.809/279.884s（第5步含保存），平均270.724s；40个rank记录已生成，HF第5步及algorithm state保存后继续第6步。该正式checkpoint5尚未单独CPU重载，不把早期2957的HF5重载冒充此目录验收。500训练、周期验证、完整最终评估与公平GRPO比较仍未完成。

2026-10-03 16:23小时检查：正式2971 Slurm仍RUNNING，实际35/500，当前第36步。最近33/34/35耗时179.306/179.873/174.744s，均值177.974s；生成量660,058/581,210/562,835token，不能把后期变短全部称为系统加速。280个rank记录逐一检查身份/12次成功优化/有限非零已有梯度范数/无skip；第35步8rank固定参数样本均非零，console所有数值有限。最近训练reward0.164063/0.196615/0.215820来自不同训练batch，不等于同题效果提升。1694次资源采样无guard事件；外来显存最高1.37569%/SM0，采样GPU峰79.5255GiB为峰值下界，尚不代表长期显存安全验收。无Traceback，最新checkpoint5。尚无第50步周期评估，最新64题F1 0.173780/EM0.125仍为step0基线；最终51713未运行。0GPU终验/终态收集任务2972/2975/2976/2977均仍等待依赖。公共负责人本地最新报告指出新GRPO正式2985在15:28:57实际5步且继续运行；不把该历史读数当当前实时值或完整对照。

2026-10-03 19:56实际小时检查（前次问询处理被中断，不能用触发17:24当实际现场时间）：2971 RUNNING，完成124/500，当前125，992rank身份/12opt/有限非零norm/无skip齐全，所有console指标有限，第124步八rank参数样本均非零；最新三步均值126.887s，生成255,263/264,746/248,218token，轨迹变短参与墙钟下降。4233次guard无事件、无Traceback，外来显存峰1.37569%/SM0，历史采样GPU峰79.5255GiB；checkpoint5/100存在。初始64 F1/EM17.3780%/12.5%，周期50为32.7198%/23.4375%（15/64），周期100为38.7574%/26.5625%（17/64），两次actual64评估墙钟32.915/32.501s；这是同64题小样本改善，尚非51713最终评估或GRPO优越性结论。100步12个HF文件/3shards和435index键仅metadata检查，不称fullHF100重载。周期原件已保存并送独立审查；最终0GPU检查仍等待依赖。公共负责人报告GRPO2985因其人类用户停止指令在24步后中断且保持停止，未完成500公平对照；本任务Tree继续原500预算，不擅自重启他对照。

周期50/100原件独立复核通过（FORMAL_PERIODIC_50_100_REVIEW.md）：20文件SHA、actualentry/config身份、7数据集counts/加权F1与15/64和17/64的EM、3次costledger全部SHA/counters、sameRayPID3489451/observer来源及step100八rank非零参数样本成立。独立scope不包含完整992rank原件重核；本方已逐项核查992rank。仅接受周期前缀证据，HF100metadata不替代fullreload，正式终验仍未完成。

2026-10-03 20:19用户问询实际复查：2971 RUNNING，135/500，当前136，最近三步116.286/118.625/123.999s（均值119.637s），1080rank身份/12opt/有限非零norm齐全，第135步八rank参数样本非零，console指标有限、guard4508次无事件、无Traceback。最新周期仍step100 F1 0.387574/EM0.265625，下一次150；HF5/100保存，最终51713尚待。

2026-10-03 20:54小时检查：2971 RUNNING，152/500，当前153，最近三步均值129.735s（包括step150周期评估21.826s）；1216rank身份/12opt/有限非零norm齐全，第152步八rank样本非零，全部console数值有限、guard4922次无事件、无Traceback。周期150实际64题，F1 0.358929/EM0.28125（18/64）；相比周期100 F1下降2.8646百分点，EM升1.5625百分点，同小集合波动不能替代全51713终测。七数据集计数/加权F1-EM-格式一致与costledger SHA检查通过，原件periodic150-prefix保存；不称独立答案重评分。最新checkpoint100，0GPU终验/清理检查等待依赖；新GRPO500完整对照仍缺。

2026-10-03 21:54小时检查：2971仍RUNNING，177/500，当前178；最近175..177步158.046/168.355/167.201s，均值164.534s。raw生成量2,380,808/2,398,326/2,308,044，相比上小时前后约25万显著上升；对应生成113.552/123.558/121.717s，actor30.781/31.225/31.702s，不能用raw token当训练保留response或宣称所有速度变化来自系统。1416rank身份/12opt/有限非零norm齐全，第177步八rank样本非零、所有console指标有限，guard5641次无事件/无Traceback，外来资源门槛符合。无新增周期评估；latest150 F1 0.358929/EM0.28125，next200，checkpoint100。终验CPU依赖仍等待；公共负责人latest报告仍保持新GRPO停止，尚无完整500对照。本任务不改数学/预算/源码或重复启动。

2026-10-03 22:56小时检查：2971 RUNNING，实际201/500，当前202，最近三步均值166.926s（含200步保存/验证）。1608rank身份/12opt/有限非零norm齐全，第201步八rank样本均非零，所有console指标有限，guard6380次无事件/无Traceback，外来显存峰1.37569%/SM0。历史采样GPU峰79.5685GiB，仅峰值下界，尚无OOM；不改变micro或预算。周期200实际64题F1 0.369271/EM0.296875（19/64），相比150两者上升，但F1仍低于100；不作最终效果宣称。checkpoint5/100/200存在，200十二HF文件/三shards/435index键及algorithm200 metadata已取证，不冒充完整重载。周期200和rank200原件送独立审查；终态/HF500 CPU检查仍待训练依赖，完整51713与GRPO500对照未完成。

周期200独立prefix审查通过（FORMAL_PERIODIC200_REVIEW.md）：24原件SHA、actualentry/config/code，64题七集加权F1/19-64 EM、五次成本账本SHA及全部counters、sameRayPID、八rank真实更新及state200身份成立。200评估rawnonpad事件65,390，相比150的10,213显著增加，账本支持但原因未确认，不能用retained生成长度遮掩实际采样预算变化；200评估墙钟28.510s。只确认200HFmetadata存在，不签完整HF重载、500/51713或GRPO比较。

2026-10-03 23:57小时检查：2971 RUNNING，224/500，当前225，最近三步144.149/145.791/171.645s，均值153.862s；raw生成2,460,900/2,559,561/2,583,069。1792rank身份/12opt/有限非零norm齐全，第224步八rank样本非零、console全部有限，guard7102次无事件/无Traceback；外来显存峰1.37569%/SM0，采样GPU峰79.5685GiB。尚无新周期评估，最新200 F1 0.369271/EM0.296875，下一次250，checkpoint5/100/200保存。最终51713及HF500依赖检查未开始，不改运行源码或重复启动。
