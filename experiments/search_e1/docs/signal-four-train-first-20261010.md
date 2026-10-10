# 四组消融先续训，补测不再阻塞

2026-10-10最新人工要求直接继续训练。原512题补测采用4卡、每卡一个HF进程逐题greedy，两批各256题；训练使用vLLM批量生成。这种实现让评测明显更慢。补测第一批256题完成后，读取退出worker的/proc/environ发生PermissionError，实际控制器退出1；256题不是完整512评测，也没有新Adam更新。

从G_POS已完成15个outer、Adam150的完整checkpoint续第16轮，原四组配置和训练代码不改。4rank、micro1、每rank梯度累积8、global32以及FP32 outer-old恢复照旧。新入口只调整CPU执行顺序：共同zero评测复用已验收结果，每组raw15/25权重保留，写明evaluation_pending=true且accepted=false；四组训练依次继续，不让补测阻断后续组。

新调度源码ed7f9769564436b115e1a58812f7f893f7dce8e2；HKUST4入口与配置见configs/signal-weight-training-priority-20261010。启动时已确认上一失败进程退出、仅原检索占用自有GPU7；原失败输出和状态保留。新supervisor1861244/controller1861436。后台启动不算训练完成，后续继续核实采样和真实更新。

八次完整512测试及最终结果仍待完成，不能用256题补测、训练EM或四组训练完成状态代替效果验收。原239/512共同zero仅适用于原HF协议；今后若改为vLLM批量评测，须统一端点条件并说明协议差异，不能混用指标。

2026-10-10 12:00 CST实测：同一G_POS第16轮完整结束，P=K307，10次新Adam（150→160），实际torchrun wait0。该轮更新前TRAIN严格EM1222/2560=47.734375%，累计pair曝光4875；末checkpoint160原生四rank adapter/optimizer摘要一致。这里只报告原生状态，不重复声称独立CPU全量tensor检查。control已创建第17轮目录，supervisor1861244仍运行，无补测启动。实际小型证明见first-resumed-outer-complete.json；全部轨迹、token和状态仍远端。

2026-10-10 12:15 CST推进：G_POS已完成17/25outer、Adam170，进入第18轮SAMPLING。第17轮完整2560采样更新前strict EM1236/2560=48.28125%，P=K303、10次实际Adam；原训练子进程实际wait0。这个TRAIN批次不同，不能据此声称固定测试EM改善。CP160已HKUST4→HKUST1远端备份完整SHA验收、实际两端传输退出0，Mac payload文件0；CP170备份刚开始时仍是pending，不冒充已完成。

后置四卡vLLM批量评测入口已发布70ef656635aa68c7ccdf7c1be8f3e5cbabbf9922，部署source/batched-eval-70ef656/opsd_signal_weight_batched_eval_v1，Mac与Node各7项CPU检查通过，尚未GPU启动/验收。它强制全部100outer完成和原supervisor真正exit0后才使用四张卡，不会卡住当前训练。复用原冻结executor、CPU PEFT/BF16合并、严格QA及固定512；greedy明确rep1.05与原EOS，8逻辑64题分片分两波。统一重新评共同zero，再评八个raw15/25端点，不拿旧HF239与新vLLM数直接相减；各保留实际生成ID/HTTP/512严格结果、总体/NQ/Hotpot和逐题新增丢失。运行后才能确认真实推理耗时与完整八次评测。

12:18 CST后续复核：第18轮实际生成trace约11.88MB继续增长、无新error；CP170远端备份已完整9成员359785236bytes逐SHA验收，两端SSH实际wait0，Mac payload文件0。上述CP170 pending已被此真实完成取代。批量评测CPU候选现为a48b2fdf6fcb39821ee2f3e42ce1870d383719e4、source/batched-eval-a48b2fd，publication SHA697261d13e6746b1615b658f0efa99860d70b171223188699d142286d03e2963；仅补齐与原训练一致的PCI_BUS_ID设备顺序，旧70ef656候选保留，Node仍7项CPU通过且没有GPU启动。下一步继续当前训练；全部100完成、真实exit0/资源释放后才按新统一协议评八端点并完成最终分析。

2026-10-10 12:34 CST推进：G_POS18/25完整outer、Adam180，当前第19轮SAMPLING实际trace约13.45MB继续增长，supervisor1861244/controller1861436实读存活、无新错误。第18轮更新前严格TRAIN EM1214/2560=47.421875%，P=K295、10次新Adam，torchrun实际wait0；CP180已HKUST1远端9成员逐SHA备份并两端实际退出0，Mac payload0。其他三组尚未开始，正式八次512仍pending；这不是总体完成验收。继续当前串行训练，不重做已有更新、不启动补测。

2026-10-10 12:50 CST推进：G_POS19/25完整outer、Adam190；第20轮SAMPLING实际生成trace约10.68MB、两原控制进程存活且无新错误。第19轮更新前TRAIN strict EM1198/2560=46.796875%，P=K300、10次新Adam、原torchrun实际wait0。CP190已HKUST1远端逐成员SHA备份，两端实际退出0、Mac payload0。原训练设置不变，其他三组待串行启动，八次正式512仍后置pending；不重复已有更新。

2026-10-10 13:05 CST推进：G_POS20/25完整outer、Adam200；第21轮SAMPLING实际生成trace约5.62MB、两原控制进程实读存活，无新错误。第20轮更新前TRAIN strict EM1146/2560=44.765625%，P=K296、10次新Adam、原torchrun实际wait0；该TRAIN题集不同，不能当作固定测试EM退化。CP200已HKUST1远端逐成员SHA备份，两端实际退出0、Mac payload0。其余三组待串行启动，八次正式512仍后置pending，继续当前训练。

2026-10-10 13:20 CST推进：G_POS21/25完整outer、Adam210；第22轮SAMPLING实际生成trace约2.53MB、两原控制进程实读存活，无新错误。第21轮更新前TRAIN strict EM1187/2560=46.3671875%，P=K301、10次新Adam、原torchrun实际wait0。CP210已HKUST1远端逐成员SHA备份，两端实际退出0、Mac payload0。其余三组仍待串行启动，八次正式512后置pending，继续当前训练。

2026-10-10 13:35 CST推进：G_POS22/25完整outer、Adam220；第23轮SAMPLING实际生成trace约1.25MB、两原控制进程实读存活，无新错误。第22轮更新前TRAIN strict EM1228/2560=47.96875%，P=K313、10次新Adam、原torchrun实际wait0。CP220已HKUST1远端逐成员SHA备份，两端实际退出0、Mac payload0。其余三组待串行启动，八次正式512后置pending，继续当前训练。

2026-10-10 13:50 CST推进：G_POS23/25完整outer、Adam230；第23轮更新前TRAIN strict EM1216/2560=47.50%，P=K311、10次新Adam、原torchrun实际wait0。CP230已HKUST1远端逐成员SHA备份，两端实际退出0、Mac payload0。第24轮进入SAMPLING，继续确认实际新生成，其余三组待串行启动，八次正式512仍后置pending。

最终CPU汇总接口4f13ffbcdbf86ffb2ba115772fe2ff34762f3c13已独立发布部署，Mac/Node各3项CPU检查通过；没有额外GPU任务或新审查轮。它仅将新批量评测九端点的完整4608行转换给原result_analysis.analyze，复用第25轮G_POS−G_NEG、H_POS−H_NEG两项exact McNemar/Holm家族和seed42/10000次共享分层配对bootstrap；旧HF239不混入新vLLM指标，TRAIN信号关系另报。当前八端点评测尚未开始，此准备不代表效果验收。

2026-10-10 14:08 CST推进：G_POS24/25完整outer、Adam240；第24轮更新前TRAIN strict EM1205/2560=47.0703125%，P=K301、10次新Adam、原torchrun实际wait0。CP240已HKUST1远端逐成员SHA备份，两端实际退出0、Mac payload0。第25轮进入SAMPLING准备，仍确认真实新生成；其余三组待串行启动，八次正式512仍后置pending。采样收尾时短暂Zombie观察提示随后自然清除，未重启、未发信号，没有实际采样失败。

2026-10-10 14:24–14:28 CST组间交接：G_POS完整25/25outer、250真实Adam、累计7598可用pair曝光；25条更新前TRAIN EM全部原生accepted且分母2560、索引0–24连续。末轮TRAIN1187/2560=46.3671875%、P=K303，torchrun和G_POS controller均实际wait0，自有后代释放。CP250远端9成员SHA备份与两端实际退出0通过；复用原audit_state.py独立CPU实读四rank全Adam250/moments/RNG/scaler/cursor一致，504个FP32 adapter张量全有限且相对共同初始全部改变，未初始化CUDA。这里只完成该组训练，raw15/25测试accepted仍false/pending。

同一supervisor1861244已自动进入G_NEG，实际controller1977766/UID1001/tick690962026/runc9a7104f9a874e5eb7efdeac87e855ec；第1轮真实生成trace约14.27MB。采样model为raw-source100、optimizer_updates0、checkpoint=null、共同初始adapter digest836590b798434f21160418c6471cbce3bdfac5bf4a9947c1e3e7723d057baf2f，未接入G_POS末权重或Adam；初始Adam还未进入trainer序列化，不能冒充已实读其完整moment。继续G_NEG25，再H_POS/H_NEG25；总25/100与0/8正式测试，补测后置不阻塞。

2026-10-10 14:40–14:43 CST推进：G_NEG第1/25outer完整结束，P=K327，11真实Adam（10×32+7尾批），原torchrun实际wait0；更新前TRAIN1280/2560=50.00%，包含全错/未选轨迹，该指标与G_POS首轮相同，不宣称全轨迹bitwise一致。独立CPU初始CP0实读四rank Adam0、504FP32 adapter与共同初始完全相同；末CP11实读四rank全Adam11/moments/RNG/scaler/cursor一致有限、504adapter全有限且相对初始全部改变，均未初始化CUDA，确认fresh组没有继承G_POS Adam。CP11已HKUST4→HKUST1远端9成员SHA备份，两端实际退出0，Mac payload0。

同controller1977766继续G_NEG第2轮SAMPLING，当前真实生成trace约10.42MB、更新前采样模型CP11、无新错误。当前TRAIN记录26/100（G_POS25、G_NEG1、H_POS/H_NEG0），八次正式512仍后置pending。训练设置与四rank累积8保持，不重做已有更新。

2026-10-10 15:00 CST推进：G_NEG2/25完整outer、累计Adam21；第2轮更新前TRAIN strict EM1230/2560=48.046875%，P=K317、10次新Adam（11→21），原torchrun实际wait0。原生四rank adapter/Adam摘要一致；本次未重复独立CPU张量实读，不冒充CP21完整数值审计。CP21已HKUST1远端9成员逐SHA备份，两端实际退出0、Mac payload0。第3轮SAMPLING实际生成trace约8.83MB、两原控制进程实读存活、无新错误；此前采样收尾Zombie观察提示自然清除，没有重启或新修复。总TRAIN27/100（G_POS25/G_NEG2/H_POS0/H_NEG0），八次正式512仍后置pending，继续当前训练。

2026-10-10 15:14–15:36 CST推进：G_NEG第3、4轮均完整结束，累计Adam30、39；第3轮P=K288、9次新Adam（21→30）、更新前TRAIN1125/2560=43.9453125%；第4轮P=K284、9次新Adam（30→39）、更新前TRAIN1075/2560=41.9921875%。两轮原torchrun实际wait0，原生四rank adapter/Adam摘要一致；没有重复独立CPU张量实读。不同outer题集不能当作固定512测试退化结论，最终测试仍后置。CP30、39已HKUST1远端各9成员逐SHA备份，两端实际退出0、Mac payload0。

同controller1977766已进入第5轮SAMPLING（outer_index4、更新前CP39），当前真实生成trace约7.23MB、两原控制进程实读存活，无新错误。总TRAIN29/100（G_POS25/G_NEG4/H_POS0/H_NEG0），八次正式512仍pending。此前权限观察提示自然清除，期间采样与后续完整训练持续推进，没有重新启动或重做更新。

2026-10-10 15:53 CST推进：G_NEG5/25完整outer、累计Adam48；第5轮更新前TRAIN strict EM1148/2560=44.84375%，P=K287、9次新Adam（39→48），原torchrun实际wait0。原生四rank adapter/Adam摘要一致，没有重复独立CPU张量实读。CP48已HKUST1远端9成员逐SHA备份，两端实际退出0、Mac payload0。第6轮SAMPLING实际生成trace约5.64MB、两原控制进程实读存活，无新错误；此前Zombie观察提示自然清除，没有重启或新修复。总TRAIN30/100（G_POS25/G_NEG5/H_POS0/H_NEG0），八次正式512仍后置pending，继续当前训练。

2026-10-10 16:10 CST推进：G_NEG6/25完整outer、累计Adam58；第6轮更新前TRAIN strict EM1229/2560=48.0078125%，P=K305、10次新Adam（48→58），原torchrun实际wait0。原生四rank adapter/Adam摘要一致，没有重复独立CPU张量实读。CP58已HKUST1远端9成员逐SHA备份，两端实际退出0、Mac payload0。第7轮SAMPLING实际生成trace约5.64MB、两原控制进程实读存活，无新错误；此前收尾观察提示自然清除，没有重启或新修复。总TRAIN31/100（G_POS25/G_NEG6/H_POS0/H_NEG0），八次正式512仍后置pending，继续当前训练。

2026-10-10 16:24 CST推进：G_NEG7/25完整outer、累计Adam68；第7轮更新前TRAIN strict EM1224/2560=47.8125%，P=K300、10次新Adam（58→68），原torchrun实际wait0。原生四rank adapter/Adam摘要一致，没有重复独立CPU张量实读。CP68已HKUST1远端9成员逐SHA备份，两端实际退出0、Mac payload0。第8轮从CP68继续SAMPLING，实际生成trace由2.52MB增长到7.22MB、两原控制进程实读存活，无新错误。总TRAIN32/100（G_POS25/G_NEG7/H_POS0/H_NEG0），八次正式512仍后置pending。上一goal回合为verified wait，本回合完成新outer及远端备份，有实际进展；训练与科学配置未变，继续同一run。

2026-10-10 16:38–16:55 CST真实完成、17:02复核：G_NEG第8、9轮均完整结束，累计Adam78、88。第8轮P=K310、10次新Adam（68→78），更新前TRAIN1244/2560=48.59375%；第9轮P=K301、10次新Adam（78→88），更新前TRAIN1195/2560=46.6796875%。两轮实际torchrun wait0、原生四rank adapter/Adam摘要一致；本次不重复独立CPU张量实读。CP78、88已HKUST1远端各9成员逐SHA备份，两端实际退出0、Mac payload0。

同controller1977766已进入第10轮SAMPLING，实际trace15.19→23.97MB继续增长；采样阶段proc/environ及动态memory-live SHA观察提示不等于实际失败，前一提示已自然清除。没有信号、重启或重复更新，冻结科学设置未变。总TRAIN34/100（G_POS25/G_NEG9/H_POS0/H_NEG0），八次完整512仍后置pending。上一goal回合完成GN7与备份，为progress；本回合完成GN8/9与备份，亦为progress，不以单次巡检结束目标。

2026-10-10第10轮端口冲突恢复：17:06:29 CST G_NEG outer9训练器在init_process_group以EADDRINUSE退出1，尚未加载模型/没有本轮Adam；教师诊断实际exit0、完整2560采样保留。原controller1977766及supervisor1861244实际退出1，旧110子进程全部ABSENT，日志/源代码/失败票据均保留。独立CPU实读CP88：四rank完整Adam88/moments/groups一致，504FP32 adapter全有限，未初始化CUDA。

20:35 CST新独立port-resume-0146350入口启动；源码0146350f7a77b87644d5639a3dac1802b6a54c79已发布与逐文件readback，仅通信启动改static c10d agent TCPStore/OS空闲本地端口；原四rank/累积8/global32/science/config原字节不变。新supervisor2297236/tick693190113、GNcontroller2300905/tick693209228、原GNrun不变、tmux%48/pane2297233。先读回已完成G_POS状态，无新增GP训练，然后复用第10轮原sampling/teacher文件SHA继续。旧5eac816候选仅CPU部署因audit嵌套字段未正确绑定而停止、没有GPU/训练启动，保留；0146350修正后实际运行。

20:45–20:47 CST真实验收：第10轮P=K308、88→98共10新Adam；原failed attempt0=1，retry attempt1实际torchrun wait0，actual argv明确static/4rank。sampling-complete/teacher-complete原SHA完全不变，没有重做采样或已完成更新。本轮更新前TRAIN1263/2560=49.3359375%。CP98已HKUST1远端9成员逐SHA备份，两端实际exit0、Mac payload0。第11轮从CP98继续采样，实际trace增长；总TRAIN35/100，正式512仍0/8pending。恢复后完整一轮通过而非仅启动。后置a48评估原终态guard绑定旧失败supervisor1861244，全部100完成后必须在独立已发布副本绑定新真实终态，不能伪造旧exit0。

2026-10-10 21:02 CST训练完成、21:08–21:09复核：同port-resume-0146350入口G_NEG11/25完整outer、累计Adam108；P=K308、10次新Adam（98→108），更新前TRAIN1216/2560=47.50%，原torchrun实际wait0。原生四rank adapter/Adam摘要一致，没有重复独立CPU全tensor实读。CP108已HKUST1远端9成员逐SHA备份，两端实际退出0、Mac payload0。第12轮从CP108采样，实际trace15.61→25.71MB增长，supervisor2297236/controller2300905活跃无新错误。总TRAIN36/100，正式512仍0/8pending。

CPU补测准备：409dda73d05ee06edd035c6d66e3e2b5ae46342d的batched_eval_v2仅修正旧失败driver的终态门禁，绑定新supervisor2297236/tick693190113/runmarker及foreground pane2297233真正exit0，并要求原生4组/100outer/100TRAIN行终态。原vLLM批量生成、CPUmerge、strictEM、八端点及共同新baseline协议不变；Mac/Node各3项针对性终态检查通过，旧7项metrics不重复审核。Node source/batched-eval-v2-409dda7已仅CPU部署，没有GPU开始，训练未热改。

完整分析尚未完成：原HF239成功B bank和旧HF endpoint/replay接口保留为历史；新的vLLM结果不能混用其baseline指标。正式新共同zero生成后、看到其余八端点输出前，需按原计划冻结该次成功参考bank；随后做固定模型posthoc g/h/KL回放并适配新端点输入，明确NA、覆盖和constant-label情况。它不增加fresh参考/额外pass采样预算，不能拿TRAIN某轮s/5冒充heldout pass。当前CPU补测终态准备不等于以上分析已完成。

21:42 CST实际复核：G_POS25、G_NEG13、H_POS0、H_NEG0，共38条完整TRAIN记录。G_NEG末raw CP128/index12，末轮更新前EM1243/2560=48.5546875%；第14轮真实采样文件从27836239增长到29186428bytes，无error，原controller2300905继续。补测未占卡、不阻塞续训。原评测慢在四卡HF逐题生成，对比训练vLLM批量生成；后续统一四卡批量评测，现有raw15/25保留。

后续CPU准备：v3源码3a3fb40已仅CPU部署source/batched-eval-v3-3a3fb40，Node新增2项bank转换测试通过，无GPU开始。新opsd_signal_weight_batched_replay_v1复用原replay_math/replay_worker/summary逐字节代码，仅将新vLLM九个全512端点与已冻结的新共同zero bank转换为明确标注新协议的posthoc输入；新增3项CPU兼容测试通过，尚未实际概率回放。无B时teacher/g/KL为NA、普通h保留，不新增参考或pass采样，不用旧HF239 bank。CP128已远端到远端备份9成员359785300bytes，两端实际exit0/逐SHA完成，Mac大payload0。

21:52 CST实际进展：G_NEG14/25、末CP138/index13 SHA b77ed31584a98ba284b227e54637b33aaf9c1a3b163d270c1364aae340646293；原生trainer实际exit0，TRAIN更新前1189/2560=46.4453125%，累计四组39/100。第15轮真实采样4071821bytes并增长，无error。CP138远端备份9成员359785300bytes/逐SHA与两端exit0完成。新posthoc兼容源码ef1db16821dcdbdca7bbe8947c2abc48908ef4a3仅CPU部署source/batched-replay-v1-ef1db16，Node3新转换检查通过，科学回放公式与worker未改，未GPU开始。训练部署仍0146350，不把分析代码部署当训练或效果完成。

后续未启动GPU的结果准备以7e7a415为准：batched_eval_v4仅将已执行的每波GPU释放检查返回值及代码身份落盘，worker/merge/admission函数体与v3逐字节相同；batched_replay_v2仅补齐完整raw_checkpoint身份及新typed端点的汇总入口，普通概率/KL/区域数学函数未改。Node原环境CPU导入成功且4项兼容检查通过，没有额外生成/训练/审核轮。旧v3/v1保留为CPU准备历史，不自动启动。G_NEG第15轮采样2560完整后已TRAINING，实际native更新143，无error；属于同一恢复后run，未开始测试。
