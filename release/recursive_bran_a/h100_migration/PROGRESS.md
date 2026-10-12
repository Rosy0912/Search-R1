# H100 实验 A 当前修复，2026-10-12

实际运行使用 h100-11、Slurm 3387，固定配置 SHA653193a4eb06bd28667f78ac14a66a2d1c97200ca8199b8521dcc2d3de35b319。

b824482 的首轮完成5450条实际轨迹、保留2560条、compose完成56.94秒，随后在旧 HKUST continuation 标记门禁退出。尚无Adam更新或完整训练轮，不能声称训练验收完成。

b103cbc 将明确身份绑定的 H100 fresh profile 接入三处原入口：不计算无目标贡献的 full beta、同采样阶段复用vLLM参数、正常数值防护消耗本轮并继续。零KL和behavior-mu约束保留；温度、reward、μ计算、模型精度及优化器数学未改。新标记经 runtime_audit 传入 Ray worker。五个变更文件已经编译、FETCH逐字节及节点部署SHA匹配。

17f6808 的一次性恢复只针对原3578892实际exit1/0Adam的失败范围，保留原output/log，释放原检索CUDA子进程后启动独立source-b103cbc。新实际owner3596768，同run48528afa6b6443cfbb35328fd8b0ff05。15:11CST实读模型加载中，未验收首轮。CPU检索3504331及已验收门禁继续复用，未重复压测；不停止外来GPU进程。

管理查询连接曾实际恢复，之后断开。连接问题不代表训练退出；恢复观察通道期间禁止重复重启训练。Mac只保留小证明，原轨迹/权重在H100。

继续要求：10完整健康outer并确认下一轮增长后完成本次goal，作业继续200；eval/checkpoint详细tree仅5/50/100/150/200。

## 15:47CST 实际新入口及生成计时

实际运行源码78eecc1，恢复helper ed47beb；owner3628041/UID0/starttick656907981/job3387，同run/config。db40459仅修改run.sh的编译环境未被独立launch采用，原失败已保留；78eecc1将三项编译路径写入实际Python入口与Ray runtime_env，CPU编译及原v11 Python导入exit0，随后实际Ray environ读到相同三项路径。没有安装或改变公共环境。

同一首轮固定cohort的生成记录：5450条实际trajectory/保留2560、32RPC、824108生成token、5381680prefill token、4049搜索调用、102padding request，历史3542960与当前3628041的这些工作计数相同。生成窗口（首RPC开始至最后RPC返回）673.286→354.781秒；RPC累计382.963→187.813秒；检索HTTP累计108.017→78.847秒。当前advance累计331.302秒、compose58.937秒、prepare_behavior_mu40.967秒。阶段有包含关系，不能相加当总时间。GPU并发占用变化，不能将整个速度差单因果归给权重复用。

当前主日志可见两项cache phase实际记录：policy_version[0,0]、32generation calls、1weight sync、1KV init、released_before_hf_or_policy_change=true；Ray日志有去重，不能将这两项冒称八rank完整直接日志核验。首轮参考前向/更新尚未验收。2秒observer累计本作业采样峰约34.7GiB，包含以前尝试，不是本轮独占连续最大值；15:47当前directfree最小17315MiB、observer age0.58秒。

用户15:38授权的子智能体仅做本地入口静态检查，未发现新的确定性阻断bug：Ray环境、guard语义、缓存版本变化、保存/评测释放和稀疏节点一致。其结果不代替实际训练验收。主线没有等待该检查。


## 20:47CST：当前运行前5轮已闭合，第6轮实际继续

下列状态取代上述历史尝试的“当前运行”：h100-11，实际八卡共享Slurm step3389.0，owner3714909/UID0/tick658086351，run1c806d15752d44a7ac99687f0737d7e3。部署运行源71d4e69d0b098ef37df00b1267ab4b9804e5024d；配置与断点声明的科学源7966a1e76d6abddb4349d57c4727d69fe62fb16d保持，两者分别记录。原20次更新后NameError且无断点的尝试已保留；当前运行是人工明确授权从官方权重重新开始，不称无损恢复该失败尝试。

5个outer已真实完成，8rank各累计100次Adam调用，正常防护跳过0。第5轮完整state manifest为b14c7793aff7e9986789270d0b88b1a2df8387d5588885bdd5b11722f1f41677，native八rank及driver同world reload全部通过；评测后state manifest为cd6fb84617aa3944ea8ac0a7d805108c52a45074ad286f0d37d0a26cf4000c6c，完整保存实际评测后状态，未另外执行重复reload。这里是原生验收和小manifest回读，不是root独立反序列化全部Adam或重新计算logits。

第5轮原生dev512完整完成：EM96/512=18.75%，NQ54/256，Hotpot42/256；采用冻结评测函数val/em，协议格式另轴（format_rate298/512）。本次从官方初始权重开始，不能与旧source100训练权重239/512直接作为同起点增益比较。未追加第10轮评测；正常评测、关键断点及详细tree仅5/50/100/150/200。

实测整轮耗时1–4为22.39、24.78、21.45、11.67分钟。第4轮原始生成窗口6.60分钟、actor更新2.79分钟；第5轮非testing部分13.32分钟，含计划内testing及保存总25.84分钟。testing750.92秒包含边界保存，不称纯生成耗时；绕过重复save计时接近0不代表真实存储没有耗时。GPU共享负载及题目工作量变化，不能把速度变化单因果归给某项修改，不承诺后续固定分钟数。

20:47:35新实读同owner无错误，第6轮真实生成/RPC及reward产物增长。只读follower改为已发布14fa4a2，处理评测outer=null，不改训练；旧只读follower已退役，远端运行未重启。私有小证据first5-closed-metrics.json、first5-closed-checkpoints.json、first-five-accepted-current.json及follow-v2目录保留在Mac，权重/原轨迹仍在H100。初期5+下一轮运行验收通过，完整goal仍要求10健康outer及实际11继续，之后让作业运行至200。


## 21:37CST：10轮初期训练验收完成，第11轮真实继续

同一H100八卡step3389.0/run1c806d15752d44a7ac99687f0737d7e3/owner3714909 UID0 tick658086351已完整完成outer1–10，八rank各累计200次真实Adam，每轮20次、跳过0，全部已记录数值标量有限。第11轮已真实记录1565个action事件、768条完成轨迹和5次RPC；这是当时进展快照，不是整轮11完成。根只读follower达到10+下一实际产物后正常exit0，它退出不等于训练退出。

按人工明确的“训练到10iter正常则goal完成，继续训练至200iter”，初期验收已达到；**200训练和最终51713评估尚未完成**。训练仍运行，下一正常dev512、关键checkpoint及详细tree在50轮，随后100/150/200；无10轮补测、无额外断点恢复。第5轮完整state的native八rank/driver同world reload及评测后实际状态保存已经接受，没有重复反序列化或重跑测试。

### 实现、发布、部署与实测范围

- 逐行动额外HF核对/无目标完整旧actorbeta取消、真实μ及训练梯度前向保留；同tree权重同步一次；search-only原生停止；轻量候选评分与同outer证明缓存；分阶段实时计时均已实现、提交推送并部署。实际训练源71d4e69d0b098ef37df00b1267ab4b9804e5024d，八个相关部署文件freshSHA与此发布源一致，后续报告/只读工具HEAD不改变正跑。
- 采样阶段八卡32768-row原tile轮转、FP32/TF32false、query64、CPU E5 batch128、双pinned/GPUbuffer、原序topk合并和CUDA阶段子进程均已实现、发布并接线。固定CPUowner3713706/tick658076829/sourceidentity75830a6d…；原完整索引有限66query门禁接受完整HTTP JSON一致。1query原路径14.100秒/候选9.967秒，65query13.730/5.957秒；只代表这两请求，不是全训练或普遍8倍保证。
- 当前运行的已记录cache phase均weight_sync_calls=1；例如第10轮版本[180,9]的30次生成调用只有1次同步，切HF前release=true。Ray日志可能去重，范围是实际记录，不声称每个rank都有独立完整日志。
- 实际10轮训练如下。第5轮含计划内评测/保存，其余普通轮；第10轮生成含整理6.42分钟、actor更新2.72分钟、整轮10.44分钟。前后GPU并发负载与题目工作量改变，没有将整轮改善单因果归给某项修改。

| outer | 整轮分钟 |
| --- | --- |
| 1 | 22.39 |
| 2 | 24.78 |
| 3 | 21.45 |
| 4 | 11.67 |
| 5 | 25.84 |
| 6 | 10.91 |
| 7 | 10.87 |
| 8 | 11.15 |
| 9 | 10.90 |
| 10 | 10.44 |

独立CPUobserver在首轮后启动，每2秒采样；本任务各卡采样峰37904/37090/40512/37482/34910/36790/40178/36118MiB，最高39.56GiB。它不是首轮全历史或连续硬件最大值；首轮曾观察到其他任务共用时GPU3 directfree210MiB，无OOM。21:37当前磁盘余1473497100288B，小证据/代码留Mac，权重及全量轨迹在H100，没有本地大资产复制。

### 边界与未解决项

当前10轮没有未修复的运行故障；历史NameError且无断点尝试按人工明确授权fresh重启保留，没有伪称无损续训。共享Slurm step依赖原分配生命周期，不能保证资源永久保留。当前证据证明初期训练执行和部署合同，不证明200最终质量、单因素加速倍数或参数/Adam全部独立CPU重算。第5轮dev512的96/512=18.75%来自官方初始权重训练，不能与旧source100的239/512作为同起点改善比较。

本地私有小证明：initial-ten-accepted-current.json、initial10-followup-first.json、recorded-timings-initial10.json、runtime-receipts-initial10.json、current-resources-initial10.json及first5-closed-checkpoints.json；不把原题/token/权重或凭据写入Git。代码入口位于本目录train.py，八卡检索在eight_server.py和retriever_overlap_candidate，主流程在verl/trainer/ppo/ray_trainer.py。
