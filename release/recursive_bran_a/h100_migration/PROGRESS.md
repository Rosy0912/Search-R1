# H100 实验 A 当前修复，2026-10-12

实际运行使用 h100-11、Slurm 3387，固定配置 SHA653193a4eb06bd28667f78ac14a66a2d1c97200ca8199b8521dcc2d3de35b319。

b824482 的首轮完成5450条实际轨迹、保留2560条、compose完成56.94秒，随后在旧 HKUST continuation 标记门禁退出。尚无Adam更新或完整训练轮，不能声称训练验收完成。

b103cbc 将明确身份绑定的 H100 fresh profile 接入三处原入口：不计算无目标贡献的 full beta、同采样阶段复用vLLM参数、正常数值防护消耗本轮并继续。零KL和behavior-mu约束保留；温度、reward、μ计算、模型精度及优化器数学未改。新标记经 runtime_audit 传入 Ray worker。五个变更文件已经编译、FETCH逐字节及节点部署SHA匹配。

17f6808 的一次性恢复只针对原3578892实际exit1/0Adam的失败范围，保留原output/log，释放原检索CUDA子进程后启动独立source-b103cbc。新实际owner3596768，同run48528afa6b6443cfbb35328fd8b0ff05。15:11CST实读模型加载中，未验收首轮。CPU检索3504331及已验收门禁继续复用，未重复压测；不停止外来GPU进程。

管理查询连接曾实际恢复，之后断开。连接问题不代表训练退出；恢复观察通道期间禁止重复重启训练。Mac只保留小证明，原轨迹/权重在H100。

继续要求：10完整健康outer并确认下一轮增长后完成本次goal，作业继续200；eval/checkpoint详细tree仅5/50/100/150/200。
