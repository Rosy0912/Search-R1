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
