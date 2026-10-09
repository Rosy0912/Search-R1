# 受控恢复 CPU 证据范围（2026-10-09）

本检查不在 GPU 训练依赖链上。原七组队列继续；C0/C1各十轮正式评估已完成，M最近闭合两轮、21次真实Adam。M独立CP1/5状态实读和原生跨outer恢复，与本页的故障恢复对照分别记录。

原冻结reverse数学、causal shift和实际尾批B1–32×普通/混合64对照，以及9个指定原方法，已在一次隔离CPU执行中实际rc0，8.52秒，最大梯度误差1.49e−8。没有重跑宽泛测试套件或占用GPU。该证据不接受完整PEFT/Adam/old/RNG/cursor恢复。

独立V1源码6a0a349实际CPU执行rc1，7.69秒，错误为fixed target/current full distributions differ。原scratch、failed.json、argv/stdout/stderr/实际退出值完整保留；不是生产作业失败，也没有因此重启M。CPUfixture捕获adapter时未clone：原state_dict和detach().cpu()可能共享活动参数storage。连续参考分支下一步会推进该引用；下一rank可能把漂移值当作原checkpoint。这个alias是确定的fixture代码缺陷，但V1日志没有rank和逐分布差异，不能断言原失败所在rank或全部根因。

V2仅在独立fixture中立即clone捕获adapter、deepcopy完整Adam，前后核原digest；每rank对照前核current/outer_old/fullAdam/scaler准确身份，并留下四分布各位置的SHA、形状、最大差和严格相等。原科学函数、nested restore、outer_old真实磁盘payload、538 setter guard及原严格相等条件不变。窄静态审查通过；截至本页提交，尚无V2实际数值通过证据。待K1连接可用后在新exclusive scratch执行一次，不覆盖或删除V1。

受控对象为真实tiny Qwen2、36层七投影的504个LoRA张量；八个磁盘rank文件分别恢复。共享Adam的八digest声明是fixture构造，不是八个独立生产rank的前序执行。CUDA RNG transport明确由CPU generator模拟，NCCL未执行；未加载source100，未覆盖BF16/FP16 GPU算术、完整atomic checkpoint writer或分布式活性。因此即使CPU对照通过，也不称为全GPU故障恢复验收。

K1当前SSH观测出现超时；最后一次实际读取确认M第三轮TEACHER_DIAGNOSTIC且controller UID/starttick正确。超时不证明训练终止，未修改网络/VPN、生产源码、科学参数或资源路由，保持同一handle重新观察。未用缺失current scalar字段或历史known_children断言故障。全目标仍为七组各10outer、11固定模型版本共5632正式槽、完整曲线/debug/持久副本及配对统计；本CPU页面不缩减验收范围。
