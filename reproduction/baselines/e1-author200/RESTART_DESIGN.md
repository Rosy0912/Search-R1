# 用户授权的新纯OPSD fresh200实验

授权记录：work/RESTART_200_AUTHORIZATION_20261006.json。2026-10-06最新人工消息明确停止旧3210并重跑；旧300继续指令被此次新指令取代，新目标未暂停。旧冻结a096与原件保留，不热改、不接续旧检查点、不计旧轮数。

新实验从官方Qwen2.5-3B-Instruct revision aa8e72537993ba99e69dfaafa59ed015b17504d1重新初始化学生与固定教师。继续读取已核对的共同manifest0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24：训练169028/验证512/阶段64/最终51713。GPU训练、教师前向及GPU E5/FAISS仅在h100-10，Wiki2018/top3/loopback。固定教师privileged user上下文、纯OPSD无PG/reference KL/entropy保持原语义；全参数更新保持，不擅自改成原作者LoRA。

用户新值：学习率5e-6（按前一张用户表格，无warmup、constant）；全模型全局L2梯度裁剪0.1；每次optimizer更新全局32条；每题1条响应；采样T1.1/top_p0.95/top_k20；词表分量裁剪0.05；horizon200，禁止第201轮。

仍然每outer取512个问题，故512轨迹/32=16次八卡同步Adam，每rank每outer16。actor micro global16/local2、oldLP global32/local4保持；localmini4由2个micro累积，完整全局监督token分母不随切分变化。首5各rank80Adam；scheduler每outer一次，LR始终5e-6。用户只修改“采样temperature”，蒸馏softmaxT保持1，不暗中把二者耦合。这与作者共享T1.1的脚本存在已明确记录的差异。

分量0.05按用户明确要求恢复，不解释成gradient clipping或PPO clipping。该旧目标可能产生反向梯度，原诊断证据保留；有限loss/梯度不等同效果改善，新验收须记录初始512、首5阶段64和行为指标，若行为退化须保存真实证据并处理，不伪称算法提升。

沿用已审查的隔离预GPU源/模型/数据/工具全文SHA检查、8rank真实CUDA/NCCL路径、本机检索核验；新源码和工具需要重新冻结并严格绑定新配置，旧签名不复用。

真实5轮保存HF和原生完整状态（8rank模型/Adam/scheduler/RNG/教师、driverRNG与数据游标、算法状态），在第6次取数前独立审查和root实际签名放行；同一作业、同一进程继续200，不重启替代连续状态。保存5/25/50/75/100/125/150/175/200，初始化及每50验证512，阶段5及每50为64，最终200后51713评估/最终HF和真实终态/资源释放检查。尚未取得实际新jobID，不预填或声称5验收。
