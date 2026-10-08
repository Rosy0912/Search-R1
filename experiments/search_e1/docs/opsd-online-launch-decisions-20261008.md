# Search-E1 在线 OPSD 新八卡运行

用户最新明确答复：“保留10，200，hkust存户到本地即可”。三项待定已解决，HKUST 本次允许节点本地保存输出；不将本地ext4称为共享盘。

从 archive GRPO checkpoint100 权重开始，冻结source100教师，fresh LoRA/Adam，不增加GRPO。学习率5e-6，每条轨迹有效token先平均、再按实际global pair数平均；KL词表分量上限10，梯度norm clip1，每卡micro1/global32，其他原适用设置保持。每外层512训练题各5条最新student轨迹，重新评分与同题配对，再对可用pairs做一遍优化；200外层不等于200次Adam。真实Adam次数与有效样本数独立记录，尾批用实际分母。

新目录：HKUST1 `/data/shaoshitong/search-e1-opsd-online-archive100-8gpu-20261008`。GPU7保留原私有18004 FP32检索，与本任务worker共享；仅身份精确匹配的本任务0–3常驻诊断检索将在源码和日志备份、四个旧资源锁和外部占用核验后退役。K4 archive原作业与其他进程、公共检索路由不改。旧OPSD/评测/失败轨迹/检查点全部保留，M3/M4 GPU不得并发。

在目标环境的CPU准备中，发现Torch2.4缺少props.uuid，已修复为实际nvidia-smi inventory与明确PCI_BUS_ID可见映射；该修复只涉及身份记录。恢复与清理扫描只收集正式原子提交checkpoint，不计入尚未rename的pending manifest，失败目录不删除。

保留实际生成轨迹、检索请求完整响应及每个有效token的实际训练loss；裁剪前后KL/触发比例记录，top32记录每外层前8192有效位置，是有上限前缀，不冒充总体分布。全部8rank Adam与Python/NumPy/Torch/CUDA RNG和精确cursor/order进入完整恢复checkpoint。源/初始/u1/u5/所有外层end与旧资产保留。完整恢复状态u0/u1/u5及当前最新状态另同步本地控制端校验；仅节点本地输出的风险已由用户明确允许。

本文件记录配置与启动前准备，不证明GPU已启动或五次真实更新已验收。实际进度与验收以新run的配置、process identity、status、checkpoint和独立CPU实读receipt为准；五次更新后同一完整运行继续，不从0重复测试。
