# CoSearch迁移到HPC3（2026-10-08人工授权）

用户明确要求“后面的实验也需要迁移到HPC上跑……也完成cosearch的迁移”。本次指令覆盖此前禁止跨集群通信中与CoSearch H100→HPC3迁移冲突的部分；仅用于该实验的已有完整检查点、源码、原始数据/检索资产和环境迁移，运行阶段的检索/训练/输出全部在HPC3内。不是向其他agent发消息或抢占其他实验资源的授权。

源：h100-11(10.33.4.14)，/dataset/from_h100-16/shaoshitong/search-r1-rl/cosearch。最新完整X checkpoint203已全24SHA验证。禁止以U12/P5或HF替代完整203。
目标：HPC3 jxu957，/data/user/jxu957/experiments/cosearch-continue300-x203-20261008。实查/data为/LK02 dpc共享文件系统，不是登录/计算节点本地盘。原件保留，传输和目标完整SHA校验完成前不得清理。使用hpc3.md已有独立SSH配置，严格hostkey，不复制私钥至远端。

目标总globalstep300，恢复first204；完整主模型/ranker/RLAdam/scheduler/data、同原始166222TRAIN/64val、同T1/BF16/PPO/4096/bounded2等冻结语义。迁移及环境/入口适配未完成前不得声称HPC训练已开始或恢复验收通过。新的纯加载0updates和真实新8GPU ACK/独占租约/状态相等/data/singleload必须验证。输出必须写已核验HPC共享盘。源码/入口/恢复/评分脚本完成适配后必须提交推送并验远端SHA。
