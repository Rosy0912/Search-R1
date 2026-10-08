# 待许可、待独审：3341 actor-inputs无损归档候选

本轮仅本地准备代码，没有部署、远端执行、压缩、删除、替换、改训练源码或配置。当前训练继续；诊断读取止于北京时间03:01:28。

候选入口是`trajectory_actorinput_archive_candidate.py`，只限h100-10现有同机ext4上本任务3341旧完整trajectory目录内的`actor-inputs.pt`。冻结格式写入证据是`package-code-only-v14/source/repo/search_r1/baselines/trajectory_log.py`：每轮保存`actor-inputs.pt`、已经gzip的`trajectories.jsonl.gz`和原始`manifest.json`，manifest持有两项原SHA。后两项不会重写或再次压缩。这里的PT是完整actor输入证据，候选按原始字节处理，不导入torch、不加载模型或执行训练。

单轮必须满足：原trajectory manifest状态和job/source/config/data/outer/attempt全部绑定；对应attempt结果明确COMPLETED_OPPORTUNITIES_OUTER_CREDITED、八rank及scheduler完成；最新complete-native指针及manifest SHA一致；所选轮至少落后最新完成轮2步。运行中轮、最近两轮、失败/拒绝轮和缺凭证轮被拒绝。

授权后的`archive-preserve`采用gzip level1、一CPU线程、nice19/ionice3、8MiB/s输入节流、同机任务路径flock单实例和独占intent。源完整SHA必须等于原manifest，压缩文件完整SHA另算，再完整解压并核原SHA及尺寸。失败保留源和自身partial证据；已有intent/archive拒绝覆盖或盲重试。成功归档默认仍保留源，因此**此阶段释放空间为0**；压缩率与将节省多少空间尚未测量。

候选支持两种未来访问方式。`restore-preserve`把已验证解压内容以不覆盖的同机hardlink恢复原`actor-inputs.pt`路径，归档保留。供未来独立CPU验收入口使用的`verified_actor_input(directory, request)`context manager一次只解压一个文件，在读取前核压缩SHA、解压全部原SHA/尺寸后返回handle；消费者只在context内读取`handle.read_path`，它是仍持有的只读文件描述符的`/proc/self/fd/...`路径，不能改为重新打开原路径。finally只清除带明确inode和独占请求名的自身临时副本。它不恢复全部历史PT，因此临时空间只增一个文件。仍在原路径的文件也需完整原SHA核验，原文件不会清除。

路径修复针对独审发现的祖先目录symlink竞态：候选从固定ROOT开始逐组件以`O_DIRECTORY|O_NOFOLLOW`打开并固定实际ROOT/RUN目录FD；后续所有元数据、源、归档、临时文件与恢复操作都相对固定父目录FD执行，不采用先检查路径再用不可信字符串打开的方式。压缩源和hash读取均为no-follow叶文件FD；临时文件的父FD在创建、消费和清理期间保持存活，容量与挂载检查使用已固定RUN的FD。未来CPU消费者通过持有的只读FD路径读取核验过的同一文件。这项修复仅本地准备并待独立复核。

当前冻结验收消费者实际读取历史原路径：`work/tree-speed-repair-20261008/package-code-only-v14/check_behavior_mu_cpu.py`第60行调用`held_tensor(directory/'actor-inputs.pt',manifest['payloads']['actor-inputs.pt'])`。本地冻结trainer中`trajectory_log.save_rollouts`负责写入，未找到训练主循环重读历史actor-inputs的调用；这**不证明原路径替换已安全**。最终验收可能再次调用该checker，所有原PT一次恢复又会吃回全部节省。若未来允许替换，需要另行批准并独立审查一个archive-aware CPU验收入口，使它逐轮使用上述单文件context manager；不能热改正在运行的checker/trainer。

本候选不提供源删除/替换动作。未来实际释放空间必须经人明确批准，审批对象应是具体旧完整目录清单、逐文件原SHA、压缩SHA、解压SHA、真实节省量及CPU单文件验收路径，并由独立恢复/替换入口执行。原manifest及轨迹gzip字节保持不动；不得把本地准备、代码推送或archive-preserve成功当成删除授权。

示例仅供审查，不执行：在未来许可后的h100-10上，先用`--action plan --directory outer-000007-attempt-000007`查看元数据选择；批准后才能使用`--action archive-preserve --directory ... --request-id ... --approval-reference ... --execute`。本轮不创建任何远端执行入口或调度作业。

原容量诊断及首失败记录在`metadata/OWN3341_DISK_DIAGNOSTIC_20261009.json/.md`和两份原始JSON。代码候选将正常提交、推送并独立读回SHA；真实小CPU fixture和独审由后续审查完成，本轮仅语法解析。
