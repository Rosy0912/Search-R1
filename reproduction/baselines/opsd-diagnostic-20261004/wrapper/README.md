# 单H100只读distribution诊断外围（仅本地软件）

未提交、未上传、未启动GPU、未重训；不修改冻结8e repo或7ac候选/正在强化的probe。仅准备供root和独审读取的bootstrap.sh、launcher.py、双文件外部wrapper manifest与CPU测试。

wrapper文件使用独立工具目录 `/mnt/data/shaoshitong/search-r1-rl/opsd/tools/distribution-launch-wrapper-20261004-v1/`，不覆盖任何96或其它诊断bundle。实际probe/source/binding目录由root通过OPSD_DISTRIBUTION_BUNDLE及路径/SHA参数传入；FAILED96目录明确拒绝，任何未来probe更改都需要新外部binding和实际SHA。

root当前提供新的实际bundle为 `/mnt/data/shaoshitong/search-r1-rl/opsd/tools/initial-distribution-7ac-be4130324322/`，probe SHA `be41303243221b8616d485bcfed5e7797ec6fecc775e91d682a97cb545533737`，source SHA `4ebe6886d2de09186676f0388e9692c488424fddf9ce003828cc04a65931e5c0`，binding SHA `ae51e1648e990afb9da3dd24d7889e61ef3da40a145f6d27a1687aa0ece07815`。只在root确认该新nativeCPU fullbinding及49CPU独审通过后使用，旧96已经实际FAILED、不覆写或追认。外围自身不写死任何未来probe SHA，而是核root传来的这三个实际外部SHA与binding字段/文件完全一致。

bootstrap要求fresh SLURM_JOB_ID、h100-10、冻历史git HEAD8e，先按要求持root gpu-job.lock fd9，独占mkdir `runs/opsd-distribution-${actualJob}`（存在即拒绝且不写），记录自己bootstrap退出码。wrapper外部manifest SHA必传，核bootstrap/launcher真实文件和实际执行路径；调用生产原preflight获得单UUID lease，并保持fd9。

launcher复核production inherited singleGPU lease/15%idle、fresh当前Slurm running单H100/node/UID/cgroup、fd9真实inode且独占、native lease不是fd9。写ownrun allocation-request.json：实际新job、实际UID/node、真实已租约UUID，不猜编号或设备。root独立核该实际请求与scheduler/UUID，再在ownrun写allocation.json及allocation.sha256（纯64hex外部审核值）。launcher最多等300秒，缺值超时，已有错误SHA/identity直接拒绝；root不回写即停止，不自行把request当授权。

allocation恰5字段version1/job_id/uid/node/selected_gpu_uuid；UID/version不允许bool替代integer。probe collect调用传外部binding和allocation两个真实SHA、唯一新ownrun/collect输出，subprocess pass_fds同时传9及实际native leasefd。再次检查lease后执行，无train/RPC/generation/optimizer；probe自身负责完整actualGPU PCI UUID/参数/冻结/finite/256case等门禁，外围不替代它。

失败仅写本次已创建ownrun中的launcher.log/bootstrap-exit.txt/collect-child-exit.json或probe自己独占collect失败；所有先前output目录不触碰。无kill。日志文件new-open，probe拒绝旧collect时不往其写failure。原rootlock/租约控制仅按任务所需持有，退出由本Slurm job清理自己的进程。

root未来实际调用前（本文不提交）须上传核对两wrapper文件/manifest，提供环境：
- OPSD_DISTRIBUTION_BUNDLE：root实际新bundle目录；OPSD_DISTRIBUTION_PROBE、OPSD_DISTRIBUTION_BINDING必须在其中。
- OPSD_DISTRIBUTION_PROBE_SHA256和OPSD_DISTRIBUTION_SOURCE_SHA256：root独审后提供实际外部值，与binding及文件再核，source必须也在同新bundle。
- OPSD_DISTRIBUTION_BINDING_SHA256：root已审当前binding外部SHA。
- OPSD_DISTRIBUTION_LAUNCHER、OPSD_DISTRIBUTION_WRAPPER_MANIFEST：独立wrapper目录内launcher.py/wrapper-manifest.json。
- OPSD_DISTRIBUTION_WRAPPER_MANIFEST_SHA256：独审最终双file manifest实际外部SHA，禁止job内自算自证。
- Slurm实际singleH100/UID/node/CUDA_VISIBLE_DEVICES，不预填未来jobID或UUID。

11 CPU tests和shell/Python语法通过，覆盖fd9保留、leasefd9拒绝、错job/UUID/UID/node、外部SHA篡改、duplicate JSON、root等待超时、bool类型、旧output不污染、实际binding/probe文件改变拒绝、wrapper+manifest共同误改拒绝。仅是软件检查，不是GPU收集/256case重放或效果证明。local-static-sha.json记录实际最终wrapper/manifest/tests SHA，供独审；任何软件修改需更新SHA再审。
