# OPSD 单H100只读分布诊断外围独立审查

结论：**固定外围静态/CPU通过，无发现的启动阻断；允许root在已授权范围提交新的自有单H100只读诊断。** 不授权训练、生成或检索服务，不是实际GPU/replay/效果验收。审查者没有提交GPU作业、操作远端或修改外围文件；只写本报告。

## 固定身份

wrapper-manifest SHA256 `6dbe7a621a6609cd0cf3f0d42a1e949eeff528dc433da6d43727af228cd3e2c0`。
bootstrap.sh `2597437e85d0ae3e9ef8786f220795e7519a4f6bf4f6688f5bee6f940be0c96a`。
launcher.py `8bfd05dabc2f59b2664fadfb521b36802c8644916490698855dd5a5f855eca9b`。
root-reviewed-bundle-parameters.json `8ec9b56ac1456c8819e392d31b9614809cbd013dde95ec375958b06c29afe3d3`。

本地逐bytes核对manifest两文件，实际相符。参数绑定新probe be413032...533737、source4ebe6886...31e5c0、binding ae51e164...e07815；禁止FAILED96目录，未来jobID不写死。bootstrap所用repo仍固定历史8e；候选teacher函数外部绑定，不覆盖生产源。

## 描述符与占用边界

bootstrap以fd9持有自己root/gpu-job.lock，保持bash父进程存活；native 8e preflight只查真实已分配GPU双15%阈值并获取对应租约，os.execvp保持9及设置inheritable的租约FD。launcher再次调用validate_inherited_gpu_leases(1)，拒绝多卡/FD9冲突，核fd9设备inode及独立open不能重获锁；传给probe subprocess.run的pass_fds明确为9+单GPU租约。probe还在硬件门禁检查锁文件inode/归属，不靠wrapper自报。

独立额外执行CPU真实OS FD probe：实际bash exec9→从固定8e AST/完整文件执行的preflight.main（GPU inventory明确mock）→实际os.execvp→实际subprocess.run(pass_fds)→末子进程fstat/flock校验。fd9与租约fd3全程存活，inode/dev对应，独立open取锁失败，returncode0。此为本机CPU/POSIX描述符证据，不假称实际Linux Slurm/GPU租约已运行。临时锁目录自动清理，无全局训练资源改动。

## 实际allocation审批与输出

launcher先核当前job RUNNING、NodeList h100-10、NumNodes1、TresPerNode单H100、实际UID、cgroup当前job及租约UUID。只生成当前allocation-request.json，再等待root提供allocation.json+外部sha最多300s。所有job/UID/node/UUID必须逐项等于实际request；超时退出，不猜未来ID或代写批准。等待后重验lease/idle再执行唯一probe collect命令，无optimizer/train/generate路径。

run路径固定runs/opsd-distribution-currentjob，bootstrap mkdir无-p且trap仅在mkdir成功后安装；原run存在时先拒绝，不写旧退出标记。collector子目录必须不存在。probe failure仅当前独占创建目录记录；wrapper日志/exit属于当前新作业，不能把returncode0当完整分布重放或效果通过。

## CPU及native记录

独立wrapper11测试 **11 passed /0.04s**，shell syntax通过。测试覆盖actualrequest schema/jobUIDUUIDnode、FD9冲突/多lease、external SHA、300s等待timeout、JSON重复key、已有output保留、source/probe/manifest bytes替换等拒绝。

已独立读取root保存真实nativeCPU记录：新be完整binding `NATIVE_CPU_FULL_BINDING_PASS ae51e164...e07815`，源/完整资产核对与函数toy self-test真实执行；旧96失败保留。另wrapper actualnative `NATIVE_CPU_WRAPPER_BYTES_PASS`，两文件和manifest SHA与本地固定值一致。这些没有模型前向/GPU执行，不将nativeCPU当实际GPU。

## 待实际证据

root提交后需保存真实新Slurm job与allocation-request/外部批准SHA、wrapper-runtime继承FD记录、实际CUDA0 PCI→UUID与ownPID/UID/cgroup、三阶段精确加载/FP32或BF16/backend/tied3085938688/finite、256原始case及manifest完整SHA、同版本CPU replay、terminal/真实child exit。当前报告不包含未来job或这些尚未发生事实。

科学范围保持64个按顺序训练初始state，base/HFstep5学生对两种teacher条件。候选条件logprob和CPU logit方向不等于生成完成率、EM/F1、后期history因果或新的训练结果。无A800/跨集群计算，无旧job/全局服务管理。
