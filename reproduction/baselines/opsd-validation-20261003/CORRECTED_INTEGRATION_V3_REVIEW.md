# v3隔离integration候选独立审查

结论：**STATIC/CPU SOFTWARE REVIEW PASS；可提交默认关闭的仓库候选；不可部署GPU/不可验收actual500/HF或效果**。审查快照work/reward-protocol-fix/integration-candidate-9fb-v3/candidate-manifest.json SHA40adc6398d1df94a95a1fa68b4b62a9e2d7a3e0a43e0738ec924dd9f8d20c036。独立逐文件核全部清单SHA一致，隔离repo父HEAD确为9fb3a74c3249a0a7a688701ce4439764605a9e1f。未改主work/opsd、源8e训练3007、既有预览或GPU状态。

## 代码范围与v2问题修复

逐字节比较9fb父版本：actor.py、distillation.py、native generation.py、fsdp_workers.py、vllm_rollout.py、ray_trainer.py六文件均不变。仓库改动只有显式corrected入口的stage传参、可选worker/call guard、main的显式SLURM_JOB_ID env传递、新capture module与测试。默认不开启仍不进入这些额外RPC/guard，不改变actor、teacher或采样数学。tools仍为独立blocked候选，不据此替换实际HF执行bundle。

v2提出的selectedGPU进程内存NaN/负值、cgroup数字/events内容已严格解析；prescription使用exact字段集合（version/phase/source/input/model/budget），不允许receipt、自身SHA或futureproof额外字段；子进程cwd=repo、实际corrected_evaluation.__file__及runtime source SHA核查避免父cwd shadow；process身份包含raw UID status/stat/cgroup/cmdline。前置--stage/--stage-sha256明确传入外部工具，与prescription→layeredreceipt→finalstage无回链顺序吻合。HF actual-final ancestry变换检查保持v2正确条件。

## Worker RPC及generation guard

现有Worker.execute_with_func_generator及DP_COMPUTE_PROTO_WITH_FUNC API确实允许FunctionType回调加DataProto；8row CPU probe按worldsize8得到每worker1row，再collect原生DataProto concat。callback在实际初始化后的worker读取sys.modules路径/摘要、raw PID/start/UID/cgroup/cmdline、rank/class、actual CUDA properties UUID和环境stage/job；无forward、parameter gather、collective、optimizer或显式CUDA tensor分配，仅返回CPU tensor/object row。当前device/property调用以前提已有CUDA初始化为限。

独立CPU cloudpickle roundtrip验证callback仍为原FunctionType/by-reference模块函数，8个原生DataProto分块序列化后audit_spec metadata保持；这不执行真实Ray RPC或GPU callback。实际cross-process模块解析/env继承、8rank/source/UUID与Slurm+sampler关联仍需真实独立烟测。

validator检查八个唯一rank/PID/UUID、worldsize8、固定h100-10、stage/job、raw身份自洽、固定worker/rollout类型和实际source paths/SHA。GenerationCallGuard替换实际driver group.generate_sequences方法，要求每次do_sample严格False/validate严格True，实际8..64行且整除8，输出同数量DataProto（n1），记录调用序号并拒绝被替换/拒绝/未使用。它不重写input/output；actual vLLM greedy override由immutable source/budget pin保证。full_final replay新增runtime audit检查，原zero-update dispatch守卫保留。

## HF sidecar安全范围

hf_launcher.run先核源8e clean与实际preflight源，再用原inherited GPU lease/preflight15%检查单GPU，核globalfd9/perGPU lock、fixed checker SHA/path，才Popen同一个immutable5586 checker。child明确继承自己锁fd；仅该Popen子进程可在失败后terminate，不操作其他作业。stdout保留并捕获真实checker子孙身份与source，写独立JSON及raw marker。lease捕获核fd/device/inode与kernel FLOCK WRITE，幂等重assert同一个open-file-description，未unlock；这避免仅凭相同inode承认外部descriptor锁。

producer与validator原语存在，但未被接入新的实际HF bundle/template/源allowlist，也未在pre-stage完全串起launcher/capture/preflight inventory/lease/raw marker与全部角色证据。现有原script精确guard-only变换不能冒称已经接受launcher插入。standalone launcher/capture CLI仍exit78；没有GPU执行、真实锁/Slurm/模型验收。

## 独立CPU与Linux结果

既有.venv-cpu环境，CUDA空、OMP1、SEARCH_R1_REQUIRE_REAL_DATAPROTO=1：
- 独立仓库四文件测试：**105 passed/2 warnings/3.74s/exit0**。
- 独立tools三文件测试：**75 passed/4.29s/exit0**，需明确PYTHONPATH指向该候选repo。第一次从tools目录未设置它得到74pass+1 missing search_r1；正确候选路径重跑全75通过。这是命令配置问题，不混用失败结果或隐藏重试。
- 独立manifest、六个native/math文件字节不变及CPU function/metadata serialization检查通过。

已读取根任务保留的v3-native-cpu-retry-result.txt：隔离Linux9fb覆盖精确40adc文件、真实DataProto/CUDA空/OMP1原生**326passed27.94s**（rawSHA83a9d6a165d9634eef597f045de04d777a77e9650b9b05b96fa594f0f7bb4247）与tools**75passed4.61s**（rawSHA36a03ad866514e9eea499ca6e38c74ee4050e2ee6038a730a3352cca7a3e1687），原训练8e clean保持。首轮环境未包含既有pytest-lib造成的失败由根任务保留；重试复用已有环境/依赖，不安装、不占GPU。上述CPU结果均不是actualCUDA/Ray8worker、Slurm、HF435/3shard或actual500证据。

## 仍未完成、部署前必须严格补齐

1. NEW HF bundle/template源门禁与actual-final ancestry要显式允许且固定sidecar插入，保留5586和全部原formal/HF数值检查；不得改冻结旧script就说原manifest已执行。
2. 前置audit全链需要实际launcher/capture/preflight源、selected inventory/perGPU/global lock、raw marker/file严格相等、PID/start/UID/cgroup/checker lineage以及独审SHA串联；当前strict bridge CLI78/internal reject不能移除后直接上线。
3. actual500正常终态、periodic64/旧final51713、HF500真实重载与cleanup仍待发生/采集；model角色要与原资产manifest及实际HF严格绑定，UNFILLED不能代替证据。
4. 真实native8worker/Ray/CUDA UUID兼容、selectedGPU PID/UUID sampler、本地GPU E5/FAISS/lease/cgroup与effective calls必须独立运行核验。
5. postreceipt必须外部绑定runtime.spec.eval_repo/job_id/uid与真实固定repo/Slurm任务及selectedGPU inventory；当前validate_completed_runtime主要核source/stage以及内部rows一致，不能由audit spec自报代替终态/资源身份。worker raw UID向量目前仅校首项为ownUID，下一更严格runtime层应要求四项UID均ownUID并与外部job UID关联；HF sidecar validator已要求四项相同。
6. 最终需要原zero-update guard0、809评分view批/尾1、逐题及aggregate CPU replay一致、model前后SHA、own normalterminal/cleanup和同输入预算base/HFpaired效果。不能把未来post证据作为prestage循环前置，也不能跳过它。

独立同意根任务curated提交默认关闭的仓库软件helper/tests及记录；这不是对blocked tools的activation批准，更不是GPU/actual500/HF500/效果验收。下一真实bundle与完整runtime/postreceipt必须新不可变身份、新证据和新独审。
