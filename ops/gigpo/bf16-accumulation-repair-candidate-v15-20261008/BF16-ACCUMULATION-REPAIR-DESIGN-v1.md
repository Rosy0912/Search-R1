# BF16 accumulation 最小运行数值候选 v1

日期：2026-10-08。状态：**仅本地候选；未执行、未测试、未创建批准、未提交或推送、未部署**。本文件不是概率独审批准，也不是运行授权文件。

## 授权和科学边界

本轮已授权 GiGPO fresh official Instruct200 的最小数值稳定修复准备；变更执行前必须经过 clean-context、非作者静态 review。不得重复 CPU、native 或 GPU 验证 job。作者仅复制隔离运行目录、编辑必要脚本、生成静态文件清单与 diff，未连接远端、未运行候选、未创建或取消 job、未设置现有进程的 flag。

现有 job3332 的 source 为 `2ddbb0a1619ce7cde28b4c5be110ffe5bd11c963`，运行目录为 `safety-calibration-fresh200-v14-evidenceclone1`；实际首批 2560 轨迹已生成、Adam0 概率暂停，概率独审仍为 BLOCK。其 v14 目录和已有材料原位保留。此前已记录 HF/vLLM selected-logit 差 1.5、current/old token-logp 差约 0.829；它们证明仍存在上游数值差异，**不证明 BF16 reduced-precision reduction 就是根因，也不证明本候选必然修好**。诊断背景见同目录 `JOB3332-PROBABILITY-MISMATCH-CODE-DIAGNOSIS-v1.md`。

唯一改变的数值设置为：在每个实际 actor_rollout/ref 角色构造的最前面执行 `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`。模型与 autocast BF16、原有 FSDP 参数/归约精度、GPU、FlashAttention、dtype、micro、科学公式、数据与预算、prompt、采样分布、guard、概率容差、old/current balance 顺序全部保持原设置。该开关只关闭 PyTorch 允许的 BF16 matmul reduced-precision reduction；不声称所有 FlashAttention 或 vLLM 自定义算子都受它控制。

备选的 driver-only 设置无法保证 Ray 角色进程继承实际值，故不采用。改模型 FP32、停用 FlashAttention、调整 micro 或放宽概率容差超出本次最小授权，故不纳入本候选。

## 隔离和真实调用链

原始目录：`work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v14-evidenceclone1`。

候选目录：`work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v15-bf16accum1`。44 个文件完整复制，4 个必要运行文件改变，其他 40 个逐字节相同；无新增/删除运行文件，未修改已有 test 文件，未写 test。

候选 `calibration_driver_v1.create_trainer` 对 `Role.ActorRollout` 和 `Role.RefPolicy` 均使用 `ray.remote(CalibrationActorRolloutRefWorker)`。这个显式名称是现有 `CalibrationWorker` 的别名，不增加 MRO；唯一 setter 在 `CalibrationWorker.__init__` 的 `super().__init__(config=config, role=role)` 之前。

冻结 source 的 `RayPPOTrainer.init_workers` 分别传 `role='actor_rollout'` 和 `role='ref'`；既有 `metadata_colocation_v1` 在同一实际 Ray worker PID 内逐角色调用真实构造函数，完成角色构造后，trainer 才调用 ref/actor_rollout 的 `init_model`。因此两个同进程角色可能各执行一次构造 setter，但均发生于所有模型初始化与前向之前。此后没有 setter、恢复 setter 或热修补路径。

actor_rollout 的原 `ActorRolloutRefWorker.init_model` 经 `_build_rollout` → `vLLMRollout` → 本地 `LLM` → `LLMEngine.from_engine_args` → `SPMDGPUExecutor` → 直接构造 `Worker` 并执行 model load/profile。executor 的 `self.worker` 和 `model_runner` 位于同一进程；候选 backend identity 读取实际 executor/worker 类，要求它们分别为 `verl.third_party.vllm.vllm_v_0_6_3.spmd_gpu_executor.SPMDGPUExecutor` 与 `verl.third_party.vllm.vllm_v_0_6_3.worker.Worker`，再保留原有真实 FlashAttention runner 和全部 36 层实现检查。此为静态调用链和拟执行时元数据检查，未冒称已观察到 v15 实际 GPU。

冻结 source 静态入口：`verl/trainer/ppo/ray_trainer.py` 的 `init_workers`；`verl/workers/fsdp_workers.py` 的构造函数、`init_model`、`_build_rollout`；`verl/workers/rollout/vllm_rollout/vllm_rollout.py`；`verl/third_party/vllm/vllm_v_0_6_3/{llm.py,llm_engine_sp.py,spmd_gpu_executor.py}`。本轮只读确认该源码 checkout HEAD 为上述 `2ddbb0a...`，`git status --porcelain` 无输出。

## 实际进程读回与拒绝规则

`_bf16_accumulation_identity` 只读当前 `torch.backends.cuda.matmul` flag，并要求它严格为布尔 `False`、constructor PID 等于当前 PID、角色一致。缺字段、不同值或进程变化直接拒绝；没有 schema 兼容 fallback，没有就地重设 flag。

driver 在正式 baseline 前、每个实际训练 batch 获取前、首批 post-sampling/Adam0 暂停、独审读回后和立即首 Adam 前，对 actor_rollout/ref 两角色分别读回 rank0–7 与 8 个不同 PID，要求双方全部 False，并与首次进程/角色读回完全相同。它们是元数据检查，不是模型前向或独立验证 job。

记录和检查覆盖：

- `actual-official-zero-initialization.json` 的独立 `runtime_numerical_execution` 字段保存两角色 all8 的实际读回；`actual-native-fsdp-before-zero.json` 保存实际 actor 初始化读回。
- `actual-vllm-backend-identity.json` 保存当前 Ray/vLLM 所在进程 flag 和真实 executor/worker identity，同时保留原有 GPU FlashAttention/dtype 检查。
- per-turn sampling 和 complete old/current/reference 的原始 PT、JSON receipt、必要 neutral receipt 保存实时 flag。driver 读取时要求 flag False、原始 PT 与 receipt 一致，per-turn 前后完整状态内的 flag 也必须一致。
- actor complete-pause 的独立 `metadata.bf16_accumulation` 在 `complete_state_sha256` 计算之前加入；模型/Adam/RNG 的原始 source pause identity 字段保持原样，以保持已有 endpoint 与 source 保存身份的精确比较。fixed reference identity 同时只读校验该 flag。driver 完整 typed state 另行绑定两角色的 `runtime_numerical_execution`。
- 原有 update 边界校验在同一已存在 matched stage 内只读要求 False；概率 tolerances、guard 和非作者批准条件原样保留。

## 新不可变 binding 与唯一实际验证

候选 launcher 在任何 guard、services、Ray 或 GPU 初始化前，要求新 binding 明确包含 `"allow_bf16_reduced_precision_reduction": false`，并在 `binding.files` 中恰好一次列出当前 v15 目录实际执行的 run、driver、worker、state 文件。既有 byte size/SHA256 校验继续逐文件运行；列旧 v14 路径或缺少新数值声明不能执行 v15。

后续由 root 将候选源码与运行入口/必要配置完整提交并推送，记录 source commit、候选备份分支、远程 SHA 与文件 SHA256 核验，再交 clean-context 非作者静态 review。作者当前未创建 binding、审批、部署或 job。**只有绑定 v15 精确文件和数值声明的新不可变 binding 经要求的独立静态 review 后才可执行；禁止向 job3332 活进程设置 flag，禁止热改其 frozen runtime。**

是否消除数值问题只能由授权的 fresh official Instruct200 **实际正式首批**证据判定；不另起 CPU/native/GPU 验证 job。若正式首批仍不满足原概率审查，继续按原 Adam0 暂停与证据保留路径，不以经验 BF16 误差、提高容差或宣称候选根因来放行。

## 改动清单和静态字节证据

以下 SHA256 是候选实际文件；完整 before/after 大小与 SHA256、44 项逐文件列表及 40 项 unchanged 列表在 `BF16-ACCUMULATION-REPAIR-FILE-MANIFEST-v1.json`。统一 diff 在 `BF16-ACCUMULATION-REPAIR-DIFF-v1.patch`。

| 改动文件 | 候选 SHA256 | 必要用途 |
| --- | --- | --- |
| `calibration_worker_v1.py` | `b936264ad7b0786a19a389aace7b55f6207450eaeacd9f3af94a62a12c6eeeef` | 唯一构造 setter；真实 PID/role/backend、状态和原始概率 flag 读回 |
| `calibration_driver_v1.py` | `07bd73a5bdf74bd60fe0a1d108fd27082b7b96d0118b43e5f1e90019024a47df` | 两角色 all8 只读准入；官方零状态和概率材料读回检查 |
| `calibration_state_v1.py` | `fb3f96c87a8a4ddd891a28cae8b8a7d3351d2e021d6ac7b3cd906d8f2cf83ebb` | 独立 numerical 执行字段的严格 False/PID/role 校验；原 digest 和 pause 验证算法原样 |
| `calibration_run_v1.py` | `371afcedfb34fe320538f948e787860c4bc884ea3711d9d760514adcb9f43aa2` | 新不可变 binding 的数值声明与实际 v15 入口文件覆盖检查 |

原 v14 全文件清单摘要：`7d6f4dce3bbfe45f287831103d555fb4063155e11c0e732a10761bc7e37876f6`。

候选 v15 全文件清单摘要：`54545ec5d6bac0c58d744d90fe12b41f8d9f22d351bccf3450e8c269670ee4ff`。

摘要定义：按相对路径排序，将每项 `{path, bytes, sha256}` 以 JSON `sort_keys=True,separators=(',', ':')` 序列化后取 SHA256。它是本地静态字节证据，不是 GPU 成功或已发布证据。原 v14 与 frozen source 不由候选作者改写。遵循 `/Users/shaoshitong/Documents/remote_codex_control/memories/code-publication-before-node-release.md`，root 的后续完整提交、推送和远端读回核验尚未完成，当前不能声称已备份完成或已获概率批准。
