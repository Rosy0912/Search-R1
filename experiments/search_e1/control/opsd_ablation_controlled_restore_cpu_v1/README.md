# 独立受控 CPU 恢复补证候选

本目录仅准备源码。发布、远端独立副本逐字节读回后，由 root 在明确空 CVD 的新进程中执行一次；GPU 七组流水不等待本补证。禁止指向 production checkpoint 或复用既有 scratch 输出。失败保留全部产物与 failed.json，不自动重试。

## 范围

实际 tiny Qwen2ForCausalLM：36 layers，hidden 8 / intermediate 16 / vocabulary 32，七个投影、LoRA rank16/alpha32，504 A/B tensors；PEFT 必须为实际生产 0.13.2。随机小型 base 在两个分支间保存并精确复用，全部模型/Adam 数值为 CPU FP32。

871 冻结原 manifest SHA `8b22e616d470d79022cc6a0b07dcb0774df21b8e35ebd7dbcaa9b19150236a41`，每个所用源文件同时核 manifest 与原 publication/readback。实际源码 bytes 编译，不读 pyc。538 独立 dtype repair manifest SHA `eee5adf6621f85274584412c19be642d8a7e4a8f6f005c84b8a90a972f9803b9`，其真实 repaired_setter 保留 504/FP32/digest/current-unchanged guard，并透传原 PEFT setter 返回。

精确 AST 提取 trainer 的 capture_actual_state、restore_accepted_rank、restore_after_targets，以及原 rank/Adam/RNG/cursor 校验、tensor digest 与 serialization 函数；输出每个原函数 line/end_line/AST digest。提取的 nested body 不改写，闭包依赖由专用 namespace 显式提供：model/optimizer/trainable/resumed/rank/config/science/opt/scaler、实际 PEFT getter/setter、TorchFacade/NumPy。实际 state_storage.optimizer_payload/hydrate_rank_optimizer 与 checkpoint.check_checkpoint 全部直接执行原 source bytes。

一次实际 Adam 产生完整 504 moments；shared payload 的八 digest 声明来自这一份实际捕获的 optimizer，属于 fixture 构造，不是八个独立生产 rank 的 pre-save 实测；后续八个磁盘 rank 状态逐个独立恢复验收。保存 default safetensors、独立八个 rank-state RNG/cursor/scaler、单个完整 shared Adam。原 checkpoint 校验要求八个真实不同 inode、全 fileSHA、schema/contract/digest/identity。随后每 rank 重建相同 tiny base，执行原 nested restore；原 initial504 FP32 张量通过原 save_torch 存成 outer-old-504-fp32.pt，完整 fileSHA/metadata 绑定 outer-old.json。每 rank 重建前重新核 old manifest 与 payload 全 SHA，实际 CPU torch.load，核 504/keys/FP32/finite/tensor digest，再由538 setter构造 original outer_old；前后再次 full fileSHA 验证。随后执行原 second RNG/scaler restore；核精确 cursor/order 和剩余一条尾批。与连续分支比较 Python/NumPy/Torch CPU 下一随机数、四分布 selected full logp、下一实际 pair_objective loss/梯度、Adam step后 current/完整 moments/groups/scaler/RNG。原 full-vocabulary reverse clip10、mixture alpha.1、mask 和梯度实现直接复用，不另写等价损失。

## 显著限制

- `CUDA_STATE_TRANSPORT_SIMULATED`：仅用八个独立 CPU generators 替代 get/set CUDA RNG 接口；原非空 uint8 schema guard 保留。不是实际 CUDA Philox 数据，不接受实际 GPU RNG 恢复。
- `NCCL_NOT_EXECUTED`：八 rank 串行 CPU 读回；不覆盖 collective、进程死锁或 GPU release。
- `SOURCE100_NUMERICS_NOT_COVERED`：没有载入 production source100；CPU FP32 前后向不接受 BF16/FP16 GPU 算术、overflow transport 或显存。
- scaler 用原 enabled state machine（scale128），前向仍 CPU FP32。这是恢复/下一更新一致性检查，不是 P/J 实跑精度证明。
- 执行真实 nested 恢复主体，但不执行 trainer.main、collectively、完整 checkpoint writer/atomic directory commit。fixture manifest 在独立 scratch 中构造，不假称 production checkpoint 已 resumed。

## 显式入口（root 填真实已发布路径与 SHA）

```text
CUDA_VISIBLE_DEVICES= python -B check.py
  --source-root /ABS/opsd_ablation_v1
  --source-manifest /ABS/opsd_ablation_v1/source-manifest.json
  --publication /ABS/source-publication-871dc46.json
  --publication-sha256 110df349c773a7a7ecfdbea4b0b9026eeeb97072a1b8e6efcc14db7c400e25c8
  --repair-root /ABS/opsd_ablation_dtype_recovery_v1
  --repair-publication /ABS/actual-538-publication.json
  --repair-publication-sha256 ACTUAL_SHA
  --output /ABS/NEW-EXCLUSIVE-SCRATCH
```

stdout 为实际 result JSON；scratch 中 result.json/failed.json 与 fixture files 全保留。没有网络、HTTP、Node 生产文件修改、global RNG 恢复到生产进程、信号、模型下载、Git 或自动重试。本候选目前只有 AST 静态检查；真实执行 stdout/returncode 必须另行保留。
