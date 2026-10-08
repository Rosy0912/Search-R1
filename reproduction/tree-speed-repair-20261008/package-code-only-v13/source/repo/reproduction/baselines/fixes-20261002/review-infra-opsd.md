# 独立复审：基础设施、共同划分与 OPSD

日期：2026-10-02。审查固定基础提交 `afc0a3b17e9f872c792af8e70a5290a0de7ea9a9` 上本次工作区改动；实现仍在修复过程中，此报告不代表最终 SHA 全部验收。仅新增独立测试，不修改实现，不启动远端任务。

## 结论

OPSD 空监督 minibatch 跳过 optimizer 的修复通过独立实际 AdamW 状态回归；共同 holdout 与早期批次整除校验通过当前 CPU 测试。GPU 启动门禁有实质改进，但最终配置的资源校验和直接入口绕过尚需补齐，生产运行不能放行。

## 已验证修复

- 全局 token count 集体归约后决定空 minibatch：所有 rank 同步决定跳过，局部空而全局非空不跳过 backward。OPSD 仍使用全局 token mean，并保持原论文固定 teacher / full-vocabulary JSD 路径。
- 独立测试先用 AdamW 做真实非零梯度更新、留下动量及 weight decay 状态，再执行全空 OPSD minibatch 两个 epoch；参数与 optimizer step 均完全不变，`optimizer_steps=0`、`skipped_empty_minibatches=2`。
- worker scheduler 现在只有实际 optimizer step >0 才前进；普通 actor 同样返回实际 step 计数。当前复审为代码核对，未实际运行 GPU worker scheduler。
- OPSD global microbatch 等于 DP rank count，SP=1/fixed microbatch；global mini/micro DP 整除提前校验，避免加载模型后得到 local microbatch=0。
- 2% holdout 按完整 prompt 分组、固定 hash 排序与 seed；重复 prompt 同组，排除最终测试 prompt，保留唯一 UID 校验与输入/输出 SHA 校验。比例针对唯一问题数，重复行可能使行比例略偏离 2%。各算法含新增 GRPO 均由同一入口应用划分。
- GPU 双阈值严格执行利用率 ≤15%、显存 ≤15%，同时更保守地拒绝已有计算进程；物理 UUID 去重避免 index/UUID 重复选同卡。共享用户级 `/tmp` advisory locks，获取后重新检测，进程 exec 继承锁。
- run.sh 默认 GPU 数随显式 allocation，独立 TMP/Ray/HF/cache/offline 输出；资产/依赖/retriever 检查在 Ray/model allocation 前执行。不调用 kill 或管理他人服务。

## 初次复审发现，已通知主 agent

1. **P1 GPU 最终配置门禁缺口**：尾部 Hydra 参数可以覆盖 `trainer.n_gpus_per_node`，使最终资源池数量超过 `CUDA_VISIBLE_DEVICES`。`main.py` 未核对两者。直接 `python -m search_r1.baselines.main` 也可以绕过 run.sh 的 GPU 检查；`SEARCH_R1_GPU_PREFLIGHT` 仅写入未消费。应统一正式入口并校验最终 allocation。
2. **P1 多节点边界**：允许 `nnodes>1`，但当前 preflight 仅检查/锁定本机。远端节点没有等价实时门禁。应明确限制该 launcher 为单节点，或实现每节点授权 allocation 门禁。

以上是本次检查时发现，不应在主 agent 修复后仍作为未修复结论；需以最终状态复核附注为准。

## 证据与限制

运行：`.venv-cpu/bin/python -m pytest -q tests/baselines/test_infra_opsd_review.py tests/baselines/test_bapo_fixes.py tests/baselines/test_numeric_safety.py`：**11 passed**。

独立新测试：`tests/baselines/test_infra_opsd_review.py`。
未验证正式 Qwen2.5-3B、vLLM/Ray GPU worker、offload/remove-padding、真实数据/检索服务、完整训练/联合 resume。advisory locks 仅保护遵守约定的同一 OS 用户任务，不能阻止外部调度器或他人之后使用这些 GPU，仍须实际独占 allocation。数据独立性以当前固定模板的完整 prompt 精确匹配为基础，不代表语义近重复去重。

Search-R1 generation/scoring 上下文不一致按用户明确指示保留，不作为本轮待修复项目。

## 最终状态复核附注

主 agent 随后新增 `validate_gpu_lease(config)`，并在训练 main 最早执行：最终 Hydra GPU 数必须与显式 `CUDA_VISIBLE_DEVICES` 一致、nnodes 必须为 1、需要 launcher 传入真实持锁 descriptor。再次 snapshot 检查物理 UUID、双阈值/进程，再以 inode 和竞争 flock 校验真实租约；仅设置环境标志不能通过。初次发现的两项门禁缺口已修复。

新增独立回归使用专属随机测试 UUID 的真实 `/tmp` flock 文件：无 launcher 租约拒绝、最终 GPU count 超配拒绝、nnodes=2 拒绝、真实持锁通过、释放 flock 后拒绝。测试结束关闭并删除仅本次测试创建的文件，没有碰触真实 GPU 租约。

最终组合测试：**12 passed**。本复审结论更新为：**所审代码修复通过 CPU 回归与代码复核；无新的阻断问题。正式多 GPU/3B/vLLM/真实 retriever/offload 资源验收仍需独立执行。** 当前启动器明确只支持单节点，不能将这项限制描述为已支持多节点。
