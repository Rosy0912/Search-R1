# A800 两卡 CUDA/FSDP 短测试

2026-10-02，`baai-a800` 的物理 GPU 0、1，代码版本 `ca95b89`，退出码 0。
独立目录 `/data/shaoshitong/search-r1-rl-validation-20261002-ca95b89`。
[原始结果](evidence/gpu-a800-20261002-run2/result.json)、[分配检查](evidence/gpu-a800-20261002-run2/allocation.json)、
[日志](evidence/gpu-a800-20261002-run2/gpu-run-2.log)、[校验清单](evidence/gpu-a800-20261002-run2/manifest.json)。

启动前连续检查利用率和显存比例均 ≤15%，另外拒绝已有 compute process 的卡。
模型准备完成后再次检查；仅向子进程暴露 GPU0/1，限制每进程 allocator 预算为该卡10%。
检查与启动之间不是集群调度器提供的原子预约；锁仅协调本测试目录的任务。
本次两卡检查时均为0%、4MiB/81920MiB，GPU7已有任务没有触碰。
测试结束GPU0/1恢复4MiB，GPU7占用仍为68494MiB，见[退出后状态](evidence/gpu-a800-20261002-run2/post-run-gpus.csv)。

| 路径 | 实际验证 | 结果 |
|---|---|---|
| GiGPO / Tree-GRPO / BAPO / TOP-D | 双进程 NCCL/FSDP，真实生产 actor 的 logprob、CUDA advantage/PPO、两次 optimizer step、有限非零梯度、非零有限参数更新、full-state checkpoint 保存 | 通过 |
| OPSD | 实际 DistillationActor，冻结 CPU tiny teacher + GPU FSDP student，特权答案条件，full-vocab loss/backward、有限非零梯度，两次 optimizer step、checkpoint 保存 | 通过 |
| CoSearch | GPU0 上真实 HF ranker forward/backward、三条事件训练、一次 optimizer step、模型/tokenizer/optimizer 保存 | 通过 |

首轮 [run-1 记录](evidence/gpu-a800-20261002/result.json)保留。独立审查指出默认 AdamW
weight decay 会削弱 delta 断言，因此最终 run-2 对 actor/ranker 显式设 decay=0，
并逐项断言/记录 actor gradient norm 有限且非零，两卡重新检查后全部通过。
审查见 [GPU 测试审查](review-gpu-smoke.md)。

测试使用离线随机初始化的小型 Qwen（hidden64、两层）和构造的 token 轨迹。
TOP-D 的 teacher logprob 是构造的非恒定值，不是独立训练教师的真实 scoring。
CoSearch 排序事件是构造的，未跑真实 ranker generation、检索器或端到端搜索。
参数更新只证明该执行路径可运行，不代表算法效果、on-policy 一致性或数值理论已验证。
FSDP checkpoint 已写出，未进行完整训练 resume 验收。

远端独立 venv 使用现有只读 Python3.12/Torch2.9.0+cu128/flash-attn2.8.1，
另装 transformers4.46.3/tensordict0.10/ray2.41 等依赖，不更改既有环境。
远端直接下载超时后按下载 memory 本地中转，两份归档均已 SHA256 校验；
[传输记录](evidence/gpu-a800-20261002/transfer.json)随证据保存。
生产 reviewed base 是 Torch2.4/vLLM0.6.3，版本差异需在正式验收中解决。
约20.1MiB是 tiny 测试的 PyTorch allocated peak，不能当成GPU总占用或3B资源预算。

尚未覆盖：3B 模型、vLLM rollout、Ray worker transport、真实 Wiki18/E5 检索、
完整 trainer/controller、remove-padding=True、offload、序列并行、BAPO真实阶段验证集、
七集指标和长上下文生成/scoring审计。前述限制保留，不把tiny测试当成正式短训练验收。

复跑必须指定当时检查通过的卡和一个新的输出目录，例如：

```bash
cd /data/shaoshitong/search-r1-rl-validation-20261002-ca95b89/repo
PYTHONPATH=. OMP_NUM_THREADS=2 ../env/bin/python reproduction/baselines/gpu_smoke.py --gpus 0,1 --output ../run-3
```

不要仅依据这个历史 GPU 编号分配生产作业。脚本会实时检查，已有输出目录会拒绝覆盖。
