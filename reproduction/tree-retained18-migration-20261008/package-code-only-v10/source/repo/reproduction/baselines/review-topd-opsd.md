# TOP-D / OPSD 实现审查

日期：2026-10-02。基线：`b251ba6b503217a59cd3395c1326d7b66b602989`。审查时工作树是开发中版本，本报告不代表已固定最终实现commit。只读审查算法代码；仅本报告写入独立baseline工程。未修改原Search-R1、未启动远端/GPU任务。

## 证据范围

- TOP-D：[作者论文 2607.04751v1 §3–4](https://arxiv.org/html/2607.04751v1)。没有已核实官方源码，当前实现是paper-based search adaptation。
- OPSD：[作者论文](https://arxiv.org/html/2601.18734v3)，[官方固定源码 ae7d2519e94920c4eb6206c0c26de46d9c50abae](https://github.com/siyan-zhao/OPSD/blob/ae7d2519e94920c4eb6206c0c26de46d9c50abae/opsd_trainer.py)。beta端点、分量clip、teacher冻结与目标位置按源码/数学方向核对。
- 本地执行 `.venv-cpu/bin/python -m pytest -q tests/baselines`：24 passed。包括真实tiny Qwen teacher采样token scoring、共享OPSD CPU/GPU loss函数的CPU反向/optimizer更新、teacher无梯度、ranker真实HF更新，以及真实fit控制流的替身worker测试。

## 已修正并复核

- 显式select_rows保持DataProto批对象，避免旧__getitem__返回DataProtoItem。
- OPSD以全DP token总数/world_size归一化local loss，匹配平均梯度后的global tokenmean；teacher在CPU冻结，不参加actor optimizer。
- TOP-D默认不使用privileged train-answer context；teacher路径canonical检查避免同目录别名，OPSD单独启用答案privileged。
- teacher目标位置用有效response attention累积映射；核查预测位置target-1，teacher/student tokenizer vocab与特殊token一致。
- 纯蒸馏关闭旧reference KL/entropy，old snapshot及TOP-D teacher奖励在内循环固定；OPSD beta=0 FKL、beta=1 RKL、中间JSD及官方词表分量级clip正确。
- OPSD限定SP=1、固定microbatch、每rank最多1；完整logits预算failfast与CPUteacher内存成本已作为工程限制，不称chunk_size提供严格内存界。

## 二轮主要问题：终轮已修正

已解决 P1：TOP-D 原本沿用等权micro/rank均值，现仅该分支使用 `global_token_denominator` 与 `token_mean_ppo`：local masked loss分子除以全DP有效token总数/world_size，不再额外除固定gradient_accumulation。old snapshot/teacher reward/advantages在epoch之间保持不变。GRPO和其它原路径不受此修复影响。

终轮复核了TOP-D全局token权重以及纯OPSD/TOP-D不创建Ref role、driver bypass KL-in-reward。实跑24个CPU测试通过：不同micro token数1和3、loss权重1与3得到梯度-2.5；两个真实CPU Gloo/DDP ranks验证global tokenmean 2.5，含空动作rank后结果3.0。空micro仍保持零loss反向的计算路径。当前CPU与静态代码审查范围内，无残留critical/major发现；这不是原论文完整上下文算法的忠实复现认证。

该结果只证明数值内核和CPU DDP平均梯度语义，不等于真实FSDP/CUDA部署验收。

## 未验证的运行条件

CPU通过不能证明CUDA/FSDP训练可用。尚未验证8卡collective同步、FSDP paddedforward/activation checkpointing反向、bf16与flash-attention精度、参数/optimizer offload、实际显存/host RAM、CPUteacher速度、长搜索context、vLLM权重同步和完整GPU optimizer step。完整actor logits及各chunk保存到backward的autograd图仍占总B×L×V内存，预算只预估logits、不是总峰值上界。

OPSD使用Search-R1短训练答案而非数学参考推理；固定独立teacher而非动态同权重teacher；student仍用原search prompt/工具循环。这些是明确适配，不是原数学recipe性能复现。TOP-D依赖明确trained teacher及其成本，不能以teacher增强冒充算法增益；论文理论不保证检索环境EM逐步提高。

## 继承的上下文一致性限制：终轮增量复核

`SearchRollout.advance` 在实际生成动作之前比较 `state.prompt + state.response` 与 `state.rolling`；默认report记录逐动作标记和`rollout/context_mismatches`，error模式在不一致动作生成之前失败。新增反例确认起始前缀裁剪不一致时error模式尚未调用rollout worker，report模式保留共同基线行为并报告计数。

README明确：实际生成的条件前缀与最终重构轨迹重算old logprob可能不同，特别是长搜索上下文；不能据此宣称TOP-D理论条件满足或严格on-policy。在正式训练前须审计报告；若修复上下文策略，原GRPO对照必须同样修复。该诊断未修复实际策略分布差异。24个CPU测试通过只支持内核、控制流和诊断行为，不支持paper-faithful full-context、真实GPU/FSDP训练或论文性能结论。
