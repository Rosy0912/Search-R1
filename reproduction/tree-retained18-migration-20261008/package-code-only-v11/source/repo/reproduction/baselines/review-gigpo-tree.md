# GiGPO / Tree-GRPO 实现审查

日期：2026-10-02。结论：**CPU 审查通过（PASS）；CUDA/FSDP/vLLM 与训练性能未验收。**

审查对象：独立 checkout `/Users/shaoshitong/Documents/Search-R1-RL-Baselines`，分支 `codex/search-r1-rl-baselines`，reviewed base `b251ba6b503217a59cd3395c1326d7b66b602989` 上的当前工作树。只读检查实现，未操作原 Search-R1 工程或远端 GPU。本报告不将准备完成、CPU mock 训练或有限 loss 宣称为论文性能复现。

## 证据与范围

- [GiGPO 论文](https://arxiv.org/html/2505.10978v2)，官方算法源码固定 `20bd331bdbc9026a5668e11362178e10ab7400c8`：[core_gigpo.py](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/gigpo/core_gigpo.py)。
- [Tree-GRPO 论文](https://arxiv.org/html/2509.21240v1)，官方固定 `19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59`：[真正树训练 dispatch](https://github.com/AMAP-ML/Tree-GRPO/blob/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59/verl/trainer/ppo/ray_trainer_ts.py)、[TreeNode](https://github.com/AMAP-ML/Tree-GRPO/blob/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59/search_r1/llm_agent/tree_node.py)。
- 重点检查 `search_r1/baselines/{advantages,rollout,controller}.py`、`ray_trainer.py` hooks、`ppo_baselines.yaml`，并执行真实本地 CPU 依赖测试。

## 原 findings 与解决证据

| 严重度 | 原问题 | 当前修复 / 回归 | 状态 |
|---|---|---|---|
| P1 | 新 rollout compose 丢失 old-logprob 所需 `temperature/micro_batch_size/use_dynamic_bsz/max_token_len` | `advance` 保存 worker output metadata、`compose` 恢复；真实 DataProto selection/old-logprob 消费回归及实际 `RayTrainer.fit` mock 运输路径 | 已解决 |
| P2 | 多轮树扩展再次加入父 snapshot，额外提高该父的采样权重 | State 稳定 node identity，新节点按 identity 去重；强制多次扩展同 post-search parent，候选计数 `[2,2]` | 已解决 |
| P2 | GiGPO 根据有 token 的步骤列表衰减，跳过空动作时间 | 按完整 anchor/action timeline 计算 discounted return；steps `[0,2]` 三动作反例优势 `±0.95125` | 已解决 |
| P2 | 无 token 动作不参加 anchor 的均值，局部基线错误 | 完整 timeline 建 keys/returns，仍只在 `step_ids==s` 位置回填；同 root reward1/0、第二动作零token反例有效优势由0.5纠正为1.0 | 已解决 |
| P2 | 预算耗尽当成环境 done，finish ratio 错报为100% | `done` 仅取环境终止，运行预算由 `pending` 判断；4次search无answer保持 active_mask=True | 已解决 |

另核对旧 DataProto `__getitem__` 返回 Item 的兼容陷阱：新增 `select_rows` 显式保存 batch tensor、non-tensor provenance 和 metadata；控制器/hooks 使用该 helper。这也是现有真实 trainer CPU 集成测试能到达 advantage/actor hook 的必要修复。

## 算法对照结论

GiGPO 当前为 Search-R1 paper-formula 适配：episode 每 trajectory 等权统计，题内 exact anchor 比较 discounted action returns，episode + omega × step advantage；环境文本不接受 action loss。完整时间轴包含空动作，无token动作参与 anchor baseline但无自身梯度。默认 observation anchor 不等于完整历史 policy context；`context` 模式可作严格前缀消融，必须在实验中记录选择。默认 `normalize=false` 采用论文允许的固定归一化因子1，并非官方 search recipe 默认 std、相似anchor0.9和长度重复行统计；禁止宣称原官方recipe逐项等价。

Tree-GRPO 为实际树 continuation：snapshot保留真实rolling前缀、响应、mask、剩余轮数；branch深拷贝共享状态，从选中前缀继续生成，不重新检索该共享prefix。默认 tree advantage 独立计算同题所有叶和同题同树叶的z-score并相加，匹配官方 `adv_estimator=tree`，不误用可选 `tree_2norm`。父节点去重按稳定identity而非文本，保留不同真实节点；选叶保持独立trajectory id及同树id。

共享prefix仅在rollout生成时复用；叶展开训练保留其重复出现，不暗加逆重复数梯度。采样账本计真实输出token，包括 GPU divisor padding 的实际额外生成、prefill、全部扩展与未选叶；实际retriever calls不含共享prefix重放。仅有账本不等于已实现跨算法等token/tool-budget的实验调度，正式报告仍须用实测累计成本比较。

配置沿用 reviewed base 的 paper-grpo 数据、同Qwen2.5-3B base和原检索接口；GiGPO/Tree未自动替换奖励、教师、ranker或原EM评价。该 base 的4-action无额外final配方与早期NQ/2-turn legacy文档不同，比较应针对同一 reviewed base，不能把旧配方成绩混入新主表。

## 实际执行的验收

命令：

```bash
.venv-cpu/bin/python -m unittest discover -s tests/baselines -v
```

结果：**24 tests，全部通过，约7秒**。包含解析式 advantage、空动作时间及均值、mask、真实DataProto选行/metadata、树续采样和父节点去重、cap/final预算、五方法实际RayTrainer.fit CPU mock运输、微型teacher/student/ranker模型，以及真实两进程CPU/DDP/Gloo测试。

Gloo测试验证辅助蒸馏的全局token归一化在不同rank长度和零action rank时得到预期梯度；它不等于在FSDP/NCCL下验证全部算法的多卡训练。RayTrainer.fit测试运行实际driver，但CUDA transport/worker被mock，不能据此保证真实worker注册、GPU内存、权重同步、vLLM engine或CUDA kernel行为。

## 尚未验收的边界

- 8卡CUDA/FSDP/NCCL/vLLM、真实Qwen rollout及其 old-logprob一致性；CPU测试不覆盖这些运行组件。
- 实际训练loss/KL稳定性、checkpoint完整保存/加载、正式验证EM及不同seed。
- 真实token/tool/GPU-hours等预算比较与树的吞吐；当前树扩展按root执行，实际性能应实测。
- GPU资源仍须另行确认；不允许因本报告PASS而打断原agent任务或在监控专用H100节点训练。

当前审查范围未发现尚未解决的critical/major问题。若工作树继续变化，本结论仅对应本轮读取与测试的版本，后续需针对新差异复核。


## 追加审查：继承上下文限制的诊断（2026-10-02）

仅复核新增 `context_alignment` 诊断和 README 对应措辞，没有改变原共同裁剪策略。`SearchRollout.advance` 在真实 generation 调用之前逐状态比较 `original prompt + retained response` 与实际 `rolling`；report 模式累计 `rollout/context_mismatches` 并在每次 action record 保存一致性布尔值，error 模式在发生不一致的那次生成之前报错。Controller 将训练配置传给 engine，默认 report。

重新运行完整CPU suite：24 tests全部通过（约7.1秒）。新回归覆盖初始 prompt 超过 max_start_length 导致的前缀不一致，并确认 strict 模式在任何 worker generation 前拒绝。审查者另作只读小cap滚动截断验证：4个纯search action、cap80，report的action一致性 `[True, True, False, False]`、mismatch2；error模式完成前2次有效生成，在第3次真正调用worker之前拒绝。诊断没有重置history、改动预算或悄悄修复共同基线。

**继承限制仍然存在，而非已解决：**长轨迹采样使用裁剪后的rolling context，最终actor old/current logprob可能由原prompt与保留response组成的另一条完整序列重算。二者条件前缀不同会使PPO importance ratio的behavior概率解释、TOP-D原论文的on-policy条件不严格成立。当前仅可称“保留共同Search-R1工程行为的算法适配”，不能将CPU PASS称为长context严格on-policy复现，也不能用诊断为理论保证背书。README明确披露该限制、report/error语义以及若修复则GRPO必须同修，措辞符合实现。

strict诊断比对token前缀，不是对真实vLLM生成概率、重tokenize行为、position IDs、CUDA数值和teacher模型的完整一致性证明。正式运行必须保存mismatch数据；存在mismatch的样本不能被混报为已满足原论文条件。若后续改变上下文/评分策略，应对所有固定环境方法和GRPO做相同修复、重建结果并重新审查。
