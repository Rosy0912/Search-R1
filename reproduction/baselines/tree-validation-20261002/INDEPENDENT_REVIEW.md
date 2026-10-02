# Tree-GRPO 独立复核（2026-10-02）

结论：**当前 Tree 算法代码/数学/CPU 数据流通过；允许进入计划规定的真实 GPU smoke，尚未完成效果复现。** 初次固定审查 HEAD `1620c2edb13809d19cc3f5af4a298ccd75488104`，最终差量复核 SHA `56e626aceaceec961d90d427dacf3bde3a7bbbb9`（含实现fix `165b80913f72ae48d9595ea0fe61814b453681be`），分支 `reproduce/tree-grpo`。本审查未修改实现、未启动远端训练、未更改其他 checkout。不存在本次查明的新 P0/P1 算法阻碍。公共部署冻结、资产/资源验收及 3B GPU 结果仍是正式运行前提。

## 来源及复核方法

重新浏览 [论文 v1](https://arxiv.org/html/2509.21240v1) 的 §3.1、§3.2 Eq.6–8 及 §4.1；重新浏览并下载 [作者固定 19bf3fa](https://github.com/AMAP-ML/Tree-GRPO/tree/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59) 的五个文件。GitHub tree 页面浏览器返回 restricted URL，但同 SHA raw 源码成功读取，不以旧 audit 结论代替本轮来源检查。源码副本与 SHA256 见 `evidence/author-*` 和 `author-source-hashes.json`。初次五个核心实现文件 SHA256 记录于 `evidence/independent-probe.json`；最终 SHA 的重复探针及核心文件 hash 见 `evidence/independent-probe-rngfix.json`，保留了旧证据。

独立运行 31 项 CPU 测试，包含树续写、跨 root batching、估计器、目标权重、2-rank CPU DDP 与真实 trainer fit 的 mock rollout 数据流；全部通过。另运行独立探针，结果与脚本保存在 `evidence/independent-probe.*`。字符 tokenizer/mock worker、CPU torch 与生产 CUDA/vLLM 有明确边界，不能代替真实 Qwen2.5-3B 验收。

## 逐项对照

| 项目 | 当前实现与独立判断 |
|---|---|
| 初始化 | trainer 对每问题重复两次；`initial_states` 为每行独立建立 tree UID。两条初始链对应 M=2。 |
| 真实分叉 | `complete` 保存动作前完整 state；`branch` 深复制共享 action/observation/prefix 及剩余 turns，仅换 trajectory UID。已执行的检索结果直接复用。 |
| 候选与续写 | 随机有放回选择 root/非终止节点，候选按 node UID 去重；与 [作者 tree_node.py](https://raw.githubusercontent.com/AMAP-ML/Tree-GRPO/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59/search_r1/llm_agent/tree_node.py) 的 root+nonleaf 选择一致。每个续写继承剩余预算，未变成独立完整采样。 |
| 并行执行 | 当前 `run` 汇总本轮所有 root 的 children 后调用一次 `complete`，与 [作者 generation_ts.py](https://raw.githubusercontent.com/AMAP-ML/Tree-GRPO/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59/search_r1/llm_agent/generation_ts.py) 的跨 root 批处理相符。旧逐 root 串行缺陷已不存在。 |
| 叶与来源 | L=1,N=2,K=3，每 root 恰有三叶，总六叶。选择叶后 sources 显式映射 QA metadata；tree/trajectory UID 随 DataProto permutation 保存；共享 prefix 在各叶路径重复参与训练，未去掉必要叶权重。 |
| 优势 | `tree_grpo` 对原始 outcome 分别计算 question 与 (question,tree) sample-zscore，epsilon=1e-6，然后相加，再乘 action mask；符合论文 Eq.6–7。对应 [作者 ray_trainer_ts.py 默认 tree 分支](https://raw.githubusercontent.com/AMAP-ML/Tree-GRPO/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59/verl/trainer/ppo/ray_trainer_ts.py)；作者 core_algos 同名 tree 函数属于另一个 tree_2norm 分支，不能混淆。 |
| token PPO | `attach_policy_weights` 每叶有效 actor token 权重为 1/length，`weighted_policy_loss` token ratio/clipping 后按叶等权。微批共享 minibatch 行分母，梯度累积无需重复除 accumulation；SP=1 下 2-rank DDP 测试通过。 |
| 观察及空样本 | observation mask=0，terminal scalar 重落到最后保留 actor token，outcome 不变；空叶权重为0但保留外层样本分母。数值 helper 在 masked input 为 infinity 时仍能安全归零；全局无有效 token minibatch同步跳过。 |
| KL/熵 | actor policy objective、reference KL及entropy使用同一逐叶 token权重。默认 actor KL独立于outcome advantage，不暗中改变tree奖励；需生产日志验证实际参数。 |
| offload/同步/保存 | 调用沿用 Search-R1 FSDP/vLLM worker；CPU gate覆盖入口和控制数据流，不证明生产权重同步、显存或checkpoint质量。无精确resume保证。 |

## 本轮实测证据

- 两 root 强制分叉于搜索后的共享状态：sources=`[0,0,0,1,1,1]`，tree UID=2、trajectory UID=6；仅两次检索，prefix 未重搜；worker batch sizes=`[8,8,8]`，padding rows=16，generated tokens=544，prefill tokens=1392。旧 audit 的 `[8,8,8,8]`/24 padding不能沿用。这是小批探针；正式1024 root 批处理开销需实际记录。
- 六叶 reward `[1,0,0,0,1,1]`：两组 zscore 的独立计算及置换等变通过，输出见 JSON。
- 一个1-token正优势叶、一个3-token负优势叶、一个空叶：逐叶权重总和 `[1,1,0]`，loss≈2e-8，梯度≈1.5e-8；当前目标达到预期等叶抵消，不再是旧 microbatch token均值的0.5偏差。
- 四轮搜索、人工 cap80探针：context mismatches=2，rolling=80、scoring=98；该现象真实存在。用户已授权统一保留 Search-R1窗口差异并report，故此审查不要求改动该协议；正式报告仍必须给出真实触发率。

## 差量复核：采样 P2 已解决

初次发现 `Controller.generate` 每次新建 `SearchRollout(seed)` 导致跨outer update重新开始同一个分叉随机序列。主agent在独立实现提交 `165b80913f72ae48d9595ea0fe61814b453681be` 中只为Tree创建持久 `tree_rng`，每次新engine复用该实例。审查diff确认修改仅涉及controller的RNG持久化和一项controller连续调用测试，其他算法仍使用原engine逻辑，未改树叶数、优势、mask或成本计数。原P2已解决，不再作为最终未解决发现。

独立重新运行 `test_tree_batching.py` 与 `test_models_and_driver.py` 的测试（结果见 `evidence/rng-fix-tests.txt`），确认真实controller的两批采样RNG状态连续演化、同seed整段可重放，其中tree连续采样三项通过，但真实fit-driver测试fixture采用 `Controller.__new__`、未初始化新 `tree_rng`，触发AttributeError；本次合计6 passed/1 failed。生产正常 `__init__` 初始化路径未发现此缺陷，fixture需同步更新后重跑，此结果保留为历史失败证据；主agent随后在 `56e626aceaceec961d90d427dacf3bde3a7bbbb9` 修复fixture，独立审查确认仅增加random import与tree/非tree初始化。最终重复相关完整32项测试全部通过（`evidence/final-independent-tests.txt`）。最终CPU gate通过，真实生产GPU仍未验收。重新运行独立probe核对其余树行为与数值不变。仍未提供精确checkpoint+RNG恢复，因此不宣称中断后等价resume。

已授权的偏离还包括四轮且不强制final、统一窗口差异report、相较五链GRPO的六叶预算，以及遵循论文逐叶token均值而非作者源码microbatch token均值。默认group至少三叶，singleton零优势保护与作者singleton返回score不同，但默认实验不触发。不能宣称逐参数复现作者训练脚本。

## 真实 3B GPU 待验收

1. 公共部署commit冻结并一致cherry-pick；h100-9 Slurm独占allocation与8张实际UUID核对；双15%阈值、既有小显存进程记录、启动前复查。未获得资源不得绕过调度。
2. 固定Qwen2.5-3B base/tokenizer、98/2 split的跨路径content_identity/产物SHA、E5/语料/index/hash、CPU检索top3及真实schema与排序一致性；周期64/最终七集与holdout隔离。
3. BF16/NCCL/flash/vLLM 8卡通信及最小推理；生产tree rollout六叶来源、shared-prefix复用、remaining turns、mask、真实generation/scoring mismatch计数。
4. 至少两次actual outer update：old/ref/current logprob、loss/grad finite，非零实际参数delta；offload、vLLM权重同步、8卡每rank1微批、optimizer step计数一致。
5. 五步保存/验证acceptance，完整HF导出与重新加载；确保正式500 actual update对应501总步入口，100步保存/50步周期评估，不宣称等价resume。
6. 正式训练及51,713题七集最终评估，与同split新GRPO比较EM/F1/格式率。单列实际生成/prefill token、工具请求、padding、六叶训练样本、GPU小时、CPU检索/RAM/墙钟/显存峰值；不能按相同步数直接推论同计算成本或论文节省率。

严格判断：本报告完成的是实现复核和CPU可验证性；没有真实3B训练指标，因此目前不能认定Tree-GRPO在search agentic RL有效果优势或完成复现。
