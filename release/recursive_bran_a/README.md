# 递归 Bran Tree-GRPO：实验 A（HKUST1 八卡）

本目录说明实际实现及训练流程。它取代旧发布分支的两根固定扩展采样。实验 B 是独立任务；本分支只实现 A 的均匀父轨迹选择。

## 实际运行身份

- 生产源码：`5045430e67860102c304ef8098e4862ddcdf6e3e`。后续文档和只读统计工具提交不热更新生产进程。
- run：`52f0bd4ef1044a15bd4ef4d366843f46`。
- HKUST1 根目录：`/data/shaoshitong/recursive-bran-a-8gpu-20261011`。
- 冻结配置：`receipts/resolved.yaml`，SHA256 `622e3c083cd8e2db5f6ef3ba126cd6d4f71babd85c97feb46f839a20fe593d03`；身份记录为 `receipts/configuration.json`。
- 训练日志：`logs/train-rolling.log`。运行在 tmux `recursive-bran-a-rolling-20261011`，使用本机 Ray，不是 Slurm allocation。
- 本分支包含源码、入口、配置生成器、只读统计和进度文档；模型、optimizer、原始轨迹与完整评测产物留在 HKUST，不进入 Git。

## 模型、数据与资源

从官方 `Qwen/Qwen2.5-3B-Instruct@aa8e72537993ba99e69dfaafa59ed015b17504d1` 开始全参数训练，fresh optimizer；不用旧 GRPO100 权重或 LoRA。

现有模型路径为 `/data/shaoshitong/search-e1-config-study-r1-aligned-8gpu-20261005/assets/model`。数据在运行根目录 `common-data-v2`：train169028、dev512（NQ256/Hotpot256）、stage64、final51713；内容身份为 `7b6369e17ebd497ea159d57dc159647eed8adad852b7468cc600b7f203e80f43`。dev/stage/final 与 train 的题目隔离由数据准备清单检查。

GPU0–7 由本任务使用；GPU7仅与已获准检索服务共享。检索 URL 为 `http://10.120.17.101:18004/retrieve`，top3。入口检查 GPU 占用和资源锁，不停止或接管其他进程。

## 每个 outer 怎样进行

1. 取512道训练题，用当前策略普通提示为每题生成一条完整根轨迹，放入接纳集合 L，generation=0。
2. 每题另有最多10次 continuation 尝试预算，根生成不占这10次。A 在具有合法分叉位置的 L 轨迹中均匀选父轨迹。
3. 根据已记录接纳轨迹和同奖励候选，计算语义共同前缀边界；只有 generation 严格低于所选父轨迹的历史记录限制该边界。在合法行动前状态中随机选起点。
4. 从真实 executor snapshot 续写：保留历史 action token、观测、行为概率，以及已经消耗的 turn/search 预算。不是重新给子轨迹额外4轮。
5. 若子轨迹与父轨迹奖励不同，则接纳到 L，child generation=parent generation+1；若相同则放入候选池 F，并在预算内向前回退继续尝试。generation 不是尝试编号。
6. 达到5条或耗尽预算后，在需要时从 F 均匀、不放回补齐。失败、预算及不足情况真实记录；不重复已有记录伪造5条。
7. 收集每题最终选中的5条完整轨迹，共2560条，计算 old/ref logprob、轨迹优势和 PPO 更新。每rank每outer20次更新机会；安全拒绝或跳过另记，不把机会数当已提交更新数。
8. 完整状态保存成功后推进下一outer。每5轮执行原生 dev512；200轮后执行 final51713。10轮是启动验收门槛，不是停止预算，训练不会因该人工验收自动暂停。

## reward、递归树与优势

完整轨迹 reward 是严格最终答案评分0/1；不累加父奖励、不按generation折扣。只有最终选中的轨迹进入优势计算；未选候选只影响构树。

`search_r1/baselines/advantages.py::tree_grpo` 保留原来的两个分量：

- inter：按 question uid，对完整轨迹 reward 做中心化并除以样本标准差+1e-6。
- intra：按 `(question uid, tree_uid)` 同样标准化。
- 最终优势为 inter+intra，并赋给有效 policy token；观测文本不承担 loss。

当前每题只有一个根，因此这两个分组相同，最终优势等于两倍同题标准化优势。**没有内部子树平均 reward、按节点局部 advantage 或沿递归边回传 reward 的实现。** 递归关系改变采样与选择，不直接改变上述估计器。五条reward全相同时，两个分量都为0。

PPO目标按 question等权 → 题内trajectory等权 → 每trajectory有效policy token均值归约。共享前缀仍属于完整轨迹训练序列，没有另做节点credit assignment。

## 科学设置与重要区别

| 设置 | 实际值 |
|---|---|
| 预算 | 200完整outer，8rank |
| PPO | mini128、epochs1、clip0.2、grad clip1 |
| Microbatch | actor全局16/每rank2；old/ref全局32/每rank4；梯度累积 |
| LR | 1e-6；57个credited outer线性warmup，随后常数 |
| 精度 | BF16 forward/rollout；FP32 master、gradient、Adam moments |
| 正则 | entropy coefficient0、reference KL coefficient0 |
| 防护 | 采样KL0.2、负优势尾部0.01、指数界30、原行为μ校正 |
| 生成 | temperature1、top_p1、XFORMERS/eager；4turn；start2048/context4096/response500/observation500；不强制最后回答 |
| 评估 | 每5轮dev512，最终51713；不是选中TRAIN正确率 |

第一outer的优化调用使用warmup起点LR0，之后scheduler推进；不能只凭Adam调用数宣称第一outer权重已变化。

原表的 DualClip3 在本实现有明确区分：按递归算法文档，`recursive_document_objective=true` 不给 PPO loss 额外加入 DualClip 项；安全监控仍保留阈值3相关统计。不能将其描述为实际启用了额外 DualClip loss。行为μ校正沿用旧实现，不声称这是完整无偏 off-policy estimator。

## checkpoint、日志和统计

每个完整outer保存八rank参数/optimizer/scheduler/RNG等状态与driver状态。只有 `checkpoints/training_state/global_step_N/manifest.json` 为 `COMPLETE_SAME_WORLD_STATE_SAVED` 才视为保存完成。

rolling保留最近两轮完整payload，以及5/50/100/150/200保护点；回收更早非保护payload时保留manifest和reclaimed记录。每5轮执行原生同world reload检查。不能从只剩manifest的已回收目录恢复。

轨迹及事件在 `output/recursive-events/<rollout UUID>/<question source index>/`，保留父子ID/generation、原生token、观测、executor snapshot、分叉与补齐事件及实际成本记录。只读统计入口：

```bash
python release/recursive_bran_a/summarize_events_v3.py \
  --events /data/shaoshitong/recursive-bran-a-8gpu-20261011/output/recursive-events
```

该统计检查已记录事件守恒、generation/预算/补齐结构，并报告group奖励熵（bits）、方差、混合奖励比例和补齐比例；它不是独立重算真实文本LCP、logits、梯度或dev EM。不同TRAIN cohort及异奖励选择会影响TRAIN正确率，不能替代固定dev结果。

## 如何部署和启动

当前作业已经启动，**不要再次执行下面命令覆盖现有冻结配置或fresh启动**。`train.py` 会拒绝已有fresh初始化后的盲目重启。

独立新部署时，将分支源码置于对应远端根目录的 `source`，准备同集群模型、数据及Python环境；根据实际节点调整 `configure.py` 和 `launch_hkust1.sh` 中的路径、主机/UID/检索身份约束，发布后在CPU阶段生成不可变配置，再经preflight取得资源锁启动：

```bash
cd /data/shaoshitong/recursive-bran-a-8gpu-20261011/source
/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python \
  release/recursive_bran_a/configure.py \
  --base /data/shaoshitong/recursive-bran-a-8gpu-20261011
bash release/recursive_bran_a/launch_hkust1.sh
```

这些入口专门用于已核对身份的HKUST1环境，不是通用集群启动脚本。续训必须绑定已完整保存的真实状态并独立记录恢复，不重新运行configure生成新run冒充续训。

相关实现：`recursive_branch.py`（Planner）、`rollout.py`（实际executor续写）、`advantages.py`（优势）、`objectives.py`（归约）、`recursive_rolling.py`（存档）、`verl/trainer/ppo/ray_trainer.py`（原生训练/评估循环）。已完成进度见 `first-three-progress.md`；进度文档是其注明时点的证据，不是实时状态。
