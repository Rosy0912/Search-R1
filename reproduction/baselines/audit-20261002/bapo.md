# Boundary-Aware BAPO 独立审查（2026-10-02）

## 结论与范围

**CONDITIONAL；当前不具备正式训练放行条件。** 核心 IDK 奖励、低多样性 gate、整组重采样和 group GRPO 的串行集成总体成立；发现一个未声明的作者 scorer 偏离（P1），以及零动作微批产生 NaN 的 Search-R1 继承缺陷（P1）。阶段验证集尚未由用户选择，必须保持 fail-fast，不能拿最终测试集切阶段。当前两卡 tiny FSDP 证据不是 Qwen2.5-3B/vLLM、offload 或指标验收。

本审查仅指 Boundary-Aware Policy Optimization，不是其他同名 BAPO。未修改实现、未启动远端/GPU、未停止任务、未 commit/push。已有 PASS 报告没有替代本次独立检查。

固定本地 SHA：`7b93fbd01742715dc14290b0a6e2d3b42e6c19f7`；Search-R1 基座：`b251ba6b503217a59cd3395c1326d7b66b602989`。本地 `bapo.py` SHA256 为 `20f3c155879b882e4bbab8463466e72bb61da83efcefb4796afe38b5ff0204c5`；`controller.py` 为 `0e10e8b0db1141e8c7717552412cb0a3695ef0cc7d81479ec43a0e256bd9ac87`。

一手依据：

- [论文 v2 §2.2、§4、附录](https://arxiv.org/html/2601.11037v2)。论文称 character-level F1，但作者实现实际为规范化 token-F1；本审查区分这两者。
- [固定作者 reward scorer](https://github.com/Liushiyu-0709/BAPO-Reliable-Search/blob/8ddbb4fa23f105d52d3621d232f839212e895c3f/RL_train/src/verl/verl/utils/reward_score/re_search_faith.py)：164–212 F1，215–242 IDK，244–247 diversity，249–336 group reward。
- [固定作者 stage manager](https://github.com/Liushiyu-0709/BAPO-Reliable-Search/blob/8ddbb4fa23f105d52d3621d232f839212e895c3f/RL_train/src/verl/verl/workers/reward_manager/re_search_faithful.py)：112–123 阶段计数。
- [固定作者 trainer](https://github.com/Liushiyu-0709/BAPO-Reliable-Search/blob/8ddbb4fa23f105d52d3621d232f839212e895c3f/RL_train/src/verl/verl/trainer/ppo/ray_trainer.py)：873–877 plateau oversample dispatch。
- [固定作者 vLLM rollout](https://github.com/Liushiyu-0709/BAPO-Reliable-Search/blob/8ddbb4fa23f105d52d3621d232f839212e895c3f/RL_train/src/verl/verl/workers/rollout/vllm_rollout/vllm_rollout.py)：617–785 重采样。

上述四份作者文件直接按固定 SHA 下载到 `/tmp/bapo-audit-20261002/`，SHA256 依次为 `a947989fcb91fcb6bb7e5c178c30f52496312c4333d840e58c332625a37c6068`、`de4472e5b1d0d41447fd1c3772b1740723e4361d4afaaf4fe7753bb1b1861350`、`c569439d507b5c787a4b3f74222df54380013241e342c41f2d4f1d91e9acacd5`、`aaaed6ee7e3f986b81b6a97966ce74a1ff1c9cb4c3be8bd722dc05ca2dff2826`。

## 串行流程与数学映射

1. `run.sh:15` 强制外部指定 `BAPO_STAGE_VALIDATION`；`main.py:9` 在 Ray 启动之前验证配置。`controller.py:67–70` 拒绝空路径及 final-test 路径交集；104–118 载入 holdout 并用完整 prompt key 排除训练/最终测试题重叠。
2. `ray_trainer.py:733` 每题 interleave 重复 G 次（默认 G=5）；`Controller.generate:135–151` 先完整 Search-R1 多轮生成。仅 plateau 阶段检查每题所有基础分≤0且不存在合法 IDK，重抽该题所有 G 条完整轨迹，最多两次追加。替换整条 state，保留原 source 行，再重新 compose；不累积旧轨迹到 GRPO group。
3. `ray_trainer.py:769、781–793` 用 source 选回 QA 行、uid=index、union 所有轨迹字段，然后统一重排。重采样在 old logprob 之前；奖励实际在重排之后计算（不是旧报告所称奖励也在 balance 之前）。`groups(uids)` 不依赖邻接顺序，重排不会直接拆错奖励组。
4. `BoundaryReward:145–154` 只从模型生成 span 提取答案，再对所有 action record 检查格式。非法轨迹 Rcorrect=-1；合法普通答案 Rcorrect=max_alias token-F1；合法 IDK 基础分0。奖励落在最后一个保留的模型 token；检索 observation 无直接奖励/损失。
5. `boundary_scores:84–94` 对全错组发0.5 IDK bonus。early gate 是整批合法 IDK 比例<0.05；plateau gate 是规范化答案去重数<G/2。None（非法答案）不计 diversity，合法 IDK 计入。任一部分 F1>0 都阻断 bonus 与重采样，符合式(5)的≤0条件而非 EM 全错。
6. `Controller.compute_advantages:211–212`：A_i=(R_i−组均值)/(组样本标准差+1e−6)，广播到模型 token。`advantages.py:7–17` 按 uid grouping；singleton 返回0而不是 NaN。PPO 使用 production actor 的 token ratio clipped loss（`dp_actor.py:264–268`）及 Search-R1 loss mask；默认 KL loss 0.001 仍启用，属于公平基座适配，论文式(2)省略 KL 不意味着实现关闭它。
7. actor 更新后，每 `test_freq=50` 步先 final-test `_validate`，随后专用 stage holdout `_validate`（`ray_trainer.py:868–875`、`controller.py:232–244`）。只有 holdout 的加权 base QA score进入 `stage.observe`。连续5次非改进（相等也计数）永久切 plateau，生效于下一训练批。初始 final-test 验证不初始化 stage.best；恒定分数时第50步建立 best，第300步切阶段。这里**没有停止训练的 early stopping**，只有 early/exploration 阶段终止。
8. 更新/验证后保存 actor，随后保存 algorithm state（`ray_trainer.py:651–665`）。`Controller.save:251–258` 同步写 best/stalled/early/patience，原子 rename JSON。最终更新会按尾部保存逻辑持久化；未实现 trainer/optimizer/dataloader/RNG/stage 恢复，manifest明确 `resume_supported=False`，不是可断点续训系统。

## 具体问题

### P1-1：F1 少作者 yes/no/noanswer 特判，且空规范化字符串被误奖1（bug／未声明偏离）

位置：`bapo.py:14–19`；作者 scorer:187–200。当前直接 token overlap，在 yes/no 类别答案不完全匹配时仍发部分正分；作者对任一侧是 yes/no/noanswer 且不同的情况给0。当前任一答案/GT被 articles/punctuation 归一化为空，且二者同时空，返回1；作者 common-token=0 返回0。

实际反例：prediction `yes indeed`，GT `yes`，本地2/3，作者0；`no answer`/`no`同样2/3；`the`/`a`本地1、作者0。五条轨迹中一条`yes indeed`且剩余全错，会阻断 IDK bonus；没有IDK时 `resample_rows`返回空，本应重采样。后果不只 scorer 数值不同，还改变 BAPO 边界判定、采样预算、奖励方向。建议补齐作者类别特判与无共同token规则，或在用户接受前明确这是独立 reward 设计；当前 README仅说 token-F1 不足以声明这种特殊偏离。正式算法一致性训练前应修正。

### P1-2：全零动作 loss_mask 微批使 PPO/entropy/KL NaN（共享继承缺陷，正式训练风险）

位置：`core_algos.py:187、192–193`，`torch_functional.py:107–109`，`dp_actor.py:264–285`。空 action/即时EOS/保留response中无模型token可能产生全零微批；BoundaryReward没有有效位置就不写入-1（`bapo.py:152–154`），仍送 actor；masked_mean 分母0。

实际调用 `compute_policy_loss(zeros,zeros,zeros,zeros,.2)` 三个返回值全 NaN。该缺陷存在于 Search-R1 shared PPO 路径，BAPO 没引入但默认沿用。一个空行不必然使含其他动作的微批 NaN；触发条件是该本地微批全部零mask。建议 shared PPO/entropy/KL 对零分母安全处理，并在多rank维持相同 backward/collective节奏，不能仅某rank跳过。此项也由 root 的统一 infra 审查汇总。

### P2-1：自定义 parquet 的 index 不唯一会合错题（继承缺陷；固定数据未实证触发）

位置：`rl_dataset.py:151–153` 从 extra_info.index 取 uid，缺省0；`ray_trainer.py:781`；`controller.py:140–142、171、211`。当两不同问题同index，正确题会关闭另一题的IDK bonus/重采样，advantage也跨题标准化。实测5条全错含IDK的题与5条正确题共index时前者bonus从0.5变0。

固定 reviewed data 的 `prepare_paper_data.py`（root已独立核对）用合并 output 长度作index并assert唯一，故**不能把此风险当当前固定 paper-grpo 数据bug**。建议入口验证index/题对应关系，或构造本批原题唯一uid，避免自定义override静默污染。

### P2-2：阶段验证依赖通用 test_freq，关闭最终测试也会关闭 phase transition（配置风险／适配）

位置：`ray_trainer.py:869–875`。即使存在holdout，`test_freq<=0` 或 `val_reward_fn=None`，也永远不observe、永远early；默认50意味着5次停滞是250个update间隔，首次恒定分数到第300步才切。不是论文“5 consecutive steps”的逐update解释。建议独立stage_validation_freq配置并声明实际频率；当前默认下有确定的可解释行为，不是 off-by-one bug。

### P2-3：长上下文截断后 rollout/scoring prefix 不一致（共享继承缺陷／已明确适配）

位置：`rollout.py:87、96–100、150–152`。生成保留 rolling最后cap token，训练保留初始prompt+response最前cap token；发生上下文截断时 old/new logprob针对的条件未必是行为策略实际条件。默认`context_alignment=report`只计数继续，故不能声称严格on-policy。已有短CPU反例能以`error`提前拒绝，正式3B尚未量化发生率。建议验收时启用error或记录实际生成上下文并逐段计算logprob；固定Search-R1 caps属于控制条件，但数学偏差必须披露。

未发现 P0。格式检查接受无think或一个think前缀、search/answer完整单action、answer只能terminal，与Search-R1协议对齐；其严格度不是作者原始 boxed/result validator 的完全复制。尚未发现另一个可实证格式奖励漏洞。

## 一致性矩阵

| 项目 | 论文／作者 | 当前实现 | 对 Search-R1 | 判断 |
|---|---|---|---|---|
| 模型/数据/检索 | 作者多种规模、5k多跳、自己TIR协议 | 固定Qwen2.5-3B、共同paper-grpo、topk3 | 共同基座配置 | 明确实验适配，不复现作者指标 |
| 基础reward | 论文写character F1；作者token-F1+类别特殊规则 | token-F1；缺类别特殊规则/空串规则 | EM替换为F1 | P1未声明偏离 |
| format | 作者think/search/result及boxed answer | model action record+answer标签，非法-1 | 使用生成span排除检索伪答案 | 明确必要适配 |
| IDK识别 | 作者boxed答案内substring匹配两种短语 | 规范化后整串匹配两种短语 | prompt不增加IDK提示 | 明确适配；拒答可能稀疏 |
| bonus条件 | 全组Rcorrect≤0，合法IDK+.5 | 一致 | 新reward driver hook | 成立（受P1 scorer影响） |
| early gate | batch IDK ratio<alpha | 一致，alpha=.05 | batch在driver完整可见 | 成立 |
| plateau diversity | 论文distinct answers<G/2；源码排除IDK且硬编码≤3/G8 | 归一化含IDK、实际G5阈值2.5；非法None忽略 | uid grouping非相邻chunk | 选论文语义的明确适配 |
| stage计数 | 论文连续5非改进；源码仅低于历史best累计、>5、不reset | 连续5、tie计数、提升reset、不可逆 | 每50次update holdout base EM | 论文语义适配；频率/EM需披露 |
| validation data | 论文80条来自四benchmark测试 | 独立holdout必须无训练/终测题重叠 | 保留共同final tests | 正确更严格适配；尚未选定 |
| resampling | 论文plateau全错无IDK up to k；作者default2实际只追加1次 | 整组完整轨迹最多追加2次 | 同Search-R1工具链 | 论文语义明确适配；非源码逐字复刻 |
| GRPO/PPO/KL | 组标准化、clipped objective；公式略KL | 组样本std+eps、token PPO、KL .001 | 沿用production actor/offload | 合理基座适配，有零mask继承风险 |
| checkpoint/resume | 作者训练框架独立 | actor+阶段JSON，明确不支持resume | 继承基座保存能力 | checkpoint可审阅；恢复未实现 |

## 实际验证及边界

执行环境为本地 `.venv-cpu/bin/python`，没有GPU/远端动作：

```text
.venv-cpu/bin/python -m unittest discover -s tests/baselines -p 'test_algorithms.py' -v
19 tests, 0.031s, OK
.venv-cpu/bin/python -m unittest discover -s tests/baselines -p 'test_models_and_driver.py' -v
4 tests, 0.938s, OK
PYTHONPATH=. .venv-cpu/bin/python /tmp/bapo-audit-20261002/probe.py
exit 0; F1 special-answer / empty-normalization mismatch；zero-mask三项nan；uid碰撞丢失bonus；按uid重排标准化对应不变
```

probe通过AST仅提取固定作者 `normalize_answer/get_f1_score` 并与本地逐例比较，不加载作者训练包/模型。库出现 urllib3/LibreSSL warning，未影响上述CPU结果。

特别注意 `test_models_and_driver.py:152–153、178、181–185` 的BAPO driver测试设置test_freq=-1、val_reward_fn=None、Controller.__new__、early=True：它验证真实fit的数据union/source/driver hook，**没有验证真实holdout初始化、stage_validation触发、plateau重采样或stage checkpoint恢复**。基础测试验证stage纯函数、gate反例、非法action、mask/source和共享rollout，但不能推导端到端正式训练已过。

代码层面已核对 FSDP strategy、state_mask、rollout.n=1限制；默认vLLM入口复用Search-R1 actor_rollout worker，不引入BAPO专用GPU模型；DP重排保留non-tensor uid且组优势在driver上计算。`fsdp_workers.py:359–401` load/offload参数与optimizer路径沿用生产实现；不宣称本次CPU运行验证了offload、remove-padding、vLLM版本/入口可用性、两卡通信、真实检索器或3B显存。

正式训练阻碍：选定且固定独立stage holdout；解决P1 scorer偏离与共享零mask NaN；以完整3B/vLLM/检索/offload设置验证至少early到plateau、重采样、阶段保存及最终checkpoint；量化context mismatch并确认接受其适配。若仅检验已披露的Search-R1算法适配而非论文指标复现，可维持CONDITIONAL；目前不能把已有CPU/tiny-FSDP PASS直接扩展为正式训练PASS。
