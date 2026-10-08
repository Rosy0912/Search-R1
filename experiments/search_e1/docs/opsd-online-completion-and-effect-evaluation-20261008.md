# 在线 OPSD 完整验收与效果评估

本计划对应最新人工选择：source GRPO100 权重、八卡在线 OPSD、词表 KL 分量上限10、200个外层采样与训练轮次、HKUST 本地存储与持久备份。当前 run 为 `1d1d67b60f0e46a2afc0536a36ab69b1`，配置 SHA 为 `d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17`。不增加 GRPO，不恢复旧训练、M3/M4 或旧七基准作业，不改变正在运行的配置和源码。

## 已核事实与当前边界

原五次真实更新已通过验收。本次独立 CPU 实读保留的 outer8 末 CP89：八 rank 的实际 Adam step 均为89，完整 moments 和 parameter groups 逐项相同且有限；504个适配器张量有限，相对明确指定的 CP5 全部改变，最大绝对变化为0.0005599805153906345。CP89 manifest SHA 为 `325c9f41f5a44d5e16a32578c5fa03dfaecec92bbdbd87a4f5e289c81c9bf75f`。

CPU 审计核对该 outer 的10次更新、292条 token 文件清单，并实际读取末批4条轨迹的全部1000个有效 token，重构原生 mask、轨迹均值和实际全局尾批分母4。该检查没有重算 logits、分词结果或早先所有 token 内容；source100 大权重的全量新 SHA 核验属于另一次健康检查。审计入口已发布于 `c75aff278f338e7e97a1fc65b289d151d85e583d`，19项 CPU fixtures 及独立审查通过；真实 CP89 审计随后 exit0。它证明真实有限更新与记录一致性，不证明 EM 提升。

随后原生完成票据已记录11/200个 outer、109次 Adam，下一 outer 正在训练。这个更新范围与独立实读 CP89 分开报告。当前在线 controller 完成200轮后仅写训练 `COMPLETE`，没有自动执行 native512 评估。因此完整训练与完整效果交付是两个里程碑。

## 训练完成里程碑

1. 对账 outer0–199 连续唯一的完成票据、采样模型身份、512 TRAIN题×5轨迹、原矿选与有效配对数 K。真实 Adam 总数应等于各轮 `ceil(K/32)` 的和，K=0时没有 Adam；累计曝光为各轮 K 的和。不能把200个 outer 当作200次 Adam。
2. 绑定 outer199 末完成票据指向的最终 checkpoint，核实八 rank 完整状态、实际 Adam、有限且改变的适配器和原配置/模型/数据/源码身份。最后 outer199 的采样 merged cache 来自 outer198 末权重，不能用它代替最终训练后权重。
3. 实际记录 controller 和全部采样、torchrun、rank 后代的成功退出及资源释放。`COMPLETE`、父进程消失或 tmux 名字单独不足以证明 exit0。原作业日志、冻结源码、检查点和失败证据继续保留。
4. 完成既有保护检查点本地传输后逐成员 SHA 验证；只有完整备份票据可称持久副本完成。最终检查点及最终评估证据也须持久保存，不能因训练终态主动删除唯一原始资产。

## 独立固定权重效果评估

使用全新的 online-results 目录、配置、run 和已发布入口。启动前先核实际原控制器/后代释放、八 GPU UUID 与直接 `memory.free`、没有外来进程、四个现有共享资源锁及新结果锁、磁盘空间；GPU7 仅允许已获准18004 oracle共享。正在训练时不运行这个阶段。

预先固定三种模型身份：原 source100 普通策略、保留的在线 outer8 末 CP89、outer199 末最终 raw adapter。CP89 是诊断保存点，不按效果选择 best；最终模型也不能由 baseline fallback 替代。源基线可用明确的原 base；CP0 的252个 LoRA B 全零须独立核验，并以实际零效果控制验证基线身份。评估不恢复 Adam，不训练。

全部模型使用同一原生普通提示 `SearchAgent.run(question, temperature=0)`，不提供成功参考。固定 BF16 base、训练后 FP32 LoRA、eval/no_grad、现有 Torch/transformers/PEFT 环境、原 generation_config 和实际 generation kwargs，包含 repetition_penalty1.05、原 EOS 与 StopTags。预算保持上下文8192、每轮500 token、5轮、最多4次搜索、top3、观测480。不能混用 vLLM 随机训练 reward 或不同 merge/dtype 的结果解释 native EM。

验证集固定原512题，SHA `75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76`，NQ/Hotpot各256。维持八逻辑 rank 的 `[rank::8]` 分片、每 rank seed42、每题单样本。全部模型同题同协议。记录全部512题 slot、真实生成 IDs、prompt、检索回包、termination 和 policy spans；透明日志 hook 须做相同 RNG 的无效应控制，不能改变生成参数或计入重复样本。

使用当前在线已准入18004路线，重新实核 oracle PID/UID/starttick、源码、完整数值门禁、健康状态及 observer。全部模型同一路线；不恢复18005或静默切换后端。源基线新评估若不匹配此前239/512，先保留全部输出并定位条件差异，暂停“复现旧指标”的结论，不改参数凑数。新最终在线模型不应被要求匹配旧离线 epoch 的203/512。

按冻结 `core.reward` 的最后 authored span 严格计分，不重定义 native EM。全 transcript 标签、搜索动作与 termination 是另列诊断。旧 M4 的四个离线适配器、18005、M1 release 与239/203双端点门禁不适配本次在线 run，不能直接恢复旧入口。

## 结果与归因交付

报告每模型准确的 hits/512、NQ/Hotpot 分集 EM、同题新增/丢失/保持/仍错四格，以及丢失题的最终协议无效和有效但 gold 不匹配分类、搜索与终止分布。每项结果绑定实际 run/model/data/scorer/retriever SHA；原始题目、轨迹、token 和权重留在私有持久证据目录，不进入 Git。

每轮不同 TRAIN512题的 temperature1 采样 reward 只反映那一批的采样结果，不能当作固定验证曲线。loss 下降、有限更新、五步运行验收也不能当作收益。source100、CP89 和最终保存点可以判断这些点上的退化或恢复，不能插值精确退化时刻；H1/H2/H3 的训练因果贡献仍需受控对照，不能仅凭最终指标确定。新训练参数或额外消融如有必要，先给出具体单因素方案供人工决定。
