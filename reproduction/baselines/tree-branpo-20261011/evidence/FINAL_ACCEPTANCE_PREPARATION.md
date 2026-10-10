# 3360 最终验收准备（本地静态清单，非最终接受）

2026-10-11；本次仅读取本地代码、小型审计摘要和公共results文字记录，不访问远端、不GPU测试、不操作其他任务。3360继续正常执行。当前本地LIVE快照epoch1791658570.6047468只含已完成outer1及outer2进行中；最终200/51713/HF重载与自身清理尚无正式完备证据。以下缺项是验收等待，**不是暂停训练理由**。

## 必须由正式产物证明

| 对象 | 最终需要的实际证据 | 当前状态/限制 |
|---|---|---|
| 完整200 credited outer | attempt journal与8rank direct记录逐轮join；20机会/rank处理完、每credited轮至少一次真实Adam；200完整记录、scheduler累计200、正式body exit0、最终supervisor终态 | outer1独审通过；其余等待。logger step/Slurm RUNNING/已提交都不能替代。拒绝/skip attempt单列，不能混作credited轮。 |
| 每轮方差 | 200个credited轮全部有纯intra `Var(Z_root(R))` population/sample版本、leaf-mass加权、raw reward variance、mixed/zero率；按attempt与真实2560行绑定、DP前完整题组；失败attempt也保留成本 | 首轮实测纯intra .082030979444、加权.089843450257、mixed树.1337890625/题.2734375，独立从叶reward复算吻合。不能把total inter+intra variance称intra，标准化方差不是PG方差，也不证明比基线改善。 |
| 七集完整终评 | 实际 final报告、逐dataset counts与F1/EM/格式/搜索统计、完整行覆盖/失败行不静默丢弃、pooled按实际count重算；固定checkpoint200与source/config/data绑定 | 等待51713实际行：NQ3610、TriviaQA11313、PopQA14267、HotpotQA7405、2WikiMultihopQA12576、MuSiQue2417、Bamboogle125。周期dev512（NQ/Hotpot各256）不能替代终评；应有每5轮一次的40个dev报告。 |
| 官方初始化与数据身份 | 800file包hash、manifest/entry/env/directrank source8b6f99d与configc9415c…；官方模型aa8e725与model manifest744d9f…；data0b002b…，train169028/dev512/final51713；无恢复checkpoint、Adam、scheduler、sampler历史 | 真实fresh driver0/checkpointfalse与8rankAdam0/scheduler0已核。配置路径和asset manifest不能冒充全参数字节验证。treeaudit algorithm68184965/base9ab694属于legacy config标签，单列；权威实际来源是包/入口/directrank，不热改旧标签。 |
| μ与PPO防护 | actual_behavior_mu native token/logp完整绑定、继承sample occurrence一致；每attempt8rank安全事件和finite参数/Adam/梯度/loss；DualClip3、clip.2、KLmonitor.2/negative-tail.01、ratio/expbound保持；初始repeat检查、无baseline subtraction；记录brake/拒绝/实际skip | 首轮480事件全allowed/finite/invalid0、denominator actual_behavior_mu、sampledKL .001230100539；原sampler/actor一致性accepted=false。该monitor为原leaf加权固定子集，不称全policy无偏KL。后199轮等待。 |
| 真实非零参数更新 | nonzero-LR轮的before/after精度对齐参数证据，最终native与HF参数finite/真正非零差异及语义映射；梯度/Adam非零不能代替参数变化 | 首轮LR=0、8rank8192坐标sampledelta0是预定warmup，不是故障；Adam20/rank和scheduler0→1仍是真实完成。需后续LR>0轮actual非零样本及最终完整重载比较；样本0不证明全模型0，样本非零不证明完整模型校验。 |
| Adam、skip与scheduler | 每轮8rank successful-return counters、opportunityledger、Adam state step、skip理由、partial attempt；总实际写入按记录求和，20机会不强制等20写入；scheduler只随credited推进 | 首轮20真实调用/rank、0skip、scheduler1。最终不得简单写4000/rank作为实测总数，须求实际ledger；不同rank身份/attempt/source必须一致。 |
| 五叶、继承、权重 | 每轮512×两root[3,2]、2560唯一leaf、1536有限新增continuation、首次末snapshot/最多一步回退、全部预算叶保留；rawledger/工具观察/token/logp继承，policybatch mask/T与非空叶mass1/空叶0；根mass.6/.4、无额外prefixloss | 首轮完整遍历通过，policy2560行empty0、权重和1±2.4e-7。后续按每轮audit核验，不以单轮推出全程。有限五叶混奖无保证，保持所有同奖叶。 |
| HF与tokenizer冷重载 | checkpoint5/50/100/150/200及rolling native边界manifest/实际文件sha；最终HF完整state独立冷加载、与native精度对齐全参数finite/差异/shape/key检查；tokenizer/vocab/merges/config/pad151643/eos151645与模型身份一致；cold reload过程实际成功且终态记录 | 未完成。保存成功、metadata存在、CPU工具准备、部分8192坐标、同进程已加载模型均不替代。模型/tensor留远端，Mac仅小receipt。 |
| 成本与比较范围 | 全部成功/拒绝/失败attempt raw generation/retained/discarded/padding/prefill/tool/retriever成本、actor/scoring/保存/验证 wall、完整scheduler GPU小时/CPU-RSS/显存峰scope；固定200终评成本另列 | 首轮gen322.530s、618857raw events；完整step730.258s含当轮保存/测试，二者不可混用。response长度含observations；prefix occurrence不等独立采样/FLOPs；缺失counter记未测，不能0填。 |
| 自身资源清理 | 最终数据/HF审计与最新源码/runtime提交推送/manifesthash完成后；仅本job cgroup/子进程/lease/loopback服务/Ray/tmp别名清理；actual executor exit、无本job活GPU进程、cgroup/lease解除与外部job未动 | 尚未完成且训练中不执行。保持checkpoint、历史checkout/branch与原始证据；不裸杀进程、不修改外部任务。最终独立receipt后才声明goal完成。 |

## plain GRPO 控制：本地公共结果匹配情况

只读范围：`/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results`。该目录未找到独立 `grpo/` 结果目录；现有plain GRPO的公共文字记录是 `gigpo/REPORT.md` 内明确标注GRPO作业的段落，另有 `tree-grpo/FAIR_CONTROL_PRECHECK.md`。这里只抽取标明plain GRPO的历史结果，不把GiGPO训练数据作为对照。

- plain GRPO2979：公共记录报告真实短程5轮验收、64题F1 .2305881892/EM .15625、HF语义重载和清理；它不是200轮/51713对照，也没有当前3360新数据/初始化/μ版本匹配证据。本次仅核公共文字，未重验其原始产物。
- plain GRPO2985：公共记录报告计划fresh base500，后由该任务用户2026-10-03明确中止，最终各rank24轮、FAILED1:0为人为中断；不是完成的500或200/七集对照。这里不联系、不恢复、不执行其他任务；旧停止仅描述该历史job，不对3360授权生效。
- `FAIR_CONTROL_PRECHECK.md` 对更早plain GRPO907a736/2973给出协议预检，后续公共REPORT已包含2978/2979成功，故不把预检中较早“2973失败”当全部控制最新状态。仍没有完整匹配控制：旧data fac19c74…（train166222/holdout3393）区别于本次0b002b…（169028/dev512）；旧base Qwen2.5-3B recipe并未提供本次Instruct aa8e725对应完整证据；旧500目标/mini256/micro64/logprob128、warmup.285、entropy.001/referenceKL.001区别于本次200/mini128/micro16/logprob32/warm57、entropy0/referenceKL0、μ修正协议。旧64dev/test50也不同于本次512dev/test5。
- 旧H20阶段速度/硬件、旧Tree3341已完成200结果、GiGPO2992/3026等均不能代替plain GRPO控制。旧协议中同样五独立路径与当前五叶路径只保证名义行数相似，不能推出同token、同状态分布、同目标或公平wall-clock。

**控制结论：本地已有产物不能支持“本次BranPO-inspired采样优于plain GRPO”的公平效果/成本声明。** 完成本训练可做本任务自身验收；要声称比较收益，需要已授权、同模型/data/eval/硬件/实际200预算的plain GRPO正式证据（原GRPO目标差异明确保留），以及更直接的原Tree随机分叉同预算采样消融。此清单不授权启动任何新控制或重启停止的旧控制。

## 现在仍未完成

完整200与40次dev512、200次variance全覆盖、终评51713七集、nonzero-LR真实参数变化与最终全参数finite/差异、native/HF/tokenizer冷重载、全部attempt成本、匹配plain GRPO对照、最终退出与自身清理均等待正式产物。首轮独审只证明首真实outer的实施与数值路径；真正最终审查在产物完整后执行，不能以清单准备或进程存活宣称验收通过。
