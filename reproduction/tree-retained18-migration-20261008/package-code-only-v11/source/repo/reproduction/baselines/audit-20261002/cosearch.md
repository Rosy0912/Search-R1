# CoSearch 独立审查（2026-10-02）

审查对象固定为 `7b93fbd01742715dc14290b0a6e2d3b42e6c19f7`，Search-R1 base 为 `b251ba6b503217a59cd3395c1326d7b66b602989`。未修改实现、未访问远端/GPU、未提交。本报告重新读代码并运行定向反例，不复用旧 PASS。适用边界：主 actor Qwen2.5-3B、原数据/search 保持；允许可训练辅助 ranker 并单独计成本。

**结论：算法接入 CONDITIONAL；正式训练验收 FAIL（放行 gate 未满足，不表示已运行生产训练后失败）。** 两个 agent 的真实参数更新路径存在，ranker 数学主体基本对应论文，但答案命中标签与作者实现不同，默认语义组还混合 availability；CPU FP32 全量 ranker、真实 top50 搜索和联合生命周期尚未生产验证。共享 context 错位、空 action-mask NaN 同样影响主 actor，不能放行正式长跑。

## 1. Primary 来源与定位

- [论文 v2](https://arxiv.org/html/2604.17555v2)：§3.1–3.4、Eq.3–6，§4.1：N=50/K=5，G=8、最多6 search，δ=.8、k_min=3、γ=.5、Hit={1,3,5}、T=1。
- [官方固定源码](https://github.com/snap-research/CoSearch/tree/763bf8c4d7caa363ad845d39ddfd53b81ae377bd) 已下载 `/tmp/cosearch-audit-20261002`，`git rev-parse HEAD` 实得 `763bf8c4d7caa363ad845d39ddfd53b81ae377bd`。
- 官方 `main_co_search_ppo.py:96–155,391` 和 `verl/verl/experimental/agent_loop/agent_loop.py:1979–2070`：独立 main/ranker Ray worker pool、rollout server；实际 CoSearch worker 为 `search_r1_reranker_agent_loop_worker.py`。
- 官方 `scripts/train_co_search_grpo.sh:152–160`：availability+subquery F1 分组，agent_threshold=.8、cond_threshold=.8；`uid_group_functions.py:145–220`；`score_assign_functions.py:60–108`。
- 官方 `verl/verl/tools/co_search_tool.py:163–183,296–302`：trivial answers=yes/no/true/false 全命中时禁用 relevance；`tools/utils/answer_match_reward.py:45–61` 是 NFD Unicode、保留标点/冠词的 token 连续精确匹配。
- 官方 `utils/reward_score/search_qa_f1_with_format_penalty.py:160–189`：普通 token F1，**没有 yes/no/noanswer 特判**。
- 本地 `search_r1/baselines/cosearch.py:16–210`、`controller.py:71–78,119–159,224–269`、`rollout.py:87–225`、`ray_trainer.py:49–61,487–491,755–866`、`ppo_baselines.yaml:97–112`。

已读取 paper-glance skill 并按其串行还原规则核证方法；本任务已明确要求算法审查，因此未重复询问菜单。

## 2. 一轮执行及公式映射

1. 原 driver 按 n_agent 扩展问题。Controller 将每行原问题 index 存入 `question_uids`；SearchRollout 初始化 source 和独立 trajectory_uid。主 actor 按原 `<search>/<answer>` 协议采样，state.source 始终指向扩展后的输入行。
2. ranker.execute 只对 search 行一次 POST，payload 明确 `topk=50,return_scores=True`。按 search 行顺序取回文档，验证批次行数；当前不验证每行确有50条，也不做超时重试。
3. ranker.rank 输入初始 prompt（由 actor tokenizer decode）、当前 query、全部候选文本，无 gold。HF FP32 模型 eval 模式按 T、top_p=1、top_k=0 生成；fresh GenerationConfig 去掉 checkpoint repetition penalty/forced tokens。随后同 eval 模式前向，保存 response token 的 `log softmax(logits/T)`、完整 token ids、trajectory uid。
4. 解析恰好一个 `<ranking>` JSON 列表，长度为 min(3,N)，ID整数、唯一、范围合法。合法输出返回指定3条；非法输出返回检索原 top3，仍保留非法 ranker event。actor 对 observation 继续使用原500 token cap，因此“选3文档”不保证3文档全文均可见，这是固定 Search-R1 对照约束。
5. 主轨迹完成，`main_f1_reward` 根据 action_records 的整轨迹合法性赋 F1 或 -.2，并放在最后一个有效 actor token。grouped_center 在原问题 uid 内算 sample-std advantage；actor PPO 只对主生成 token 的 info_mask 更新。此处 F1 reward 取代原 EM 属 CoSearch 必要变化；原 actor reference KL 仍按共同配置存在，不能称论文无 KL 的逐字 Eq.3 实现。
6. 主 actor 更新后，after_actor 用**重排后的 trajectory_uid**建立 `main F1,targets,legal` 字典。ranker.update 根据字典查 event，不依赖 batch 顺序；非法主轨迹先过滤，因此不会把 -.2 主格式罚传播到 ranker。gold 仅在此离线计算 reward/availability，未进入生成 prompt。
7. 合法主轨迹中的 ranker event：非法 ranking 得 -.2；否则 `r_rel=(Hit1+Hit3+Hit5)/3`，`I_ans=any(candidate contains answer)`。有答案且 r_rel≤.5 时只用 r_rel；高于.5 时 r_rel+main；无答案时只用 main。这对应论文 Eq.6；K=3 时 Hit5=Hit3，分布只能 {0,1/3,2/3,1}，是保留原top3下必须注明的变化。
8. 每问题内按 query token F1 与第一代表 greedy 分组，δ=.8，默认不拆 availability；少于3条不训练。组内标准化 reward 作为每个 response token 的共享 advantage。独立 AdamW 更新 `-min(ratio*A,clip(ratio)*A)`，每个连续 batch 按其总 response token 数归一化；不是逐序列等权。
9. 验证 decorator 临时置 ranker.validation=True，Adapter 通过同 SearchRollout 使用当前 ranker，greedy 生成，不记录 training event。训练 checkpoint 分别保存 actor 与 ranker HF/tokenizer/optimizer.pt；manifest 明示 resume_supported=False，无 joint reload。

## 3. 四维审查

| 维度 | 结论 | 证据与限制 |
|---|---|---|
| 代码接入 | CONDITIONAL | 两独立 optimizer/参数、actor 后 ranker update、同 trained ranker 验证接入存在；定向 real CPU HF generation 成功，但 tiny tokenizer需修正测试夹具输入字段，真实服务/完整trainer尚未跑 |
| 数学 | CONDITIONAL | Eq.4/6、semantic representative、ratio温度及clip方向正确；K3使Hit5重复、availability标签偏差、官方recipe奖励与论文不一致，必须声明采用哪一个 |
| 逻辑 | CONDITIONAL | gold不入prompt，trajectory uid查表抗行重排，非法主轨迹过滤、非法ranker fallback正确；trivial答案与标签正则化造成伪监督，成本/合法率缺少细分 |
| infra | FAIL（正式训练验收） | shared-infra两个P1成立；driver CPU FP32 eager ranker 的规模预算未验证；top50真实服务未测；无联合resume，tiny GPU evidence不是生产整链 |

## 4. 问题与级别

### CS-01 / P1：yes/no/true/false 仍作为 relevance 命中标签（作者一致性缺陷）

本地 `cosearch.py:31–45,169–170` 对任何答案做 contains_answer；官方 tool明确排除全trivial答案，配置 `config/search_r1_tools_goal_guided_config_hit.yaml:28–29` 给出该集合。

实际反例：`ranker_reward(['Yes this is unrelated','noise','noise'],[1,2,3],['yes'],0)` 输出 **1.0**。最终答案错，但无关含 yes 文档给满分 relevance；作者配置下 answer_in_docs=False，成功ranking tool_score=0，随后仅主奖励=0。真实 HotpotQA yes/no 类问题可触发，可能奖励检索布尔词而非证据。建议统一训练 reward/availability 的 trivial 判定，并记录类别计数；不是要求主F1加入yes/no特判。

### CS-02 / P1：命中检测套用 QA F1 normalize 改变作者 pseudo-label（作者一致性缺陷）

本地 `cosearch.py:31–36` 调用 BAPO normalize（删除冠词、把标点直接删除），作者 answer_match_reward 使用保留标点/冠词的 SimpleTokenizer连续精确匹配。实际导入两函数比较：gold `The Hague`、文档 `I live in Hague` → **本地True / 官方False**。因此 I_ans、Hit@k及复合奖励分支会变，不能称与官方标签逻辑一致。建议依据官方 token matcher实现并做针对人名/地名/标点/Unicode的差分验证；若保留本实现，必须作为有意标签改动报告。

### CS-03 / P1：共享长上下文错位、全零mask NaN（继承问题，非CoSearch新引入）

参见 [shared-infra](shared-infra.md) INF-01/02 的实际反例。本地 `rollout.py:87,96–100,149–152` 中生成尾部rolling与评分初始prompt+头部response不同；默认context_alignment=report只计数。`core_algos.py:187–193` / masked_mean空mask除0。CoSearch主actor使用该路径，ranker正确logp不能修复主actor的重要性ratio。正式实验须共同修复/共同门禁，保留GRPO公平对照；报告不宣称远端生产已发生NaN。

### CS-04 / P1：真实 top50 与 CPU 全量ranker规模未验收（未验证的训练阻碍）

`cosearch.py:74–88,109–113,139–141,194–199` 是driver内逐请求串行 FP32 HF eager，全量生成后还做一次完整 score forward；训练另做完整 forward/backward，无FSDP/低精度/FlashAttention/异步批rank。相比官方双Ray/vLLM是明确部署适配，但CPU默认“能启动”不能证明7B可用。

7B模型仅 FP32参数+梯度+两份Adam moments下界约112GB（十进制，未计激活、KV、tokenizer、actor/retriever），并非实测；50候选产生的全序列 logits 约为 `4*L*V` bytes，eager attention随L平方增长。若prompt+max_new_tokens超模型limit直接raise，未设置候选文本预算；retrieval POST 60s失败也终止整批。建议先明确ranker型号/预算、验证50候选实际长度分布、延迟/峰值/吞吐；需要完整 actor→服务top50→真实 rank generation→controller双更新→validation→save 的最小验收。不得把tiny constructed events冒充该证据。

### CS-05 / P2：无信号group仍optimizer.step，训练事件计数不等于有效学习

`cosearch.py:77,177–202` 只过滤group大小，不过滤同奖组。实际用16参数CPU toy model运行本类update：3条同奖、advantage全0，`training_events=3,optimizer_steps=1,weight_decay=.01,delta_l1=.00016021728515625`。有旧Adam动量时无新信号step也可能继续漂移。官方actor默认weight_decay=.01也存在；官方worker通常先过滤同奖group，且其fallback可保留同奖，因此不能把AdamW本身说成CoSearch公式方向bug。建议暴露明确weight_decay并分别记有效非零adv事件/同奖组/真实gradient；现GPU烟雾测试手工设weight_decay=0，对此默认行为未验收。

### CS-06 / P2：成本状态/恢复与推理可靠性边界（能力缺口）

`save:206–210` 保存model/tokenizer/optimizer而未存ranker cost/RNG/config snapshot；`controller.py:251–258` 宣告resume_supported=False，未加载独立optimizer或joint step。长期实验中断不能保证恢复，可重新开始但须记录额外成本。cost有参数、生成/prefill/score/training token计数，但validation score混入总score_forward，验证后不一定立即推送累计metrics；未记录CPU时间/峰值/50文档token/非法ranking率/超时/fallback。注释“invalid action remains -1”与实际-.2不符，仅文档问题。

额外兼容观察：使用仓库 TinyModels 的泛型 PreTrainedTokenizerFast 不patch generate 的真实rank调用首次失败，`ValueError: model_kwargs not used ['token_type_ids']`。probe仅把测试tokenizer.model_input_names设为input_ids/attention_mask后成功。正式Qwen tokenizer通常不提供该字段，故这里不当作已证实Qwen3B生产故障；若宣称任意HF ranker可用，应明确输入字段契约。

## 5. 论文、作者、Search-R1一致性矩阵

| 项目 | 论文 / 作者固定源码 | 本地 | 判断 |
|---|---|---|---|
| 联合优化 | main+generative ranker都更新；作者独立Ray pool/server | main FSDP+driver HF独立AdamW | 核心一致，backend适配 |
| 主模型/data/search | 论文3B-Instruct+7B ranker；作者recipe可配 | 固定Search-R1 Qwen2.5-3B原数据，ranker路径需用户给定 | 有意公平控制，不等价原绝对F1复现 |
| N/K | 50/5 | 50/3，validate强制keep=base topk | 用户约束下允许，须注明Hit5重复及obs500 cap |
| G/search预算 | 论文8/最多6 | 原n_agent/4轮，原500 action tokens | 有意保留基座预算 |
| 分组 | 论文仅同q0 greedy F1；作者recipe先availability再F1 | 默认split_availability=False，可显式true | 符合论文描述，偏离官方默认；不能同时宣称两者一致 |
| query normalize | 作者去冠词、标点改空格 | 标点直接删除 | 细微差异，如 foo-bar token分组可能改变 |
| 组过滤 | 论文min3；实际CoSearch worker min4、同奖过滤、duplicate重分奖、cap/align与fallback | min3，不同奖过滤/duplicate/cap | 论文优先适配，官方偏离须披露 |
| ranker reward | 论文连续F1+γ=.5；作者recipe agent F1≥.8二值、cond=.8 | 连续F1+.5 | 论文一致、官方recipe有意偏离；边界≤也不同但默认离散Hit不出现.5 |
| pseudo relevance | 作者精确token串、排除trivial | QA normalize且含trivial | CS-01/02，需修复或明确改法 |
| 主F1 yes/no及空normalize | 官方普通非空F1，无特殊布尔/noanswer分支；零公共token返回0 | 普通非空F1一致；双方normalize为空返回1，另从BAPO继承IDK=0 | “yes no”对“yes”都2/3，不算CoSearch bug；空normalize反例见CS-07，IDK若gold也是IDK才有差异，当前固定数据未证实 |
| format | 作者reason+tool_call/answer严格chat块；论文ReAct | 基座think可选+search/answer、全records合法检查 | 为保留原actor协议的有意适配；不能直接用官方reason规范判原基座非法 |
| 非法主轨迹 | 论文剔除ranker训练 | before grouping跳过 | 一致；官方最终执行路径的过滤细节未逐个reward hook动态验证 |
| invalid ranker | -.2与检索fallback | 同-.2+top3 fallback | 一致（标签tag/prompt是本地协议） |
| inference gold | 仅训练标签 | gold仅update，validation也不读取gold排行 | 静态数据流通过 |
| rollout policy logp | 作者vLLM/FSDP | fresh HF GenerationConfig+同T/eval logits | tiny真实采样独立验证通过，生产模型未验 |
| 验证/重排 | current ranker / uid grouping | decorator+Adapter，event按trajectory_uid lookup | 静态正确；验证主指标保留base EM而非论文F1须注明 |
| checkpoint | 作者独立workers checkpoint | ranker权重/tokenizer/optimizer+actor保存，无joint resume | 保存存在，完整恢复不支持 |

## 6. 实际命令、结果与未覆盖

本轮执行：git SHA/base diff复核；固定官方git clone+checkout；primary论文网页读取；`.venv-cpu/bin/python` 导入本地与官方F1/answer matcher并打印上述差分；CPU toy直接调用GenerativeRanker.update确认同奖行为。官方answer matcher导入时只stub未使用的pytrec_eval模块，**未替换SimpleTokenizer/has_answer_string**，因此差分是实际作者匹配函数结果。

独立真实HF generation probe：调用仓库TinyModels.setUpClass建立随机小Qwen（单层hidden16、vocab≈300），未patch generate；测试T=.7、max_new_tokens8。修正测试tokenizer字段后，`response_tokens=8,ranking=None,fallback_count=3,max_old_logp_error=0.0`；再greedy validation，`validation_events_unchanged=1,validation_generated=8`。它证明该受限真实生成/打分/invalid fallback/validation事件路径；模型随机，**未验证合法ranking生成、真实ranker质量、完整retriever/controller联合执行**。

已有2GPU tiny证据中ranker.events人为构造、old_logp另算、weight_decay设0；可证明真实辅助optimizer和保存，不能代替生产50候选/真实生成。root本轮共享24CPU/39reproduction通过与本文差分同时成立。

未覆盖：目标真实parquet/模型资产、真实top50检索服务/FAISS索引、Qwen规模ranker prompt上下文分布、CPU峰值/延迟、GPU正式full FSDP/vLLM/offload/同步、合法rank生成概率、trainer validation动态回调/metadata完整性、联合save后reload、长跑公平F1统计。

正式放行条件：先处理共同P1 context/mask；明确辅助ranker及CPU/GPU资源与成本预算；统一CS-01/02的作者标签策略或声明改法；选定“论文优先”分组/奖励配方并固定配置；跑真实服务top50和双agent完整最小生命周期，再完成资源规模验收。当前允许继续小规模隔离验证，不应启动宣称等价复现的正式训练。

定向pytest实际命令：`.venv-cpu/bin/python -m pytest -q tests/baselines/test_algorithms.py -k 'ranker or ranking or semantic' tests/baselines/test_models_and_driver.py -k 'ranker'`，结果 **7 passed, 16 deselected**，1.54s；pytest最后一个-k为实际选择条件。该测试的generate patch限制已在上文列明。

### CS-07 / P2：主F1的双方空normalize边界与作者不同（条件性差异）

本轮追加独立导入复核：`answer_f1('the',['a'])` 输出 **1.0**，官方 `compute_f1('the',['a'])` 输出 **0.0**。本地 `bapo.py:18–21` 对双方normalize后空token执行 `float(a == b)`；官方 `search_qa_f1_with_format_penalty.py:175–179` 对公共token数0直接返回0。因此主F1一致性只适用于通常的非空normalize答案，不能泛称所有输入一致。此分数也会进入ranker复合奖励。固定parquet是否含冠词/纯标点等normalize为空的gold未读取验证，**不声称实际训练已经触发**；建议改成作者零公共token=0的定义，或明确披露边界变化并统计数据覆盖。该边界与yes/no/noanswer特判是不同问题，仍保留“官方CoSearch无该布尔特判，本地不加并非CoSearch bug”的结论。
