# Preparation3 OPSD配对与教师语义只读审查（2026-10-08）

实际源根：`/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runtime/opsd-preparation3-xformers/search_e1/`。下述行号对应本目录保留的实际冻结源码副本。仅CPU文件读取、AST比较与一次pairs流式聚合；没有GPU、模型推理、检索请求、运行修改或publisher编辑。

## 已产生结果与归因边界

实际`runs/opsd/history.json`：source100 native512 baseline 239/512=46.6796875%；raw epoch1 203/512=39.6484375%，减少36题、7.03125百分点；平均训练loss 0.17748732205277348。此loss是沿预存轨迹的分布蒸馏loss，不是问答正确率，也不能证明EM提升。既有native评测审计已确认baseline/post同条件；本审查不重做模型验收，也不把archive GRPO另一路512分数当OPSD基线。raw epoch1最终七基准成绩尚未由本报告确认。

## 实际执行与配对

`rollout.py:100–102,117`：vLLM生成后以严格reward评分；`vllm_rollout.py:9–10,15–17,32–40,61–64`：BF16 source100，每题5条，temperature1、top_p1，每段500 token，8192完整上下文上限。`parallel_rollout.py:27,35–39,55–81`：题目按8worker交错切分，seed42–49，每题sibling0–4完整唯一覆盖。此采样种子不同于OPSD初始化seed42。最多5段/4次真实检索，top3、每次结果480token；observations加入轨迹，policy_spans只标记模型实际生成段（vllm_rollout:18–59）。

`mine.py:10–13`一次读取845515轨迹，调用`core.py:53–73`，产100974pairs：

- 按question_id将5条同题轨迹分组。没有任何reward1的题整组略过；不是将所有失败都用于训练。
- reference取成功轨迹中的最少实际search_calls，再最少完整response token_count；包含思考、检索query、检索observations、正确最终答案。
- 优先在失败轨迹中取与reference完整字符串归一化Levenshtein距离最大的一个。没有失败时，改取另一成功轨迹中距离最大的一个。每题恰好一个pair。
- `answers`用于选成功与失败，保留进pair，但`encoded_pair`不读取answers。reward0可来自错误答案，也可来自严格协议/终止不合格；不等价于所有失败均有一条显式错误最终答案（core:23–38）。

本次一次流式证据`pair-stat-evidence.json`：100974个唯一question_id、全部reference_reward1；失败student44079（43.65%），成功student56895（56.35%）。在5条完整覆盖前提下，成功student正是全成功题的fallback。NQ42704pairs、HotpotQA58270pairs；原训练题169103，因此68129题（40.29%）不提供OPSD监督。多数pair是成功/成功，而非失败纠错。该分布统计已通知root，后续不重复全量扫描。

## 教师到底看到什么，token怎样对齐

`core.py:12–17`明确在teacher user prompt附加完整“Expert reference trajectory”与“Read the reference, then attempt ... in your own words.”。教师实际看见成功轨迹中的正确最终答案及检索内容；没有额外直接拼接answers数组。reference是source100自身采样后由EM筛选出的成功路径，并非另一个专家模型生成或逐步验证过的推理。因此“成功”只证明其最终答案满足评分，不保证每条思考/检索合理。

`opsd.py:16–22,29–37`把**同一student response token序列**接在两个不同prefix之后：student只见问题，teacher还见成功reference。teacher在eval、no_grad、disable_adapter下使用冻结source100；student开启当前LoRA。teacher/student logits切片都从各自prefix末token预测response第一个token，之后按相同response位置对齐，没有把成功reference的第j个token强行对应失败轨迹第j个token。

教师也沿student历史进行teacher forcing，包括此前生成的错误思考、query、answer及该路径真实retrieval观察。训练目标是teacher的全词表软分布，不是对student原始错误token做硬标签交叉熵；不能把当前实现描述为“直接教学生背错误答案”。teacher LoRA始终关闭；本轮训练没有重采样更新后的student轨迹。

## loss覆盖与权重

`core.py:41–51`将information标签及内部内容排除loss，并要求token完全落在模型policy_spans；外部结果仍作为双方上下文。于是合法模型作者的think、search动作/查询、最终answer均进入loss，forced budget提示不进入loss。未针对最终答案或搜索关键决策另设更大权重，也没有额外SFT答案CE、reward advantage或在线RL反馈。

`core.py:75–80`为teacher到student的forward KL，FP32 logsoftmax，逐词表贡献仅上截clip10；`opsd.py:38–46`先按有效policy token取mean；`:127–147`再按真实全局batch轨迹数取mean、手工SUM，完整batch32，尾14，合计3156updates。配置usable_pairs100974、skipped_pairs0，故本次没有8192超限或空mask过滤导致新增样本选择变化。

## 与压缩包的相同和区别

保留archive `work/search-e1-publish/experiments/search_e1/archive/search_e1/core.py`的prompt、response_encoding、mine_pairs、clipped_forward_kl，以及opsd.py的encoded_pair和pair_loss，和实际runtime各函数source段**逐字节相同**；证据`archive-function-equivalence.json`。原archive行号分别core12–17/36–42/44–64/66–71、opsd16–46。不能将privileged参考提示、成功fallback、字符距离选择或同student历史KL当作本轮运行overlay新引入的算法改动。

确切不同：archive core:23–30在整个response中找最后answer标签；runtime core:23–38改为严格作者policy spans终结答案评分。因此实际成功/失败集合可能不同于旧宽松评分；这是明确实验条件差异，但没有对本次845515轨迹额外重评分，无法报告差多少。执行overlay增加全rank有限值/更新证据和global validation；pair_loss数学不变。

## 能解释下降的风险，尚不是已证实原因

1. 特权上下文蒸馏的分布落差：teacher预先知道正确答案和成功观察，student推理时没有该参考；teacher在坏student历史上的软目标是否可由无参考student实现，尚无分布/逐题证据。监督可能同时改变已有解题和检索习惯。
2. 配对选择偏向已会题：56.35%成功/成功fallback，完全失败的40.29%训练题没有监督。可能主要改变容易题路径/文风，而不提供最难题纠错信号；此处分布是事实，“导致下降”仍是假说。
3. 最大字符距离包括retrieval观察，虽观察不计loss，仍能主导选哪条student；不是按错误类型、搜索决策差距或最终答案差距挑选。reference最少搜索的偏好也可能偏向少检索策略，但loss并非直接惩罚搜索次数。
4. 一条轨迹所有有效token等权，长思考/query文字可稀释关键最终答案/决策监督；本次长度字符统计只能支持存在长短差异，不是token级答案loss比例实测。
5. 一个固定学习率1e-5的一epoch、3156次更新可以改变source100行为；有限loss/参数一致只证明执行正常，不证明该优化目标与EM一致。不能把训练loss0.177或仅36题净下降解释成特定机制已经坐实。

最诚实结论：已观察同条件native512 raw adapter下降；源码揭示上述明确目标和样本选择机制，并未发现teacher/student错位token或直接硬标签训练错误答案。当前best选择仍保留baseline（opsd:114–118,188–202），因此selected merged七基准与raw epoch1七基准必须分别报告。定位原因需要后续保留逐题baseline/raw配对成败与动作变化、raw七基准结果；目前不应改科学超参后把新结果混为原轮结果。

## Root复核与持久证据链接

本报告的 `core.py`/`opsd.py` 行号对应实际冻结runtime的只读副本，SHA分别为 `0fbfbe63d0aafb41f79db702e08c9a0efc5f9bbf29a2964310594ec2ec0a83fc` / `61df9cc630bca7466f3edfb06da18ad92ddff1c713bf7c6c4a104ba88672c037`。Root于10月8日再次通过只读SSH读取实际源并核对两者。此科学代码未修改。

[首次配对聚合](receipts/opsd-epoch1-pair-stat-evidence-20261008.json)、[聚合探针源码](receipts/opsd-epoch1-pair-stat-probe-20261008.py)、[归一化函数逐字节比较](receipts/opsd-epoch1-function-equivalence-20261008.json)、[独立完整配对聚合](receipts/opsd-epoch1-full-pair-statistics-20261008.json)。两次独立聚合的pairs内容SHA均为 `a81a3ae61bb402f6b99dd35ae552c4bb76df8da391f3620c17fbd742ba2b1f40`，成功/失败计数一致。

虽然统计入口是mined pairs，[实际native配置](receipts/native-opsd-config-20261008.json)确认输入同一pairs路径、usable_pairs=100974、skipped_pairs=0。因此本次没有任何encoding/context过滤后的样本丢弃，统计比例适用于本轮全部训练样本。

完整字符长度包括检索观察；不能将student平均4363.25字符、reference平均3413.08字符或字符长度分位当作policy-token长度、答案loss比重或实际搜索调用数。完整answer标签只是语法检查，不能替代严格终结评分；reward0可能同时包含内容错误与协议错误。

原压缩包宽松评分与本次严格评分的差异可能改变历史样本筛选，但本次native512 baseline/raw采用同一评分和推理条件。当前下降不能简单归因为训练前后评分规则改变。学习率为原压缩包复现假设，不是已核实的作者公开OPSD超参；没有新增超参实验。

[10:30 archive独立审计](receipts/archive-audit-20261008T023007Z.json)记录159/200完成、160处理中，reward0.472与裁剪前gradient norm18.276有限，实际自有4卡与检索门禁/三个669文件runtime通过。该GRPO对照与新OPSD3156更新分别计数。10:34 Root读取selected/baseline七集评测已有14080/51713条完整分片行，仍正常运行；raw epoch1七集GPU评测尚未启动。raw CPUmerge已accepted，不重复合并、不抢占GPU、不重新训练GRPO。

未进行teacher分布校准、逐tokenKL诊断或超参消融。本报告列出的下降机制均为待验证假设；下一步继续现有baseline/raw同条件七基准，比较同题成败、搜索次数、输出终止和长度的变化，然后决定哪些假设获得支持。
