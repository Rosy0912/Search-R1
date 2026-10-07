# Prep3 native512 baseline／post-OPSD条件审查

只读范围：实际HKUST1 prep3冻结OPSD、HF SearchAgent、严格评分、检索源码/配置；同HKUST集群archive实际GRPO冻结路径；已保留原压缩包代码和旧收据。没有GPU生成、HTTP推理负载、作业变更、代码修改或线程通知。当前0 OPSD更新；以下是评测设计和代码路径结论，没有native512新收益结果。

## 主结论

prep3训练前archive100基线与post-epoch LoRA验证同条件，可用post_native512_EM − baseline_native512_EM描述此次验证上的训练变化；两次都启用同一个LoRA wrapper，基线B零初始化使adapter为恒等增量，训练后adapter生效。它们和当前archive GRPO step100的512验证不是同一推理条件，不能拿旧44.7265625%直接相减归因为OPSD。进一步独立测试集收益仍需相同评测条件，验证集还参与best_adapter选择。

## native baseline与post-epoch逐项核对

实际冻结native-opsd.py:81默认seed42，92–97先seed CPU/CUDA与python再加载同一assets/archive100、明确BF16/SDPA、LoRA rank16 alpha32 dropout0；没有rank seed偏移。113创建一次SearchAgent，114及182调用同一个_global_validation（56–65），相同validation[rank::world]；八rank每人64题，正确数/题数int64 SUM，覆盖原512。验证均为同一个model/tokenizer对象，不重新加载、merge或切换vLLM；HF backend由SearchAgent直接model.generate决定，pipeline的VLLM_ATTENTION_BACKEND=XFORMERS是外部vLLM生成配置，不能称native验证采用vLLM/XFORMERS。

native-rollout.py:69–74两次均agent.run(question, temperature=0)，35令do_sample=False，即greedy。每题一次rollout、最多5段，每段500新token（20、29、35），最大完整上下文8192；33–34在下一段会超限时直接终止，不滑窗截取。第5段明确插入预算耗尽强制最终回答提示（30）。搜索最多4次，topk3、return_scores=True、单query原文.strip发送（51–54）；无客户端query token单独上限。查询受该段500新token与标签生成范围约束，检索服务另截断。

native-retriever-server.py:76–83两次同URL18004、同完整21,015,324×768 FP32 E5/wiki2018索引和CPU FP32 E5 encoder，batch128；未覆盖retrieval_query_max_length。native-original-retrieval-server.py:301默认query max256，223传入encoder，87–96加query:前缀、tokenizer truncation=True，因此实际检索encoder输入最多256 token。LLM问题上下文8192、单段500和检索query256是不同预算，不能混写。top3内容按Doc i:原contents组织（native-rollout.py:58–65），正文合并后先截480 tokenizer token，再添加闭合information标签。

生成继承assets/archive100/generation_config.json，但显式覆盖max_new_tokens=500、do_sample=False、pad_token_id=tokenizer.eos_token_id。该文件repetition_penalty=1.05仍有效；temperature0.7/top_p0.8/top_k20在do_sample=False下不构成采样（模型generate可能发不匹配warning，不等于随机抽样）。两次继承相同配置，eos集合151645/151643相同。不应仅说temperature=0就遗漏repetition penalty等其他生效设置。

评分两次同native-core.py:23–38：只取policy_spans，经native-qa-protocol.py:31–37取最后 authored span完整独立answer动作；拒绝非终结、多动作、嵌套协议、空答案。normalize后与同answers列表EM比对。信息区、强制提示、问题示例不参与答案作者跨度。评测不把teacher reference加入prompt；teacher disable_adapter仅在pair_loss（opsd:32），不作用baseline或post验证。SearchAgent每题model.eval（rollout:26），两次LoRA与attention dropout0，不存在随机dropout条件变化。

基线wrapper与postwrapper一致，B=0数学上增量为零；它不是另起裸模型验证，不能声称测得与任意其他backend裸模型逐位输出一致。评测前后改变的是训练后LoRA参数；检索固定服务正常情况下条件相同，但当前尚无两次实际评测时间窗服务身份/每query证据，服务后续身份仍需运行记录闭环。

## native与当前archive GRPO step100的确切差异

1. 数据已证明一致。native/archive-live-validation.jsonl逐字节同SHA75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76。dataset-cpu-proof.json用实际parquet读取，512项题ID、answers、source、question全部一一匹配，NQ/Hotpot各256；parquet SHA30d8a6175f3f3f7cc6d4e106521035be672cf9b35e98d371a8e81ac487859e71。因此native overall准确率与旧两组等权macro数学上相同分母，但推理条件不同。

2. greedy标志存在实际源码路径问题。archive-live-ray_trainer.py:504–510虽设置do_sample=False/validate=True/recompute_log_prob=False，archive-live-generation.py:243–245和286–288重建active DataProto时只传tensor dict，不传meta_info。archive-live-protocol.py:281、292–293的from_dict默认meta_info=None后变{}。is_validation只在generation.py:36赋值，从未用于生成分支。_generate_with_gpu_padding整除路径185–186直接传active batch；非整除197重建padded batch依然不传meta。因此主循环/末轮、有无padding都丢失False。

archive-live-fsdp-workers.py:432–451入口只补eos/pad，没有恢复do_sample=False；archive-live-fsdp-vllm-sharding.py:112–119只allgather batch并返回data，不恢复标志。archive-live-vllm_rollout.py:162 get('do_sample',True)遂取True，163–171的temperature0覆盖分支未触发。grpo.sh:34明确temperature1/top_p1，SamplingParams:106–122从该配置构建。已有日志kwargs记录temperature1.0、top_p1.0、max_tokens500、n1，与此推断一致，但没有逐次validation有效SamplingParams日志，不能把初始化kwargs当作每次validation实测采样记录。

cpu-meta-probe.py直接在实际archive环境以实际DataProto和实际冻结AST运行CPU tensors/fake actor，主循环243与末轮286各测试32条整除、31条padding，共4种路径，全都使worker收到空meta，实际vLLM do_sample表达式为True。结果保存cpu-meta-probe-result.json。它明确证明该代码路径的元数据丢失，结合入口检查支持“旧archive512走默认温度1采样”的结论；没有新增GPU推理，也没有改变现场逻辑。

3. 后端和模型计算路径：archive由GRPO FSDP actor同步到vLLM、批32调度，native HF/SDPA每rank串行64题；两者生成名义BF16（archive-live-ppo_trainer.yaml:75 vs opsd:95），但src100磁盘checkpoint/float32 config被native显式转BF16，训练前LoRA wrapper额外存在。backend、kernel、batch调度不同，不能宣称逐位等价。archive vLLM SamplingParams未指定repetition_penalty，实际env vllm/sampling_params.py:178默认1.0；native继承checkpoint1.05，可能改变greedy候选分数。

4. 上下文与动作：archive grpo.sh:26起始上限2048、rolling prompt4096、段500、obs500，generation.py:107–114左侧滑窗保留最近4096，164–167累计响应右侧截至4096；native保留完整历史直到下一段超8192即终止。archive四轮+最后一轮（234、278–297）名义最多5段/4次真搜索，但末轮没有native的预算耗尽强制提示。archive遇非法动作注入重试提示并继续（395–401），native遇无search/answer直接终止（51–53）。archive响应后处理优先search closing，即使answer closing更早（61–65）；native在两种closing中取最早（42–44）。这些会改变完整轨迹。

5. 观察文本：archive generation.py:460–467解析标题并格式化Doc i(Title: title)正文，391用双换行information包裹，再将包括标签的obs整段截500（77–89），可能截断闭标签；native Doc i:原contents、正文480然后闭标签。检索逻辑资产同类型完整FP32，但archive用既有stream-batched服务，native用HKUST1 segmented service18004；不是同一服务PID/执行实现。只凭资产类型不主张每query bitwise top3相同，应引用各自已验收服务身份与固定query校验；本次无HTTP请求。

6. 当前archive评分主线同strict protocol，不应误称旧legacy reward。native-qa-protocol.py与archive-live-qa_protocol.py字节相同。archive-live-main_ppo.py:73–88从info_mask decode模型跨度并调用compute_score_generated_em，archive-live-qa_em.py:140–146同extract_final_answer、EM；legacy只旁路诊断。与native区别是跨度来源（token info_mask vs char policy_spans），以及上述响应/obs截断导致跨度本身可能不同。归一化算法相同。原压缩包main_ppo.py:69–78则解码完整prompt+response，qa_em.py:73–82需至少两个answer标签并取末尾，不能把原包reward混同本次运行strict reward。

## 收据与结果边界

当前archive source100是HKUST archive运行：已保存archive-audit-20261007T071827Z.json的NQ0.422/Hotpot0.473，从各256题与三位日志唯一推得108+121=229/512=44.7265625%。这是旧GRPO验证，不是native基线，也不是OPSD结果。原压缩包reports/step100_validation_result.json的0.402/0.461属另一历史运行，不能移作当前源100成绩。

native baseline记录在opsd history epoch0；postepoch记录同history epoch1；best_adapter保留baseline并只在EM更高时替换（native-opsd.py:115–116、188–193），所以最终selected validation score不低于baseline是选择规则的结果，不能独立证明训练提升。应该同时报epoch1原始EM、baseline及selected；独立最终测试另以统一backend比较。

## publisher docs用词搜索（只搜不改）

对work/search-e1-publish/experiments/search_e1/docs所有.md执行忽略大小写greedy、deterministic、temperature紧邻0、贪心、确定性、温度紧邻0搜索，没有匹配，未发现需纠正的具体Markdown行。JSON原始收据/代码中的do_sample=False仅是请求配置，不能改写为生成有效参数；这次保存源码和探针结果供root解释。未编辑publisher任何文件。

扩大到publisher全仓Markdown搜索后：reproduction/review/logic-review.md:15已经正确指出metadata缺失使validation落回training sampling；:89的greedy-tail是该独立review测试，不是archive512实测描述。archive/README.md:92与README.lumia_snapshot.md:92的“贪心解码”明确指最终七数据集独立并行评测，不是512 GRPO验证，也无需用本问题修改不可变archive。没有发现误称archive512 greedy/temperature0/deterministic的具体文档行。
