# OPSD 修复后独立 infra / performance / scoring / effect 边界审查

2026-10-04。固定审查 HEAD `fae5e3295c18a0c6b80e314b2fdce61ad571fec6`。读取 canonical results/opsd/STATUS.json、REPORT.md，固定3007/3146原记录、原infra报告及当前配置/入口/评分/OPSD计算源码。本次仅CPU和只读，不修改生产源码、不提交GPU、训练或Git。结论：**无裁剪目标与教师提示结构修复有明确证据；新训练部署的默认安全边界仍有阻断，性能与真实Search效果尚未全部修复/验收。**

## 具体阻断与风险

### [P1] OPSD通用入口默认仍执行CPU教师，并未强制同节点本机检索

`verl/trainer/config/ppo_baselines.yaml:106` 为 `teacher_device: cpu`；`actor.py:18-19` 将该值直接传入 `FrozenTeacher`，未设置时也回退CPU。`distillation.py:200-202` 接受CPU；`execution_device` 决定 update 的实际执行设备。CPU存储回迁本身是正常机制，但这里CPU是**执行设备**，不能用“生成期间teacher放CPU”解释掉。

`ppo_baselines.yaml:73-74` 默认 `local_runtime_only:false / local_runtime_host:null`。`local_runtime.py:16-17` 在关闭时直接return。`reproduction/baselines/run.sh:25-30` OPSD设置path、privileged、纯OPSD损失与micro，但不覆盖teacher设备或local-only。`main.py:12`虽然调用local检查，默认关闭使它不阻止远端retriever/Ray。`preflight.py:125-126` 是对配置URL直接HTTP请求；assets/API结构通过不证明编码/索引CUDA、本机所属PID或无跨集群。

所以：用新HEAD直接走通用 `run.sh opsd`、只设置teacher资产路径，仍可能回到FP32 CPU teacher；配置远端URL也未被默认local边界拒绝。这是**未来部署风险，未观察到3007已经回退**。旧3007的resolved overrides和独立门禁已经证明CUDA执行/本机GPU检索，不能替新配置担保。新本任务launcher/config必须明确并验收 `teacher_device:cuda`、`local_runtime_only:true`、host h100-10、loopback检索及Ray；同时验真实E5/FAISS与worker CUDA/PID/UUID来源。不能仅改摘要booleans宣称通过。

### [P1] 历史被冻结入口不会自动采用fae的null目标

新函数 `distillation.py:12` 默认None，配置:115 null，实际CPU Hydra入口核验通过；但显式`.05`仍作为历史control接受。旧8e的原工具、resolved YAML、receipt/source pins仍是历史来源；不能拿新分支HEAD推断这些入口已切换。`opsd-runtime-20261003/gpu_job_entry.sh:6-7`仅要求环境指定commit一致；它不自动认定该commit就是本次批准source。旧blocked-e3-v2入口固定e3且exit78，旧HF500入口固定8e且必须500，不应解除阻塞或重标成当前fae。

未来必须重新固定source/config全部SHA及实际进口路径，并显式核 `component_clip:null / teacher_prompt_mode:user_context / privileged_answers:true`，避免只测Python默认None却实际调用设置仍.05或关闭私有teacher内容。保留历史控制是合理的，但控制与修复运行需要不同immutable记录。当前不存在fae重新训练的实际证据。

### [P2] 性能差异仍在，尚无修复后同节点update基准

fae `distillation.py:119-123` 仍调用padded student_model并保留矩形完整词表；不会因use_remove_padding自动走GRPO packed forward。:137-163仍逐chunk16完整teacher head/KL，:152张量finite转Python条件会host同步；teacher仍固定FP32/eager且update前后转移。无裁剪删除的是分量clamp，不删除teacher、full vocab、padding、微批循环或checkpoint成本。当前没有同节点真实修复后training forward/backward/update性能。

固定3007前5平均update407.0216秒、step80 update401.573秒；H20用户55.59秒分别约7.32/7.22倍。这是额外teacher/fullKL与micro2 vs原local8等工作量/部署差异的诊断对照，非严格matched benchmark。不能把所有倍数归因padding或FP32，更不能将3146单卡225秒分布诊断当training update速度。最小等价padding方案尚待独立代码、有效位/完整梯度与原真实数值验证，再实际容量/性能测量；本报告不预先通过该方案。

## 已通过与严格边界

当前qa_protocol.py:parse_model_action/extract_final_answer、evaluation.py:batch_answer_metrics使用attention&info_mask分隔模型署名span，评分只采用最后模型span的非嵌套terminal answer。think/search/观察中的答案不应冒充final；EM/F1奖励与format使用修复后的协议，实际CPU反例测试通过。`evaluation.py:search_calls`仍计标签regex出现次数，不能把它作为有效工具调用次数/真实retrieval成功率；可非法嵌套标签也可能被计数。必须结合原生合法动作/实际HTTP记录报告。

OPSD的reward/EM/F1只作记录，目标仍全词表teacher蒸馏；评分修复本身不增加正确性reward梯度。旧3007初始/50阶段EM.125→.09375、F1.17378→.11891属于旧协议历史，不能直接叫修复后效果；原生finish/action退化也不能因为重评分而抹掉。

周期validation默认data/val_files指向validation-64、val_batch_size64、test_freq50；BAPO独立phase_num64也保留。OPSD并非每步执行BAPO phase。3007实际初始/50各64配对JSON-console门禁是历史真实通过；只验证默认文件名不是所有未来输入都64的proof，未来resolved数据文件/实际rowcount/console必须重新配对，不能换3393或51713到每步阶段。

当前 `corrected_evaluation.py` 默认opt-in，完整评分原输入外pin、mask/tokens克隆raw view、CPU native replay、每64顺序batch与51713尾1、零update runtime guard/source角色分离机制仍在；其严格前置明确training3007 actual500/HF500，**现在3007仅80 CANCELLED**，所以旧最终评价工具保持blocked正是正确行为。这些CPU工具不是现成的修复后fresh5证明；未来若用新受控训练/模型，需要另行新的外部身份与门禁，不能冒充旧500completion或把评分视图称完整pre-truncation轨迹。

## 实际诊断证明了什么

3146同h100-10自有单H100 COMPLETED0:0，225秒，manifest3618cc9e2ccefe5be052b2b2ee13adaac25734f1895d0879011518c64ba11002。原256NPZ完整保留在本节点，独立逐SHA+解析公式重算已通过。64有序train**初始state**，base/HF5学生与固定base旧/新teacher；无检索、完整rollout、optimizer或parameter更新。

新user_context+旧.05在真实logits上产生base30/64、HF5 34/64与真KL相反的logit方向；无裁剪重算256/256小logit步真KL下降，0零/0上升。这证明该批分布的目标方向性质，**不是实际Adam/FSDP参数更新、晚期history/搜索教师能力或reward效果改善**。新提示还改变教学文字，不能隔离位置的因果。标签概率不等于有效完成率；完整首alias直接回答概率不等于思考/检索后的最终accuracy。保存的实际HF只有step5，不存在80/500。

canonical STATUS与REPORT最新段落明确not_deployed/not_trained/effect_accepted:false，保留formal80 CANCELLED、正式500未完成与上述因果限制，未发现将诊断完整性冒充效果通过的新主张。但旧历史段落仍有continue500文字，执行者必须服从最新停止/修复指令，不能仅检索旧段重新自动长跑。

## 本轮验证记录

本机CUDA隐藏、OMP1、CPU环境，检查local-only、phase64、terminal answer反例、真实reward/validation入口和corrected-view replay测试：**91 PASS /4.49秒**，原输出 `work/opsd-final-infra-review-cpu-retry.txt`。首次缺tests/baselines sibling模块PYTHONPATH的collection error保留于 `work/opsd-final-infra-review-cpu.txt`，重试仅增加正确fixture路径，没有修改源码或安装依赖。

这些是当前fae软件/CPU契约证明，不是实际GPU检索/训练吞吐、teacher搜索轨迹、paired最终EM/F1或正式完整验收。此review不放行“全部4项已修完”或自动训练；defaultCPU/local-disabled部署风险、性能方案与真实history/effect仍需分别关闭。
