# Agent 6 提交后独立复审

对象：`work/cosearch@9c3d7ccd37e67b71ed8b225eba451065f33ab480`，追加复审 `c5d9c83` probe修复，最终追加至 `30d3aebf78ee66ecaf08e17afce16407376f0133`。共同部署source `297e1c1` / cherry-pick `cd86c9d`采用owner已独立冻结的审查，不在此重复作全面门禁审查。没有修改被审查repo或启动训练。

**最终结论：ranker核心诊断/strict50补丁通过代码与CPU复审；probe prompt问题在c5d9c83已修复，可以继续真实能力探测。真实模型/检索/联合GPU/效果仍未放行。** 全部来源和方法偏离沿用初审 `work/review-cosearch.md`，不把CPU条件通过写成效果通过。

## 关闭或改善的初审问题

1. `execute`现在逐行严格要求50个candidate且document.contents为str，缩成top3、49条、非法内容会在rank前拒绝。成功取50但内容重复是retriever品质问题，未要求额外去重改变任务。
2. rank成本分别记录训练/验证合法ranking、fallback、prompt最大token、生成/score token、CPU与wall time；update记录CPU/wall、informative group、distinct rankings、preclip gradient norm和每参数最多8点delta。未改变数学/奖励/温度/optimizer默认weight_decay=.01。
3. optimizer回归改为不同ranking、weight_decay=0、gradient norm>1e-5，修复初审“相同序列仅靠参数变化不足以证明RL信号”的证据问题。该测试仍patch generate；它证明真实tiny HF forward/backward/AdamW的有效信号路径，不证明base能自然生成合法ranking。
4. validation decorator在函数返回后添加cost_metrics并打印JSON，最终评估能捕获全部累计ranker成本，关闭“最后一次验证成本仅留在内存”的缺口。保存于final evaluation之前的checkpoint cost snapshot仍不含最终validation，最终JSON是补充会计来源。

## 独立实际验证

运行 `test_cosearch_fixes.py` 与 `test_models_and_driver.py -k 'not real_fit_driver'`：**10 passed,1 deselected**。覆盖strict50/schema fail-fast、同奖组Adam状态保持、真实未patch tiny generate、验证训练计数隔离、修改后的有效ranker梯度/参数probe等。没有重新运行完整140项，不把owner结果冒称独立执行。

本地按README脚本路径 `python reproduction/baselines/cosearch_probe.py --help` 在未editable安装该checkout的旧CPU环境报 `ModuleNotFoundError: search_r1`；按module运行成功。检查bootstrap第62行已 `-e repo`安装，故已按计划生产环境此项有明确解决路径，不列生产P1。

## 已修复：R-CS-F01 / 原P1 probe与生产初始问题上下文不一致

`cosearch_probe.py:44`无条件调用tokenizer.apply_chat_template；实际 `RLHFDataset.__getitem__:128–131`只在tokenizer.chat_template存在时使用，否则直接chat[0]['content']。缺template的base tokenizer会在probe失败，而训练dataset可运行；即使template存在，probe没有复现dataset tokenization/truncation和SearchRollout max_start_length2048→decode的初始ranker question context。长问题可导致probe测得与训练不同的50文档prompt长度和溢出结论。

建议用与dataset一致的template fallback和tokenize/postprocess参数，再取原max_start2048窗口、skip_special_tokens decode，最后传ranker；最好复用一个不含gold的公共helper以防prompt字符串漂移。输入query=原问题是允许的standalone检索能力探测，但不代表actor实际生成subquery的分布；报告必须保留该限制。不能为“修probe”截断50 passages。

追加实际核验c5d9c83：现在template存在才apply，否则chat[0].content；add_special_tokens=False与dataset一致；超过4096拒绝；取最后2048 token后skip_special_tokens decode，与SearchRollout.initial_states的unpadded prompt一致。未截断50候选。sys.path引导可靠，独立再次运行README脚本`--help`成功；`--questions 0`实际exit2并在读取模型/数据之前拒绝。Darwin RSS换算也已修正。这些关闭该P1及非正值/单位问题。

## 诊断实现边界与剩余项

- CPU参数抽样每个有梯度tensor最多8点，保存量随tensor数而非3B元素数增长；当前HF Qwen参数连续存储，reshape是view，不复制全权重。非连续参数reshape可能分配副本，泛型HF扩展须注意；当前固定Qwen没有已证实RAM风险。delta=0只表示抽样点未变化，不能推出所有参数不更新；delta>0也可能含decay/momentum/roundoff。grad norm与distinct-ranking指标能改善证据但不自动证明生产学习质量。
- gradient_norm_max是整个update各optimizer batch的preclip最大值，不是累计norm或平均；parameter_probe_delta_l1是各step抽样绝对差之和，不是完整权重L1且非最终与起点net delta。报告字段解释应保持这些语义。
- rank time包括prompt tokenization、generate、score及parse，不包括HTTP retrieve或完整actor rollout；update time包含训练梯度/optimizer及diagnostics，CPU process_time累加本进程线程时间，不能和wall直接等同。峰值RSS为driver全生命周期，不是ranker独占或全机job峰值；retriever进程/Ray worker/GPU峰值须额外采集。
- `ranker.cost_metrics`与修复后的probe.save RSS都正确区分Darwin/Linux。
- probe --questions/--repeats/--threads已做正值校验，但空train文件仍可能complete却无生成，建议reject empty data。默认3题×3重复有效，但小样本只反映该样本，并不能放行500-update。
- probe retrieval/schema、model load或generate异常时没有保存明确failure status；部分已生成报告仍可能停在in_progress或根本没有文件。建议try/finally写失败阶段/exception（不含凭据），保留已有证据。contexts overflow路径已写状态并exit2，不截断候选，符合用户约束。
- 共同cd86c9d已提供batch_answer_metrics和aggregate_answer_metrics；实际检查_validate的两条生成路径都调用collect_qa，并在返回metrics及JSON添加F1/EM/answer_format/format_rate/token/search_calls。因此初审R-CS-03实现缺口已关闭（此前增量复审只聚焦RewardManager而漏看独立QA metrics，该判断已纠正）。指标真实七集运行仍待验证。checkpoint未提供joint optimizer/RNG/loader exact resume。

## 后续真实验收门槛

修正probe上下文后跑冻结3B base+真实50候选合法率/长度/CPU/RSS能力；按共同Slurm/UUID门禁通过8卡基础验收；实际actor→retriever→ranker→双更新的2 outer update，记录有效奖励差异、实质梯度及参数变化而非仅events；5步save/load；再500实际更新与新GRPO公平比较、最终51,713题F1/EM/format及累计成本。真实失败若需要换ranker/截断passage/改预算，仍按计划由用户选方案。

## 2026-10-03 最终增量复审：9db1568、30d3aeb

**提交后复审通过（静态/受控CPU范围）；此前probe空train和异常报告两个P2均已关闭。真实联合效果仍未验证。** 当前git worktree clean，本轮没有编辑repo。

9db1568注册sys.excepthook，在tokenizer/config/data/retrieval/model/generate未捕获异常时写status=failed、error_type/error_message/RSS，并保留已记录query/generation。读取data后空frame明确ValueError。独立用实际tiny本地tokenizer/config、真正空parquet在独立子进程调用完整probe脚本：exit1，输出JSON status=failed、error_type=ValueError、message=Cannot probe an empty training dataset；另用不存在model目录触发真实AutoTokenizer失败：exit1且报告OSError/failed。两例真实Darwin RSS分别约255.30/244.41MiB，换算正确。无需mock rank generation或伪造reward。context overflow原先显式status+SystemExit(2)路径不经excepthook覆盖，保留用户选择状态。

hook不是kill/OOM崩溃保险：SIGKILL或进程外终止不能触发Python异常hook，写盘自身失败也不保证报告；这属于正常边界，无新增要求阻止probe。

30d3aeb只改变dp_actor metrics reduction：paper_objective各micro已经以整个mini分母归一，pg/entropy/reference-KL现在累加到mini一次记录；clipfrac/PPO-KL用micro policy-weight mass与跨rank mini mass/world比值加权。所有新增weight mass、micro_mass及metric值detach/取item，新增all_reduce仅操作mask weight的副本sum；policy_loss、loss.backward、optimizer step、优势/ratio/clip/entropy/KL数学未更改。CoSearch属于policy_weights的序列等权token PPO分支，不传GiGPO step_ids，core ranker optimizer完全不触及。

独立执行新原始logging回归：**1 passed**（原测试GiGPO）。额外将该actual actor测试夹具的method与attach_policy_weights均改为CoSearch在内存运行，同样比较micro1/micro2：pg、entropy、reference-KL、clipfrac、PPO-KL一致，实际SGD参数更新一致，通过。测试用实际DataParallelPPOActor.update_policy但CPU替代cuda和简单model forward，范围如实限制，不称FSDP/CUDA通过。

跨rank额外collective发生于所有rank共同的非空mini分支；局部空rank仍参与，global空mini继续先共同跳过。报告权重归约修复未修改有效policy梯度，且未发现CoSearch新增数学缺陷。现实资产/环境已准备进度来自owner，本审查没有独立重复72GiB SHA校验/Slurm硬件验收，不将其记录为本reviewer真实验收证据。

## 2026-10-03 固定Linux依赖兼容增量：cda530e

实际读取 `work/linux-cpu-gate-retry.log`，141 passed/1 failed 的异常来自TensorDict0.5 `get('step_ids')`缺省KeyError。dp_actor只在GiGPO select step_ids，其他paper_objective故意缺此字段，以调用token-ratio路径。cda530e唯一改为`data.get('step_ids', None)`，正确恢复optional契约；GiGPO select仍强制required step_ids，未默认伪造action信息，也没有更改ratio/梯度。扫描verl/search_r1的batch/micro/data/tensor get，没有找到另一处同类TensorDict optional访问。

独立阅读 `outputs/evidence/linux-cpu-gate-fixed.log` 结尾确认固定Linux环境 **142 passed**。该运行由owner执行，本reviewer核实日志；本reviewer另在本地实际运行paper_metric_logging和paper_ddp，**2 passed**。因此兼容修复复审通过，不表示GPU/真实效果已完成。

### work/retrieval_compare.py 只读静态审查（尚未运行结果）

该脚本从resolved训练文件取14条原题query，无gold或最终测试。直接复用当前Encoder，CPU/GPU均E5 FP16、mean pool、256 query tokens、同query前缀和归一化；FAISS沿用同Flat index，GPU cloner useFloat16+8卡shard，与原DenseRetriever配置一致。CPU-cq/CPU-gq/GPU-gq三路ID对照可区分encoder与FAISS排序差异。语法检查通过，本reviewer未运行真实index/模型。

建议在作为验收证据前补齐：

- `score_max_absolute_delta`当前cs-gs按rank位置比较，不同ID时是“第k名检索分数差”，不是“同文档FP16数值误差”。应标明语义并保存中间iscores/iids，另对相同ID对齐计算encoder-only/FAISS-only score delta；只看未对齐最大差容易错误归因。
- 断言实际lenqueries=14、三路ID全部>=0且<ntotal、iscores finite（当前只检查cs/gs）。防止短输入/小index返-1导致虚假overlap；记录实际数据源和query token长度。
- head14可能全属NQ，不能称覆盖NQ+Hotpot分布。该小样本能进行能力对照，不能证明全部训练subquery等价。计划要求固定真实query，必要时补训练Hotpot或长短分层，不使用最终题选择。
- index IDs只有在同一资产hash/index→corpus映射下才代表同内容；本轮外部manifest已经冻结，建议结果JSON引用实际模型/index/corpus manifest SHA以使证据可移植。

top3/top50 exact order不一致并不自动意味着脚本错误，FP16 encoder/index可能改变邻近分数和排序；应保留每query对照，报告幅度并在用户既定协议下处理差异，不能发现差异后换mock/不同retriever或隐瞒。真实检索结果尚未运行，不能提前签等价通过。

## 2026-10-03 实际更新预算与检索诊断最后增量：7617ee9

只读检查0ec6412（共同已review767cc9b的cherry-pick）与7617ee9。新guard在actor/update与after_actor之后、周期validation/save/global_steps递增之前检查actor/optimizer_steps；CoSearch现在和GRPO/GiGPO一样，0或缺省0直接RuntimeError，actual_outer_updates和global_steps都不前进，亦不保存虚假的新checkpoint。critic_warmup导致本轮未调用actor同样停止而不消费预算。它是明确fail-fast，**不是自动重采样或跳过本轮继续训练**；失败轮仍有rollout/检索等实际成本，预算不递增不能当成本为0。

真正执行多于1个actor optimizer step的outer iteration仍仅actual_outer_updates+1，符合500实际outer update预算。actor KL/advantage/loss/梯度/ranker reward均未修改。位置上after_actor仍先于guard，因此本轮失败不提供事务rollback；不承诺已经发生的辅助更新可撤销。这不改变“零actor/warmup不能消费budget”的验收结论。

独立本地实际运行 `tests/baselines/test_actual_update_budget.py`：**9 passed**（GRPO/GiGPO/CoSearch × 零更新、多step单outer计数、warmup三情形）。测试提取并执行真实driver尾部，运输/actor返回为夹具，不称GPU整链通过。owner报告远端固定Linux `logs/linux-cpu-gate-budget.log` 为151 passed；该日志本轮尚未落本reviewer本地，不冒称独立运行或全文核实151项。

再次只读检查work/retrieval_compare.py：现已明确first14非代表性、保存actual data_sources、严格len14、三路IDs范围合法与iscores有限、保存中间GPU encoder/CPU index IDs及scores；原cs-gs字段更名sorted_rank_score_max_absolute_delta，另三路对shared document ID对齐计算score差。这些关闭前述主要诊断建议，比较分解保持正确，没有gold或最终题。shared-ID score仅覆盖top50交集，不是全部候选全集数值误差；JSON计数已明确该范围。token长度/manifest SHA引用可作为后续增强，不改变此能力对照的静态条件通过结论。

**最终状态：7617ee9预算guard与更新后的检索对照静态复审通过。真正CPU/GPU retrieval equivalence、base合法rank生成/非零联合学习、500更新与七集合效果仍必须实际验证，本报告不提前签字。**
