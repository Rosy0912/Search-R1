Current: job3360 on h100-10/10.33.0.18 completed104/200, actual105 started;832 direct rank records,2080 successful Adam/rank,0 skips,scheduler104. Rolling104 SAVED. Outer104 810.678455s (13.511308min), leaf reward0.476953119040, pure intra variance0.063964632896,mixed100/512. Dev100 unchanged:F1 53.226531%,EM44.140625%. HF100 saved;cold reload pending. Final200,51713,HF/tokenizer,costs,cleanup incomplete.

# Tree-GRPO BranPO sampling：执行准备与证据

第9轮历史记录：3360当时已完成9/200轮，第10轮实际开始（measured_task pid4011013 epoch0 step10）。8rank各累计180次成功Adam、180机会、0skip，scheduler9；72份直接rank记录及9份credited attempt一致。第9轮719.647897秒（11.994分钟），纯intra总体方差0.080728899512，mixed题26.3671875%，2560叶平均训练score/reward均0.171484381（0–1范围），训练叶reward不混作dev评分。rolling global_step_8/9已保存。周期评分未更新，仍第5轮dev512 F1 24.475422%、EM 18.1640625%；第10轮按原计划进行dev512。完整200/51713/HF冷重载及自身清理仍待完成。

第1轮记录：8rank各20次成功Adam、20机会、0skip，scheduler0→1；首轮LR0所以参数delta为0，随后第2轮使用非零warmup LR。完整第一轮730.257855秒（12.171分钟），含rolling存档；单独采样322.530420秒，生成618857 token、1720检索调用。

每轮方差记录已实际落日志并独立复算：第一轮按树平均纯intra总体方差0.082030979444，按3/2叶mass加权0.089843450257；raw树内奖励方差0.030924479167，实际总优势树内方差0.418341560800。五叶混奖140/512题=27.34375%，混奖137/1024树=13.37890625%；不可用此不同轮次/题目的数值直接宣称优于旧run。该段为第1轮历史记录；当前评分和完整轮数见本文顶部。

独审JOB3360_FIRST_ROLLOUT_REVIEW.json确认512题/2560唯一叶、1536续写全部保留与前缀继承、原权重、finite安全诊断及8rank真实成功返回。treeaudit中algorithm_source_commit/base_commit两项沿用旧config标签，不能作为当前源码身份；当前执行版本由部署800文件SHA、入口与direct-rank绑定的8b6f99d及config c9415c0096853f78d870e205fc956f20045cde985836de5225f609eaf091d4b3证明，未热改运行实例。

目标保持新鲜官方初始化的200个完整outer及最终51713题。历史3341已经完成200轮，其最终F1为52.957946%、EM为45.984569%；它属于旧实验，不计入本次新实验更新数。

本次采样保持两根、叶配额[3,2]，总计五叶。先在末次模型动作之前续写；根0首次续写与原叶同奖时，第二个续写槽回退一个原始动作状态。原始叶与所有新增叶都参与训练，不额外采候选、不按奖励丢弃。保留原Tree树间/树内优势、叶权重、PPO及行为概率校正。

五次有限采样不能保证混合奖励。每轮同时记录原始奖励方差、纯树内优势方差、实际总优势方差、混奖比例和零优势比例；不能单凭标准化优势方差断言探索改善。

最终执行源码为8b6f99d1b3353145200eb3fa7c16a579a79a987c，实际git远端SHA一致。独立审查已绑定源码与28个运行文件，18项相关CPU测试通过。历史测试中缺native stopping参数的旧fixture问题已在未修改基线复现，没有降低运行检查。

H10018/h100-10/10.33.0.18只读探测：epoch1791656672.9163759，队列空、8卡0MiB/0%SM、可用空间5780740136960字节。提交与分配时重新检查；这些观察不代表GPU租约。

代码包已部署并逐文件SHA通过，manifest SHA为e2503977c8746edbc83b722d8224548f060909df9fc25d91a9cb88a947a79a21。模型、数据和索引沿用H10018本机资产，未传到Mac。北京时间2026-10-11 02:32:33单次提交正式作业3360，run为tree-branpo-fresh200-3360。早期实测1791657165.856704→1791657205.8087997确认从GPU预检推进到固定split核验、guard及retriever加载；后续完整更新事实见本文顶部。五轮训练与第六轮接续已实际通过，当前保持同实例连续推进200轮。

原30分钟automation实际状态为PAUSED，未擅自恢复；当前新goal执行继续。历史3341监控指令不能覆盖本次新授权，活动身份以本STATUS及当前真实job为准。

证据：PLAN.md、MATH_REVIEW.md、CODE_REVIEW.md、INDEPENDENT_RELEASE_BINDING.json、RESOURCE_PROBE_20261011.json及STATUS.json。执行前接受不代表训练、评分或最终HF重载验收。

成本观察截至epoch1791662068：已完成五轮实际3153412生成token、12690908 prefill token、8672检索调用、12800逻辑叶及136 padding rows；全卡967个5秒样本最高50208768000字节（46.76GiB），外来显存/SM最大0。Slurm运行4914秒、预约8GPU累计10.92 GPU小时，独占allocated192 CPU累计262.08 core小时（task请求32 CPU为43.68 core小时）；这是预约量而非实测CPU时间，均非最终成本。第6轮在两次根新只读epoch1791662017/2139间采样日志继续增长，未终态。

六轮实际防护审计（SAFETY_SUMMARY-3360.json）：2880条rank事件，均allowed，invalid/overflow记录0；固定每rank32行监测的采样KL最高0.001682260948、negative-tail fraction最高0.000191350932，分别低于0.2/0.01阈值。分母为actual_behavior_mu；不声称原始sampler/actor一致，也不把固定监测样本当全token总体KL。六轮8rank各20梯度范数有限，普通global gradient clip1生效；裁剪前范数>1不等于策略刹车。第7轮实际采样audit继续新增，未退出、未重启。
