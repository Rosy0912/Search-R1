# GiGPO Instruct300：完整状态恢复75与有限反传

用户最新目标明确授权意外停止后从checkpoint RESUME，不再等待额外重启确认。原3185八rank完成83，84梯度元素非有限而拒绝；GPU空、cleanup{}、cgroup已释放。恢复唯一来源为本新Instruct实验global_step_75/training_state，manifest SHA65f5eef374d0eb5378aa50d56b9c883755e961b154526a26c05eefcd127968a9。旧2992/3026/3130和3185全部代码、日志、checkpoint原位保留。新源码、辅助程序和续训目录独立版本化。

## 保持数学含义的数值表示

已读实际Torch2.4 CPU原目标：4token同动作delta各25，negative advantage -1，FP64前向有限而float32反传Inf。实际84精确输入未保存，不能称已重演84。禁止改joint ratio、截断优势/概率比、改变sequence_mean、PG/KL相对权重或跳坏梯度。

在每个完整全局mini128更新前进行no-grad范围探针；探针前后恢复CPU/CUDA/Python/NumPy RNG，训练态和参数不变，不能多推进数据/scheduler/Adam。探针逐微批计算原GiGPO joint log-ratio，仅对有正policy weight、非零优势且PPO未被裁剪封住的导数分支取上界。全8rank MAX归约后得到统一正标量s=exp(min(0,40-max_active_logratio))；必须detach。同一完整mini全部微批及rank对完整PG+KL(+原entropy项)施以同s，不逐micro独立缩放。

真实反传累积h=s*g。FSDP sharded/replicated归约保持原计数及DP平均；以FP64计算||h||，有限检查不能省略。保持原epsilon=1e-6和C=1的等价裁剪：h*min(1/s, C/(||h||+s*epsilon))。返回并审计未缩放的原范数||h||/s；该值也必须有限，不采用clip(h)/s，不先把h/s写回低精度梯度造成溢出。Adam收到原数学裁剪向量；若真实损失/输入/缩放/梯度/原范数仍非有限则拒绝且保存失败mini详细证据。前向原目标及原诊断数值保持不变。

这是原数学损失/裁剪的数值表示扩展，旧程序拒绝的低精度中间量范围可以被覆盖，必须公开记录；不宣称旧拒绝域完全相同。默认不开启，只有新GiGPO续训显式启用。无梯度缩放prepass的旧配置路径保持。

## 完整恢复与进度

从官方Instruct及固定reference构建相同36层3B架构，然后在第一次训练取数前加载75所有8rank模型、_local_shard/buffers、Adam、scheduler、train/gen RNG及driver的sampler排列/cursor38400、KL controller、CPU/CUDA RNG和GiGPO算法状态。固定reference仍为官方固定revision，不能恢复成训练actor。恢复前全文件SHA、源revision/数据manifest/实际参数配置/layout/状态字段逐项审计。旧来源code/config身份保留，迁移白名单仅数值表示、resume控制、新目录与审计身份；不宽松忽略hyperparameters。

scheduler直接加载75/LR1e-6/_step_count76，不能重建Adam/重走57warmup，第一次新增outer是logical76，旧76–83分支不计入恢复后连续轨迹。选定轨迹是原1–75加恢复后76–300，共300；额外历史回滚8轮单独记录算力，不能83+225冒充轨迹进度。原进程已退出，新的job/进程明确记录；不承诺vLLM引擎重建后的bitwise exact sample replay。

真实恢复首5个outer为76–80，计入225个后续更新；80保存完整HF+状态，在81取数前同一续训进程暂停独审，通过后原位继续300。80是额外恢复门禁保存，不替代原5/25/50/75。后续100、125…300每25保存全状态；100/150/200/250/300同512周期验证，额外80阶段64，300最终51713七集不缩水。保留原init512/step50与first5 stage64证据；最终汇总验证事件要跨原与续训原始日志按来源/step身份合并，不能依赖Ray转发漏行或精确固定事件数。

## 验收与资源

先CPU数学/梯度/微批/全局归约/clip epsilon/失效拒绝/RNG探针测试，再同h100-8真实8rank Torch2.4 FSDP小Qwen2 decoder native（高joint ratio、一个空mask rank、联合PG+KL、正常有限梯度与高精度oracle、全状态恢复及next-update replay）。CPU不能替代GPU。新源码/环境/输入/工具全SHA部署与独立审查通过才提交唯一正式resume75作业，不并行重复提交。

原通用512x5、mini128/actor globalmicro16/oldref32、epochs1/20Adam/outer、57warmup、LR1e-6、entropy0/KL.001/lowvar/clip.2/grad1、所有长度及共同instruction/template/data均保持。线上仅h100-8 GPU E5+完整八GPU FAISS/top3/loopback18081/Ray16381。Slurm独占/双15/UUID-OFD/pidfd-cgroup/60秒清理全部保留。不杀他人、不降micro/预算、不跳badgrad。

数值修复、恢复和部署均须独立接受；native不算训练轮数，恢复后的76–80不能从Base重新训练替代。自动监督继续每小时；用户已要求每小时问询/状态更新，报告真实logical完成、恢复新增N/225、checkpoint及评估。失败再按最新授权从最新完整检查点恢复，先诊断真实terminal状态，不因观测timeout重复启动。
