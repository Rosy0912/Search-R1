# OPSD 五次真实迭代前独立审查

日期：2026-10-03（北京时间）。固定本地代码：`98c9661b030d8c2dcded1a298089ff3c83ab9292`。公共source：`297e1c140677dc621302f1d49fefdb31023fd2b7`，合入OPSD为`f60897e6212bed0c9c54f61bc17e67746948a094`，观测提交`ea0a78b`。本轮只读当前OPSD checkout/PLAN/既有报告，仅写本文件；未启动远端、未修改实现或他人checkout。

**当前结论：未发现SP1下OPSD padded forward与保留的use_remove_padding配置之间的静态启动冲突。五迭代与checkpoint路径已核对，允许按已授权协议继续准备和实际验收；本报告尚不能认定真实五iter验收通过。Linux固定环境、真实资产、Slurm授权与通信/检索门禁必须先通过。最终必须独立读取真实日志、checkpoint重载及资源证据后重新给结论。**

## 一、精确五iter协议

五iter接受性运行应使用新的run-id及独立输出路径，沿用OPSD launcher并明确：

```text
trainer.total_training_steps=6
trainer.smoke_only=true
trainer.val_before_train=false
actor_rollout_ref.actor.ppo_micro_batch_size=8
retriever.url=http://127.0.0.1:18084/retrieve
```

`run.sh opsd`还需OPSD_TEACHER为不可变同初始Qwen2.5-3B base目录、OPSD_GLOBAL_MICROBATCH=8，8卡实际allocation与Ray resource pool一致。保持train512/mini256、n_agent5/n1、SP1、ppo_epochs1、lr1e-6/warmup.285、已批准的所有搜索长度/预算、CPU教师和无额外KL/entropy；不得为了过验收静默缩模型、数据、问题batch、轮次或长度。

`fit()`从global_steps=1开始，每个完整outer循环结束后加1，再判断>=total_training_steps。因此total=6实际更新日志是1、2、3、4、5；退出时counter=6，**不是第6次outer update**。total=5只运行4次，不满足本轮要求。main限制smoke_only总计数≤6；smoke只跳过终轮51,713题全评估，默认test_freq50在前5步不触发周期评估；若保留val_before_train=true，初始评估仍会执行。正式运行必须恢复total501、smoke_only=false，最终七集全评估路径仍保留。

每outer默认512问题×5轨迹=2560 trajectories；8rank各320轨迹，mini256全局对应localmini32；micro8全局对应localmicro1。每rank通常**10次minibatch optimizer调用/outer**，不是每outer一次，也不是“五次optimizer调用就通过”。共五outer为12800训练轨迹（另计padding生成）。全局监督为空的mini会协调跳过optimizer，要读取实际计数而非只按配置推算。

warmup步数`int(.285*6)=1`，LambdaLR初始化current_step0时lr=0，worker只在outer完成且有optimizer调用时推进scheduler。因此首outer的10次optimizer调用可能全部使用lr0，即使梯度非零也没有参数变化；日志返回的actor/lr是scheduler推进后的值，不能直接当该outer真正使用的lr。不要为此改warmup，也不要以首步delta0误判失败；必须在后续outer/最终checkpoint证明非零学生权重变化。

本轮进行了CPU-only resolved Hydra与validate_config检查，world8/globalmicro8/SP1/use_remove_padding=true通过。另执行实际fit终轮AST CPU替身五次，事件update1–5、save5后无terminal eval。这仅确认控制流，**不是五次真实训练、GPU或3B证据**。

## 二、真实checkpoint路径及重载契约

`ray_trainer.py:_save_checkpoint`实际HF actor路径：

```text
${trainer.default_local_dir}/actor/global_step_5
```

默认展开为：

```text
${SEARCH_R1_BASELINE_OUTPUT}/checkpoints/opsd-qwen2.5-3b/actor/global_step_5
```

算法metadata另在`${trainer.default_local_dir}/algorithm/global_step_5/state.json`。保存触发要求save_freq>0；默认100时acceptance_step5和终轮条件同时触发也只保存一次。不得把日志的“Saving actor”或仅存在目录当作保存成功。

FSDP以FullStateDictConfig(offload_to_cpu=True,rank0_only=True)聚合actor；rank0用save_pretrained写HF weights/config、tokenizer，全rank barrier完成后返回。需要记录实际文件清单/大小/SHA、shard索引完整性及config/tokenizer；必须从上述实际路径重新加载并做真实推理/评分，验证输出finite及权重区别于初始base。只检查state.json不证明HF导出成功。

HF目录不含完整actor optimizer/scheduler/RNG/dataloader精确恢复状态，state明确resume_supported=false。重载是验证产物可用，不是等价resume；断连只reattach同一tmux进程，不新起第二个训练实例。checkpoint保存本身可能有额外GPU全参数加载及CPU state_dict峰值，资源观测须覆盖此阶段。

### 已证实的共享路径缺陷，非OPSD五iter启动依赖

共同holdout-only main两处regex要求`/global_step_N/actor$`，但实际trainer保存的是`/actor/global_step_N`。对自然产生的GRPO checkpoint运行公共teacher选择评估会在加载前被拒绝。共同REPORT也按反向布局给示例；已通知执行负责人协调公共修复，不能手工移动/拼假路径后称协议一致。

这是未来GRPO→TOP-D teacher selection的真实阻断；**OPSD使用初始base frozen teacher、五步HF重载走实际路径，不依赖该选择器，因此不应等待此修复才运行OPSD五步**。此前组合审查未发现该路径不匹配，本次实际路径交叉核对后补充纠正。

## 三、FSDP padded forward与remove-padding边界

- model.use_remove_padding=true先检查model_type是否支持；registry包含qwen2，Qwen2.5-3B属于该架构。只有remove-padding且SP>1才apply_monkey_patch；OPSD config/actor均拒绝SP>1，因此SP1不会被改成只能处理展平无pad输入的SP forward。
- DistillationActor继承父actor仅用于logprob/_optimizer_step；update_policy调用自己的opsd_loss，显式使用B×L input_ids/attention_mask/position_ids的**padded student forward**。model flag true不强制该loss走父actor的unpad路径。
- old_log_prob仍经继承compute_log_prob，其remove-padding路径用于评分；这与loss的padded路径可并存。真实production必须比较有效actor token logprob的finite/shape/对齐，保留已授权生成窗口与评分历史差异并报告context_mismatches，不能拿该差异重新要求改算法。
- FSDP student由flash_attention_2加载，mixed precision默认bf16参数计算、fp32 reduction/buffer，梯度checkpoint use_reentrant=false；teacher CPU FP32/eager且eval/no-grad。每rank一份teacher不归actor optimizer，纯蒸馏无Ref role。OPSD padded激活/全logits与FP32 divergence图的峰值不可用普通token-PPO经验估算。
- worker在update前load参数/梯度/optimizer，update后offload；SP1 manager不需要序列切分。CPU审查没有证明生产Torch2.4/FSDP非重入checkpoint/Flash backward、offload、NCCL、vLLM权重同步可用，这些必须由真实连续五outer证明。

## 四、资源峰值观察必须覆盖的阶段

必须采集加载teacher/学生、vLLM初始化/rollout、CPUteacher长context前向、student backward/optimizer、offload及step5保存/重载。作业获配CPU/RAM以实际Slurm/cgroup为准，不能引用主机2TB当256GB作业的预算。

八份FP32 3B teacher权重约96GB十进制，另有student/offloaded optimizer、loading copies、eager attention/hidden等；这是估算，不是测量或OOM断言。student原始logits预算cap4GiB检查在teacher forward之后，而且仅估logits；chunk16不限制整张student logits或累计autograd图，不能当总峰值上界。

当前log_gpu_memory_usage仅打印默认rank0的当时memory_allocated/reserved，默认DEBUG还可能不输出；它不调用max_memory_allocated/reserved，不是8rank峰值。`teacher_worker/peak_rss_bytes_per_rank`为整个actor进程high-water，driver reduce_metrics取rank均值，既不是teacher独立RAM也不是最坏rank/cgroup峰值。需要独立、按PID/UUID绑定的各rank峰值/整作业采样，并记录采样周期；GPU nvidia-smi进程占用含vLLM/其他上下文，不等于PyTorch allocator。外部周期采样若没有框架峰值只能称采样高水位，不能称严格瞬时峰值。

记录teacher加载与forward CPU/wall区间、模型tensor bytes、worker RSS、retrieval CPU/RAM/吞吐、GPU占用、作业cgroup memory current/peak及OOM事件、墙钟/GPU小时。timer排除CPU→CUDA传输；process_time包含区间内全部worker CPU线程，不是exclusive attribution。per_rank平均CPU×8可算rank和，累计counter不跨step重复求和，wall均值不等于端到端墙钟。

## 五、真实启动门禁及五步接受证据

启动前必须有以下实际记录；缺任何已要求门禁不能用本报告替代：

1. 远端唯一checkout实际HEAD=98c9661或明示review过的后续commit，完整upstream/source存在；resolved Hydra与Linux env freeze，Torch2.4.0/vLLM.6.3/Transformers4.46.3/Ray2.40.0及锁定依赖、CUDA/Flash/NCCL实际import与通信通过。确认bootstrap/tmux旧任务状态，避免重复启动。
2. 模型/原parquet/Wiki18 corpus/index/E5固定revision及实际SHA，student初始与teacher权重同源一致、tokenizer vocab/BOS/EOS兼容且teacher不可变；实际98/2 split content_identity/产物SHA与公共任务一致，不覆盖原train。
3. 本任务Slurm独占allocation/JobID、获配CPU/RAM、CUDA_VISIBLE_DEVICES实际8个UUID，启动前双15%快照/既有进程记录/租约检查与真实8卡通信。保留已有低占用进程，不能绕scheduler或借别人allocation。
4. 独立CPU检索18084健康、真实E5/index schema和top3/top50与原检索排序/内容/误差对照及吞吐/RAM记录；Ray16384及独立目录/端口，不能拿mock检索通过。
5. 连续真实日志中step1–5每次完整生成、teacher scoring、backward/optimizer与outer结束，真实3B/8rank而非tiny/stub；各step finite OPSD loss/old logprob/gradient norm，实际optimizer_steps/skipped_empty计数、globalmask/response/actionmask对齐、raw生成tokens/实际retrieval_calls/trajectories及context_mismatches。普通非空配置预计每rank optimizer_steps10/outer，异常要解释，scheduler推进和teacher冻结需核对。
6. 后续学生非零参数变化证据及step5HF导出/全部文件hash、真实HF重新加载与至少真实推理/评分通过；teacher初末hash或等价不可变证明，teacher不进入optimizer/无梯度。仅config不证明更新发生。
7. 全过程资源/成本观测和无OOM/非finite/NCCL hang/worker重启，step5结束的日志exit状态/无重复实例；提交完整日志、artifact清单、resolved配置及成本口径给独立subagent重新审查。验收报告需区别“五次真实迭代可运行”与“完整500步/51,713题效果验证”。

目前此审查者没有收到任何上述真实五iter日志或step5产物；因此**实际接受状态仍PENDING**。此前139passed/1skipped、32/20项CPU回归及tiny数学等价支持代码基础，不抵扣本轮任何一次真实iter。五步短跑通过也不能声称OPSD优于GRPO；正式效果须完成用户授权完整协议或明确报告尚未完成。
