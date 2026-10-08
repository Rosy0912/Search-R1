from pathlib import Path
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia')
out=root/'outputs/tree-bapo-e1-v1';out.mkdir(exist_ok=True)
archive=root/'work/search-e1-audit/search-e1-reproduction-20261003-233328'
common='''actor_rollout_ref.model.path="$MODEL" \\
  data.train_files="$DATA/train.parquet" data.val_files="$DATA/validation-512.parquet" \\
  data.final_val_files="$DATA/test.parquet" \\
  data.train_batch_size=512 data.val_batch_size=32 \\
  data.max_prompt_length=4096 data.max_response_length=500 \\
  data.max_start_length=2048 data.max_obs_length=500 \\
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \\
  actor_rollout_ref.actor.ppo_micro_batch_size=16 \\
  actor_rollout_ref.actor.ppo_epochs=1 \\
  actor_rollout_ref.actor.optim.lr=1e-6 \\
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \\
  actor_rollout_ref.actor.entropy_coeff=0 \\
  actor_rollout_ref.actor.use_kl_loss=true \\
  actor_rollout_ref.actor.kl_loss_coef=0.001 \\
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \\
  actor_rollout_ref.actor.clip_ratio=0.2 actor_rollout_ref.actor.grad_clip=1.0 \\
  actor_rollout_ref.actor.state_masking=true \\
  actor_rollout_ref.actor.fsdp_config.param_offload=true \\
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \\
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \\
  actor_rollout_ref.ref.fsdp_config.param_offload=true \\
  actor_rollout_ref.rollout.temperature=1 actor_rollout_ref.rollout.top_p=1 \\
  actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.n=1 \\
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \\
  actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \\
  actor_rollout_ref.ref.log_prob_micro_batch_size=32 \\
  trainer.total_epochs=20 trainer.total_training_steps=201 \\
  trainer.save_freq=25 trainer.test_freq=50 trainer.acceptance_step=5 \\
  trainer.smoke_only=false trainer.val_before_train=false \\
  trainer.logger=[console] trainer.default_hdfs_dir=null \\
  trainer.default_local_dir="$OUTPUT/checkpoints" \\
  max_turns=4 force_final_answer=false retriever.topk=3 \\
  retriever.url="$RETRIEVER_URL" baselines.seed=42 \\
  +baselines.loss_reduction=sequence_mean'''
commands={}
for method,node,port,extra in [('tree_grpo','h100-9',18082,'actor_rollout_ref.rollout.n_agent=2 baselines.tree.roots=2 baselines.tree.rounds=1 baselines.tree.expansions=2 baselines.tree.leaves=3'),('bapo','h100-5',18083,'actor_rollout_ref.rollout.n_agent=5 baselines.bapo.stage_validation_files="$DATA/stage-64.parquet" baselines.bapo.stage_validation_size=64 baselines.bapo.stage_validation_freq=1 baselines.bapo.patience=5 baselines.bapo.alpha=0.05 baselines.bapo.bonus=0.5 baselines.bapo.resample_attempts=2')]:
 name=method.replace('_','-');home=f'/dataset/shaoshitong/search-r1-rl/{name}-e1-instruct-v1'
 command=f'''# 仅供审核：尚未部署。以下是八卡独立作业内的算法命令，不是资源提交器。
# 新loss_reduction开关必须先实现并通过真实归约测试；当前不能直接提交此命令。
# 已审核的同节点GPU检索/Ray/资源监督包装应在外层启动、清理服务。
NEW_ROOT={home}
MODEL="$NEW_ROOT/assets/Qwen2.5-3B-Instruct"
DATA="$NEW_ROOT/data"
OUTPUT="$NEW_ROOT/runs/APPROVED_UNIQUE_RUN_ID/output"
RETRIEVER_URL=http://127.0.0.1:{port}/retrieve
cd "$NEW_ROOT/repo"
bash reproduction/baselines/run.sh {method} \\
  {common} \\
  trainer.experiment_name={name}-e1-instruct-v1 \\
  baselines.local_runtime_only=true baselines.local_runtime_host={node} \\
  {extra}
'''
 commands[name]=command
text='''# Tree-GRPO / BAPO：Search-E1 压缩包配置迁移 v1（待用户审核）

日期：2026-10-04（Asia/Shanghai）。这份文档和命令都是草案。没有修改两个算法的生产源码、没有提交作业、没有向两个会话发送启动指令。新ROOT和模型/数据路径尚未部署，不表示资产已经存在。用户批准后才实施和运行。

## 目标与推荐选择

从官方 Qwen2.5-3B-Instruct 相同初始权重分别新开 Tree-GRPO 和 BAPO 实验。采用压缩包 GRPO 阶段的通用优化设置，保留 Tree 树采样/优势与 BAPO 边界奖励/阶段控制。不要启动 Search-E1 的 OPSD、轨迹挖掘或循环训练，不把两算法改成普通GRPO，不承诺效果一定改善。

推荐采用压缩包200真实outer的训练长度，配置total_training_steps=201，warmup=floor(201×0.285)=57。另一个可选方案是300或500真实outer但保持warmup57，这时必须显式设置独立warmup horizon并核实最终resolved配置；不能沿用0.285得到85/142再声称相同早期schedule。200是当前待审核默认，不是用户已经批准的训练长度。GiGPO的300轮续训授权不自动适用于本实验。

## 已核对的压缩包设置与新实验

| 项目 | 新方案 | 依据/说明 |
|---|---|---|
| 初始模型 | Qwen2.5-3B-Instruct | run_phase.sh grpo1；全量模型权重/词表/config/模板内容SHA须一致 |
| 训练方式 | 全参数PPO，fresh Adam、fresh scheduler | 不是从旧base checkpoint续训，不使用OPSD LoRA |
| 每outer题数 | 512 | scripts/grpo.sh |
| mini-batch | 全局128 | 八卡每rank16 |
| actor micro-batch | 全局16，每卡2 | run_phase.sh当前每卡2，不照搬其4卡全局8 |
| old/ref评分micro | 全局32，每卡4 | run_phase.sh SCORE_PER_GPU=4 |
| PPO epochs | 1 | 不借更多内循环隐式增预算 |
| 最大lr | 1e-6 | constant warmup，57外层调度步；精确打印实际lr，不使用三位小数推断 |
| 熵系数 | 0 | 无任务信号时不再由熵项推动策略 |
| reference KL | 0.001，low_var_kl | ref是同一Instruct初始权重；保持已验证数值稳定实现 |
| PPO clip / grad clip | 0.2 / 1.0 | 保留原算法数学和数值拒绝门禁 |
| loss measure | 每轨迹actor token均值，再对轨迹均值 | PG、KL用同一mask/权重，空序列零贡献且不除零；微批/rank划分不改变目标 |
| rollout | T1、top_p1、top_k禁用、TP1、n=1 | GPU memory utilization .4，vLLM；同节点GPU检索 |
| FSDP offload | param/grad/optimizer及ref param均true | 压缩包设置，成本实测，不承诺H100上的速度 |
| 长度 | prompt4096/start2048/action500/obs500/最多4轮 | 沿用Search-E1 GRPO预算；不使用其后续独立rollout8192 |
| 提示词 | 压缩包INSTRUCTION及官方Instruct chat template | prompt应重建到新数据文件；不添加IDK示例/新拒答指令 |
| 检索 | E5-base-v2/Wiki2018/top3 | 保留两个节点已验证GPU FAISS FP16分片，区别于压缩包float32索引，单列部署差异 |
| 周期开发集 | NQ256＋Hotpot256，共512，seed42 | 相同题ID、顺序、问题文本与评分；每50轮完整评估，batch32 |
| 保存 | 每25轮，额外第5轮与末轮200 | 用户可审核的新保存配置，压缩包原为每50轮 |
| 初始评估 | 训练前额外评估同512题并单列成本 | 不计入outer；算法trainer.val_before_train=false，额外评估是方案新增诊断 |
| 末轮评估 | 七集完整51713题 | 保留原完整效果预算，与512题阶段成绩分别报告 |

## 数据与BAPO阶段控制的例外（必须明确批准）

按包内prepare.py的source顺序、seed42逐source打乱，重建NQ/Hotpot512题开发划分；FlashRAG原始版本与实际问题/答案内容要冻结，不能因为题数一致就声称是相同数据。

BAPO现有实现要求stage_validation_files与periodic/final文件不同。推荐额外保留独立64题阶段集（NQ/Hotpot各32）用于BAPO每轮early/late控制。从E1的169103训练候选中确定性留出这64题，两个算法共同排除，以保持新实验训练题一致。预计共同训练169039题；实际数量、重复问题、split交集和原始版本必须在准备时核对后记录，不能硬凑数量。阶段64与训练/开发512/最终集内容交集需核查，不只是路径不同。若严格隔离导致计数改变，先报告实际manifest，不静默改划分。

这是与原包169103训练题不同的明确适配：多留64题，不复制512开发集换文件名绕过守卫，不未经授权放宽guard。BAPO stage size64/freq1/patience5/alpha.05/bonus.5/resample_attempts2保持原值。周期开发512不被冒称最终独立测试。

## 两算法必须保留的差异

Tree保留roots2/rounds1/expansions2/leaves3，每题6叶，外层512×6=3072条训练叶，mini128时每rank/同步约24次optimizer.step。n_agent=2代表根数量；不要设为5，否则与roots契约冲突或改变树设计。

BAPO每题5轨迹，512×5=2560行，mini128时约20次optimizer.step。重采样额外生成成本另计。因此二者相同outer并不表示相同采样/优化预算。报告实际叶数、token数、重采样数、optimizer步数、GPU时间。

Tree现有paper objective已为叶内token平均、叶间平均；不要误改成每题token平均或节点去重。BAPO当前审计路径仍为micro内token平均，需要实现显式sequence_mean适配，使PG和KL都按同一序列权重、完整minibatch分母累积。新命令的+baselines.loss_reduction=sequence_mean只是待实现接口，单独传参不产生效果，必须以真实DataProto/实际函数检验再启用。不得直接装载压缩包monkey-patch覆盖现有算法分支。

## 启动与验收

用户审核通过后，由原两会话在新独立源码和输出目录中实施；同步已修复的最终答案判定、已验证数值修复以及必要的新归约适配，不恢复旧漏洞。源代码、配置、模型、数据及检索资产manifest固定。保留原2971/307轮及2948/500轮等所有目录、权重、日志和结果，不覆盖或删除。两节点只使用原允许的独立八卡分配；不占用他人进程，不跨集群训练/检索。

先执行新配置的真实5轮验收，包括全rank loss/grad有限、非零学习信号和参数变化、实际mini/micro及optimizer次数、独立内容一致的评估、checkpoint和完整恢复状态保存/重载、独立审查。5轮检查须沿用正式200轮scheduler horizon，不能用total_steps6使warmup变成1。

首5轮计入正式200轮，验收后继续6–200；保留完整Adam/scheduler/RNG/data/BAPOstage或Tree采样状态。原运行器现有acceptance_step=5只保存，不会自动等待独立验收；批准后需要最小门禁适配，使第6轮之前暂停、独立验收通过后原位继续。不允许把checkpoint的HF权重误当完整恢复、不透明重置Adam、或自动再从零重跑。若该原位门禁/重载验收不可满足，先报告具体限制及替代方案再请求用户决定，不能虚构已有能力。

第200轮完成后保存、执行约定完整评估、停止，不自动改成500或其他实验。若用户最终选择300/500，先更新命令和审批版本。

## 尚未执行的工作

官方Instruct模型的目标节点部署和全SHA、准确split/prompt重建、BAPO新loss measure、首5轮原位验收门禁、完整恢复状态和实际Slurm包装全部在用户批准后实施。本地静态阅读不等于目标GPU验收，不重用旧base配置的独立验收receipt。

算法启动参数草案在下面；只在自有八卡作业中，由经审查同节点GPU检索/Ray/资源监督包装启动。这里没有sbatch提交、没有对活跃进程的操作，当前不能直接当作可执行部署完成品。
'''
for name,command in commands.items():text+='\n### '+name+'\n\n```bash\n'+command+'```\n'
text+='\n## 一手配置来源\n\n'+''.join('- `'+str(archive/path)+'`\n' for path in ['scripts/run_phase.sh','scripts/grpo.sh','search_e1/prepare.py','search_e1/core.py','search_e1/grpo_batching.py'])
(out/'PLAN-v1.md').write_text(text)
for name,command in commands.items():(out/(name+'-command-draft.txt')).write_text(command)
print(out/'PLAN-v1.md')
