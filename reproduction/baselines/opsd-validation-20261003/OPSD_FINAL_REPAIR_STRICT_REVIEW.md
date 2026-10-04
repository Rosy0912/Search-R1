# OPSD 最终修复严格独立审查（fae5e329）

**结论：教师提示结构、无裁剪目标、loss日志三项通过源码及CPU独立审查；没有发现这三项新的可复现工程阻断。整条“修复有效/可重训”尚不能通过：真实历史教师行为未验证，且旧GPU门禁未覆盖新配置。性能项由另一审查agent负责，本报告不代签。** 未修改生产、未提交Git、未部署、未启动GPU或训练。

固定源码 `fae5e3295c18a0c6b80e314b2fdce61ad571fec6`，实际使用git show读取；核对所审本地关键文件与固定commit字节一致，不混冻结8e或其他候选。已重读 prompt/logging/unclipped/3146各独审报告，本次不重复已合格的61case CPU回归，仅补未正式保存的多工具历史证据。

## 全链路源码

`distillation.py:36–100` teacher_inputs把accepted target提示插到最后user结束之前，验证精确Qwen ChatML生成suffix；原prompt token IDs保持，原assistant开始suffix保持，相同response/history通过attention选择保留。未知模板/mode、错suffix、特殊控制token或超出teacher最大上下文拒绝，不静默回落旧构造。该adapter限定当前Qwen模板，不冒充通用chat模板。targets来自actor当前micro的reward_model ground_truth，select_rows同时切分tensor/non-tensor，学生输入中不新增答案。

`distillation.py:104–166` teacher以完整prompt+相同已生成历史预测response，位置是prefix_len+cumsum(response_attention)-2，即预测目标token前一个位置；观察token留在attention/context，loss_mask只控制监督。chunk teacher base/LM-head处于no_grad，generalized_jsd另detachteacher；global冻结/audit机制保持，teacher未加入学生optimizer。学生teacher并不共享privileged prompt，区别是有意条件监督，而不是对齐错误。学生全trajectory scoring与实际滚动生成窗口差异仍是既有批准适配，不能假称每个历史prefix严格on-policy。

`distillation.py:12–33` 默认None时beta0为标准full-vocab KL(teacher||student)，温度梯度(p−q)/T，保留显式.05历史控制；纯OPSD没有importance ratio/PPO reward loss。run.sh生成参数不会覆盖component_clip，真实Hydra入口默认None、CLI.05可明确覆写，已独审原始JSON保存。controller→actor→opsd_loss→普通/checkpoint loss都传显式settings值，没有缺省还原.05。actor.grad_clip1.0仍是参数梯度范数裁剪；PPO clip_ratio.2配置保留但不在此路径启用，不能作为OPSD保护宣称。

`actor.py:27–68` 每mini一次全局监督token数，denominator=global_count/world；各micro总loss用同分母累加backward，再每mini一次optimizer。全局空mini统一跳过；本地空micro仍connected-zero参与必要FSDP同步。新日志把同一mini内完整贡献求和后append一次，再mini均值；没有改backward/分母/更新次数。rank-local贡献和等权mini均值不是直接outer token-weighted全局KL，应维持报告口径。loss console三位小数不能代替实际精度证据。

## 新增小CPU历史检查

固定fae字节加载的teacher_inputs/opsd_loss，真实ByteLevel tokenizer+Qwen ChatML、tiny真实Qwen teacher，2行synthetic history：think/search1→information1→think/search2→information2→answer。没有新模型资产、真实3B前向、实际trajectory抽样或GPU。

独立检查全部通过：hint位于user内部；相同history token逐个保持；所有attention有效目标的teacher prediction positions+1等于原response token；观察loss位置直接student-logit梯度0，监督位置非0；teacher所有grad None；改未来history token不影响此前teacher logits（同CPU因果前向torch.equal）。原始JSON `opsd-final-history-independent-cpu.json`。观察token可以通过因果上下文影响后续模型参数梯度；直接观察logit不监督不意味着观察embedding参数梯度必须为0。

此前61相关case与checkpoint-off/on、temperature、teacherfreeze、masked NaN/Inf、全空mask、micro1/2/4及末小mini证据仍有效。没有扩大这些CPU证据为8卡FSDP全参数数学或新source实际性能通过。

## 实质未闭环项

1. **新source GPU门禁配置身份不符。** 固定repo `reproduction/baselines/opsd-runtime-20261003/gpu_teacher_probe-factor-dtype.py:74`、`gpu_loss_checkpoint_equivalence.py:111` 仍显式component_clip=.05且未指定teacher_prompt_mode，因此默认legacy；microbatch_probe的同类settings也是旧配置。旧控制门禁是保留的历史科学证据，但不能用它批准user_context+None新source完整数值、容量或fresh5。当前没有重新训练授权，本审查不修runtime；后续实际准备必须绑定当前mode/clip及源码并独立验收，不借旧BF16 device门禁放宽新目标测试。

2. **角色修复不等于真实历史上的工具监督正确。** actual3146严格只有有序训练前64个initial state，student base/HF5、固定teacher legacy/user，未采actual搜索/观察后的监督位置。其完整256NPZ、来源/runtime/CPU replay/解析梯度通过，但不能证明所有trajectory位置teacher偏向合法search、有效query、正确终止/答案。新teacher search概率均值仍低于学生base，完整首别名候选55/64降低；这是真实初始诊断风险，不是最终准确率或晚期80轮崩溃因果。应明确保持“内容监督有效性待验证”，不能被synthetic tiny历史结构测试覆盖。

3. **无裁剪修复目标方向，不自动修复内容/效果。** 同actual3146四条件256配对独立解析无裁剪小logit步全部降低KL；.05下user教师30/34个反向被移除。这里不是实际FSDP/Adam参数更新，也不是新生成completion/reward。新默认None偏离作者主脚本.05，实验应标候选修复，旧8e原设定和失败历史保留。

4. **更新速度修复不在这三项diff内。** fae没有新的吞吐修复；此前教师GPU放置、loss checkpoint和空chunk跳过仅具有各自已冻结旧配置scope。实际更长privileged上下文、无裁剪对显存和wall time的成本还未测。另一agent的性能设计/证据不能仅凭本报告得到联合通过。

## 固定文件身份

- `search_r1/baselines/distillation.py`: `a457ca7b1237bb3accffec89f8cc21327bd43feb8e488d96f40f1c5c3b775747`
- `search_r1/baselines/actor.py`: `92cb9237170260c1d625007dc006db86ba82b66a7e264e004901147c502a99e3`
- `search_r1/baselines/controller.py`: `0711bb78fde91b33d48b159786a7babb12b10719591eadb36594d4b59af72b54`
- `verl/trainer/config/ppo_baselines.yaml`: `ef2c055b8142acf8eab9a34f0787665fc21f06e7a2e18587985a42f32b6540f8`
- `reproduction/baselines/run.sh`: `578e9d0b36325159ba9849a7113cd2b92ec660c7bd6ca420d25a98971736646a`

整体验收状态：**三项代码/CPU修复通过；性能项及真实3B历史监督/新配置GPU效果未验收。** 没有新增主源码反例可据本次审查认定三项修复错误；上述门禁覆盖缺口和教师目标内容风险是明确待闭环事项，不以“未发现工程bug”代替算法有效性。
