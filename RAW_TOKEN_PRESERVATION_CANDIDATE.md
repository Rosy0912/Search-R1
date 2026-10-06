# Tree 原始采样 token 保留候选

状态：本地源码与合成 CPU 候选，默认关闭；没有部署、远端资产读取、GPU 执行或生产训练放行。基于 `a632ba91840fb116aacfc0e770681a860647b295`，独立分支 `codex/tree-raw-token-preservation`。设计依据为 `../tree-token-preservation-design/REVIEW.md`，SHA256 `0e42b1f75edc9db0ca3a9a5e65fd286c73dcd08dda9f711541bb8727b07f5b95`。

## 策略 ID 与旧语义的双表示

原 vLLM facade 在补 PAD 前，Tree opt-in 返回第三项行元数据：实际原生 request ID、调用 UUID／rank／物理行、源／配置／数据／模型／作业身份、原始 emitted ID、原始 selected scalar、真实长度、finish／stop 与实际 SamplingParams。原双元返回保持默认行为。adapter 逐行要求真实长度 mask 与原来的 EOS mask 完全相等，不补造 EOS 或修改 mask 来使检查通过。

`sampled_action` 用固定 tokenizer 对最多 500 个完整原始前缀逐一解码；保留覆盖旧 canonical 文本 C 结尾的最短完整原始前缀。完整 raw 解码和 C 仍采用旧 `skip_special_tokens=True`、优先第一个 `</search>`，否则第一个 `</answer>` 的规则。策略 action 和后续 generation context 只使用 `raw_emitted[:cut]`；其原生 sampled scalar 按相同 cut 保留。重新编码 C 的 ID 只进入语义和审计 ledger。

parser／环境继续直接使用旧 `_postprocess_responses` 得到的 C。reward／evaluation 从绑定 ledger 重建旧 canonical-ID 时间线及 model mask，按旧的连续 model spans 逐段调用 **`tokenizer.decode(ids)` 的默认 flags**。这里没有改成 `skip_special_tokens=True`，也不假定 `decode(encode(C)) == C`。观测 ID 和 loss-mask=0 的位置均保留；无观测的相邻 action 仍按原 contiguous span 合并后解码。实际源码调用 fixture 证明带 `</answer>tail` 的不可拆 token 保留 raw ID，而旧 Paris QA 的 reward、F1、EM 仍为 1。

raw ID、scalar、完整 context、position、attention、cut、canonical text／ID、原生 finish／length、tokenizer flags 和采样身份随 `State.branch()` 深拷贝。actor 更新前再次绑定到实际 batch prompt／response／info-mask 并验证。greedy validation 使用同样的双表示，但保持原单轨迹评估，不扩成六叶。

## 明确的协议影响与执行限制

这不是旧新轨迹逐 token 等价。标签末尾位于 token 内时，其不可分尾字符仍进入下轮模型上下文及整个 token 的 loss，ledger 明记尾字节；特殊 token／EOS-only 原来可能产生零 canonical token，现在可以有 raw 有效 token。有效长度、每 token 权重和后续采样分布会因此变化；原等叶权重公式和总质量不变，但不声称每叶数值相同。

候选拒绝 PAD==EOS、真正 emitted PAD、空 emitted、缺长度／scalar／finish、原生 abort、超 500、U+FFFD／非法 UTF8、任何非单调完整前缀解码、采样身份重复，以及 native 真长度和原 EOS mask 不一致。实际 tokenizer 固定 PAD151643、EOS151645；BOS 允许原来的 None。未实现另一套显式有效 mask 协议来支持 PAD 情形。

start 保持 2048、rolling／response 4096、action／obs 500、四轮、无强制最终回答。观测仍使用原 `_process_next_obs` 的 500 右截断。为避免语义投影复活已丢弃答案，本候选 **拒绝任何 rolling 左裁、raw response 或 canonical semantic timeline 越过 4096**，也拒绝起始 prompt 已超过 2048。常见长轨迹确实可能触发这种拒绝，批次停止而不是默默缩短、增加预算或继续更新。因此当前不能声明 500/300 长训可执行、five 接受或 GPU READY；长度支持的后续扩展需另立协议与审查。

driver 收到可检查的 emission 后发生拒绝，会在自身 output 下保存 `token-preservation-rejections` JSON，包含错误、现有成本计数与拒绝上下文；RPC 在返回前失败时只有实际已收到的证据，不捏造未知 emitted 数或 GPU 成本。原运行 observer／Slurm／资源终态仍需独立保留，不能用这里的计数替代完整作业成本。

## 配置和成本口径

仅 `ppo_tree_safety300.yaml` 增加默认 false 的 `actor_rollout_ref.rollout.tree_raw_emission` 和 `baselines.tree.token_preservation.enabled`。启用两者时 protocol 必须为 `tree_sampled_prefix_legacy_semantics_v1`，且明确提供同机已冻结 `tokenizer.json` 的 SHA256；没有猜测或自动下载该身份。采样 capture 要求运行包提供原 source/config/data/model/run/job 六项 provenance 环境变量。

预算、LR、train640／mini384／micro16、epoch1、Tree2 roots × 每 root3 leaves、原树分叉／优势／等叶分母、四轮与评估 cadence 均未修改。`advantages.py`、`objectives.py`、原 actor 和 policy-safety 数学文件保持 base 字节。

新增成本把物理槽位、实际 emitted、保留 action、丢弃 sampled 尾部／填充行、最终叶中重复继承的 sampled-token occurrence 分列。继承按 `(sample_id, original_offset)` 计重复，不冒充新采样。原 generated_tokens 计数仍保留。全前缀扫描可能二次方解码，单列 CPU／wall；这些合成 fixture 耗时不代表真实 tokenizer 或训练成本。

## 证据范围和待验收项

15 项合成 CPU fixtures 使用实际源码中的 `_postprocess_responses`、`_process_next_obs`、parser／env、SearchRollout、RewardManager／evaluation、facade 以及原等叶权重函数；外部 DataProto 与 native 请求仅作小桩。覆盖整 token 尾字节、分拆／双标签优先／多标签、中文／特殊 ID／EOS-only、UTF8／非单调拒绝、PAD／长度拒绝、恰好 500、真实旧 obs500 截断及 mask0／完整下一前缀、Tree 身份继承／deep-copy、score／context／position 拒绝和 reward／F1／EM 不变。它们不是实际 Ray、vLLM、FSDP、DataProto 或 tokenizer 重载验收。

本机缺 `tensordict`，原 `tests.baselines.test_tree_batching` 与 `test_algorithms` 在 import 阶段失败，未执行回归；没有安装依赖掩盖这个限制。完整真实 DataProto row 重排／RPC 元数据保留、原 facade 实际链、固定 tokenizer 加载来源、分布式同步失败与成本终态尚需冻结包后的同机独立核验。

候选没有改变 old-logprob 重算、native 归一化、概率容差或三项防护数学。保留 raw scalar 不意味着原 sampler 与 actor backend 数值相等；**不声称修复已观测 raw backend 1.8738 偏差**。梯度／参数 finite、真实非零更新、零写入概率、five／F1 及完整最终效果仍无新凭证。无需重跑独立窗口诊断或修改其冻结源码来使用这份 source-only 候选。
