# 原生搜索行动停止：不改变既有SEARCH-first解析规则的续训方案

启用字段仅为：

```yaml
actor_rollout_ref:
  rollout:
    tree_action_tag_stop: true
    tree_action_stop_mode: search_only
```

默认关闭，不影响正在运行的d554源作业。必须在真实完整CP边界、原owner正常停止且自有GPU后代释放后，部署新不可变源和新config身份，完整恢复原模型/Adam/scheduler/driver/RNG/cursor。不得热改正跑、不得从0重采样已提交前缀。root应把配置比较允许变化精确限定为这两个rollout字段；其余训练数学、温度1、预算、检索、奖励和更新次数保持。

## 为什么只停止SEARCH

旧parser是整段SEARCH-first：只要500token/EOS完整输出含`</search>`，就优先把第一段SEARCH作为行动；仅完全没有SEARCH才取ANSWER。因此，若先输出ANSWER后又输出SEARCH，双标签stop会隐藏后面的SEARCH，将原搜索行动改成回答。这不是纯性能等价。`search_only`只在第一段搜索结束标签完成时停止，ANSWER且无SEARCH仍按原EOS/500规则生成。显式`tree_action_stop_mode: both`只保留为诊断/明确协议变更，不能作为此语义保持方案生产启用。

在耦合相同的原生随机token序列下，一旦SEARCH闭合，旧规则必然丢弃其后的全部尾部；早停只省这段不参与行动的生成。没有SEARCH时不早停。原canonical parser、QA和训练行动选择均未修改。早停减少随机数消费、改变调度和后续随机轨迹，故不保证bitwise复现或逐次EM相同，也不能拿这个改动许诺EM提升。

## μ与停止证据

- vLLM实际设置`stop=['</search>']`、`include_stop_str_in_output=True`、`detokenize=True`。
- 从真实`RequestOutput`原样保留token IDs及每个已选token的logprob；新增原生返回text，只在真实字符串stop发生时记录。
- 返回text必须实际保留并结束于原native `stop_reason`。完整原token decode必须覆盖返回text；在更早完整token边界已经出现stop而仍继续生成则拒绝。
- 字符串标签可能结束于最后一个完整token内部。该token及其原logprob保持不可分割，原规则的`indivisible_extra_text`继续记录；不把重编码token用作policy token，不补造/替换概率。
- 旧EOS/length分支保持；温度、sampling支持和μ/PPO损失未改。metadata完整不等于实际GPU验收，首新采样还需观察真实native停止记录与原prefix-mask/μ验收。

## 有限检查

4项新增synthetic CPU fixture通过：原EOS输出与SEARCH-stop保留相同原token/μ；真实停止text与ID不符/缺失/NaN拒绝；越过搜索停止token边界拒绝；ANSWER→SEARCH反例证明双stop不等价。另原3项stop metadata检查通过。它们不冒充GPU采样或EM测试。

此前同16/32 prompt的实际单GPU双标签诊断显示丢弃尾部约46%，32条生成19.50→11.22秒、整体49.20→39.58秒。该结果支持尾部生成有开销，但不是search_only新生产停止合同的GPU验收，也不能宣称整体35→8分钟。另最大项仍是实际HTTP等待；详细结果在sampling-profile-16-32-20261012.md。
