# 历史 TRAIN R 重构：独立数学与数值协议审查

2026-10-10。只读本地绑定源码及已保存的小型现场审计；未启动远端GPU、未改训练/探针/检索，未另写重复replay实现。结论是**原协议可明确重建，但R应标记`R_reconstructed`**。原逐行在线R/b/d没有保存，完整协议和聚合值匹配也不能证明找回了原在线逐行bitwise值。

## 1. 原打分排列必须恢复

`verl/trainer/ppo/ray_trainer.py:722–733`使用每条完整`attention_mask`的sum作为长度，包含有效prompt、policy、observation和correction；不是policy token数。k来自实际actor worker world size。本次run使用训练GPU0–3，实际world4，2560条均分为四个640条分区。不要被`common8`目录名误导。

原`verl/utils/seqlen_balancing.py:25–130,152–183`为Karmarkar–Karp：

- 初始按`(seqlen,index)`升序；`equal_size=True`以相邻四项建立State。
- Set排序依次比较sum、items数量、**保留插入顺序的items列表**；不是先排序items再比较。
- State最小堆使用反向spread比较，使最大spread优先；spread相同时按最大首Set排序。
- merge把一个State的第i个Set与另一个State的第`k−1−i`个Set合并，随后Sets reverse-sort。
- 最后wrapper仅将每个partition内部的index升序；保留partition外层顺序。按此顺序flatten后`DataProto.reorder`同时重排tensor及所有non-tensor字段。

应AST提取并执行原函数，不能改成greedy、总长度排序或按group排列。使用原Python环境与heapq；无RNG依赖。独立本地AST执行得到的小型期望：

| seqlen / world4 | partitions |
| --- | --- |
| `[10,10,10,10,10,10,10,10]` | `[[0,7],[1,6],[2,5],[3,4]]` |
| `[2,2,4,4,2,2,4,4]` | `[[0,7],[1,6],[3,4],[2,5]]` |
| `[4,4,4,4,8,8,8,8,12,12,12,12]` | `[[0,4,11],[1,5,10],[2,6,9],[3,7,8]]` |

C本身不依赖此排列；R的GPU重构需要恢复它，以保持micro4的组成、padding长度和GEMM形状。稳定join使用原trajectory ID，不能依赖重排前后的zip。

## 2. 原probe输入与顺序

`main_ppo_eco.py:394–417,451–456`首先使用**原initial tokenizer**解码：取`prompt_ids[-valid_plen:]`；response取`[:valid_rlen]`，再按`info_mask`调用`decode_segments`。传给probe的prompt就是这个decoded prompt，response是分段解码后拼接的response。不能改为全序列一次decode或重做chat template。

client发送重排后的**全2560项**，包括无效、错误和未搜索项。它发送一个完整请求，不是四个640请求。`_probe_call`重试会重发同一完整items/order；不会改变chunk。原服务在整次请求的LOCK/evaluator锁内先完成全部Hc，再完成全部Hb。

`probe_server.py:161–169`中`Xa`是prompt加response截至answer payload末，**包含原payload空白，不包含`</answer>`**；合法性和C由trimmed answer决定。无合法最终answer时使用`prompt+response`fallback，也仍进入probe。b使用prompt本身。不能去重同题prompt、先删wrong/invalid、只处理correct或把Hc/Hb合并为一次多layer forward。

## 3. 每个旧version的encoder/tokenizer/readout

实际ledger绑定：steps1–50用round0，51–100用round50，101–146用round100。iter50和100的R仍使用上一version；refit发生在完成该轮之后。必须按每条committed update的version选择，不能凭保存点名字推测。

已保存head元数据：

| version round | commit | boundary |
| --- | --- | --- |
| 0 | layer24 / Xa / MLP256 Pipeline | layer24 / Xp / LogisticRegression |
| 50 | layer24 / Xa / MLP128 Pipeline | layer27 / Xp / LogisticRegression |
| 100 | layer21 / Xa / MLP64 Pipeline | layer27 / Xp / MLP64 Pipeline |

原服务`_load`使用该version自己的encoder目录tokenizer，left-padding、left-truncation，pad缺失才替换EOS。round0与50/100的`tokenizer.json`和`tokenizer_config.json`字节SHA实际不同；不能用initial tokenizer替所有probe version重新编码。checkpoint目录还有added/special-token/merges等补充文件，应保留并fresh-hash记录；原encoder manifest没有覆盖的补充文件须标“当前补充身份”，不能冒称历史manifest已覆盖。

原数值路径：BF16、HF SDPA、model.eval、`torch.no_grad`，`base=model.model`，decoder layer`layer−1` forward hook，取tuple的第0项或tensor；每个micro4使用`add_special_tokens=False,padding=True,truncation=True,max_length=4096`。这是按该四项截断后的最长长度padding，不是每项固定pad4096。取`h[:,-1,:].float().cpu().numpy()`。left-padding保证最后位置是有效token；不得右pad后仍取−1，也不得改position_ids、pooling或取最终norm后的hidden。Hc全640个micro后再Hb全640个micro。分类器调用原`predict_proba`并取positive class1，先核classes`[0,1]`；不提前把概率降为FP32或clip。

## 4. R的数学与FP32落点

当前入口wrong-ground-scale/probe-base/probe-alpha-cost均为0，因此：

\[
R=\begin{cases}
-.1&无合法最终answer,\\
.05&合法answer但C=0,\\
1+.5b&C=1且n=0,\\
1+.6(1-b)d-.02\min(1,\max(0,n-1))&C=1且n>0.
\end{cases}
\]

n是完整policy生成的`<search>…</search>`数量，不是实际HTTP检索次数，也不包含prompt/observation中的示例。保留原Python float运算的顺序，随后原`reward_tensor`赋值将score转换成FP32。每行仅最后有效response位置有该值；序列sum即该FP32 score。struct的binary32 round-trip可表达这个落点。例如-.1实际存储为约-.10000000149，.05约.050000000745。当前四个分支都不产生−0。

**不要对R再乘info_mask。** 无效行最后有效位置可能是observation，原terminal reward仍在那里；GRPO先对完整token rewards求sum，之后policy loss才排除observation。原`use_kl_loss=True`使scores直接复制给rewards；当前KL系数0，无须也不能把KL项塞入这个R。每轮的native reward mean/min/max和EM必须核验，但三位打印精度只构成必要条件。

## 5. 实现验收必须覆盖的fixtures

1. 上述KK tie向量、真实2560/world4完整排列/逆排列、四分区640及长度包含obs/prompt的反例；比较原AST函数的完整输出，不只比较balanced sums。
2. original actor ID/gold/C和replay ID逐行一致；错序/漏行/重复/sibling或version错配均拒绝。
3. 保留invalid/wrong/nosearch；probe调用的两趟顺序、micro4边界、每组padding长度、4096左截断和特殊token设置记录一致；不得dedup b。
4. mask-aware分段decode、无效fallback、Xa payload端点/Unicode/空白/closing-tag边界；不得用模型自己生成的observation文本当参考答案。
5. layer−1、tuple/tensor hook、BF16→FP32 feature转换、各version自身clf/layer/tokenizer资产和classes校验；禁止最新head替代。
6. 四奖励分支及search成本饱和、FP32存储bit pattern、terminal reward在obs位置仍保留；wrong-ground-scale必须是实际0而非构造器默认.15。
7. version边界50/51、100/101及未知/missing version HOLD；所有范围输入缺失或R非有限时不能给全量pooledρ。
8. 在独立GPU准入后，至少以同一固定小型输入比较原冻结feature函数与候选函数，确认完整numeric/context/shape记录；静态CPU fixtures不是GPU数值门禁。实际资源/独立锁/已绑定retriever共享/UUID/PID/显存准入仍须由root执行，不能因空闲显存自动分配。

## 来源边界

本地原文件SHA：`seqlen_balancing.py`836a381b88a7c2185aea684b8e8c26e7eb0c25c6183e8f2f4d1686c85636fefb；`ray_trainer.py`0c9042cb1e414a34e8928df761fbf850505ddcb29cfddbc54f0ed767066635d0；`main_ppo_eco.py`bc0ac16ea7b4d671bc7d8c80b6525a9b467b41fdc92cbb097e7e3beaa00a285d；`probe_server.py`284df6e025822eb654d5858c394ce5c27fe80bb0d538a7d7dfa247081c461900。新preflight须再次从原run manifest逐文件fresh核验，不拿本说明代替准入。

现有小型审计证明encoder文件存在、heads完整SHA匹配和tokenizer小文件身份；大权重完整fresh SHA由独立preflight处理。无法证明旧GPU每次kernel实现/并发/allocator状态与新replay bitwise相同。报告应发布逐轮及全flatten的`Corr(C_raw,R_reconstructed)`，保持与“原始在线R”区别；正相关不验证b/d忠实性或性能提升。
