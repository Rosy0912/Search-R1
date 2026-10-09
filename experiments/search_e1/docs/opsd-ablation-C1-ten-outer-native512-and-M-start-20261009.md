# C1 十轮正式评估与后续组接续（2026-10-09）

截至09:39:37 CST，C1同一运行完成10个在线outer、101次真实Adam、3023个有效pair曝光、629099个有效policy token，原训练与评估父进程实际wait0且自有GPU后代释放。原顺序队列已经自动进入M第0轮采样；没有等待Mac备份或重跑C1。

| 固定模型 | NQ /256 | Hotpot /256 | 全部 /512 | 严格EM |
|---|---:|---:|---:|---:|
| 共同BF16零adapter起点 |110|129|239|46.6796875%|
| C0：reverse，LR5e-6，10outer末raw |109|118|227|44.3359375%|
| C1：reverse，LR1e-6，10outer末raw |109|127|236|46.09375%|
| 共同FP16零adapter起点（独立精度端点） |111|134|245|47.8515625%|

C1评估明确加载末CP101的BF16 base + 未合并FP32 LoRA，未回退best。普通提示、原HF SDPA greedy/repetition_penalty1.05、8192上下文/500token每轮/5turn/4search/top3/observation480及固定NQ256+Hotpot256均与共同BF16及C0主评一致。FP16起点是另一个精度端点，不作为C1主评起点。

配对重用原严格逐题分数，按题bootstrap10000次、seed42，未新生成或重新评分：

- BF16起点→C1：新增13、丢失16、保留223、仍错260，净−3题（−0.5859375pp）；95%描述性区间[−2.734375,+1.3671875]pp。
- C0→C1：新增29、丢失20、保留207、仍错256，净+9题（+1.7578125pp）；区间[−0.78125,+4.4921875]pp。
- C0→C1的净差来自Hotpot +9题，NQ净0。上述丢失题的最终行动均协议有效但gold不匹配，无丢失题零搜索。C1总体另有1题最终行动无效，该题不属于以上丢失集合。

当前点估计与较小学习率减轻退化的解释相容，但两个总体区间都跨0，C1也没有超过起点，不能确认效果提升或宣称学习率是唯一根因。两组各自在线生成，pair/Adam/token曝光有差异：C0为3012/100/583074，C1为3023/101/629099。10outer相同不代表曝光完全相等。

正式完整覆盖目前为4个模型、2048/5632槽，七个模型版本仍待完成。09:39快照中M正在采样；09:47:20实读同一M运行已完成outer0，生成2560条轨迹、P=K327、11次真实Adam更新。全队列已闭合21/70 outer、212次Adam、6362个pair曝光，尚余49个outer。M有效token总量尚未独立汇总，不能把此前20轮的1212173 token当作21轮合计。已保存的TRAIN曲线仍为20轮，未把训练batch reward当heldout曲线。

M运行b255c1eb72e84ca0bd6315e57e3527d0，controller2630209、UID1005、starttick930160381，原driver1894801自然接续。09:56实读M已闭合两轮，P/K分别327和308，共21次原生Adam提交；全队列22/70轮、222次Adam、6670个pair曝光。CP1/5独立CPU实读各实际rc0、八rank完整Adam/groups/moments有限且一致、步数准确1/5；504个adapter张量有限，首步252个B改变，第五步504个A/B改变。此核验没有加载模型或初始化CUDA，也不是效果提升证明。显存observer鲜活，历史每卡直接余量均不低于3175MiB，没有低于1024MiB门限的记录。

C1十轮raw4087成员已持久备份。新增闭合全局票据151成员/1279704B独立gzip100221B，实际inventory/transfer/after/memberSHA均通过；39个原phase-exit引用已独立绑定。正式C1 native512完整原轨迹、HTTP/生成ID、worker与父进程票据另591成员/33084301B，独立gzip5038635B；源freshSHA、实际传输rc0、源metadata-after及全部解压成员SHA均通过，本地完成票据SHA29273eb14d6b512c7f6884b706efc93ab88689f7a1ec788074bed1f4dbfdc314。两份副本均保留Node原资产，备份始终不挡M训练。不重复备份已验收raw，不改生产source。V1接口误用stage导致只读inventory失败，原rc1/日志保留；独立V2按真实group phase字段修复并发表后实际成功。

补足必要数学证据只运行一次隔离CPU检查，8.52秒实际rc0，真实reverse目标的B1..32含空rank×普通/混合64对照及9个原指定数学方法通过，最大梯度误差1.49e−8。CUDA未初始化。它不验证NCCL或完整生产故障恢复；完整current/old/Adam/scaler/全RNG/cursor的受控恢复证据仍需补足，不以toy或scaler-only代替。

统计产物在 `reports/seven-group-recorded4-models-progress-20261009/`；该报告与统计均是部分结果，不是七组完整验收。
