# 两组20轮已完成，后续五组沿原队列推进

2026-10-09 08:53:50 CST，一次直接连接HKUST1的读取返回0，原study driver的UID、启动时间及启动参数一致。C1 native512的八个rank均为RUNNING，共已写入207条正式结果。C1完整正确数尚未产出，不能将这207条作为完整512评测。C0、C1训练各10轮已自然退出0；后续M/R/W/P/J由原队列依次执行，备份不作为启动条件。

实际CPU收集和原已发布formatter均返回0，取得20条闭合outer链及70槽曲线。合计201次唯一已提交Adam、6035对实际曝光、1,212,173个有效policy token。C0为100次Adam、3012对、583074 token；C1为101次Adam、3023对、629099 token。其余50槽保留NA，没有插值。两组第一轮相同起点，后续采样各自更新后的raw student，不能假设10轮对应相同Adam、样本或token预算。

C0第一轮至最后一轮的原始/裁剪后loss为0.403615/0.393250→0.345832/0.339097；C1为0.404793/0.394526→0.402945/0.395478。每轮使用不同TRAIN cohort，因此这些是实际在线训练记录，不是固定heldout曲线；loss下降也不构成EM改善证据。C0独立native512为227/512，普通提示BF16零更新基线为239/512；C1仍等待本次正式评测自然完成。

之前的直接SSH超时未被当作训练故障。连接恢复后，C1 closed outer0–9的三个原始debug gzip备份全部验收：4087文件、2,568,006,231成员字节，其中8份原NPY保持原字节。第三份outer5–9的实际传输、源after核验和本地全部成员校验均返回0；最终验收SHA为`6fae8276f30b258dd691af772318c2c3d90ca2c31396e025c905875c581c6ebf`。后台checkpoint队列随后完成当前两组全部20个outer末包，原失败与Node原件继续保留。debug备份覆盖outer目录，不替代独立checkpoint数值验收、native512完整输出或driver终态。

本机既有clipmass聚合器实际返回0，完整检查C1十轮、101个committed update、3023条轨迹和629099个有效token。1193个词表分量被上限10裁剪，分布于1188个token（0.1888415%）；该集合的student质量在命中token上的均值为0.89016864，而teacher为4.4352068e-6。说明少数位置存在强分布分歧，不能将命中token比例混同概率质量、也不能直接归因为EM变化。全部被检查的记录指标有限，固定前向记录的等轨迹raw/post loss为0.4026358140/0.3940404729。完整聚合见`reports/C1-committed-clipmass-20261009/`，无额外GPU测试。

独立静态审查发现最终11模型CPU汇总的确定接口问题：原`summary.compare`直接比较含整数键的search histogram与JSON读取后的字符串键，正确记录也会比较不等。此问题位于全部训练与评测结束后的统计步骤，不影响正在执行的各组训练和native512。原正跑源码保持冻结；最终统计复用已发布且已在1536条真实记录上返回0的`opsd_ablation_stratified_effects_v1/derive.py`，其JSON规范化已处理这一边界，并保留原评分、逐题配对、18项预设比较及三分层bootstrap。无需重训、重推理或新增GPU测试。

原driver完成所有11模型后会先持久保存`eleven-model-inputs.json`，再调用存在问题的比较函数。若其最后统计返回非零，必须保留真实返回值和错误原文，不能将独立CPU统计成功改写为原driver返回0。最终科学验收将分别对账70轮、5632正式预测、各原子作业实际退出和GPU释放、debug/persistence及独立统计产物；当前目标仍为ACTIVE。

真实20轮facts SHA：`8b902baf444a9dd98b216fa9257e25d26c64aa7ad305fdf551c9f28166c871ad`；派生产物manifest SHA：`9b49b7066f62f4b498ee974c594fc08683207491222a9774b46c40972d1e2a61`。私有原始证据留在本地`work/search-e1-analysis-output/opsd-ablation-execution-20261008/goal-continuation-20261009-19/`。本报告不包含原题、逐token数据、词表数组、模型或凭据。
