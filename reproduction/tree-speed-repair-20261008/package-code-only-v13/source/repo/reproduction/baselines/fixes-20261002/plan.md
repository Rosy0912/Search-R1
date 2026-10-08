# 独立审查后的顺序修复设计

用户授权：2026-10-02 依次修复审查问题，一般问题按对应原文算法；Search-R1自身冲突先说明并询问。
修复基点 afc0a3b；只改独立baseline分支，不干预原agent checkout或活动任务。

1. 共享PPO数值：空mask均值有限0；exp前屏蔽无效位置，对有效非有限输入fail-fast；
   KL输入同样屏蔽；全局空mini统一跳过optimizer和scheduler，单rank空而他rank非空保留backward。
2. BAPO reward：按固定作者scorer补yes/no/noanswer规则，空normalized token F1=0；
   保留CoSearch普通F1语义，避免把BAPO特判施加给另一算法。用作者差分与bonus/resample反例验证。
3. CoSearch标签：按作者NFD Unicode精确token串匹配，排除全trivial答案的relevance/availability；
   query分组与文档命中各自独立normalize；无信息group不执行无监督optimizer step，显式记录。
4. 配置/infra：DP/SP整除、local micro非零、OPSD global micro=DP ranks；launcher可见卡与Ray数量匹配，
   明确隔离Ray/临时缓存及依赖/资产/检索预验收，GPU实时双阈值和已有process检查，不停止他人。
5. 按论文目标修正action/trajectory/question聚合权重，跨micro/rank验证相同梯度；
   保留固定主模型/数据/搜索预算，辅助教师/ranker的授权和成本记录。
6. Tree continuation跨root合批，保留树prefix/预算/叶来源/成本统计。
7. 每步针对性测试；最后独立review、必要GPU短验收、修复报告与commit推送。

用户已决定：
- 保留原Search-R1 rolling生成/整轨迹评分的不一致，所有六算法和GRPO统一沿用。只记录已知限制。
- 原训练集固定2%阶段验证/98%训练，所有算法和对照共用，最终七集测试集不参与阶段切换。

现有tiny test不当作3B/vLLM生产验收。尚未验证的资源风险需要测量，不能通过修改文档宣称关闭。
完整resume是明确缺失能力，若本轮不能实现/验收，保持manifest false和修复报告pending。
