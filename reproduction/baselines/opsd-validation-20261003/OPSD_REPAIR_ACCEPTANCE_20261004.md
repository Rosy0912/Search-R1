# OPSD 修复及严格审查结论（2026-10-04）

用户要求修复后独立 code review。本次不重新训练、不新增 GPU 作业，自动发送保持暂停。原3007实际80轮 CANCELLED/0:15，冻结8e及原始证据保持，只有HFstep5，无HF80/HF500。

## 已修复及独审通过的软件问题

- 教师特权解答与指令置于最后用户消息内部，随后开启assistant；学生原输入与已生成历史不加入答案。真实tokenizer与tiny Qwen多工具历史验证角色、预测位置、因果性、观察mask、冻结教师通过。修复7ac。
- 默认取消KL分量0.05裁剪（None/null），恢复标准KL(teacher||student)。保留显式旧裁剪用于历史对照，参数梯度范数clip1.0仍在。实际3146完整256组原分布的独立解析小logit步全部降低真KL；这是方向证明，不是实际Adam/搜索效果。修复fae。
- loss日志按mini完整贡献记录，消除重复平均；独立同partition梯度及更新不变。修复5cae。
- 本轮补 canonical run.sh opsd 默认CUDA教师、user_context、null及h100-10本机限制。任务入口设置的边界标记使CPU/关闭local/错host/改method覆盖被运行检查拒绝。其它方法默认保持；直接绕过任务入口的main未获本任务部署验收。

根验证70项相关CPU测试PASS（3.72秒）；入口独审13项CPU测试PASS及真实Hydra入口/四种覆盖检查；已有61项核心回归独审PASS。测试原始日志和各独立报告均保留，不能相加当不同独立用例总数。

## 未验收，不能称全部问题已解决

性能问题仍存在：学生完整padding前向、全词表蒸馏、FP32/eager教师与小微批。共同外侧padding移除已完成隔离设计、12项CPU契约测试及toy两次Adam等价证明，但未采用到生产；真实BF16/FSDP形状变化可能改变数值，CPU不能保证GPU等价或实际提速。

旧GPU门禁显式.05且未指定新teacher模式，不能用于新user_context+None放行。实际3146只有初始状态分布，没有完整真实工具历史、Adam更新或80轮模型，因此没有证明修复后的教师会正确教搜索，也没有证明3007退化唯一根因。

结论：三项核心代码修复及本任务启动边界修复通过独立软件审查；性能和真实Search效果未验收。未经后续明确训练授权，不提交fresh5/500；后续应使用新源码绑定的门禁，先验证真实历史教师监督、新目标GPU全梯度/容量及性能，再执行受控fresh5、64题行为门禁与独立验收。不得复用旧配置测试冒充通过。

## 独立证据

- OPSD_FINAL_REPAIR_STRICT_REVIEW.md：核心源码及tiny真实Qwen多工具历史。
- OPSD_FINAL_INFRA_REPAIR_REVIEW.md：infra/评分/64题/效果与性能边界。
- OPSD_LOCAL_ENTRY_FIX_REVIEW.md：本轮默认入口风险修复及绕过拒绝。
- OPSD_PERFORMANCE_FIX_DESIGN.md：隔离性能候选，明确未采用。
- opsd-final-repair-root-cpu.txt：70PASS原始输出。
- opsd-reviewed-entry-resolved.yaml：本轮真实Hydra配置检查，无GPU加载。
