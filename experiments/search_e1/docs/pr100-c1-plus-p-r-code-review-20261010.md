# 当前PR100是否确实为C1＋P＋R：代码与实际运行审查

2026-10-10。按用户最新要求，本报告只判断机制是否正确接入，不用其它组合组或效果置信区间代替代码审查。

**结论：已检查的部署代码、冻结配置与实际运行证据一致支持“C1基础＋原P精度机制＋原R随机学生选择”。未发现漏加、接错这两个机制，或额外改变C1科学设置的确定缺陷。** 用户另行指定的10→100轮、每10轮评测、输出精简是运行预算与管理变化。这个结论覆盖下述实际检查，不是对所有可能缺陷的无条件保证。

## 逐项机制对照

| 项目 | C1基线 | 当前PR实际实现 | 结论 |
| --- | --- | --- | --- |
| 学习率与优化器 | LR1e-6 constant；AdamW(.9,.999)、eps1e-8、wd.01；无warmup/scheduler | 冻结配置完全相同 | C1保持 |
| 教师与目标 | 固定source100，正确参考B提供特权提示，同学生D历史；reverse KL | 教师固定、teacher-only；p和logp保留梯度，教师stopgrad | C1保持 |
| 损失 | 正词表分量cap10、负分量保留；每轨迹有效token均值后实际全局pair均值 | 同文件/同数学；global32、micro1、八rank SUM、真实尾批分母 | C1保持 |
| 样本权重 | 单位权重 | 实际pair_weight全部1 | C1保持 |
| LoRA与clip | r16/alpha32/dropout0、七projection、gradnormclip1、chunk32 | 同冻结配置 | C1保持 |
| P：训练精度 | BF16 base/主要计算，FP32 LoRA | FP32 base/master参数、梯度存储与累积、Adam；FP16主要前向/重算；FP32 norm/logsoftmax/KL例外 | 原P正确启用 |
| P：采样精度 | CPU BF16 merge、BF16 vLLM | CPU FP32 base＋FP32 adapter merge，再一次FP16转换；FP16 vLLM | 原P正确启用 |
| R：学生D | 优先错误候选、完整response距离最远；全正确时fallback | 本题五个sibling均匀选择，包含B；相同seed42/qid/outer的SHA拒绝采样函数 | 原R正确启用 |
| 成功参考B | 严格正确中最少search，再token，再原顺序；全五错跳过 | B规则及全错跳过保持；仅D选择改变 | C1保持 |
| 数据与起点 | 共同source100、共同初始504 adapter、seed42 TRAIN顺序 | 前10轮题目及原始行顺序一致；原序列扩展至100轮 | 未换起点或前10题集 |
| 固定512评测 | HF SDPA、BF16 base＋未merge FP32 raw adapter、普通greedy、原生成/检索预算 | 相同正式协议，每10轮选该轮末raw CP | 未混用best或旧cache |

实际PR冻结scientific字段相对真实C1配置，**唯一差值为outer_iterations=10→100**。PR的precision字段与原P完全相同；采样字段与原P只差TRAIN序列从5120题扩展至51200题。前10前缀已另行实读确认一致。

## 证明机制确实运行的直接证据

1. **P不仅是配置名称。** 原P与当前PR的precision.py、targets.py、objective.py部署SHA一致；sampling.py只扩展cohort界限。当前第1、10、30轮八rank票据记录FP32模型/master状态、同步scaler，前向及重算观察为FP16。CPU merge记录938个FP32 PEFT参数全CPU，合并后一次转换成434个FP16参数。生产环境2×2 CPU autograd核对FP32参数通过FP16计算可回传FP32梯度存储；未调用GPU或加载模型做新实验。
2. **R不仅是分支标签。** 实际前10轮3002个PR pair重算B/D选择、policy spans和单位权重，0违规。D的五个index计数为573/592/614/630/593；其中596个D=B，实际允许自配。原R与PR对同qid/outer的随机函数一致。
3. **同一首轮能直接区分P和R是否接入。** 原P与PR首轮2560/2560生成response、reward、search计数完全相同，313个可用参考B也完全相同；D只有59/313相同，差异对应原P最远选择改为PR的R随机选择。原R与PR的294个共同可用题，随机D sibling index为294/294一致；轨迹正文不同来自P的FP16采样，不能将“随机函数相同”误写成“训练文本也相同”。检查只输出计数与内容SHA，原题/轨迹留HKUST。
4. **共用数学代码也做了检查，不只比较文件。** 小CPU复现中reverse KL autograd与解析梯度最大误差2.98e-8，teacher无梯度；普通及触发cap10条件下PR与原目标输出/梯度一致。八rank SUM/actual在32、7、3、1条真实批次均匹配直接全局mean，没有多除8。
5. **评测插入未重置训练。** 第30评测实际CP296、八worker各64、父进程wait0及释放接受；随后同run真实CP296→297→306。新adapter digest改变，八rank optimizer digest一致，恢复原Adam/scaler/Python/NumPy/Torch CPU-CUDA RNG/order/cursor；没有重新训练已提交前缀。当前正式评测选原始adapter，不是best或前轮采样模型。

## 源码定位

生产科学源码commit：`dc08aad44fe42adbc24ce102a774735a2c66f4f8`；周期控制commit：`e72eb3cdbf444625824c3d8369890e1c4d76c688`。文档发布不改变部署版本。

- `control/opsd_pr100_training_v1/prepare.py:24`：P精度模式；`:33`：C1科学参数；`:41`：FP16采样；`:44`：FP32 merge后一次FP16转换。
- `control/opsd_pr100_training_v1/miner_v1.py:43`：与原R相同的拒绝采样；`:99`：B规则；`:103`：本题全部五个sibling选D；`:121`：单位权重。
- `control/opsd_pr100_training_v1/precision.py:22`：FP16 autocast；`:58`：FP32参数约束；`:106`：FP32 optimizer/master与SUM、unscale、finite、clip、step。
- `control/opsd_pr100_training_v1/objective.py:58`：FP32 logsoftmax/KL与teacher detach、学生概率梯度、cap10。
- `control/opsd_pr100_training_v1/trainer.py:442`：原Adam/adapter/RNG恢复；`:525`：named-adapter setup后恢复scaler/RNG；`:716`：teacher-only目标；`:719`与`:729`：真实pair分母与八rank SUM。
- `control/opsd_pr100_periodic_recovery_v1/worker.py:92`：BF16正式评测加载FP32原始adapter；`supervisor.py:136`：边界后同状态resume。

## 小型证据与检查边界

本机审计目录：`outputs/search-e1-pr100-20261010/code-review/`。三路报告为precision-review.md、selection-loss-review.md、resume-eval-review.md；实际配置差为c1-plus-pr-actual-config-diff.json，首10选择与数学核对见selection-live-first10-readonly.json、selection-firstouter-content-differences.json、selection-loss-small-cpu.json。

真实配置SHA：PR `651dfb40b1bf22fb5f604b9b17c45ffe8619f20baf521f399bc8059b11d6aa91`；C1 `13fdbd51cc290bc6d23b0577c67bf8160a17292fab6f315fbfb07ea3f6ed3310`；P `dcc0a03750dec59651550609889a8b82ba0d967a728da01626216af6d7714412`；R `8cc81ae28882f0fb32a924d7a4d96fc6f7f0702b210ef62847e409c404d36ab9`。

审查没有修改生产、重启训练、增加GPU评测或拷贝大产物。既有八rank票据支持实际dtype/恢复，未冒称本次又独立反序列化全历史Adam或重算全部CUDA kernel。FP32梯度这里指master的存储/累积精度，不指所有反向矩阵乘法都使用FP32；这与原P定义一致。

**因此，本次审查确认了“机制是否忠实加入”，没有发现需要修复后重启的确定代码错误。EM下降本身不能据此解释为漏加P/R；下降根因仍需区别于本次实现一致性判断。** 当前100轮与每10轮固定512任务保持原授权运行。
