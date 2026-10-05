# GiGPO 三项防护接入设计（2026-10-06）

依据用户提供的《PPO三项防护实施与验收说明》实施。该授权新增 Dual-Clip目标和更新刹车，覆盖旧“不得clip联合比”的约束，仅限文档定义的负优势Dual-Clip，不改变GiGPO联合文本动作、优势或原权重。继续健康原3185 step25完整状态恢复，mini256、10Adam/outer、micro16、LR1e-6/warmup57、reference KL、数据/提示/采样/预算均不改。旧实验原位保留。

## 方案

采用有界符号分支的Dual-Clip(c候选3.0)；不采用先exp再clip，也不采用将联合比改成token或平均logratio。有效logprob先FP64转换，再相减和动作求和；正优势上界log1.2、负优势范围log.8至log3、精确零优势PG=0；原policy_weights和完整globalmini行分母保持。

每个outer固定监控第一globalmini256条轨迹（每rank32）；同一old/token/mask/positionIDs/温度，完整动作不截断。明确仅覆盖该固定子集，不能称全rollout。每次Adam前后，TransparentProbe保护RNG/buffer/FSDP边界，重新forward实际当前策略。token k3按原policy_weights全局分子/质量归约；联合动作尾部只每action计一次，同时记录原目标动作质量权重。精确全局p95/p99通过全rank样本聚合计算，非局部分位数均值。有效样本为空/非有限/delta超过kl_log_bound30直接拒绝，不exp该值，不报告截断KL为真实KL。

全rank同顺序forward/collective，先归约拒绝标志再一致退出。每个gradmicro仍同步检查有效输入和loss；有效reference/current差值在FP64相减并检查有限后，沿用原low_var指数前等价有界实现，保持原referenceKL目标；不对reference差值另设30阈值。梯度全局有限和clip保留，Adam后权重/状态有限检查保留。拒绝后受控退出，落全部rank原始mini/固定监控old与current/logicalstep/真实Adam写入数和原因；不fresh-rollout、不自动回滚、不假装完成outer；从已接受完整checkpoint后续恢复仍按授权执行。

## 阈值及验收

配置显式enabled、dual_clip3、kl_log_bound30、target_old_policy_kl、negative_tail_limit、monitor_first_global_rows256、阈值标定证据身份。生产KL/尾部阈值初始为空，禁止正式启动；0.01只作待验证KL候选。先健康25实际GPU隔离标定（更新前后、原概率一致性、10真实Adam步）；基于正常负优势尾部及长度选阈值并独审，不以存在单一r>3拒绝。CPU公式/极端800/NaNmask/联合梯度/globalmini微批归约、实际8rank集体拒绝无额外写入、postforward读新权重/old固定、记账及完整状态验收后才READY。

新增轨迹记录bug修复独立列示：结束4轨迹后compact2556要按原State.source映射完整2560；聚合后/obs截断前记录，不改变训练输入或数学。已有26项本地CPU回归通过，不代替GPU。

实际当前3248失败新增0更新，保留25/300；同512 F1 .3231979640745351/EM .26171875健康。h100-8另有Search-G1作业3250，保持Slurm独占等待，不干预。


## 明确统计口径与目前状态

negative_tail_limit比较“联合动作delta>log(c)的负优势独立动作数/全部负优势独立动作数”；另记录原目标质量加权比例，但不用它冒充计数阈值。本地无有效/负优势动作只贡献零并匹配collective；全局有效质量零拒绝，全局负优势零可正常通过。所有输入mask必须有限0/1；无效位置old/new/adv NaN不计入。采样策略k3在token层，联合动作p95/p99/max与长度只每动作一次。

新增安全配置文件 `verl/trainer/config/ppo_gigpo_instruct300_quality10_safe.yaml` 继承所有原共同配置；阈值及标定证据路径/SHA尚为null，生产严格拒绝。生产要求实际读取独审校准artifact，SHA、scope、source、25恢复manifest和阈值全部一致。CPU测试中的synthetic-calibration-fixture是解析器夹具，不能作正式证据。旧native3243和正式f1 READY不是新安全目标的验收。必须新版本native/formal接受器理解安全audit(联合rawexp不计算，只保留真实完整logratio)，禁止将空raw-ratio字段冒充旧scope通过。

CPU已覆盖真实actor循环10Adam写入/160gradmicro和post新权重读取、实际post拒绝只写1Adam不finishouter，以及真实2rankGloo的空rank/坏rank/精确全局分位数；仍不代替真实8GPU/FSDP/Flash/完整恢复与短段质量验收。FSDP/NCCL内部fatal错误须超时及资源清理，不能承诺通过外部badflag消除全部框架死锁。h100-8在稍后只读复查已空闲，但新GPU标定/接受helper仍待独审，不按旧helper启动。

Native execution scope separation

The GPU diagnostic uses an independently reviewed native-only binding containing the exact source commit, complete selected step25 path and manifest, and explicit fixture thresholds [100, 1]. These are permissive diagnostic controls, not calibrated production thresholds. Every guard logs its execution and admission scope. Calibration may only collect healthy actual rollout statistics while retaining numeric rejection; formal continuation accepts production mode only and requires a separately accepted calibration artifact. Neither a fixture PASS nor a calibration allowed row authorizes production. The native validator rejects empty scalar loss, nonfinite KL, empty effective monitor and inflated self-reported fixture thresholds.
