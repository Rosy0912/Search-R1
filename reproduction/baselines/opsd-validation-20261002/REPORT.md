# OPSD / Search Agentic RL 验证记录

更新时间：2026-10-02T21:34:56.316524+08:00。

当前结论：**CPU 数学与实现检查通过；真实 3B 搜索训练及效果验证未完成。不能据此声称 OPSD 优于 GRPO。**

## 已完成

- 根据 six-agent-plan-20261002/PLAN.md 与关联会话执行 agent 4；独立分支 `reproduce/opsd`，已推送提交 `ea0a78b`。算法冻结起点 `0723dca0c1458482ac3e78f8ed43af04893955d2`，计划提交 `1620c2edb13809d19cc3f5af4a298ccd75488104`。公共提交 `297e1c140677dc621302f1d49fefdb31023fd2b7` 已冻结并 cherry-pick 为 `f60897e`，已推送；组合完整CPU gate为139 passed / 1 skipped。
- 完整 CPU gate：134 passed / 1 skipped（macOS 无 Linux procfs），补充观测后的复测同样通过；合入公共提交后最终复测为139 passed / 1 skipped。日志 `cpu-common-merged.txt`。
- 独立 subagent 对固定作者论文/源码及数学、逻辑、infra 审查：16 组损失值与学生梯度一致；32 项定向回归通过；真实 tiny 模型师生对齐检查通过。成本观测修正后，损失和15个学生梯度张量与冻结前逐位一致。tiny 结果不替代生产3B/GPU验收。
- 增加教师加载/评分CPU区间时间与墙钟、模型tensor字节和worker RSS高水位。CPU→student CUDA传输排除在教师CPU计时外；process CPU时间覆盖区间内所有worker CPU线程，不是专属profiler归因。RSS包含student/optimizer/worker，不能称纯teacher RAM。
- h100-10 首次快照8张H100 80GB，利用率0%，显存0MiB，无compute process；Slurm gpu partition支持该节点8GPU。快照只是候选资源检查，尚无Slurm预约。
- 远端独立目录 `/mnt/data/shaoshitong/search-r1-rl/opsd` 已建立；3.6MB完整历史bundle SHA256核验通过，远端checkout在1620c2e，固定上游commit存在；项目内工具uv0.12.21已安装。核心训练环境尚未完成。

## 必须保留的协议

Qwen2.5-3B base；冻结初始base教师、CPU每rank一份；teacher答案特权仅训练使用，学生及验证保持原输入；全词表FKL，beta0/T1/component clip.05；无额外KL/entropy；SP1、每rank micro1/globalmicro8；train512/mini256/lr1e-6；98%train/2%holdout/seed42；2048起始、4096rolling、动作及观测500、最多4轮、不强制final；保留用户已授权的生成窗口/训练历史差异。正式500个实际outer update，完整51,713题最终评估，与新GRPO共同协议比较。

## 尚待完成的门禁

| 门禁 | 当前状态 |
|---|---|
| 统一common-setup提交与独立review | 297e1c1 已冻结，f60897e 已合入且通过增量独立审查 |
| 固定依赖及全部71.72GiB不可变资产hash | 未完成；远端镜像HTTP200，但后续跳板SSH间歇超时 |
| 跨路径split content_identity与parquet SHA | 等待公共准备及真实数据 |
| Slurm独占8GPU分配、UUID租约、通信/检索一致性 | 未启动 |
| 生产3B两outer-update smoke/finite/logprob/非零更新 | 未启动 |
| 五update检查点保存、HF重载 | 未启动 |
| 正式500update与完整七集最终评估 | 未启动 |
| F1/EM/格式率/检索次数/token/成本与GRPO差值 | 无真实数值；将沿用公共评估器 |

## 当前阻碍与续接

`reproduce/common-setup`由agent1集中准备，所有算法必须cherry-pick同一审查提交。目前已合入共享F1/EM/格式评估器，OPSD不重复实现。2026-10-02晚间多个远程查询/传输命令发生连接超时，之后曾恢复并确认bundle、repo及uv，最新bootstrap与状态查询再次在mgmt-host入口超时。bootstrap是否产生任何远端残留需连接恢复后检查 `logs/bootstrap.txt`、`logs/bootstrap.exit` 和tmux `opsd-bootstrap-20261002`，不能根据本地命令异常重复启动。

恢复后先检查本任务进程及环境状态；共同提交已合入且CPU gate完成；校验并更新远端唯一checkout至f60897e或后续纯报告提交；完成固定环境和只读资产核验；申请Slurm后进行真实GPU/检索/算法smoke；5步验收后再正式预算。不得绕过preflight、kill原进程或把CPU/tiny证据当效果结果。当前没有资源预留、GPU训练或已启动的自动监控。

## 指标解释

带per_rank的成本被现有reduce_metrics聚合为rank均值：累计CPU/model bytes乘8可估计rank合计，但worker RSS均值不是最坏rank或cgroup峰值；需原始rank/外部采样补充。累计计数不得跨step重复相加；teacher wall均值也不是端到端墙钟。生成窗口差异、固定教师及answer-only是明确授权的Search-R1适配，最终报告须说明，不能宣称作者数学任务的原始recipe复现。

## 最终组合增量审查

独立review复核f60897e（common来源297e1c1）：20项focused CPU回归及5组DataProto评估泄漏反例通过；低占用有进程允许、超阈值/NaN拒绝、split路径无关且行改变可检测、smoke≤5与正式500保存/终评路径通过。无新增已证实代码阻断。

口径提醒：公共 `val/generated_tokens` 是保留的后处理/再编码model tokens，`val/search_calls` 是完整search标签计数；不是实际采样总token或HTTP调用成本。训练SearchRollout有raw生成/实际retrieval成本计数；正式成本报告需结合原始训练计数、外部资源采样和检索服务观测，不把验证保留量冒充完整成本。

远端最新更新bundle传输因mgmt-host连接超时失败；没有确认远端checkout更新到f60897e。下一次连接后先查询repo SHA、partial bundle和本任务tmux/logs，再执行续接，不能声称已经部署最终版本。
