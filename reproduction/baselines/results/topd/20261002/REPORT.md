# Agent 5：TOP-D / Search Agentic RL 验证记录

更新时间：2026-10-02（北京时间）。**CPU 准备与独立审查已完成，真实效果未验证。** 本轮新 GRPO 选出的教师和公共部署冻结提交尚未交付；没有启动 TOP-D GPU 训练，没有预留 GPU。

| 项目 | 实测结果 |
|---|---|
| 完整 CPU gate | 134 passed / 1 skipped，17.81 秒 |
| 算法 unittest | 44 项通过 |
| reproduction unittest | 61 项，1 跳过，其余通过 |
| 聚焦 CPU 测试 | 29 项通过，9.33 秒 |
| 独立审查 | 核心数学有条件通过；另有 4 项定向 CPU 检查通过 |
| h100-11 GPU 快照 | 8 张 H100 80GB，全部 0% 利用率、1135/81559 MiB；原进程未改 |
| 远端准备 | 独立目录可写，Python 3.10.12；env 解释器存在，依赖未安装 |
| 端口 | 18085 / 16385 当次未监听 |
| Slurm | gpu partition、h100-11 IDLE；未预约 |
| F1 / EM / 格式率 | 未测，不能填 0 或推算 |
| 500 更新 / 51,713 题最终评估 | 未开始 |

CPU 验收使用本机已有 Python 3.9 环境，不能充当远端 Linux Python 3.10/CUDA 固定环境验收。tiny Qwen 和 Gloo/DDP 测试验证部分数值与控制流程，不能替代真实 3B 搜索训练。

## 版本和部署

分支 `reproduce/topd`，本地独立工作树 `work/topd`。当前实现提交 `1620c2edb13809d19cc3f5af4a298ccd75488104`，算法祖先 `0723dca0c1458482ac3e78f8ed43af04893955d2`。后续必须接入公共 owner 审查冻结的同一个提交；本次没有修改核心算法或其他分支。

远端根目录 `/dataset/shaoshitong/search-r1-rl/topd`，HTTP 18085、Ray 16385。系统缺 ensurepip，已保留隔离 env 解释器供公共 uv bootstrap；没有安装系统包。首次完整 Git bundle 上传超时，尚未验证代码部署，本地副本保留供重传。

`topd-config-preview.yaml` 由实际 launcher 的配置预览生成，确认 alpha=.1、2 PPO epochs、privileged_answers=false、额外 KL/熵系数 0、microbatch 8、5 轨迹、4 轮、train512/mini256。PENDING 教师/输出路径仅用于 CPU 配置预览，禁止作为训练配置启动；实际训练须另存 resolved 配置。

## 数学与来源

当前 logaddexp proximal reward、排除当前 token 的未来有效动作 token 均值、同题 pooled whitening、全 rollout `B/(Q*Nq)` 权重、固定 old snapshot 的两轮 PPO 已通过静态/CPU 检查。历史全 token 跨题权重和 masked 溢出问题已修复。固定 minibatch 是完整 rollout 逐题目标的估计，不宣称每批恰好等同完整问题目标。

详见 `topd-independent-review.md`，对照一手论文 [Trust Region Policy Distillation v1](https://arxiv.org/html/2607.04751v1)。没有核实官方作者代码 commit，不声称源码级复现。Search-R1 检索 MDP、Qwen2.5-3B base、同规模已训练 GRPO 教师属于本次适配。生成窗口与评分上下文差异按已授权计划保留，生产须量化报告。

## 继续执行的依赖

1. 公共部署审查冻结提交：路径适配、双 15% 门禁、CPU 检索、实际卡数通信、smoke 评估开关、holdout-only 教师选择器。
2. 固定模型/数据/语料/index/E5 manifest；跨节点源 SHA、路径无关 split content_identity、产出 SHA 一致。
3. 新 GRPO 完成 500 次实际 outer update；从 update 5/100/200/300/400/500 的检查点仅按 2% holdout 平均答案 F1 选教师，平局选最早。
4. 不可变完整教师 HF 目录与 teacher-manifest：词表/BOS/EOS 兼容、完整分片/索引、学生教师权重 hash 不同、训练来源/holdout 指标和训练成本。

这些依赖尚未交付。PLAN 禁止教师交付前运行 TOP-D 算法 smoke，因此没有拿 base 或 Instruct 教师替代，也未加载学生占用 GPU。

教师就绪后按 `TOPD-RUNBOOK.md` 继续：公共提交/环境验收 → Slurm allocation 和 UUID 门禁 → 通用硬件/检索检查 → 两次实际 outer update smoke → 5 步 checkpoint 保存/加载 → 500 次实际 outer update → 51,713 题最终评估 → 与同协议新 GRPO 比较。

报告必须包含教师 CPU 时间/RAM及训练/选择成本、检索 CPU 成本、两轮 PPO 额外更新、搜索/生成 token/samples/outer updates、GPU 小时/墙钟/显存峰值。当前没有生产实测成本。教师串行 CPU FP32 评分，仅词表 head 分块，并不限制完整 transformer forward 峰值，须按实际 Slurm 配额验收。

报告和 STATE 同步到用户指定 `remote_codex_control/research/search-r1-baselines/results/topd/`。日志、配置预览、执行手册和独立审查保留在本线程 outputs。**目前不能判断 TOP-D 是否提升 Search agentic RL 效果。**
