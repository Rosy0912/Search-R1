# TOP-D 执行记录与后续验收

本任务分支 `reproduce/topd`，算法冻结祖先 `0723dca0c1458482ac3e78f8ed43af04893955d2`。当前配置预览来自 `1620c2edb13809d19cc3f5af4a298ccd75488104`；只有公共部署审查冻结后才可更新训练代码。节点 `h100-11`，目录 `/dataset/shaoshitong/search-r1-rl/topd`，CPU检索18085，Ray16385。

## 已执行

- 独立本地工作树就绪；完整 CPU gate 134 passed / 1 skipped。
- focused CPU tests 29 passed；真实 tiny-Qwen teacher、actor和Gloo/DDP测试不代表真实3B效果。
- 远端目录就绪，Python 3.10.12，Linux x86_64；项目env解释器存在，缺pip/训练依赖，等待公共bootstrap。未安装系统包。
- 八张 H100 全部利用率0%、显存1135/81559 MiB，已记录UUID和原进程；未持有预约。
- 配置预览确认 alpha=.1、2 PPO epochs、teacher不含答案、全部额外KL和熵系数0、microbatch8、5轨迹、4轮、train512/mini256。预览中 PENDING 路径必须更换，不能启动。

## 必须交付的依赖

1. 同一个已审查common-setup提交，含路径适配、双15%门禁、CPU检索、可配置通信、路径无关split身份、smoke专用最终评估开关和holdout-only teacher selector。
2. 固定模型、源parquet、语料/index/E5的只读manifest，所有节点split的源SHA/content_identity/产出SHA一致。
3. 新GRPO完成500实际outer updates；从5/100/200/300/400/500 checkpoints按2%holdout平均答案F1选教师，平局取最早。不得用最终七集或旧H20训练checkpoint选教师。
4. 不可变完整HF教师目录、teacher-manifest.json和teacher-path.txt；须核训练来源、holdout指标、候选选择、完整权重分片/索引、词表/BOS/EOS匹配、与学生权重hash不同。教师成本随指标单列。

## 教师就绪后的执行顺序

Cherry-pick相同公共提交并重新CPU验收 → 项目bootstrap/环境freeze/资源清单 → 正式Slurm单节点8GPU allocation内复查UUID及双阈值 → BF16/NCCL/flash/vLLM通信与基础推理 → CPU检索top3/top50与原检索对比/健康检查 → TOP-D两实际outer updates smoke → 5步保存、验证并加载checkpoint → 500实际outer updates → 七集51713题最终评估。教师交付前不运行TOP-D算法smoke，不加载学生占用卡。

训练入口遵循 PLAN section6 的独立Ray/持久输出，在调度分配内设置下列必要覆盖：

```bash
export TOPD_TEACHER="$(cat "$AGENT_ROOT/teacher-path.txt")"
bash reproduction/baselines/run.sh topd \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  retriever.url=http://127.0.0.1:18085/retrieve
```

CPU预览不是最终resolved训练配置。实际配置须固定alpha .1、epochs2、privileged_answers=false，记录全部model/data/teacher SHA、common commit和环境。源码允许某些Hydra override；本次调用不能因此改变协议。

## 报告判据

有限loss/logprob/梯度、非零学生参数变化、冻结教师不更新、observation/pad不计loss、当前token不纳入future、同题token pooled whitening、全rollout B/(Q*Nq)权重、2epochs复用old snapshot。完整目标权重在切成固定minibatches后只是全rollout估计，不宣称每minibatch恰好实现完整逐题目标。

报告F1/EM/格式率、搜索次数、生成tokens、samples、outer updates、teacher CPU时间/RAM、retriever CPU成本、GPU小时、墙钟和显存峰值；与同协议新GRPO对照比较，并单列教师训练/评分和TOP-D多epoch成本。当前没有F1/EM实测，效果结论未验证。

不得终止其他进程、全局ray stop、修改共享只读资产或凭据。中断不宣称精确optimizer/RNG/dataloader恢复；只attach原tmux活跃实例。
