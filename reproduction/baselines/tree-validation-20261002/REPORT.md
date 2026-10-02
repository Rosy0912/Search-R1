# Tree-GRPO Search-R1 验证执行记录

截至 2026-10-03。本任务仍在执行；目前没有真实 3B 训练或效果指标，不能宣称复现成功或优于 GRPO。

## 已完成

分支 `reproduce/tree-grpo`，生产代码审查版本 `c7091e792d930c6e2488c395a1dbd324ce0eec73`。冻结起点 `0723dca0c1458482ac3e78f8ed43af04893955d2`，公共部署来源 `297e1c140677dc621302f1d49fefdb31023fd2b7`，本分支公共 cherry-pick 为 `b5bd22605bdc1cbb5853efbb0a7a127e4d0c8334`。

1. 修复外层迭代重复初始化 Tree 随机数：Controller 持有持续推进的 seed42 RNG，避免每次 outer update 重放同一 parent/leaf 选择序列。
2. 继承公共部署与微批日志修复，增加实际 actor outer update 计数。零 optimizer update 或 critic warmup 跳过不能消耗预算；一次 outer 中的多个 optimizer step 只计一次。真实 driver 尾部独立控制流探针验证 500 次更新对应 global_steps501，保存实际第5/100/200/300/400/500次更新。
3. 修复分布式日志只保留首 rank 的问题：五项 detached paper objective 日志先跨 rank 平均，再交旧版 DataProto 收集器。真实两 rank CPU Gloo/DDP 测试覆盖不等长叶、空叶与空 rank，日志与完整 batch 计算一致；开关日志归约时每 rank 梯度/参数完全相同。未改变梯度目标。
4. 最新完整 CPU gate：**151 passed / 1 skipped / 4 warnings**，见 `cpu-all-dp-metrics-20261003.txt`。独立审查与原始来源对照见 `INDEPENDENT_REVIEW.md`；不得将各次 gate 数字相加。
5. h100-9 独立 checkout、Python3.10 venv 骨架、项目内 uv0.8.22、完整 Git 历史和固定 FlashAttention wheel 已部署/哈希验证。Slurm 短预约探针2844已完成并释放；不是当前 GPU 预约，也不是 GPU 验收。

## 实验协议

Qwen2.5-3B base、原固定 E5/Wiki18/数据/prompt、按完整问题 seed42 划分98/2且排除周期/最终评测题；四轮、不强制 final，沿用户决定保留 Search-R1 生成/训练窗口差异。2 roots、1轮、2 expansions、每 root3 leaves，共6叶；真实共享前缀、问题和树内 zscore 相加、每叶 token 均值后叶等权。相对通常5轨迹 GRPO，多出的叶、丢弃扩展和重复前缀须单列成本。

h100-9 独占8张 H100，经 Slurm 取得后重查 UUID/双15%阈值，保留原小显存进程。TP/SP1，全局 micro8（每 rank1），train512/mini256、lr1e-6、warmup .285。CPU检索端口18082，独立Ray16382。模型/长度/batch/数据/奖励未改变。

## 远端准备进展与限制

SSH已通过既有本地 SOCKS 路由恢复；临时设置未改全局配置、密钥或凭证。先尝试目标下载，之后建立只绑定 loopback、仅允许固定 manifest 路径的原H20只读源，源训练/检索进程未动；复制文件仍按原 manifest 验证，未通过者保留 partial。随后确认 h100-9 可用既有集群 HTTP proxy `10.33.0.16:3128`，固定 wiki32MiB Range 实测约1.03MB/s，切换为目标端直接下载。

资产/依赖下载各用自有 tmux、独立目录和日志；不持有训练 GPU。Linux wheel 按固定版本、Python3.10/x86_64兼容标签与 PyPI SHA256 选择；固定 Flash wheel SHA256 为 `a8719a86d0cd12a105739151f9e45ac8c2ecb840a83ed7a080a88d1645a98440`。大文件、完整环境和路径无关 split identity 尚未验收。已发现 aria2 HTTPS proxy 环境优先级导致首轮大文件连接失败，正在用显式 HTTPS proxy 重试；此问题没有改动算法。

未获得生产 GPU NCCL/BF16/Flash/vLLM 检查、真实 CPU/GPU检索 top3/top50 一致性、两更新 smoke、五更新 checkpoint 重加载、500更新正式训练或51,713题最终评估。CPU Gloo测试不能替代 NCCL/GPU验收。当前生产 outer updates、训练 samples、GPU小时与已评测题数均为0；F1/EM/格式率/峰值显存等为未测，不能填写0分或负结果。

## 下一步

1. 下载完整固定资产/依赖并核对 SHA，离线 bootstrap，保存环境 freeze；生成公共数据与split，核对路径无关身份。
2. 自有 Slurm 独占8 GPU，检查 NCCL/BF16/Flash/vLLM及真实检索一致性/吞吐/RAM。
3. 新运行目录完成两次实际更新 smoke、五更新checkpoint导出/加载验收；不绕过 GPU gate，不静默减小实验预算。
4. 正式500实际更新和七集合51,713题评测，与 Agent1 同协议新GRPO比较；记录EM/F1/格式、搜索/token/sample、GPU小时、检索CPU/RAM、墙钟、显存峰值及Tree额外成本。

完整 actor/optimizer/RNG/dataloader 精确续跑尚不支持；断连应重新连接同一进程，不能另启动“等价恢复”任务。未经实际日志和checkpoint核对，不宣布效果验证完成。
