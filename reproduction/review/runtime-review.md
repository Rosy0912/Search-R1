# Search-R1 运行适配独立审查

本报告保留初审事实与修复历程。最新独立复查结论见文末“发布前复查”，不应把初审 findings 当成当前未修状态。

**最终结论（2026-10-02）：当前审查范围内的代码发布阻断已解决，可提交为受审查的 Search-R1 复现实现。40项CPU回归已在真实目标环境通过；完整训练指标尚未产出，不能声称论文性能复现完成。**

审查时间：2026-10-02。范围是项目运行、离线数据、资源完整性、环境和验收门禁；不代替另一位 reviewer 对算法及论文配方的审查。初审对象为固定上游提交 `598e61bd1d36895726d28a8d06b3a15bed19f5d3` 与本次本地适配脚本。只读远端检查，没有额外运行 GPU 计算或停止任务。发现问题后，仅应主 agent 指定修改了最终 checkpoint 保存条件及其 CPU 回归测试。

**结论：环境适配在已有实测范围内合理，但初审版本不能作为完整、可重建的论文复现成品发布。** 必须补齐环境重建、资源内容验证和终止验收。短训练证明工程链路与权重更新，不证明最终 EM 或检索数值等价。各项修复由主 agent 负责集成；本报告未把尚未复审的修改自动视为已解决。

## 可确认的部分

- 初审时远端 Git 工作区干净，说明 Torch/cuBLAS/NCCL 的兼容处理没有私自修改上游 Python 算法。`prepare_data.py`、`run_pipeline.sh`、`train_full.sh`、`gpu_guard.py`、`download_assets.py`、`retriever_local.py` 六文件的远端 SHA256 与本地逐一相等。
- NQ 转换逐字段保留上游输入：同一 `make_prefix(..., 'base')`、去两端空白并补问号、`golden_answers` 全列表、`data_source=nq`、`ability=fact-reasoning`、reward rule/target、split/index。空问题处理由上游索引最后字符改成 `endswith`，对正常 NQ 没有语义变化。上游 dataset.map 会保留源列，本地只写训练需要的字段；`RLHFDataset` 消费的是 prompt/reward/extra_info，未发现源列删除造成奖励信息丢失。没有把答案拼入 prompt。
- `train_full.sh` 在初审中逐参数对应上游根目录 `train_grpo.sh`，主要差异是修复未定义的数据路径变量、改成本地模型路径、console 日志、缓存目录、兼容环境变量。此结论是“对应根目录 quickstart”，不能据此声称对应论文最终 NQ+HotpotQA 的全部设置。
- 模型/NQ/索引使用固定 HF revision。大文件具有官方 LFS SHA256；成功时从 `.partial` 原子替换。失败状态不会写 complete=true。HF/数据集 offline 开关、模型本地路径与 `/data` 缓存适合此服务器网络及根盘限制。
- cuBLAS 12.4.5.8 override 的依据不只有导入：本地 GDB 证据显示 SIGFPE 位于 `cublasLtMatmulAlgoGetHeuristic` 的 BF16 线性运算路径，之后真实生成、8 卡短训练、参数变化均已通过。`NCCL_NVLS_ENABLE=0` 后 8 卡 all-reduce 与 BF16 检查成功。这些兼容调整不改变 GRPO 奖励、组归一化或 observation mask；浮点执行仍不应承诺逐 bit 一致。
- CPU FAISS 分支本来就是上游支持的路径；E5 的 query 前缀、mean pooling、normalize、top-k3、原始 Wiki18 索引及语料映射沿用上游。只将 uvicorn 绑定改为 localhost，HTTP JSON 协议不变。

## Findings

| 优先级 | 初审问题与证据 | 实际影响与必要处置 |
|---|---|---|
| P1，发布门禁 | 远端 `bootstrap.sh` 不创建 venv；只安装 Torch2.4 默认 cuBLAS12.1，没有 cuBLAS12.4.5.8 override；本地没有完整 bootstrap。`install_flash.sh` 才补 setuptools70.3/FlashAttention，但需要已存在 wheel，并不记录完整取得流程。 | 当前机器是逐次修复成功，新机器不能靠提交内容重建。须发布可执行安装顺序、全部必要 pins、Flash wheel 来源/SHA、明确使用 `--no-deps` 的依赖覆盖例外。完整 freeze 不应被当成直接可由 resolver 安装的 lock。 |
| P1，结果可复评 | 上游 `ray_trainer.py` 仅按 save_freq 保存；1005 计数上限实际最后更新1004，save_freq100只留下1000，随后最终EM来自1004权重。 | 终止 EM 与最后保存模型不一致。已完成最小 final-save 修复，见下节；需要部署被审查版本并记录新 checkpoint 和 EM 的对应关系。 |
| P1，资源/评估完整性 | 初始 `assets.json` 23 项中只有8项有 SHA256，其余15项包括 NQ test、config、tokenizer 仅大小检查。 | 从第三方 mirror 得到等长错误内容不会发现。“全部官方哈希通过”描述不准确。官方 manifests 已含 Git blobId，应补普通文件的 Git blob SHA1 校验或由 canonical 文件计算可信 SHA256。 |
| P2，资源身份 | `gpu_guard.py` 无条件把 `retriever.pid` 的裸 PID 从 GPU 进程中排除；pidfile 不在退出时清除。EXIT trap 也只保存数字 PID。 | 文件陈旧且 PID 复用时可能豁免别人的进程；退出时若原 child 已消失/被回收也有误杀风险。须绑定 PID+启动时刻+实际命令身份，清理自己 pidfile；kill 前核对身份。 |
| P2，验收假阳性 | `run_pipeline.sh` 67行只检查 step1目录存在，不运行 `verify_checkpoint.py`，目录可能陈旧或分片残缺；最后 bash返回0就写FINISHED。 | 当前 step2抽查证据有效，但自动门禁不足。须验证 fresh run、全部分片/index/config、有限数值、第二步实际变化，最终判定必须包含最后 checkpoint、最终EM日志和计数。 |
| P2，下载恢复 | SHA mismatch 的满长 `.partial` 留在原名；后续 curl `-C -`/aria2续传不能自行修复完整坏文件。 | 正确阻止使用坏资源，但重试可能永久失败。须将坏副本隔离并清除续传状态，在重新取得正确文件后再原子替换。 |
| P2，检索口径 | 官方 launch用GPU float16 FAISS，本项目CPU float32；真实 retrieval-test 只检查返回3条。 | 两者都是同一 Flat 检索算法，但 float16 舍入可能改变近分值 top-k；CPU吞吐也可能限制正式rollout。必须明确这是数值/实现差异，不能未经对照声称检索逐结果相同或论文性能已复现。未来用固定问题集比较文档ID/top-k及吞吐；不应为审查抢用当前训练GPU。 |
| P2，入口锁 | flock 只在 pipeline 外层，手动直接运行 train_full.sh/train_smoke.sh 不经过锁或 GPU guard。 | 按现有README可安全查看tmux，但直接训练脚本可重复启动。须明确唯一受控启动入口，或把锁/身份检查封装到统一启动器；避免内外层重复锁造成死锁。 |

P1 指必须在宣告完整可复现、发布结果前解决的问题，不表示当前权重更新一定错误。P2 也应在交付自动控制脚本前修复或明确限制。

## 已完成的最小最终保存修复

修改 `verl/trainer/ppo/ray_trainer.py` 的保存条件：`save_freq > 0` 时满足原周期边界 **或** `global_steps + 1 >= total_training_steps` 即保存。调用仍在最后更新后、计数递增前，1005 上限对应文件名 `global_step_1004`；最终验证日志计数1005对应同一份1004更新后的权重。没有改变优化步数、学习率调度、奖励、loss mask、生成、参考策略或验证行为。`save_freq=0` 的显式禁用保持不变。

CPU测试 `tests/reproduction/test_final_checkpoint.py` 用 AST 直接抽取真实 `fit()` 的 update→save→counter→final-validation 末尾控制流；仅 RPC worker/metrics/timer 用CPU doubles，不导入或调用 CUDA/Ray。四项通过：非周期末步保存并与最终EM共享权重、周期末步只保存一次、非末步非周期不新增保存、显式禁用仍禁用。另对内存中恢复的原保存 predicate 做 negative control，第一项明确失败，能检出原缺陷。此测试验证控制流，不能替代真实分布式 checkpoint 载入和数值验收。

命令：`python3 tests/reproduction/test_final_checkpoint.py`；4 tests OK。补丁需主 agent 集成后同步到远端。初审时正式任务还在初始验证，主 agent 随后已停自己的 quickstart trainer 做配方修正；本报告没有执行停止。

## 证据及边界

固定上游代码：[Search-R1 commit](https://github.com/PeterGriffinJin/Search-R1/tree/598e61bd1d36895726d28a8d06b3a15bed19f5d3)。本地 `evidence/model-smoke-gdb.log`、`model-smoke-fixed.log`、`env-cuda-preflight.log`、`search-r1-smoke.log`、`checkpoint-update.json` 与 `environment-freeze.txt` 是采集快照。cuBLAS冲突/兼容资料已有主记录的[PyTorch H20 issue](https://github.com/pytorch/pytorch/issues/152017)；[NCCL NVLS变量文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2205/user-guide/docs/env.html)说明开关作用。

没有在繁忙服务器重复做 GPU 推理、对照 FAISS GPU 检索或全资源76GB重新哈希；下载成功日志不能取代尚缺的15个普通文件内容校验。本审查不承诺不存在其它错误，不把已启动正式任务等同于论文指标复现。最终论文性能需要明确配方、完整训练和固定checkpoint的正式评估。

## 环境重建补充

应主 agent 指定，随后已新增本地 `bootstrap.sh`、`requirements-h20.txt`、`constraints-h20.txt`、`check_environment.py` 与 `environment-setup.md`。23项直接需求与155项活跃依赖闭包由目标隔离venv的元数据取得，排除与这些依赖无关的系统包；cuBLAS从resolver constraints中排除，以最后no-deps override明确处理。脚本补齐venv创建、固定安装顺序、Flash wheel官方SHA核验/失败保留/离线优先、项目锁、来源记录和严格唯一声明冲突验收。对现有venv仅执行元数据审查，确认为唯一预期Torch/cuBLAS冲突，无其它缺依赖或冲突；shell/Python静态检查通过。**没有执行新环境clean rebuild、GPU计算或远端安装**，发布前须将这一边界保留，首次重建运行仍要通过实际集成门禁。

## 发布前复查

2026-10-02对主 agent 后续适配版本再次独立静态检查：

- `process_identity.py`/`gpu_guard.py` 现在核对PID、proc start_ticks、uid、argv及retriever脚本身份，不再裸PID豁免。
- 新增资源后manifest共30项，全部明确SHA256或Git blob SHA1；`verify_assets.py` 全部逐内容验收。下载器隔离坏满长partial和aria2状态后重新下载，并保留失败副本。
- `prepare_paper_data.py` 使用NQ+HotpotQA、全局唯一prompt index、7源评估，并写dataset-audit。训练脚本明确top_p1、force_final_answer=false、max_turns4，计数上限501产生500次update；int(501×0.285)=142与按500计算的142相同。
- pipeline默认在 reviewed-smoke 后停止，只有显式 `SEARCH_R1_FULL_RUN=1` 才跑paper-GRPO；短训练不被自动升级成“论文指标复现”。
- 环境checker已将版本比较从错误的base_version改为public，保留postrelease版本；主 agent 报告真实clean-env bootstrap安装成功、唯一cuBLAS声明例外，额外GPU集成验收仍以其实际日志为准。此 reviewer 没有重新执行安装或GPU计算。

新增CPU回归 `repo/tests/reproduction/test_runtime_adaptations.py`：10项；Mac9通过、1项真实Linuxprocfs跳过。覆盖身份精确匹配、PID/starttime/uid/args变化拒绝、malformed/missing fail-closed、Git blob前缀、同长度内容篡改、坏满长partial及aria状态保留、网络失败不提升最终文件、全部manifest存在内容digest。测试不联网、不调用nvidia-smi、不执行GPU。远端可设置 `SEARCH_R1_ADAPTATION_DIR=/data/search-r1` 再运行，补上procfs检查。

**当前判定：未发现新增的论文算法阻断；资源校验、环境重建和身份保护的主要修复合理。** 本轮仍报告两项验收缺口供主 agent 收口：smoke应拒绝上一轮遗留step2并明确本轮产物时间/identity；正式阶段不应仅凭train脚本exit0写FINISHED，应核验global_step500完整权重与最终7源EM。发布代码可以清楚注明复现进行中，不能把尚未完成的完整训练/固定checkpoint复评当成既成结果。

两个额外P2运行边界：stop的身份check→os.kill仍有极窄PID复用窗口，可用Linux pidfd；retriever的EXIT清理trap在identity record成功后才安装，record失败可能漏掉新子进程，应提前建立可靠清理。CPU float32与官方GPU float16检索的排序/吞吐对照仍未执行，必须保留差异说明。后续由主 agent 的最终集成验收报告确定这些项是否已解决；本报告不自动假定并发修改已完成。

## 最终收口与发布判定

主 agent 最后修改已独立逐项静态复查：

- smoke分片与index必须在本轮 `SEARCH_R1_RUN_STARTED` 之后写入、非空存在，并要求抽查参数真实变化；不再仅看目录存在。
- full必须有fresh `global_step_500` 全部映射分片，逐分片用真实safe_open核对header key集合，要求最终7源evaluated_count与dataset-audit全量一致、EM范围[0,1]。只有验收脚本成功才写FINISHED。
- 复查发现新的真实阻断：上游pprint将最终metrics打印为带引号的多行字符串，不能逐行literal_eval。主 agent 已修复为保留人读pprint并额外输出单行 `SEARCH_R1_FINAL_METRICS_JSON=`，verifier仅按JSON解析；该修复只改变日志。已检查producer和consumer实际代码一致。
- retriever stop使用pidfd并再次确认身份，避免PID复用窗口；EXIT trap在identity record之前安装，record另有短暂exec身份重试。

新增 `test_acceptance_gates.py` 的11个CPU测试实际执行整个验收helper，仅AST替换项目路径，使用真实小型safetensors，不替换校验算法。涵盖新鲜checkpoint通过、旧分片拒绝、不变权重拒绝、最终分片缺失/损坏/陈旧拒绝、缺少最终marker/某源metric或不完整evaluated_count/无效EM拒绝。

主 agent 统一在目标实际环境报告：**40 tests全部通过（0skip）**，含真实DataProto、Linuxprocfs与11个实际CPU safetensors门禁；另已真实完成fresh bootstrap、fresh8卡NCCL/BF16、真实Qwen生成验收。此 reviewer 未重复占用GPU。mixed8卡smoke已到step1并产生checkpoint、非零梯度/奖励，9行验证尾部均计数；step2及完整正式任务以运行记录的后续实际证据为准，不据此提前断言完成。

`turns_stats` 原始计数从1开始会使显示动作数多1；主 agent 将仅将统计初值改0并补回归断言。静态调用链确认该字段用于日志动作数/有效动作比例，不决定生成循环或优化；真实worker调用次数已有测试约束。统计修复后应重新跑相应CPU回归，不为修日志指标重启已有训练。

**发布判定：审查范围内未剩余代码发布阻断，允许发布已审查实现与明确的复现进度/兼容差异。** 这不是“保证不存在任何错误”，也不是500步论文性能验收。CPU float32检索相对官方GPU float16排序差异仍须保留说明；完整训练结果须通过新增full验收门禁及固定checkpoint复评后才能报告。更广泛的硬件迁移或依赖变更需要重新运行实测集成门禁。
