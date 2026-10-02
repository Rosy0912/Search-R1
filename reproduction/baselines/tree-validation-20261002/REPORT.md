# Tree-GRPO Search-R1 验证执行记录

截至 2026-10-03。本任务仍在执行；目前没有真实 3B 训练或效果指标，不能宣称复现成功或优于 GRPO。

## 已完成

分支 `reproduce/tree-grpo`，生产代码冻结版本 `aed28e176da946dc6cff73cfb23fac5cf2a879a8`（仅增加17位有效数字日志，指标数学不变）。冻结起点 `0723dca0c1458482ac3e78f8ed43af04893955d2`，公共部署来源 `297e1c140677dc621302f1d49fefdb31023fd2b7`，本分支公共 cherry-pick 为 `b5bd22605bdc1cbb5853efbb0a7a127e4d0c8334`。

1. 修复外层迭代重复初始化 Tree 随机数：Controller 持有持续推进的 seed42 RNG，避免每次 outer update 重放同一 parent/leaf 选择序列。
2. 继承公共部署与微批日志修复，增加实际 actor outer update 计数。零 optimizer update 或 critic warmup 跳过不能消耗预算；一次 outer 中的多个 optimizer step 只计一次。真实 driver 尾部独立控制流探针验证 500 次更新对应 global_steps501，保存实际第5/100/200/300/400/500次更新。
3. 修复分布式日志只保留首 rank 的问题：五项 detached paper objective 日志先跨 rank 平均，再交旧版 DataProto 收集器。真实两 rank CPU Gloo/DDP 测试覆盖不等长叶、空叶与空 rank，日志与完整 batch 计算一致；开关日志归约时每 rank 梯度/参数完全相同。未改变梯度目标。
4. 最新完整 CPU gate：**151 passed / 1 skipped / 4 warnings**，见 `cpu-all-dp-metrics-20261003.txt`。独立审查与原始来源对照见 `INDEPENDENT_REVIEW.md`；不得将各次 gate 数字相加。
5. h100-9 独立 checkout、固定Python3.10 venv、项目内 uv0.8.22、完整 Git 历史和固定 FlashAttention wheel 已部署/哈希验证。Slurm 短预约探针2844已完成并释放；不是当前 GPU 预约，也不是 GPU 验收。

## 实验协议

Qwen2.5-3B base、原固定 E5/Wiki18/数据/prompt、按完整问题 seed42 划分98/2且排除周期/最终评测题；四轮、不强制 final，沿用户决定保留 Search-R1 生成/训练窗口差异。2 roots、1轮、2 expansions、每 root3 leaves，共6叶；真实共享前缀、问题和树内 zscore 相加、每叶 token 均值后叶等权。相对通常5轨迹 GRPO，多出的叶、丢弃扩展和重复前缀须单列成本。

h100-9 独占8张 H100，经 Slurm 取得后重查 UUID/双15%阈值，保留原小显存进程。TP/SP1，全局 micro8（每 rank1），train512/mini256、lr1e-6、warmup .285。CPU检索端口18082，独立Ray16382。模型/长度/batch/数据/奖励未改变。

## 已实测的环境、数据与GPU基础门禁

h100-9自己的固定环境已离线安装成功，163个Linux依赖源文件全部SHA验证。核心版本、完整freeze、唯一允许的既有cuBLAS覆盖例外和零意外依赖冲突见 `evidence/environment-acceptance.json`。Pyairports构建额外需要pathlib1.0.1纯Pythonwheel，已按PyPI SHA验证，仅补构建依赖，核心协议未升级。全部目标依赖校验后已清理本次Mac中转副本；远端最终文件保留。

原始数据转换实际生成169,615训练行、51,713最终评测行。公共seed42问题分组split保留166,222训练行、3,393 holdout行，排除周期64及完整最终测试题；路径无关内容身份为 `fac19c74fe28df2016bdff12f74f035fe272bed0d65d37e9740071e674195524`，源与产物SHA见 `evidence/split-manifest.json`。其他方法真实split产物尚未逐一对照，不将本地身份生成当作跨方法核对完成。`evidence/formal-protocol-resolved.yaml`是正式配置的预备解析，不是已运行训练配置。

Slurm job2849完成并释放：8张预定UUID的H100上NCCL **FP32** all-reduce rank和为28，每卡BF16矩阵乘finite。Flash BF16在GPU0对一个shape与显式FP32 attention比对，最大误差0.0023816824。job2850完成并释放：逐文件验证固定Qwen2.5-3B base哈希后，真实vLLM TP1在GPU0生成16token，耗时8.4415秒、Torch GPU0峰值allocated8.37GB。日志/记录见 `evidence/hardware-*`、`evidence/model-*`。这些门禁不能替代8卡3B FSDP/offload、Tree搜索、反向或正式效果验证。

CPU E5 FP16实际编码2条固定query通过，输出2×768且finite，编码0.0335秒；只覆盖编码，不覆盖FAISS/Wiki内容/排序。独立GPU FAISS wheel安装在自己的 `retriever-gpu-lib` overlay，已验证可导入GPU API；生产trainer环境仍保留CPU FAISS。overlay wheel SHA为 `7ea5e6aa82897fd0c1af3b1decc9c8659a4398fed42de1d7746715b64bbf9bf9`。

## 资产与真实检索测量

通过既有集群HTTP代理在目标直接下载原固定revision，无全局VPN/SSH配置、凭证改动。模型、E5和数据已校验；Wiki18压缩语料和两个索引分片已从校验过的同集群只读源复制并在目标再次完整校验。TLS中断后保留aria2分段状态，改为有限续传监督，每次重用已验证文件；稀疏partial的文件表观大小不代表已下载完成。

此前h100-9自有tmux `tree-assets-supervised-20261003` 执行有限重试；随后只读核对h100-11 TOPD独立目录的三份完整Wiki文件，实际源SHA全部符合原manifest。现改为head的 `tree-peer-assets-20261003` 只读转发到Tree独立目录，目标streaming size/SHA再核验后才发布；源目录未修改。自有外网下载进程已停止，所有失败partial保留。`tree-derived-assets-hashcheck-20261003` 等全部文件哈希通过，再对实际源分片SHA复核并拼接index、校验compressed SHA并gzip解码语料。status原子发布、旧JSON写入竞态兼容和损坏源拒绝经过独立复审及4项toy验证。已实际派生index64,559,075,373字节，SHA `69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166`；解压corpus14,393,579,520字节，SHA `85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9`。见 `evidence/derived-assets.json`。派生ready只证明源/输出文件。

head的自有tmux `tree-retrieval-wait-20261003` 等派生ready后仅派发一次Slurm h100-9独占8GPU、32CPU/256G/2h真实检索测量。等待期间没有GPU预约；资产等待deadline24h（允许当前SSH迭代的少量越界，scheduler排队另计），flock/marker拒绝重复派发。失败保留日志和marker，不自动重复GPU作业。

Slurm2854现已实际取得并运行检索测量，固定8UUID/占用preflight通过。CPU路径完整加载FAISS/Wiki21,015,324条，14query top3+top50两轮结果稳定；load73.95s、search36.75/22.54s、self peak RAM67.48GB。该14条小batch不足以推断512条或完整训练吞吐。GPU比较与独立差异审查已完成，作业退出0并释放。14/14查询top3 ID/顺序/内容一致，top50有9/14重排；固定CPU embeddings的GPU index对照仅1条同分排序交换。仅接受有限Tree top3 sanity，不能外推全查询或top50等价。

检索沿生产DenseRetriever，固定14条真实query，top3/top50、重复2次、E5 FP16；分别运行生产CPU FAISS/CPU encoder、独立overlay的GPU encoder/CPU index、8GPU FP16 shard，以及用保存CPU embedding运行GPU index。逐query记录ID/排序/score/内容SHA、实际FAISS导入路径、完整index/corpus SHA和资源。不同backend/build、encoder与索引量化或tie ordering都可能影响结果，ablation只区分路径，不先断言差异原因。**只输出差异报告，需审查真实结果后才放行训练。** 历史H20 FP32 top3 overlap .976不能替代本次FP16验收。

## 效果与成本状态

真实Tree-GRPO outer updates、训练samples和训练GPU小时目前仍为0；2更新smoke、5更新checkpoint导出/加载、500正式更新和51,713题最终评测均未完成，F1/EM/格式率等为未测，不能写成0分、负结果或优于GRPO。

基础硬件/模型作业已经消耗GPU资源，Slurm accounting关闭且未保存其完整分配起止，故总GPU小时目前未知，不能报0。模型推理的8.4415秒不是整项Slurm预约时长。后续检索wrapper记录gate body起止、8卡数量和退出码，明确不包含scheduler prolog/epilog；Torch allocator峰值不包含GPU FAISS分配，还需联合显存快照判断。

## 后续必须完成

1. 完整固定Wiki资产、派生哈希和真实检索差异审查/吞吐/RAM；差异影响实际观察内容时按PLAN处理，不能用平均overlap静默放行。
2. 新独立Slurm运行目录完成两次实际更新smoke，核对finite loss/logprob/grad、nonzero真实参数delta、mask/shared prefix/每题6叶和实际split。
3. 五次实际更新保存及完整HF/tokenizer重载、验证前向/参数；每个GPU阶段独立作业/cgroup保留原小显存进程，不用global ray stop；资源变化只停止自身作业。
4. 正式500实际outer更新，保存5/100/200/300/400/500和完整七集51,713题评测，对比Agent1同协议的新GRPO。报告F1/EM/格式、搜索/token/sample、GPU/CPU/墙钟/RAM/峰值及Tree多叶/丢弃扩展/重复前缀成本。

完整actor/optimizer/RNG/dataloader精确续跑尚不支持；断连重新连接同一进程，不能另启动并声称等价恢复。未核对真实训练和checkpoint之前，不宣布效果验证完成。

CPU checkpoint验收器已独立复核：完整HF加载信息缺失/不兼容拒绝、非零base参数delta、tokenizer兼容和finite CPU前向；tiny正例/未更新/缺权重/tied共4场景正确，不替代真实五步checkpoint。训练launcher/资源守卫已准备与复核；Slurm2866于2026-10-03 05:23启动两更新smoke，目前尚无完成更新记录；守卫NVML外国显存/SM峰值、完整cgroup job segment及父shell双服务存活检查，未知/超15%只停止自身作业，保守峰值和过期owner可能造成提前停止。

检索复核详情见 `TREE_RETRIEVAL_REVIEW_2854.md` 与真实measurement JSON。两更新作业采用新目录 `/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-two-2866`，冻结代码aed28e1；接受记录将真实检索证据SHA绑定。后续五步/正式训练仍须真实产物验收，未自动放行。

Slurm2866检索服务就绪后，在Ray CLI导入alias的deepcopy阶段因Click8.5 Sentinel报错；退出1，78秒stage body对应8卡0.17333 GPU小时，不含scheduler prolog/epilog。没有训练更新。只在Tree ownenv将未固定的CLI传递依赖Click固定8.2.1（PyPI SHA61a3265b914e850b85317d0b3109c7f8cd35a670f963866005d6ef1d5175a12b），真实ray --version2.40.0通过，依赖检查仍仅有原允许cuBLAS覆盖。Ray官方issue56747实报8.3，对8.5同栈的归因由本次实测支持。新独立目录Slurm2867已启动，保留旧失败marker/日志，不计失败作业为训练成功。

Slurm2867在Ray创建AF_UNIX plasma socket时报路径超107字节，退出1且实际更新0。独立复核后用唯一 `/tmp/sst-tree-SLURM_JOB_ID` symlink指向原持久RUN/runtime，使socket拼写缩至86字节；-e/-L检查与GNU ln-sT拒绝覆盖，算法和产物路径不变。失败保留，不宣称任何smoke通过。

Slurm2869在RayParams发现默认worker10002..19999与计划GCS16382重叠时报错，退出1且实际更新0。仅将worker范围设22082..23081，保留计划GCS/检索端口；Ray2.40官方tag CLI与范围已独立核验，实际bind仍需新作业验证。
