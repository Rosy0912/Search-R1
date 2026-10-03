# Tree-GRPO Search-R1 验证执行记录

截至 2026-10-03。本任务仍在执行；已完成真实3B Tree rollout，但尚无成功的实际actor outer update或效果指标，不能宣称复现成功或优于 GRPO。

## 已完成

分支 `reproduce/tree-grpo`，生产代码冻结版本 `f4f457df55fbfe99691d466d5c6f1d3fd7175c65`；Slurm2873运行的是此前 `aed28e176da946dc6cff73cfb23fac5cf2a879a8`，失败后才部署新版本。冻结起点 `0723dca0c1458482ac3e78f8ed43af04893955d2`，公共部署来源 `297e1c140677dc621302f1d49fefdb31023fd2b7`，本分支公共 cherry-pick 为 `b5bd22605bdc1cbb5853efbb0a7a127e4d0c8334`。

1. 修复外层迭代重复初始化 Tree 随机数：Controller 持有持续推进的 seed42 RNG，避免每次 outer update 重放同一 parent/leaf 选择序列。
2. 继承公共部署与微批日志修复，增加实际 actor outer update 计数。零 optimizer update 或 critic warmup 跳过不能消耗预算；一次 outer 中的多个 optimizer step 只计一次。真实 driver 尾部独立控制流探针验证 500 次更新对应 global_steps501，保存实际第5/100/200/300/400/500次更新。
3. 修复分布式日志只保留首 rank 的问题：五项 detached paper objective 日志先跨 rank 平均，再交旧版 DataProto 收集器。真实两 rank CPU Gloo/DDP 测试覆盖不等长叶、空叶与空 rank，日志与完整 batch 计算一致；开关日志归约时每 rank 梯度/参数完全相同。未改变梯度目标。
4. 最新完整 CPU gate：**163 passed / 1 skipped / 4 warnings**，见 `evidence/cpu-all-step-ids-fix.txt`。独立审查与原始来源对照见 `INDEPENDENT_REVIEW.md`；不得将各次 gate 数字相加。
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

## 真实训练结果与修复

Slurm2873取得h100-9全部8张预定H100，实际完成FSDP/vLLM初始化和首轮Tree搜索rollout，随后actor更新在 `TensorDict.get('step_ids')` 缺字段时报错，退出1。Tree actor只选择自己的policy字段，不包含GiGPO专用step_ids；TensorDict0.5的无default get会抛异常。修复为显式default None，不改变GiGPO必须选择step_ids的路径或Tree目标数学。最新提交f4f457d已推送且部署；真实node9 Torch2.4.0/TensorDict0.5 CPU actor测试以及独立两rank测试通过。实际完成outer updates仍为0，不能计作两步smoke成功。

此前初始化失败已全部保留：2866 Click8.5/Ray CLI deepcopy；2867 UNIX socket路径超107bytes；2869默认worker端口与GCS重叠；2870 Torch.compile缺Python.h。修复只在ownenv/ownrun作用：Click8.2.1、唯一短symlink指向持久runtime、worker22082–23081、精确匹配3.10.12开发包解到ownroot。Slurm2873真实GPU entropy编译能力通过，未禁用compile或改算法。

Tree生产审计dc92f23已部署，下一次独立run显式启用：分支继承、共享前缀/观测mask、每题2root/每root3leaf共6叶、问题映射、action/observation step锚点及最终有效policy mask/叶等权均输出hash/count证据；验收器强制两份审计JSON每步完整，绑定rollout SHA。2873未启用此诊断，不能以它证明完整生产树结构。审计不记录原始文本，不额外抽样或改梯度；hash仍是实验数据，不能视为隐私保证。

## 检索瓶颈与辅助GPU候选

2873的实际CPU检索请求包含503.67、222.68、226.92、244.21、313.22、177.92秒等耗时，CPU检索严重限制8卡训练吞吐。FAISS CPU轮子的实际SGEMM绑定到其内置单线程OpenBLAS，OMP8不等于BLAS8。独立own OpenBLAS pthread preload小型synthetic测试在128/512行约快4倍、有限top3逐值一致，但512行CPU时间增加；尚未跑生产64GB index的完整parity，因此未部署。见 `TREE_CPU_BLAS_CANDIDATE_REVIEW.md`，该微测试不能外推真实512吞吐。

按PLAN允许的未分配辅助检索GPU，准备A800 GPU4（UUID GPU-c63282c6-1d05-224e-234b-2a8c794b1f04），在已有平台container allocation中通过公共UUID lease和occupancy复核，仅使用4号；0–3训练候补及高占用7号保持不动。ownenv复制后重建路径并移除owncopy的peer editable .pth，sys.prefix/imports验证独立；peer资产只读复用，index/corpus全SHA独立重算与h100-9完全一致。

实际单GPU FP32全index差异gate退出0：固定14 CPU embeddings，真实k3查询14/14文档/顺序一致，约0.195秒；同14条重复成512行的k3 512/512一致，约0.971秒。k50集合一致，但14条中3条同分排序变化，含tallest-building在k50 rank1/2互换；真实独立k3仍同序，不能取k50前3替代。最大score差约1.19e-6。64.56GB FP32索引、512MiB scratch，实测GPU峰值67.52GB，foreign监测无新进程。此gate仅隔离index，尚须实际A800 CPU编码+HTTP内容差异审查；未切换训练检索。见 `evidence/a800-fp32-index-difference.json` 和 `TREE_A800_INDEX_REVIEW.md`。

## 效果与成本状态

两更新smoke、五更新完整HF/tokenizer保存重载、500正式actual outer updates和51,713题最终七集合评测均未通过。F1/EM/格式率及相对新GRPO收益为未测；不能写为0分、负结果或优于GRPO。跨方法split产物内容身份实际对照尚未完成。真实生成samples/token/丢弃扩展/重复前缀成本在2873报错前未完整发布，因此未测，不能写为0。

2873 stage body3194秒，8卡为7.09778 GPU小时，包含初始化、CPU检索等待、rollout与失败更新尝试；不包含scheduler prolog/epilog，不代表500正式训练成本。CPU检索自有进程最后累计CPU时间11639.92秒；采样从加载后才开始，末次RSS在退出时为0不能当峰值，早期峰值约69.52GB。各失败作业body成本单列于 `evidence/training-failed-stage-costs.json`。基础硬件/model gate的完整预约起止未保存且Slurm accounting关闭，总GPU小时仍未知；不得报0。

辅助GPU index probe body44.7585秒约0.01243单卡GPU小时，来源env准备/平台已有闲置allocation不在此数。HTTP候选的加载、gate、服务闲置与请求都将累计到完整服务生命周期成本。微测延迟改善与总CPU/GPU成本必须分别报告。

## 仍须完成

1. 审查实际CPU编码+辅助GPU HTTP固定真实query的top3/top50、内容和分数差异，选择并冻结可用检索路径。
2. 新Slurm目录、冻结f4f457d、启用Tree审计，完成两次实际更新；验收finite loss/logprob/grad、非零真实HF参数delta和树/mask/叶等权证据。
3. 五次实际更新及完整HF/tokenizer CPU重载、有限前向验收，再放行正式500更新。阶段验证器只产生READY_FOR_INDEPENDENT_REVIEW，不能替代独立接受记录。
4. 正式500实际outer updates、完整最终七集51,713题评测，对比同协议的新GRPO；完整报告额外Tree样本与检索/训练/辅助成本。

完整actor+optimizer+RNG+dataloader精确恢复尚不支持。每阶段持久会话、唯一marker/run-id，失败保留且重新开始独立预算，不声称等价resume。所有未知/foreign越限只终止自身任务，从未停止原有小显存进程；未核对真实训练和评测之前，不宣布效果验证完成。

## 继续执行：H100 编码与辅助 GPU 索引候选

此前 A800 CPU 编码加 GPU 索引的完整 HTTP 候选在512查询测试中超时300秒，未接受、未用于训练，已仅停止本任务服务。当前候选保留 H100 原CPU E5 FP16编码、Wiki文档读取和HTTP接口，仅将原始FP32查询向量通过本任务回环隧道交给辅助GPU4 FP32 IndexFlatIP。真实二进制门禁两次重复均稳定，true-k3在14/14与512/512行文档顺序完全相同；k50分别11/14与402/512行顺序相同，其他为已记录近分数排序差异。14查询端到端约2.59–3.50秒，重复512约14.64–18.46秒；512仅有14个独立问题，不能视为独立512问题效果证据。全HTTP候选门禁及生产阶段仍待完成。

完整HTTP CPU预约2876实测exit0，top3 rank/content14/14与512/512一致且两次稳定；k50文档ID排序11/14一致、内容排序14/14一致，重复Wiki内容使部分ID换位不改变内容。512 HTTP耗时24.274/22.916秒，H100 CPU编码4.605秒（首轮），原始向量SHA与2854保存向量完全一致。CPU门禁已释放；训练接受与实际参数更新仍待独立复核及运行。

独立真实复核与严格服务绑定验收完成后，Slurm2877已启动带Tree审计的两次更新smoke，生产冻结代码f4。预约h100-9全部8H100、32CPU/256G/4h上限，原八个小显存进程保留；辅助GPU4索引为固定独立实例。检索接受receipt SHA `ec62a3a1b1d40f43f7d9e2c10e1f789311aefe31b784ca34a5a344ab64d7e9ca`。截至提交，成功outer更新仍为0；不是已完成训练验收。
