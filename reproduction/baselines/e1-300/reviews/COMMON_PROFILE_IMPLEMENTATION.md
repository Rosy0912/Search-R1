# Search-E1 纯 OPSD 300：共同数据消费者与固定入口

2026-10-04，本地隔离 checkout `work/opsd-e1-300`，分支 `codex/opsd-e1-300`。本次仅 CPU 实现与测试；没有提交 Git、远端作业或运行 GPU。其它 agent 同时负责 state/gate/worker/数学状态接口，本文件不把它们的未验收运行继承为通过。

## 新文件与接入

- `search_r1/baselines/sharedholdout.py`：从 BAPO 最新 `bapo-e1-300-countfix/search_r1/baselines/holdout.py` 的 prebuilt 消费路径移植，依赖旧 `holdout` 的无变化序列化/哈希工具。旧 `holdout.py` 仅增加明确 `input_mode=prebuilt_manifest` 委托；legacy split 不变，未知 mode 拒绝。
- `verl/trainer/config/ppo_opsd_e1_300.yaml`：独立配置，无旧 Base 模型/旧重划数据继承。
- `search_r1/baselines/e1_profile.py`：真实 Hydra compose 后的固定条件检查、官方原 manifest+全部模型文件校验、外部 binding/public-key JSON SHA 校验。
- `reproduction/baselines/run_opsd_e1_300.sh`：`--cfg` CPU 检查，`--validate` CPU 全资产校验，`--run` 仅明确八卡 H100 本节点现有 Slurm allocation；不接受任意 Hydra override，不调用会覆盖微批的旧通用 run.sh。训练 stdout/stderr tee 到唯一 own run_root/train.log，pipefail 保留失败状态；生产 preflight 负责 UUID/cgroup/资源/lease，与传入 inherited lease 共存。

## 共用数据完全消费，不再划分

原始共同 manifest 外部固定 SHA `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`，seed42、protocol search-e1-common-300。实际既有产物 train169028、dev512(NQ/Hotpot各256)、stage64(各32)、final51713(七来源)。消费者不写数据、不建缓存、不重划、不抽样。

每个产物核：原文件 SHA、行数、列、来源与 source:split:index ID 契约、ID唯一/顺序 SHA、全行和完整 prompt 身份 SHA、canonical question SHA、共同原 instruction/单user消息、非空答案列表。stage/dev canonical 问题独立；四split所有两两 ID/canonical question 交集必须0。内部 numpy parquet 结构用既有 plain 工具规范化。路径必须数据根内相对路径，越界/外部 symlink 拒绝。

raw provenance 保留严格九份原始 source 文件 SHA，覆盖两train+七final，preparation policy/源码哈希/来源序；新增 exclusions-audit.json 原 SHA 核验；原 manifest 重复 JSON key 拒绝。开始/结束 manifest和产物再次哈希，`assert_prebuilt_unchanged` 供训练结束状态审查调用。禁止 `train_data_num/val_data_num` 子采样。stage路径写入 **opsd_e1.stage_files**，不使用 BAPO stage状态或算法。

因此部署必须包含 manifest、四parquet、exclusions-audit.json、raw/九文件。h100-9 源四parquet既有 SHA 通过，不代表 h100-10 全部署/内容校验已通过。新消费者实际 h100-10 核验结果才是证据。

## 官方模型与固定训练条件

官方原模型 manifest SHA `7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44`，repo Qwen/Qwen2.5-3B-Instruct，revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`，实际11条文件；逐条 bytes/SHA/revision 与根内路径核对。actor 和 frozen teacher 同一官方初始目录；不会指向训练后HF5。额外 root普通文件仅允许 `VERIFIED` 标记及真实 manifest_path 正好位于模型根直属目录、且原始字节已经精确官方manifest SHA核验的那个实际文件名；其它一律拒绝，避免未manifest tokenizer/weights改变加载行为；**VERIFIED标记只算库存元数据，不是模型或训练验收证据**。不能把旧13文件叙述、任意 self-reported passed 或其它目录模型哈希继承到本实验。

512问题×n_agent5/n1=2560轨迹；global mini128，每rank16；globalmicro16，每rank2；每mini八micro，每非空outer20 Adam。old/ref logprobglobal32/perrank4。pureOPSD：PG=0(算法既有实现)，reference KL/entropy coefficients/use_kl_loss均0，固定特权 teacher FKL(beta0、temperature1、user_context、component_clip=null)、global supervised token denominator 不变。保持当前模型、教师 FP32/eager CUDA 与 loss checkpoint/空chunk实现，不采纳未经验证padding裁剪。

lr1e-6、整数warmup57优先于ratio0，每成功outer调度一次；首outer既有lr0允许，必须记录lr before/after以及各rank真实参数变化，不能将首步零delta自判故障。gradient clip1、PPO clip_ratio0.2仅为既有元数据（OPSD component_clip仍null）；采样temp1/top_p1/top_k-1/TP1，prompt/start/action/obs4096/2048/500/500，maxturn4，same instruction+official chat template，本节点GPU E5/GPU FAISS/Wiki2018/top3/loopback。

配置 `trainer.total_training_steps=300`，由新E1 dedicated fit按actual_outer_updates==300终止，global_step保持300，不沿用旧非E1 sentinel301。禁用E1的旧行为保持。dev初始512+每50；stage64第5和每50（频率由root明确选定，与dev独立）；save5+25/50/.../300的HF和完整state由state agent实现。第5后第6取数前原进程签名门禁阻塞，签名内容必须实际独审身份/nonce/source/config/data/assets/state/evidence绑定，不能self-passed放行；fullstate真实新进程重载/原位继续需各自实证。

max_student_logits_gib=8只提高明确上界避免原4GiB提前拒绝，**不是显存容量通过**。满形状实际8rank容量/前后向/双Adam与检索共存门禁必须重新执行，不能静默降micro、改teacher精度或借历史8e短样本结果。

## 外部入口信息与职责边界

调用方必须提供：OPSD_E1_OFFICIAL_MODEL、OPSD_E1_OFFICIAL_MANIFEST、OPSD_E1_COMMON_MANIFEST、OPSD_E1_COMMON_DATA_ROOT、SEARCH_R1_BASELINE_RUN_ROOT、OPSD_E1_REVIEW_PUBLIC_KEY/_SHA256、OPSD_E1_BINDING_MANIFEST/_SHA256；OPSD_E1_RETRIEVER_URL可显式设置本作业服务端口，只允许本机loopback /retrieve。run_root绝对、现有本作业独占新目录，不猜未来SlurmID；receipt唯一 `${run_root}/acceptance-5-receipt.json`，实际日志 `${run_root}/train.log`。stage_num64明示不可改。公钥签名算法与binding完整schema由state/gate agent审核；当前绑定schema为source_commit/run_root/files必备roles allocation/source/import_closure/data_manifest/model_manifest/tool_manifest/environment，公钥严格RSA-SHA256 n_hex/e65537，完整语义由checkpoint_gate读取。本模块只做外部字节SHA/uniqueJSON，不以任意绑定JSON自报flags作为科学接受。其它内容如controller确认actual300、20Adam、first5签名、source/tools和完整state需要门禁的实际验证。

CPU planned resolved JSON 中 `/PLANNED/...`、示例 key/binding SHA仅用于证明compose和固定条件，**绝不能部署或作为当前job绑定**。真实部署应写新的完整resolved记录，冻结source/config/tool SHA并独立review。

## CPU 测试结果与剩余验收

`common-profile-cpu-tests.txt`：当前冻结重跑57 PASS/38.23秒，含真实169038 retained fixture（合成重叠排除例，不是原产物169028）、dev512/stage64/final51713 parquet，DataProto保留ID和答案，旧instruction/ID来源/内容/顺序/计数/哈希/越界/leakage/canonical重复/子采样/exclusion tamper/duplicateJSON拒绝。配置漂移拒绝包括跨节点检索URL、微批/mini/warmup/预算/教师CPU/clip/提示/refKL/采样n/local/gate/stage/subsample/教师HF5/外部log。

此前新增fast CPU重跑记录见 `common-profile-fast-cpu-tests.txt`（含启动器拒绝unknown/额外override及bool伪integer）；`bash -n` PASS。未运行官方3B forward、GPU容量、FP32/BF16数学门禁、真实newjobfirst5、fullstate重载、签名恢复、300outer或final51713。因此本模块完成的是共同条件接入和CPU软件检查，整体新实验不能据此宣称通过。


## 实际导入闭包与启动路径

新增 e1_import_bootstrap.py 与专项 CPU tests。launcher 从自身位置解析物理新repo，校验 SEARCH_R1_REPO，前置并export PYTHONPATH，切换到该repo；这会传播到Ray/torchrun新进程。CPU cfg/validate、GPU preflight 与 main 均调用实际新repo脚本，run调用已核验的实际main模块对象。

闭包 exact schema为 {modules:{name:{path,sha256}}}；它本身由外部binding字节SHA固定，包含checker自身reproduction.baselines.e1_import_bootstrap，以及实际运行环境hydra.main/torch.utils._contextlib/contextlib/dataclasses四库源文件。每个实际已加载project模块都须匹配物理repo路径/完整SHA；driver/rank允许不同已加载子集，但缺任何实际模块pin拒绝。原始源码编译得到结构code fingerprint与当前实际functions及wrapper各层对比，常量递归规范化tuple/frozenset，浮点按IEEE字节。未知filename的project function、property三accessors、stale project-function aliases均拒绝。dataclass仅源码明确声明、未显式实现的已知生成method可例外，不算source body核验；导入class在其实际定义模块核验。

这用于真实源码/导入一致性和典型stale对象检测，不宣称通用所有对象slot语义、防任意恶意monkeypatch或库生成方法源码证明。driver保留七关键数学/状态函数额外确切code核验。当前实际CPU collect已exit0，67个模块库存写入 import-closure-cpu-only.json；CPU库路径不能用于原生H100外部pin，须在实际环境重新采集、独审和固定。旧失败frozenset/导入dataclass集成记录保留，不能隐去。

本轮57个profile/consumer/bootstrap CPU tests通过，包含真实合成parquet完整预算、漂移拒绝、old editable、已缓存旧源、源SHA/路径/实际函数替换、wrapper、property、dataclass冒认及导入alias正反例。实际collect不调用模型forward/训练。无GPU、完整新进程RL恢复、官方3B满形状、首5独审签名或300outer验收声明。
