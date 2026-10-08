# 待明确许可：3341同机actor-inputs归档替换与CPU验收候选

本轮只在Mac本地准备和发布源码；没有新增远端读取、部署、压缩、删除、替换、GPU测试或Slurm作业。主线3341继续到200。已PASS的79f3归档入口原样保留，本计划补齐实际释放空间和最终验收的候选入口，不能把保留原文件的归档（释放0 B）视作解决容量缺口。

固定范围是h100-10/H10018原ext4、run `tree-current200-continue-3341`的`output/trajectories/outer-######-attempt-######/actor-inputs.pt`。候选只选择已有完整credited attempt、八rank实际JSON/SHA、原manifest/source/config/data/model身份绑定的轮，并满足outer≤最新完整native−2。原manifest及已gzip的`trajectories.jsonl.gz`不改；native、模型权重、数据集、训练源码和保存协议不改。所有新增产物留在同一run的`.trajectory-actorinput-archives-20261009`，没有Mac/NFS/其它节点资产中转。

`trajectory_actorinput_replacement_candidate.py`默认为metadata-only plan。未来明确批准`archive-preserve`后先调用固定SHA79f3接口，以nice19/ionice3、一CPU线程、8MiB/s、gzip level1生成归档；核原SHA、压缩SHA、完整解压原SHA/bytes且压缩更小，原文件仍保留。压缩率和真实可节省量当前未知，不能承诺足以覆盖04:04预算缺口34.67–41.49GB或49.38GB缓冲。

未来独立CPU入口是`check_behavior_mu_archive_cpu_candidate.py`，默认只打印计划，不打开tensor或写运行文件。实际verify同样需要明确许可、execute、approval reference和唯一request；CPU只绑定一个核心、低优先级，CUDA必须显式为空。它在固定V14 package目录用FD读入SHA绑定的原checker `c56e3c79959c501d6256ba51f0cd455b2de0a7540ea0c1ff9231fae066afe5bc`及四份原helpers，并保留原behavior-mu/DataProto/真实原SHA、Tree身份、beta、tokenizer、无optimizer/forward/CUDA等校验逻辑。只把文件访问改为同机固定FD；attempt枚举仅作机械IO替换，其派生源码SHA另记，不改变原accounting验证分支。原checker/trainer文件不会被热改。

CPU入口优先读已有完整归档，即使原文件仍在，从而能在首次删除前实际验证archive reader。每次只解压一个actor-input，先完整核压缩SHA和原解压SHA/bytes，再让原`held_tensor`从仍持有的只读FD执行原hash→CPU反序列化→fstat稳定→再次hash逻辑；finally只清理创建过且inode吻合的自有临时文件。最近两轮或未归档轮保留原路径并通过固定FD读取。不恢复所有历史PT。原checker的package绑定tokenizer元数据/CPU tokenizer加载仍保留，模型权重不会加载，模型forward和optimizer写入为0。本候选限目前all-credited、无allskip/partial的正常链；出现其它历史类别会拒绝，不能丢弃那部分证据或顺带读partial native PT。

未来`replace-existing`的具体审批对象必须含：目录名、原actor SHA、压缩SHA、归档.complete SHA、实际成功的archive-aware CPU.complete SHA及request、源/压缩真实分配blocks差、request与明确的人批准reference。字符串reference本身不是授权。执行前重新校验原manifest、credited/eight-rank证据、归档完整SHA/解压SHA/bytes和对应CPU receipt；CPU必须确实用该归档完成原逻辑校验。全部成功后，durable原manifest原字节、归档.complete原字节、intent和ready先保存并逐SHA读回，file/dir fsync不能静默忽略。

随后重新打开并固定源inode和实际父目录FD，确认源稳定、唯一hardlink、归档仍是同一已核验文件且实际分配blocks更小。为避免旧check后unlink错删另一个leaf，先在同机exclusive自有staging目录用Linux `renameat2(RENAME_NOREPLACE)`原子捕获源，再核捕获inode，最后只unlink已证明的自有staging文件。若rename支持不可用则不fallback，原source保留，intent待检查。若捕获不同inode，保留staging且不删除；不自动覆盖恢复目标。压缩档/原metadata始终保留，可用79f3原路径恢复接口或新CPU单文件reader访问。

每个目录只有一个不覆盖的replacement intent；新提交遇已有intent/partial/final拒绝盲重试。READY之前失败源保留；READY后source仍在的reconcile只返回未删除/需检查，不继续删除。source已不在时，显式批准的reconcile要求原intent+READY pins吻合，重新完整验证归档；若有自有staging，只有原inode和全SHA吻合才清理，随后发布新的不覆盖完成凭证，并记明恢复边界与缺失时间/原因未知。CPU audit有独立lease、唯一request目录；intent或pending无.complete意味着尚未验收，不能拿来释放源。任何失败只影响自己的候选操作，不能停训练。

移除源路径与立即释放磁盘空间不同：其它已打开reader可能延迟blocks回收，global df还可能变化。完成receipt记录实际源/归档blocks及可释放上界，不能写保证df已增加。未来状态仍需按常规heartbeat观察自身容量；本轮不改reservedblocks或碰其它任务。

可审查的许可顺序：先批准限定旧完整目录的preserve压缩与独立CPU archive读取验证（源保留）；拿真实压缩率、四类SHA和CPU成功receipt后，针对具体清单批准replace-existing/reconcile。若原逻辑CPU验收或压缩验证失败，不删除源，也不能把准备好的代码当作已执行。当前实际训练数据、Linux/proc消费与ext4 rename路径没有本轮运行测试；仅本地语法和独审必要tiny CPU fixture，实际部署/读取/替换均尚未授权。
