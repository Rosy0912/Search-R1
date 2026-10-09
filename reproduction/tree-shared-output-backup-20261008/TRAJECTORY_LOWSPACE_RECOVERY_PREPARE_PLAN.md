# 3341低空间恢复候选：仅本地准备，PENDING_NOT_GRANTED

本轮未新增任何远端读取、部署、归档、删除、替换、GPU/Slurm测试或作业。运行中ebef训练、原checker、配置、输出路径和保存协议不改。实际释放0 B。此前PASS79f3保留源码原样；本次增加显式低空间预算，不通过静默移除64 GiB门槛来宣称可执行。approval-reference字符串本身不构成人工许可。

## 当前是否有安全路径

已有本地冻结证据`RETAINED18_HEARTBEAT3341_20261009T102435Z_CAPACITY_SUPPLEMENT.json`（SHA256 `6ee186330452d346fb4f415e87f9d0a4070510132daa09e3ad0fd7b3725629ee`）在10:25:49Z记录bavail 50,134,011,904 B（46.690937 GiB）。一份native实际分配49,376,825,344 B（45.985752 GiB），余757,186,560 B（0.705184 GiB）。此时145完成、146开始，未见写失败；空间是该时点快照，不能当作此刻最新余量。

严格磁盘临时文件路径目前**拒绝新增归档**。旧库存最小actor实际分配616,378,368 B（outer42，约0.574 GiB；作为源逻辑尺寸的保守示例），硬压缩写上限约623.6 MB，再加单文件CPU解压616.4 MB，已超过0.757 GB余量；还没加下一轨迹、凭证及未知全盘增长。当前没有实际压缩档或真实archive-aware CPU成功receipt，不能跳过这些阶段直接移除源。没有已经证明可立即安全执行的低空间回收路径。

若未来已有完整、较小、正分配blocks收益的归档，以及对应真实原checker逻辑CPU成功receipt，则移除那一份原副本只需小凭证空间，不重新要求再留一份解压临时文件或native空间。这是已完成验证后的释放路径，不是绕过首次验证的捷径。当前并不具备这些前提。

## 显式预算与停止条件

新`trajectory_lowspace_budget_candidate.py` profile为`own3341-bounded-disk-v1`，SHA由replacement固定。源文件已存在，源本体增量为0；压缩上限为ceil(1.01×源逻辑bytes)+1 MiB再向文件系统block取整。这是强制写上限，不是预计压缩率或gzip数学保证，超过则候选失败并保留源。

新归档开始前，bavail必须覆盖：训练参考峰值 + 16 MiB序列化余量 + 2 GiB下一轨迹allowance + 1 GiB未知全盘增长allowance + 压缩硬上限 + 一份源尺寸的CPU临时解压 + 128 MiB小凭证余量。2 GiB和1 GiB是明示操作allowance，不保证将来增长有界；将来native实际大小超过参考或其它进程突增仍可能造成训练风险。此方案不能宣称保证健康训练不会空间不足。

普通训练参考峰值为上述一份native。已完整latest在149/150或199/200时保留native+HF参考62,981,308,416 B；直到151的完整边界证明150保存路径已经经过才回到普通参考。200之后仍保守使用永久峰值，不通过猜测保存完成降门槛。固定源码证据：rolling hook在保存完整新native、发布latest后才prune；trainer先验证/stage_validation（hook保存滚动），再_save_checkpoint，永久HF随后完整native。

每个gzip输出块都受硬bytes上限和剩余最坏预算约束；每个CPU解压块也检查剩余原bytes+训练参考+凭证余量。过程中重新读own RUN最新native小manifest/SHA/eight-rank身份及同一固定RUN FD的fstatvfs，不能使用过期145相位跨过150却仍只保留普通参考。低空间/竞态/身份失配只停止候选自身，不调用任何训练控制。开始拒绝发生在目录intent之前；已有本调用未发布partial若失败，只清理固定父FD下相同自有inode；未知final/intent不自动清除或重试。

128 MiB覆盖两个各不超过32 MiB CPU结果及小intent/READY/manifest/dir凭证。替换开始、READY前、原source捕获前和reconcile staging删除前都重新检查这份小余量。归档CPU临时文件清理只处理本调用创建并仍吻合的inode。若阶段空间不足就停自身，保留源/已发布归档/所有unknown凭证；不自动改reserved blocks、停训或选择其它目录释放空间。

## 原副本何时可以释放

范围仍仅同一H10018 ext4、own3341旧已credited trajectory的actor-inputs.pt，outer≤最新完整native−2；必须核原manifest、source/config/data/model、8实际direct rank记录与SHA、attempt/native身份。原manifest和trajectories.jsonl.gz字节不变，不读或删native/model/data。

未来先明确批准preserve归档+独立CPU访问：核原SHA/bytes、压缩SHA/bytes、完整解压原SHA/bytes、final路径独立SHA回读，file/dir fsync成功，且压缩逻辑bytes更小、真实st_blocks×512差为正。随后独立CPU入口逐文件解压到同机唯一自有临时FD、核全SHA，让冻结原held_tensor继续原hash→CPU读入→稳定fstat→再次hash逻辑；CPU必须真实成功并明确记录该归档SHA。当前没有真实Torch/Linux消费测试。

只有以上真实receipt齐全且具体清单/四类SHA/CPU request/实际分配收益经人明确批准，才可以execute+approval reference执行replacement。再次完整核归档/解压和源SHA/inode稳定，durable不覆盖intent、原metadata、READY先落。重新pin源/归档、固定父FD，以RENAME_NOREPLACE捕获到exclusive自有staging，再核捕获inode后只unlink该文件。不支持rename时不fallback。压缩/CPU/回读/收益/空间验证失败不删源。

原FD关闭后他人已有reader仍可延迟回收，global增长可遮蔽df，故不保证unlink马上增加bavail。完成只能记原路径已移除、可释放blocks上界和归档保留。观察真实空间后才考虑下一件，禁止预先批量按估计收益操作；任一intent/partial/final未知必须先inspect，再显式reconcile，不盲重试。reconcile继续严格核原intent/READY状态、full decode flag和实际具体CPU receipt；缺字段/错SHA/缺实际CPU在staging动作之前拒绝。

## 普通滚动、150及到200预算

一份普通滚动瞬时约49.377 GB；新快照完成再prune，不能把两份常驻快照之外的瞬时需求漏掉。150永久HF+native合计62.981 GB（58.656 GiB），较10:25:49Z余量缺12.847 GB（11.965 GiB），尚未计未来轨迹和临时归档。源码顺序显示滚动先完成/prune、永久随后，因此同一150保存峰值取两者较大，不能无依据相加为同时额外两份；但要继续151，150永久增量保留后还得再有一份普通滚动空间。也就是150前可用需至少62.981+49.377 GB再加到151轨迹/运行增长，当前较这一继续训练要求约缺62.224 GB（未加其它增长）。

145到200仍有150和200两组永久资产，按100实测参考合计125.963 GB。约55轮未来轨迹用旧03:01–04:04自身近期均值约0.804 GB/轮或旧全观察均值约0.850 GB/轮作情景，约44.2–46.7 GB；这不是上限或预测保证。终态净预算缺口约120.1–122.6 GB，还不含未知全盘增长/运行日志。200终态不再要求201滚动；中间跨150仍需普通滚动瞬时reserve。不能把旧74时三组永久188.944 GB的预算直接套145，也不能把终态净缺口误称包含所有峰值。

真实压缩率、真实可批准释放量及最新RAM/系统余量未知，本候选不承诺补足这些缺口。RAM/memfd单件解压能降低磁盘增量，但会改变CPU消费资源预算，尚无当前RAM与真实Linux/Torch验证，不作为本轮已经具备的安全捷径，也未实现或获准。若训练出现写失败，先保留故障证据，不自动执行资产变更或新训练。

## 可审查的未来许可事项

第一项：限定旧完整目录、以本profile门槛运行preserve和真实archive-aware CPU验证，源保留；门槛不满足则不执行。第二项：持真实SHA/CPU receipt/正blocks收益的具体清单，逐件批准replacement/reconcile，清单外全部不动。当前两项均PENDING_NOT_GRANTED，释放0 B。方案有明确条件与拒绝边界，当前容量下不能称已经解决。
