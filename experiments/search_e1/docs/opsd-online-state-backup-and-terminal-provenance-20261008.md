# 在线 OPSD 状态、备份与终态证据

本次保持同一在线 run `1d1d67b60f0e46a2afc0536a36ab69b1`、配置 `d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17` 和实际 CPU-merge recovery `51eb6c574dd726b929dffcd6c0abfdc7b441a5d4`。上限10、outer200、LR5e-6 constant、原矿选、固定 teacher、轨迹内 token 均值后实际 pair 均值及所有科学设置未改。

## 实际状态审计

2026-10-08 20:35 CST 只读快照绑定原 controller 的 PID/UID/starttick/完整 argv，实际核验305文件冻结 runtime、23个原助手文件、两个恢复源码及出版链、source100全部12文件与 TRAIN/native512 数据的新 SHA。没有修改节点文件、生产 RNG 或模型，也没有启动或停止任务；只请求既有18004的 health。

该快照记录17个完整 outer、169次已完成轮次的 Adam 和5153对累计曝光，第18个 outer（索引17）训练中、最新已提交更新177。连续完成票据、outer/checkpoint manifest 与最新完整 CP169 的文件 SHA 对账通过；这不等于反序列化了177次 Adam 的全部内容。八卡当前最少实际余量15003MiB，observer新鲜、历史低于门槛计数0；既有 oracle 身份未变，没有外来 GPU 进程。本次读取 /data 实际可用1545887328KiB。所有数字是对应快照，不自动作为更晚进度。

独立 CPU 读取此前明确固定的 CP149/outer14，manifest `5079cabf70e7cb3c13eb7d93f675efa7dcff1d8af14f2c37237ad41224dc7d01`：八 rank 的真实 Adam 均为149，完整 moments/groups 逐项一致有限；504个 adapter 张量全部有限并相对保留 CP5 改变，最大绝对变化0.0010421955958008766。该 outer 的10次更新与296条 token 文件清单对账通过；末批8条轨迹的全部1510个有效 token、原 mask、轨迹均值及实际尾批分母8通过。没有重算 logits、分词器或更早所有 token。真实执行 exit0，私有独立审计 SHA `9a96dae7dde0229f99550e7d2c84c6b62b00d722363fcf83d20db567bda390a3`。

HKUST4 原 archive 同次只读记录174/200 GRPO已完成、175进行中，174的 reward0.525/gradnorm0.662有限；最新验证与模型检查点仍为150，不能将其称为174的验证。五个正确映射的669文件冻结源码目录均逐 SHA 一致，自有 GPU4–7/原控制与检索进程保持。未停止或改变 archive。

## 持久副本验收

原保护 CP0/1/5/11、关联输入及紧凑验收记录已完成本地备份：58个文件，成员总字节6239760772。传输实际 exit0，逐成员完整 SHA 与节点 inventory 一致，安全解包后的完整副本存在；原节点数据继续保留。完整票据 SHA 为 `0169e6b4f9be5429e3fb0d39211cec16cf6a234e208eb32f6af6ad9a6718db65`，归档 SHA 为 `0646f2da8b338108f753f880b30859a96831681cc99767cc6c47a50da93e9dd1`。

root另核文件清单、覆盖、字节数、SHA声明一致与本地实际存在；没有再进行第二次6GB逐字节重算或将全部成员反序列化。以后最终状态和评估原始证据仍须备份；这次完成不代表整个200轮的资产已经全部备份。

## 后续终态与效果边界

原200轮训练完成后没有自动 native512 评估，继续按已发布完整验收与效果计划执行。预选 source100、CP89 与最终 outer199 raw adapter；不能以 outer199 的采样 cache（outer198权重）代替最终模型。真实训练完成、终态 rc/release、完整科学状态对账和固定权重效果是分别验收的证据。

为保留真实退出证据，新增辅助入口仅核 live→terminal 的同一 tmux server/socket/pane、单前台 Bash 命令、原 PID/UID/starttick/argv和已发布源码链。节点既有 pane 已启用 remain-on-exit；助手不修改 tmux 选项。只有该同一保留 pane 的实际 dead status0、同一 controller的 COMPLETE200、所有自有后代释放，才可确认这个 controller 的 exit0。历史失败 attempt 不由该状态推断为成功。读取不确定、进程消失或源码兼容问题是 UNKNOWN，需要另查，不能当作训练失败或成功。

没有产生新的 native512 EM；随机 TRAIN reward、loss和状态验收均不表示效果提升。不会恢复旧 M3/M4、旧 GRPO或旧八卡七基准作业，也不会自动追加训练消融。

## 实际 live 来源验收补充

新增终态助手已在 `f6cd787e2040e5c17c5f38ff7fdcba8d26b2f6cb` 提交、推送并逐文件从 FETCH_HEAD 回读相等，清单 SHA 为 `d93c99e9ac099c7a8ba3f6905bb72157d8351b383f156f3a44d3d4ade751ad5b`。独立部署到本实验的新 source 子目录，四源文件、发布票据与未修改完整效果计划均重新读回节点 SHA 相等。2026-10-08 20:41:27 CST，真实 live CLI exit0、accepted，绑定仍 TRAINING 的原控制器720753、前台 Bash720750、tmux server2373025、原socket/pane%70及单前台命令。实际输出原字节 SHA 为 `a9f91f1187231b9139607d221a7893b3e436db41b422ca5f5d28e97ec24b41d0`，独立本地只读复核通过。

live输出没有最终退出码声明；这个 CLI 的 exit0 不等于训练控制器已 exit0。保持这份原字节用于未来 terminal 的明确 SHA 输入，终态另存输出，不能覆盖 live 原件。原部署、配置、三个科学/恢复源版本及 GPU工作均未热改。

## 21:14 CST 独立进度和数值复核

同一控制器新只读快照记录22个完整 outer、218次完成链 Adam、6665对累计曝光；outer22（第23轮）正在使用最新 CP218 raw adapter 采样。八个自有采样进程与既有 oracle 身份正确，八卡实际最少余量3279MiB，observer新鲜，历史低于运行门槛计数仍为0。305科学源码、23原助手、两次恢复源码、12个 source100 文件和两个数据的新 SHA 均一致。此快照的原始证据 SHA 为 `669484e04859d2e8bfd7a8c008f297f95850b0642e267871b79938503d919b1c`；它不代替后续读取。

另用已发布的只读 CPU reader 实际打开明确绑定 CP218/outer21，manifest SHA `a42f7c8987ba2bdb1c02d3f5fc5bde2e19b3511e15e52c31625f582e2281c3bb`。八 rank 真实 Adam218、完整 moments/groups 逐项一致有限；504个 adapter 张量全部有限并相对 CP5 改变，最大绝对变化0.0016676150262355804。该 outer 10次更新和319个 token 文件清单对账通过；末批31条轨迹的全部7947个有效 token、原 mask、原记录 loss、轨迹均值与实际尾批分母31通过。没有重算 logits/分词器或更早全部 token；CPU前后没有初始化 CUDA，实际执行 exit0。独立票据 SHA 为 `6084683095420a38e5b12c6db1677bc0783777f8fd87b0c7dd96792bb565e835`。

HKUST4 21:12 CST 新只读记录176/200 GRPO已完成、177进行中，176 reward0.496/gradnorm0.529有限，最新512验证与模型检查点仍为150；五个669文件目录的新 SHA 均一致。自有四卡与原检索/observer身份保持，没有改变或停止 archive。

独立在线效果入口按已发布设计实现于 `control/opsd_online_results_eval_v1`。完整冻结清单 SHA 为 `5c8fb3d21f5c3c1ab33a349cb7923bedeeb212acf090c1becc819c354a3ac664`，10个负载文件、9个 Python AST/逐文件字节与 SHA 均通过；实际本地40个 CPU fixtures、0 skip通过。独立科学审查将终态、原始 nested wait、CP218真实 CPU execution、实际 CPUmerge 四票据与生产者接口逐项对齐。候选阶段发现的接口/短命子进程/CPU控制器 CUDA准入问题均在运行前修正，没有改变正跑源码。

新入口保留六个资源锁、精确 oracle 与鲜活 observer，分别核训练200轮完整链/真实退出/最终八rank实际状态，再评 source100启用的CP0、预选CP89和最终outer199 raw。所有额外零效与hook控制排除在1536正式样本之外；最终模型不由采样cache或best替代。缺失真实最终输入会HOLD；源239条件不符只标条件诊断，不改参数凑数。训练200轮、终态和效果均尚未完成；本次没有新 EM，也没有启动该评估的 GPU 工作。

双独立审查均正式批准后，新入口已在 `fb0932a019379b7a17f7b74ebc8037cefb28774c` 提交、推送，12个新源码/文档逐文件从 FETCH_HEAD 回读一致。节点独立目录 `/data/shaoshitong/search-e1-opsd-online-results-20261008/source/results-eval-fb0932a/opsd_online_results_eval_v1` 的11个文件逐字节与本地相同。首 Node CPU fixture 缺少测试使用的仓库 sibling admission.py，四项路径错误已保存；补入字节相同、已经发布的原 acceptance.py 作为测试依赖，未修改 evaluator 或训练源码。原环境40项 CPU tests 随后全部通过，0 skip；这不是一次训练故障或 GPU 恢复。完整复跑票据 SHA 为 `5a311563a19e323f56d250784111fa186ba5ffbabeb482f7cd1bd7b39727f2a4`。

实际 CPU prepare 对 source100/305源码/数据/CP0/CP89 身份检查通过，产生新独立 run `0aa669133f74418e818f14d217b4d9b7`。发布绑定的 `receipts/config-published-not-ready.json` SHA 为 `fe3d6eb5903fdc55334922f020bf53b9009d63d0a8155da5e3758dd90eb92d33`，自身出版与原 observer 已绑定；真实最终 checkpoint、终态和全200验收仍明确缺失，因此 ready=false/HOLD。实际 CLI exit0只表示准备完成，不能当作效果评估开始。配置绑定过程没有 import Torch，所有新阶段均未启动 GPU。

## 22:18 CST 新读状态与 CP286

同一720753控制器继续运行，最新只读快照实际记录29个完整 outer、286次完成链 Adam、8744对累计曝光，outer29（第30轮）用最新 CP286 raw adapter 采样。原科学源码、恢复入口、source100及两份数据均通过新 SHA 核验，八个自有采样进程与原18004 oracle身份正确；八卡实际最少余量3143MiB，observer年龄0.35秒，历史低于运行门槛计数0。快照 SHA 为 `3951946d33aa9796a9c4a2525b6ec5ecd3e4b54e09492adca8c831d12164a0ec`，数据只对应22:18的读取。

独立 CPU reader 实际读取明确固定 CP286/outer28，manifest SHA `ea208123e618e7574fcc4b7a1d3c702e45cb892bdfc67ed197f6eb7f92489e3f`。八rank真实 Adam286、完整 moments/groups逐项相同且有限，504个 adapter 张量全有限并相对 CP5 改变，最大变化0.002025941386818886。此轮279对/9次更新与 token 清单对账通过；末批23条轨迹5800个有效 token 的文件 SHA、原有效 mask、记录损失、轨迹均值及尾批分母23通过。没有重算分词器或 logits，也没有声称早先所有 token 已重新实读；CPU前后 CUDA未初始化、实际 exit0。独立审计 SHA 为 `92b023b301176ee3366954a15b626df6ac4fa35b235fa785181861881057cde8`。

HKUST4 22:13:58 CST 实读177/200 GRPO已完成、178进行中，177 reward0.529/gradnorm0.619，末四轮有限；最新 native512 与检查点仍为150。五个669文件 runtime 的新 SHA、14个已知进程身份和鲜活 observer 通过。GPU4–7直接余量至少27651MiB，原共享检索718MiB；没有停止或修改 archive。

这些是训练健康、真实状态与逐token证据的验收，不是新效果结果。新在线三模型 native512入口仍等待原200轮完成及真实终态，不抢占训练八卡，不恢复旧评测。

为提前保全预选 CP89，新增独立 stdlib 备份入口 `control/opsd_online_pinned_backup_v1`，只接收明确 manifest/config SHA、run、update/outer与已发布源码票据，不猜 latest，也不重发已完成 CP0/1/5/11。29项 CPU fixtures通过，覆盖真实18成员结构、SHA/stat/路径/符号链接、完整 USTAR成员及失败 partial保留；清单 SHA `1cc12fd6b9215d177480c1c46ae0c04c2f0b8cae641c16287f08b0e260338163`，两个新本地目录根权限0700。必须先推送、读回、独立部署后才执行真实备份；全部成员安全提取、另一次本地磁盘 SHA读回及真实传输rc0后才能完成。入口就绪不等于 CP89已备份或200轮完成。

另复核旧192题原始诊断本地副本：当前只有1050文件/4414030129字节，0/56 NPY；1017份可对照文件 SHA相符，一份 token日志尚短，32份仅有原 metadata与本地 SHA。原 rsync已不在，旧返回码未知，不能改写成 exit0。节点固定 run 的6401份已验收路径 metadata未变，当前完整目录6603文件/39215210380字节；曾记总量39224070870与当前目录范围相差8860490字节，来源范围尚未定位，不能据此称数据丢失。新只读 rsync已复用原 partial继续传输，无 delete、未改节点或启动GPU。该副本仍标 PARTIAL，实际新退出码和全内容核验完成前不称全部备份成功。

独立源码审查正式批准后，固定备份入口已在 `f26e9e47d6165c0417f2c2173227fe53f8c3b20b` 推送并从 FETCH_HEAD逐文件读回。新节点目录 `source/pinned-backup-f26e9e4/opsd_online_pinned_backup_v1` 的四源码文件与出版票据均新 SHA相同，原环境29项 stdlib CPU fixtures也实际 exit0/0skip。随后启动真正 CP89备份，18份源文件新 SHA/stat库存验收通过，总字节2041167791，库存票据 SHA `edb64652fa0269e6bf4d015133b8071a64e92ba057d67af0b6a2dd4ac7c6fe7a`；新 `.partial` 已实际增长。只有字节库存和传输开始已确认，完整传输、独立磁盘读回与最终 completion仍待完成；没有改变训练或评估源码。


### 23:18 CST read-only continuation and independent CP344 audit

The same authorized controller720753/run1d1d67b60f0e46a2afc0536a36ab69b1 is healthy. At23:18:03CST,35 outers were complete, with344 completed-chain Adam updates and10515 usable pair exposures. Outer35 was training; its latest native scalar was347. The source/runtime305/recovery/config/model twelve files and both dataset fresh SHA checks passed. Eight owned ranks plus the approved18004 oracle were the only observed GPU applications. Direct memory.free was at least15295MiB; the observer was0.0025seconds old, with10396 samples and zero below-floor samples. This is a timestamped health result, not native512 effectiveness. Monitor stdout SHA:90c3de7ecf451ad01bd136cc765ea2e0269da2a20daba95c350a0719ac8bbb01.

An independent CPU read of the protected CP344/outer34 completed with actual SSHrc0: manifest9129a36c80f4eb8192caa577b3a4458f93afac50bf6e0857a0305bcc729aa18a; all eight actual Adam344 state groups and full moments were identical and finite. All504 adapter tensors were finite and changed versus explicit CP5; maximum absolute delta0.002392503432929516. The current outer has318 token-trace references; the complete latest mini of30 trajectories/8351 valid-token rows passed native mask/loss/offset reconstruction with the actual tail denominator30. Earlier token files were identity/inventory checked, not all numerically reconstructed; tokenizer/logits were not rerun. The read-only helper did not restore production RNG, load a model, create an optimizer, initialize CUDA, write to the node or call HTTP. Private CPU proof SHA318cd9ab13172de9289178fa5d15512a401b54137f357ca29d387fe94310f5a8. Later native updates do not inherit this numeric acceptance.

K4 remains untouched: at23:17:38CST archive179/200 was complete and180 had entered. Latest reward0.489 and preclip norm0.670 were finite. Step178 had a finite402.139 norm, followed by179 returning to0.670; this isolated event and the checked logs do not establish a crash. Four owned archive actors, the original shared retriever and their exact UID/starttick identities were verified;5x669 frozen source files matched fresh SHA. Direct memory.free on owned GPU4-7 was at least23081MiB; retriever memory738MiB. The latest native512 log and model checkpoint remain step150, with rounded means compatible with230/512; no new validation is claimed. Private compact proof SHA962b06a66e570beb2bcf5c99bb5917ddd49770bcc802542a6eba92b7bd51d7b9.

Both persistent local transfers are progressing. Root polled unified sessions99920(CP89) and91110(original192 evidence); each returned still running without error. Independent local PID/argv/provenance checks found CP89 partial growing681975808 to728211456bytes with zero transfer stderr, no complete or failed receipt, and the original18-file/2041167791-byte inventory bound by fresh SHA. Published f26e helper payload/manifest/publication and local Node inventory remain byte identical. The existing M1 rsync partial grew5924022176 to5971342390bytes/1554files; old transfer rcUNKNOWN stays separate from the currently active PID39162. Local disk had81331101696 bytes available. No original Node asset was modified or removed, and no completed-backup/full-member-SHA claim is made. Private local backup proof SHA13c48c08d126720b77f8b28e668401fb04380d2aef070239b995452632b48d75. The same200-outer training continues; the published independent result evaluator remains CPU-prepared and has not acquired GPUs or produced new EM.
