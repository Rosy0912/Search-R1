# GiGPO 第一次实验 checkpoint200 续训到300

## 人类已批准的范围
协调会话01a101e0-90d3-7731-bae9-58c8f1fdf7e2在2026-10-04明确批准从2992唯一checkpoint200继续100次实际更新，先201–205独立接受，之后同一轨迹到300；205/225/250/275/300保存，300进行64题评估并停止。覆盖旧fresh500和全量终测安排。无需重复授权权重恢复。

恢复目录：/mnt/data/shaoshitong/search-r1-rl/gigpo/runs/gigpo-formal500-job2992-1791016340597739432/checkpoints/actor/global_step_200。
完整库存证据：outputs/evidence/original2992-checkpoint200-resume-inventory.json，12个HF文件已完整SHA核对；没有优化器、调度器、RNG或数据迭代恢复文件。旧201–227更新无法恢复。本次仅权重恢复，不是精确恢复。

## 配置与恢复行为
actor和rollout从checkpoint200初始化；reference仍从第一次实验原base路径加载，禁止把reference错误换成checkpoint200。数值和奖励采用已验证21d50f代码；严格joint GiGPO数学目标、数据、micro64/rank8/train512/mini256/nagent5/turn4及长度不变。
优化器为新建AdamW，无旧动量。调度器重建原horizon501/warmup142，并将逻辑计数置200，恢复段第一步学习率为原日程第201轮的值；不能因stop300把horizon变301。RNG和数据排列重建并记录seed；不能声称恢复旧迭代位置。逻辑201–300对应新增actual1–100。

## 同进程五轮检查
在201–205完成真实八rank更新、finite loss/gradient及非零参数变化后，205强制64题验证和保存。driver写冻结ready收据，然后等待绑定本作业及完整产物SHA的独立PASS文件。等待期间driver与八rank仍存活，optimizer、scheduler、RNG、数据迭代器均不重建。独立检查用CPU完整HF重载，不能覆盖GPU演员或重启训练。拒绝/超时/资源不合格终止自身并留证据；禁止自动从base重试。
继续许可文件的作业身份、恢复源、205状态清单SHA和独立审查SHA必须匹配；单独一个PASS字符串不允许继续。通过才原进程推进206。

## 可恢复训练状态
每次HF保存额外保存八rank本地优化器状态、scheduler状态、CPU/CUDA/Python/NumPy RNG以及FSDP shard布局身份。driver保存实际数据排列及已消费cursor、epoch、RNG、KL控制器和实际更新计数。模型由完整HF恢复，优化器状态按同world8相同分片布局恢复，loader需验证布局/形状/超参后才载入。完整manifest包括真实文件SHA与尺寸，原子发布完成标记。
checkpoint状态保存和恢复应在真实Torch2.4八rank FSDP、AdamW、CPU offload下验证恢复前后下一次更新一致，而非只证明torch.save成功。HF之外不宣称vLLM内部RNG精确恢复；未来重启生成会重建引擎并明确记录。当前205暂停不会重建引擎，因此本任务205前后保持同一轨迹。

## 停止和保存
205/225/250/275/300保存完整模型和上述训练状态。300验证只64题，记录qa_count并在counter递增前返回，禁止执行301或_prepare_final_validation全量入口。新独立输出目录，不覆盖2992/3026源码或产物。

## 资源与失败
仍使用h100-8本机GPU E5与八GPU FAISS、loopback18081/Ray16381；保留Slurm独占、双15%、全UUID/OFD租约、pidfd/cgroup本任务检查、60秒严格清理。远端清除proxy，管理SSH显式Mac代理。所有失败保留原始日志/微批证据；不得跳坏梯度、降micro或额外训练。

## 实施和验证顺序
1. 实现受显式resume配置控制的恢复逻辑、状态保存/加载、可保存排列sampler与205暂停许可。
2. CPU测试状态加载拒绝错误布局/损坏SHA、数据cursor恢复和停止语义。
3. 隔离真实Torch2.4八rank FSDP下一更新等价测试与独立审查；只检查实现，不从base训练3B。
4. 冻结源码与版本化runner、独立部署READY和全部SHA绑定，再启动唯一200到300续训。
5. 原作业205等待独审，通过后继续同作业300；最后完整保存/重载/64题和退出资源独审，交付报告并结束监督。

独立设计复核已完成；补充保存sharding manager的gen_random_states/torch_random_states，并绑定FSDP原名/原shape/shard offset与optimizer参数组顺序。205新Adam step50而scheduler205；201起每轮非零变化不能豁免。

实现收敛：增加真实64题QA计数；205许可前后RPC验证八rank的PID/starttime、模型/optimizer/RNG内容SHA和scheduler/LR未改变。所有state receipts发布manifest前重新SHA，RNG只访问已初始化当前rankCUDA而不初始化其他卡。扩展坏LR/坏RNG/坏optimizer反例证明加载拒绝前不修改live模型。生产层reference真实base身份由guardedrunner资产SHA绑定，保存205仍保持原进程。
