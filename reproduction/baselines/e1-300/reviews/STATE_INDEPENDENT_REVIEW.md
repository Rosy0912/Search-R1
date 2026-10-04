# OPSD E1 rank state 独立审查

2026-10-04。gpu_probe未编写full_state.py/fsdp_workers状态接口/nativefixture。审查最终作者清单 state-implementation-final-files.json SHA d513f9a0c56a04813ad923cdf1ce30f3bed25e767b4f73456b3290662d77f4f7，四文件与checker依赖实际SHA全部匹配。独立重跑state+trainer/gate CPU tests共63 PASS/16.04秒，原始记录 state-trainer-independent-current-cpu-tests.txt；运行后再次核对最终源全部匹配。此前32/37CPU记录及旧暂停报告事实属于历史，不能代替当前冻结证据。

结论：当前限定CPU状态/恢复原语、同活进程门禁未发现可复现P0/P1阻断；不签发Torch2.4 NCCL/FSDP八卡、官方3B完整state或完整RL新进程恢复PASS。原生fixture仅源码/编译检查，未运行GPU。

## 核对结论

1. 采用公开 SHARDED_STATE_DICT/offload_to_cpu=True，生产已有 DeviceMesh/FULL_SHARD/use_orig_params=False；不是 DeviceMesh 不兼容的 LOCAL_STATE_DICT。DTensor digest 包含global shape/stride/dtype/placements、mesh类型/ranks/dimnames/coordinate和local bytes；不做私有mesh改写、full_tensor或redistribute。PyTorch2.4官方 FSDP 在设置DeviceMesh时自动使用DTensor，加载CPU shard时内部转回compute mesh device再gather，因此CPU落盘仍需真实GPU collective测试，不能凭world-one CPU替代。[Torch2.4官方实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.0/torch/distributed/fsdp/_state_dict_utils.py)
2. optimizer使用原生当前rank flatparam local Adam状态；严格保存/恢复paramgroups/moments/step/scheduler。仅同对象/topology/PID使用，避免把local shard当作换worldsize restart。Torch2.4的Adam load按当前参数映射并转moment到参数dtype/device，普通step标量保持原state设备；后续生产已有load_fsdp_optimizer转当前CUDA，保存finally返回CPU。参数master必须仍FP32；不以BF16compute当作FP32Adam存储。[Torch2.4 Optimizer源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.0/torch/optim/optimizer.py)
3. 实际磁盘torch.save/read/load，再重取完整payload digest，含rank/world/step/PID/model/optimizer/scheduler/teacher对象ID；旧version/算法/LOCAL/异对象/异rank/step/PID/内容均拒绝。atomic_save不覆盖已有最终文件或他人tmp，fsync后独占hardlink发布。失败不产生release receipt；个别rank失败后其它rank可能等待FSDP/barrier，外层作业必须负责fail-fast/cleanup，不自行宣称能无条件同步恢复。
4. CUDA RNG只捕获当前rank设备，恢复时显式拒当前device改变；Python/NumPy/CPUtorch一起保存。actor/rollout/oldLP/ref数据入口显式cuda,current_device，12个AST实际函数入口覆盖logical0/3/7；这是API路由证明，尚非真实显存/device执行证明。
5. teacher需idle CPU/eval/FP32eager，requires_grad/grad/optimizer membership全0。记录实际资产全部regular文件SHA和参数全内容/config/count/bytes/execution/storage身份；symlink拒绝。dynamic buffer（包括nonpersistent）/rope缓存和八项实际counter同对象恢复。静态teacher参数/资产/配置变更拒绝，savedcounter不是重新伪造BAPO状态。所有时间counter在save/restore点精确连续，但真实不同后续branch的elapsed不能要求物理逐位相等，toy fixture没有区间计时，不证明该种相等。
6. warmup显式lr_warmup_steps严格整数57优先ratio；每成功outer scheduler一次，首outerlr0允许。native fixture含57步独立scalar oracle；实际生产每outer20Adam依赖actor/trainer真实审计，不是该fixture的20initial总calls。模型/优化预算未被state模块修改。
7. sampler记录epoch/order/emitted+consumed cursor/generator，strict拒未commit行，num_workers0；同活跃iterator的load可重现下一行，未重新造iterator/reset数据源。20合法Adam后提交实际512问题、下一取数前gate及driver完整状态归另一agent集成，此审查不扩展为其实际运行通过。

## 证据范围与限定

独立32tests中的model/Adam/磁盘/RNG均真实CPU，FSDP context/CUDA API明确mock；world-one Gloo DTensor真实保存读取并拒layout/content差异。作者native fixture只编译且未运行：两个默认profile FP32、nested3wrappers/BF16compute/FP32reduce/buffer/Adam/manualoffload；各toy五outer×四Adam=20initial，两条第六各4，总28。它准确记录自己预算，**不能叫用户五outer×20=100 Adam的真实验收**。CPUOffload为可选额外profile，未冒充生产manualoffload。

完整“状态”范围有一个需要保持明确的边界：student侧SHARDED state_dict不含nonpersistent buffer和任意Python模型cache；teacher有显式额外buffer/cache保存。固定官方aa8配置rope_scaling=null，Transformers4.46.3 Qwen2在default RoPE不会进入动态更新，故本审查没有证明该遗漏导致当前目标模型的next-update错误。但不得推广为任意动态RoPE模型全state恢复；若修改rope_scaling/其它nonpersistent state，必须补保存/拒变检查和下一步oracle。PyTorch官方也显式跳过不在state_dict中的nonpersistent buffers；Qwen2 inv_freq为nonpersistent，动态更新只有dynamic rope_type触发。[固定官方配置](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/aa8e72537993ba99e69dfaafa59ed015b17504d1/config.json)，[固定Transformers源码](https://raw.githubusercontent.com/huggingface/transformers/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py)

还需实际GPU验证：Torch2.4/DeviceMesh CPU-offloaded DTensor pickle+load；嵌套FSDP manual offload模型/Adam/RNG完整第六更新分支oracle；全8rank相同源设备和collective生命周期；官方3B exact save/load后20Adam的模型/Adam/RNG/教师离散状态连续、非finite拒绝、固定资产身份。原生toy通过后仍需first5真实检索/采样/64+512验证、完整HF/reload、签名gate同PID原位继续、300与final实证。所读旧snapshot没有new-process actor restart支持：worker默认严格PID/object IDs拒绝异进程；作者当前正在新增opt-in不同PID路径，本轮尚未审查，不能把独立HF加载与同对象rankstate恢复混为一项。

## 给root的决策

旧路径源码/CPU核查可作为后续准备参考；新source冻结/补审完成之前不可据本报告放行owned8GPU门禁。后续若补审通过可准备实际3B门禁，前提固定默认官方配置、保留同对象恢复范围、外层source/import/resource/audit与失败清理齐全。保持未验收状态，不签GPU或RL release。本轮未修改作者state源、未放宽门禁或自写接受凭证。


## 最终增量核对与边界

- 当前recovery_source_closure固定整个search_r1/verl所有.py文件与GitHEAD清单，保存/恢复不再比较易漂移的已import集合；当前已loaded模块物理origin仍必须正确。原动态集合设计已被替换，不能描述为最终设计。
- rank自身使用外部SHA绑定closure+checker，读取一次已核bytes编译执行并验证实际当前rank模块/函数；不继承driver单独PASS。四字段配置由driver在Ray worker创建前传递。checker依赖04304d563a71b3123e0da6c5350398bf0b8510e0b477ce47742798214bd5b3f0。
- 显式authorized_new_process只排除payload.pid/object_ids，保存模型/Adam/scheduler/RNG/teacher/source/config/rank/layout core完全比较；真实CPU新PID/新对象读盘第6Adam oracle通过。默认接口继续拒不同PID/对象，未借receipt或自报passed绕过。
- 完整restart仍未编排：fit当前重新初始化counter；driver.restore_driver与worker.load_training_state是各自外部授权原语，没有完整跨进程driver+rank+sampler/迭代器/资源重新建立的入口。不得称整体full recoverable验收完成。
- nativefixture每synthetic outer4Adam：五outer20calls，加两条第六分支各4，共28；不是正式每outer20Adam、首5共100Adam。8rank/NCCL、DTensorCPU卸载回GPU、官方3B双Adam/检索共存与实际第5后第6继续均需要后续native实证。
- 官方固定aa8默认RoPE目标下未发现student非persistent动态buffer恢复问题；不要扩大为任意动态RoPE模型支持。teacher全部buffer/caches/counters已保存并严格恢复。
