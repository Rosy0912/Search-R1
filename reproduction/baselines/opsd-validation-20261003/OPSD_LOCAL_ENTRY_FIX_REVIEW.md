# OPSD 本机 CUDA 入口修复独立审查

2026-10-04。只读审查 root 新修改的 `reproduction/baselines/run.sh`、`search_r1/baselines/local_runtime.py`、`tests/baselines/test_local_runtime.py`。**通过这一入口的软件/CPU契约审查；不批准GPU/训练或效果验收。** 不修改source、远端或任务资源。

## 已验证的修复

`run.sh` 的opsd case固定export `SEARCH_R1_OPSD_LOCAL_HOST=h100-10`，增加teacher_device=cuda、local_runtime_only=true、local_runtime_host=h100-10。真实 `bash run.sh opsd --cfg job` CPU-only inspection输出验证实际Hydra组合结果：CUDA teacher、本机true/h100-10、privileged_answers=true、user_context、component_clip:null；阶段数据validation-64、val_batch_size64、test_freq50均保留。Inspection只是配置解析，不初始化模型/CUDA或加载资产。

Hydra用户CLI参数仍可以在解析结果中覆盖这些值，但是 `check_local_runtime` 先检查来自任务launcher的独立policy env，再进入原有条件检查。CPU teacher、禁用local_only、换host、将method伪装grpo均在任何资产检查前报错。因此不会因为config.local_runtime_only被CLI改false而提前return；更不能仅method改名逃出policy。原main已有local检查的调用顺序仍在，真正在线执行后续仍检查actual hostname、matching Slurm节点/job、loopback retriever、explicit loopback Ray、无proxy以及唯一实际Ray node。CPU inspection允许查看unsafe组合，但不等于可以执行。

本review另通过**真实launcher/Hydra**分别生成四种unsafe CLI配置，再执行实际policy function，四种全部拒绝，原输出在 `work/opsd-local-entry-review-cli-negative.json`。没有用模型或GPU模拟。

其它method case没有新增OPSD flags。真实 `run.sh grpo --cfg job` 仍localonly false、shared teacher_device cpu；在没有OPSD policy env时通用检查行为不变。共享YAML保持CPU不影响本任务opsd canonical入口已组合为CUDA，也避免擅改TOPD等共享默认数学/部署设置。若调用者主动在其它method进程注入OPSD policy env，被拒绝是policy一致性保护，不是方法正常入口回归。

## 通过范围与保留限制

13项 CPU tests通过（local runtime含原proxy/Slurm/Ray反例及新增policy四覆盖拒绝、phase64），原输出 `work/opsd-local-entry-review-tests.txt`。真实default配置输出 `work/opsd-local-entry-review-actual-cfg.txt`。未加载teacher、FAISS、GPU或Ray实际集群。

这关闭了先前 `OPSD_FINAL_INFRA_REPAIR_REVIEW.md` 的**canonical run.sh OPSD默认CPU/可CLI关闭本机policy**问题。通用直接 `python -m search_r1.baselines.main` 没有任务policy env时仍可选择CPU或关闭local检查；它不能被冒充本任务批准入口。调用者主动删环境变量/绕过launcher也不是此canonical入口的保证范围；未来immutable task wrapper仍需精确source/config/argv/实际进口SHA和外部批准绑定。

此policy只强制设备与host边界，不代替实际teacherCUDA、E5CUDA/FAISSGPU、同job PID/UUID、资源容量、raw receipt核验。loopback地址本身也不证明后端GPU/计算位置，需要已有semantic/runtime门禁的新实际证据。CLI仍可显式选择历史clip=.05或关闭private answers；这样的配置不得冒充本次null/user_context/privileged修复运行，须由未来外pin配置审核拒绝错误身份。

64问题contract/default保持，但未来实际input与JSON-console配对仍要验证。此修复没有更改loss/padding性能或旧8e/e3被冻结入口，不解除500/HF500 blocked，不证明搜索reward/finish已恢复。`OPSD_FINAL_INFRA_REPAIR_REVIEW.md` 的性能/真实history/effect待验收部分仍有效。

## 最终字节确认

root追加了run.sh明确user_context/null覆盖（用户显式legacy/.05仍是标记历史控制）；独立重新读取diff与实际run.sh --cfg job，defaults仍全部匹配。最终审查字节SHA：run.sh `eeeccaf4f17a0312c422f329ec93272d6fd5fa3c21d7eaf22b65ac079cda98de`；local_runtime.py `527fb1400cf337c684e93baa589e6231b7147c24b0e2b02244fdf99ef2771636`；test_local_runtime.py `7b9e261fb5c0c789febe413db5fbf6e5d647811a5c0772fa6e188aaeb0c7a3ce`。接受限上述固定软件/CPU入口字节，不扩展为运行时GPU/训练效果验收。
