# OPSD 无裁剪候选独立修复审查

**结论：本地候选的源代码、真实Hydra入口与CPU数学/边界验收通过。未部署、未训练，不构成GPU或效果验收。** 冻结8e历史训练和 `.05` 结果不改写。

## 范围与入口

本次实际源码差异仅 `generalized_jsd` 默认 `component_clip=None`、docstring及 `ppo_baselines.yaml` 的 `component_clip:null`/注释。actor、controller、采样mask、教师prompt/权重、mini/micro循环、optimizer没有本次diff。新测试文件是未跟踪CPU测试；其它已存在review文档的工作区差异不属于修复。

独立实际执行 `reproduction/baselines/run.sh opsd --cfg job`，没有加载模型或GPU，Hydra真实解析值为 method=opsd、component_clip=None、beta=0、temperature=1、actor.grad_clip=1、PPO clip_ratio=.2。实际执行 `run.sh opsd baselines.distillation.component_clip=0.05 --cfg job` 后仅该值变为.05，其余保护值相同。用户额外CLI参数位于run.sh生成覆盖参数之后，允许明确历史opt-in。检查cfg的CLI override需位于 `--cfg job` 前，避免Hydra argparse把flag后的override报成未知参数；正常训练参数不存在该inspection flag顺序问题。

controller不强制恢复.05；actor直接把config.distillation传入opsd_loss，opsd_loss以显式 `settings.component_clip` 作为参数传到普通及非重入checkpoint的同一 generalized_jsd，None不会被隐式替换。TOP-D当前路径没有调用generalized_jsd，不将此变更描述成TOP-D的实际目标改动。

## 数学及CPU证据

beta=0、clip=None确实为 `KL(teacher || student)`。两方logits先FP32/T做log_softmax，teacher.detach阻断教师梯度；温度T下学生梯度是 `(p-q)/T`，没有新增T²乘因子。测试T=1.1的解析梯度、teacher.grad None通过。teacher冻结参数/optimizer排除的已有机制未改。

显式.05仍逐词保留原 `parts.clamp(max=.05)` 目标。真实toy teacher(.9,.1)、student(.1,.9)测试独立通过：无裁剪小logit步减少真KL，.05小步增大真KL；不把.05负分量裁剪风险误判为全局符号bug。实际3146的256配对无裁剪全部降低KL，详见 `OPSD_UNCLIPPED_ACTUAL_REVIEW.md`，这是独立真实分布证据而非新模型试训。

独立CPU运行6个相关文件，最终 **61 passed，3.35秒，exit0**：新无裁剪测试、JSD checkpoint、loss logging、teacher prompt、真实tiny模型/driver、algorithms。原始stdout `opsd-unclipped-fix-independent-cpu-final.txt`。首次测试60pass/1fail的真实输出保留在 `opsd-unclipped-fix-independent-cpu.txt`；失败是新测试仅load子YAML忽略父ppo_trainer导致grad_clip key不存在，最终已改为Hydra initialize_config_dir+compose且没有放宽保护值。

已有checkpoint测试真实覆盖clip=None的beta0/T2、混合beta及beta1，逐参数loss/gradient对checkpoint off/on精确一致；teacher投影不重放，teacher无梯度；masked非有限值和全空mask边界有限且空梯度为0。actor logging测试采用实际DataProto/actor loop、不均等监督量、micro1/2/4和末小mini，核每mini完整分母、两次SGD及梯度/权重一致；它的loss是独立可微探针，并非声称tiny真实OPSD全模型micro逐位一致。源码中实际OPSD仍每mini all-reduce监督token计数，denominator=global_count/world，每micro贡献相加，backward/step位置无变化。response-only loss_mask、观测排除、teacher-only targets及位置对齐不受此次默认变更影响。

## 冻结字节与限制

- `search_r1/baselines/distillation.py`: `a457ca7b1237bb3accffec89f8cc21327bd43feb8e488d96f40f1c5c3b775747`
- `verl/trainer/config/ppo_baselines.yaml`: `ef2c055b8142acf8eab9a34f0787665fc21f06e7a2e18587985a42f32b6540f8`
- `tests/baselines/test_opsd_unclipped.py`: `f8f24631e14dff9e08934baf3580a0e27e3f9a92923426d4d729f1dc64bbe176`

保留标准梯度范数裁剪1.0；PPO clip_ratio.2虽保留配置，pure OPSD路径不使用importance ratio/PPO reward目标。不能将保留这个配置声称OPSD启用了PPO保护。未对无裁剪的真实GPU数值/容量/吞吐做新验证，也未对真实历史上的教师工具监督和完整答案弱化作修复。本次去掉分量裁剪是候选目标修复，偏离作者主脚本默认.05应在后续实验标签中明确，不能覆盖原复现设定。当前无重新训练授权，不放行自动GPU/fresh5/500。
