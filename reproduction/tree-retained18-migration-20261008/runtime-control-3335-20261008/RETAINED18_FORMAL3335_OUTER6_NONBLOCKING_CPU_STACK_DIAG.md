# 3335 第6轮非阻塞 CPU 栈诊断

当前真实慢点已定位：北京时间13:48:57、13:49:12、13:49:27三次独立非阻塞快照，栈顶全部是 `tree_raw_tokens._encoder_binding:276`；该行源码为 `serialized=backend.to_str()`。调用来自训练生成路径 `sampled_action` 的 live tokenizer identity validation。这些采样中 driver 不是 Ray get 等待。

采样窗口约30.21秒，PID143490实际CPU计数增加30.31 CPU秒，约单核100.34%。6次前后/proc身份均为R，三个Python MainThread均active且owns_gil=true。结合相同栈顶，这支持**当前生成阶段存在持续占用CPU的tokenizer完整状态序列化开销**；3个快照不等于完整profile，不能把该函数精确占比或整个旧session所有慢点都归为它。

## 身份、安全范围与工具复用

每个快照前后都确认：hostname=h100-10；PID143490；starttick1849458711；cmdline=`ray::measured_task`；cgroup含`job_3335/step_batch`；最新supervisor `process-resources.jsonl` tracked成员包含PID且age小于30秒。全部6次核验通过，返回码0、stderr为空，原始身份和CPU计数在JSON中。

复用原隔离目录的py-spy0.4.2，没有再下载或安装。二进制SHA采样前重新核验：`9b4d1f39b2a47ae44f4c6a46f615dcc0287d7755beba5065f32391951e07d594`。实际`dump --help`原始输出沿用旧凭证并保存在本JSON，明确`--nonblocking`不暂停目标，也提示非阻塞读取可能不准确。

本次只读自身PID/栈/CPU计数及绑定源码，未读取其他PID或原集群；没有signals、kill、暂停、重启、新训练、额外测试、模型/GPU调用、训练源码或环境修改。仅写新诊断快照和本次凭证，保留旧outer5文件。

实际命令：

```text
/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/.node-migration/diag-tools/pyspy-nonblocking-20261008T052239Z/py-spy dump --pid 143490 --nonblocking --full-filenames --json
```

两次间隔15秒，合计3次。完整实际远程命令及inline payload在同名JSON `remote_command`/`remote_source`。

## 实际栈与对应源码

| 样本 | 北京时间 | 真实栈顶 | 进程/Python线程 | dump耗时 | 返回/身份 |
|---|---|---|---|---|---|
| 1 | 2026-10-08T13:48:57.165618+08:00 | `_encoder_binding:276` | R / active=True / GIL=True | 45.9 ms | 0 / stable |
| 2 | 2026-10-08T13:49:12.244745+08:00 | `_encoder_binding:276` | R / active=True / GIL=True | 45.4 ms | 0 / stable |
| 3 | 2026-10-08T13:49:27.322660+08:00 | `_encoder_binding:276` | R / active=True / GIL=True | 44.4 ms | 0 / stable |

三个实际调用路径相同：

```text
_encoder_binding:276
← _current_tokenizer_identity:308
← _TokenizerValidationContext.__init__:315
← sampled_action:440
← rollout._advance:195
← SearchRollout.complete:277 / run:378
← controller.generate:266
← training_cost_hooks.generate:33
← ray_trainer.fit:989
```

实际18路径与本地冻结v11的tree_raw_tokens.py、rollout.py、controller.py、training_cost_hooks.py、ray_trainer.py SHA重新读取比较，全一致。完整bindings和源码片段在JSON。

源码对应关系：

- `_encoder_binding:276`先调用`backend.to_str()`，之后`:278`才检查序列化字符串缓存。因此即使cache hit，完整序列化调用仍会执行；cache只省去之后的JSON解析/语义hash。
- `_TokenizerValidationContext.__init__:315`检查当前完整tokenizer identity。
- `sampled_action:440`无条件新建该validation context；当前函数签名不接收外部共享context。
- `rollout._advance:195`在逐行生成输出处理处调用sampled_action；当前真实栈正处该路径。

`semantic_prefix_scan_cpu_seconds`由rollout汇总`ledger.prefix_scan_cpu_seconds`，后者计时从`sampled_action:431`开始，包含`:440`的tokenizer identity/context建立。因此已观测的624秒该指标不能解释为纯UTF8/逐token前缀扫描；其内包括这里的完整encoder序列化。`:472`另有`prefix_scan_tokenizer_validation_cpu_seconds=context.cpu_seconds`细分字段。

没有使用native采样，所以C/Rust内部细分位置未知；没有采到`_scan_utf8_prefixes`作为栈顶。当前结果不能倒推上一轮长CPU区间的精确分布，也不替代root已有的生成/adv/update_actor耗时与semantic_prefix_scan原生计数。

## 原始真实栈

JSON `dump_samples[*].stdout`保留原始输出；`parsed_stack`为解析结构。18工具目录保留`outer6-nonblocking-stack-1.json`、`-2.json`、`-3.json`。下列为可读转写：

### Sample 1 — 2026-10-08T13:48:57.165618+08:00

```text
PID=143490 MainThread active=True owns_gil=True
_encoder_binding (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:276)
_current_tokenizer_identity (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:308)
__init__ (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:315)
sampled_action (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:440)
_advance (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:195)
advance (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:127)
complete (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:277)
run (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:378)
generate (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/controller.py:266)
generate (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/training_cost_hooks.py:33)
fit (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/verl/trainer/ppo/ray_trainer.py:989)
main_task (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/verl/trainer/main_ppo.py:223)
measured_task (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/validation_cost_hooks_instruct300.py:165)
main_loop (/dataset/shaoshitong/search-r1-rl/tree-grpo/env/lib/python3.10/site-packages/ray/_private/worker.py:922)
<module> (/dataset/shaoshitong/search-r1-rl/tree-grpo/env/lib/python3.10/site-packages/ray/_private/workers/default_worker.py:289)
```

### Sample 2 — 2026-10-08T13:49:12.244745+08:00

```text
PID=143490 MainThread active=True owns_gil=True
_encoder_binding (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:276)
_current_tokenizer_identity (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:308)
__init__ (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:315)
sampled_action (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:440)
_advance (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:195)
advance (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:127)
complete (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:277)
run (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:378)
generate (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/controller.py:266)
generate (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/training_cost_hooks.py:33)
fit (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/verl/trainer/ppo/ray_trainer.py:989)
main_task (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/verl/trainer/main_ppo.py:223)
measured_task (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/validation_cost_hooks_instruct300.py:165)
main_loop (/dataset/shaoshitong/search-r1-rl/tree-grpo/env/lib/python3.10/site-packages/ray/_private/worker.py:922)
<module> (/dataset/shaoshitong/search-r1-rl/tree-grpo/env/lib/python3.10/site-packages/ray/_private/workers/default_worker.py:289)
```

### Sample 3 — 2026-10-08T13:49:27.322660+08:00

```text
PID=143490 MainThread active=True owns_gil=True
_encoder_binding (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:276)
_current_tokenizer_identity (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:308)
__init__ (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:315)
sampled_action (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/tree_raw_tokens.py:440)
_advance (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:195)
advance (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:127)
complete (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:277)
run (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/rollout.py:378)
generate (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/search_r1/baselines/controller.py:266)
generate (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/training_cost_hooks.py:33)
fit (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/verl/trainer/ppo/ray_trainer.py:989)
main_task (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/source/repo/verl/trainer/main_ppo.py:223)
measured_task (/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/candidates/current200-retained18-v11-6efb604/validation_cost_hooks_instruct300.py:165)
main_loop (/dataset/shaoshitong/search-r1-rl/tree-grpo/env/lib/python3.10/site-packages/ray/_private/worker.py:922)
<module> (/dataset/shaoshitong/search-r1-rl/tree-grpo/env/lib/python3.10/site-packages/ray/_private/workers/default_worker.py:289)
```

