# Native manifest 元数据历史保存

读取冻结源码 `ba9f6627ec9a4e9749830d7b6acca53f6feec6b4` 的 `reproduction/tree-nested-branch-runtime-20261012/rolling_training_state_hooks.py`：每次发布完整rolling后只保留最近2个 `global_step_*`，替换 `latest.json`，旧目录整删；5/50/100/150/200里程碑另保留。原实现没有每轮完整manifest的持久历史副本。2026-10-14实际读取确认母0bef及子ba9两个run都尚无 `output/native-manifest-history`。

新增独立 `native_manifest_history_collector.py`，接到本地 `poll_nested_shared.py` 的同一次SSH控制请求；不改冻结package、训练入口/配置/服务。同步脚本单列history的 `audit_error`，不把采集错误当成训练错误，不自动停止或重启训练。

目标目录：
`/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:ba9f6627ec9a4e9749830d7b6acca53f6feec6b4/output/native-manifest-history`

- `source-<完整source SHA>-outer-<六位outer>.manifest.json`：原始完整manifest JSON按原字节保存，原SHA与字节数不变。
- 同名 `.receipt.json`：采集时间、来源path/SHA/bytes、8rank finite/Adam/scheduler、worker/driver的原始file SHA与logical SHA及仅stat取得的PT字节/mtime。母0bef仅第1轮；子ba9仅第2轮及以后。原失败母2不纳入有效历史。
- 只有COMPLETE状态、8rank/driver身份与计数一致、源JSON两次稳定读取且PT只stat确认完整时写入。保存中/不完整/已清理源跳过；字段或既有SHA不一致报独立audit_error。
- 小JSON先写临时文件并fsync，用 `renameat2(RENAME_NOREPLACE)` 原子发布并目录fsync；相同SHA幂等，不覆盖已有不同记录。

两次实际保存/读取：母1 manifest 16752B，SHA `90f0420ccdc30a3e15245861a73b405f3f760d333722314d08b96b95765f9a1f`；子2 16742B，SHA `1b4a45088b56ba8d146e397691cb032e40f7b34cad00d62a417a854029109964`。首次created=true，第二次created=false，audit_error=null；母8rank20/scheduler1，子8rank40/scheduler2，各rank原始full_worker_state_finite=true。

边界：历史保存的是metadata，PT以后可能被rolling清理。file SHA与logical SHA是原native writer记录，collector只对小manifest重新hash，不读取/hash PT、不重算有限性、不冷重载。manifest提供整体worker/driver logical hash和saved_components标志，没有独立逐parameter/optimizer/RNG组件hash；不补造这些字段。没有执行测试、code review、模型/奖励重算或训练操作。
