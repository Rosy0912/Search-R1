# V14 rolling6 绑定独立只读审查

2026-10-08 07:05:42 UTC：已审查 core `ebef786697b78abafc2248c0c794827606c7214a`、当前 runtime-template、V14 builder/deployer/submitter。未发现确定的可行动源码或包装绑定问题。封卷时 V14 package 与 actual3339 stop receipt 尚未在本地出现，本结论限于源码/包装及已保存的 parent JSON，不声称实际部署或恢复已完成。

- rolling6 路径是 run3339 下 `checkpoints/training_state/rolling-completed/global_step_6/manifest.json`；该路径 `parents[4]` 正确返回 run 根。local parent manifest SHA `2421b7f1f8f8db0d5b73e342e9de1a9c8b61a5fe08cb6596da492a0d3a04fbc3`、run3339 migration receipt SHA `6384b8350550ef767c8d1e0b5c27705842913517c642d8556d59b37058a98c3e` 均独立读取核验。
- 实际 parent 的 status/snapshot_kind 是 native rolling completed outer；source6227108/config98a88882/run3339/outer6、8 rank Adam120/scheduler6 和 guard accounting 均与 core/runtime 期望相符。worker/driver reload 字段真实保持 false，HF6 checkpoint/identity 为 null；terminal 报告保留 ancestry3335 的历史 HF5，与 native6 区分。
- core startup 用同一 FD 校验全部 state file SHA 后反序列化，检查 serialized logical digest，再把恢复后的 worker model/Adam/scheduler/RNG 和 driver/sampler/Tree RNG 完整快照与 parent logical digest 比较。父 ancestry bytes 与 saved provenance 也必须一致；wrapper 只消费实际 core receipt，没有把 prospective spec 或父 reload=false 改称 true。
- restore_driver 承接 completed6、training_attempts6、rank writes120、scheduler6、cursor3072、epoch0；fit 在 migration 后进入 outer7。direct training/terminal consumers 从6/120与attempt7开始；allskip 保留原 attempt-based sampler_position。旧 from0 `attempt_results` 留在非 direct 验收分支，direct terminal 显式传入包含 parent120 的实际 accounting。
- builder 要求 clean supplied source HEAD、实际3339 CANCELLED/end/exit/cgroup/unknown partial7 receipt，生成不可覆盖 V14 namespace；deployer 从 hash-verified V13 clone 传 code delta 并逐文件复核；submitter 复检 parent metadata、已取消3339/无 own cgroup、exclusive once journal，unknown 不重试。
- native prefix production SHA 仍为 `8798793a60eb093981b532a490acbab7eafe4688811ab484dcb519d2d63ef384`。新 core diff 仅 rolling6 continuation binding，没有改5leaf `[3,2]`、四轮、2048/4096/500/500、PPO或μ。

仅本地读取源码/diff/JSON/hash并写本审查文件；没有运行测试、SSH、GPU或job，也没有停止或修改训练。本结论不证明 actual child migration 或普通 outer≤420s，不作为主线暂停门禁。实际新 startup 仍必须完成文件SHA、逻辑 equality 与6/120/7/cursor3072记录，之后由运行证据判定。
