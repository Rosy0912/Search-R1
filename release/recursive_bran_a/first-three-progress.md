# 实验A：前三轮完整训练记录

生产源码为 `5045430e67860102c304ef8098e4862ddcdf6e3e`，run 为 `52f0bd4ef1044a15bd4ef4d366843f46`。本报告不改变已部署源码。

| 完整outer | 最终轨迹 | 续写次数 | 选中TRAIN正确数 | 八rank各累计Adam |
|---|---:|---:|---:|---:|
| 1 | 2560 | 4896 | 454 | 20 |
| 2 | 2560 | 4929 | 456 | 40 |
| 3 | 2560 | 4848 | 496 | 60 |

每轮512题、每题最终5条、每题续写不超过10次；三轮结构记录检查均无不足组、违例或执行错误。选中TRAIN正确数来自变化cohort和选择过程，不是固定dev EM，也不证明学习收益。

第三轮完整manifest位于 HKUST1 `/data/shaoshitong/recursive-bran-a-8gpu-20261011/checkpoints/training_state/global_step_3/manifest.json`，SHA256 `603942a5f0adbef4c21234c78d470f87d25dccdb1ffe54c1fe4ee007612d8dfc`。状态 `COMPLETE_SAME_WORLD_STATE_SAVED`，原生完整更新账本八rank各60次，拒绝0。第4轮已生成约61MB初始轨迹文件。该证据没有独立反序列化第三轮全部Adam或重算logits。

第一轮已独立CPU读回八rank参数/Adam类型：FP32主参数和moments；CUDA未初始化。后续不重复此重型审计。第5轮原生同world reload与dev512仍待实际完成。

rolling已回收第1轮非保护payload，保留其manifest/reclaimed记录；第2、3轮完整状态保留。磁盘仍约475GB可用，Ray的95%容量告警尚未导致实际写入失败。保存期间实际磁盘写入持续增长。

目标仍为10完整健康outer并确认继续运行，随后让原生200轮任务继续；目前仅完成3轮，未验收整个goal。自动发送任务保持删除。
