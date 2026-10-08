已闭合 BF16 零适配器基线单阶段的原生直连备份。仅复制固定的 588 个 regular files（33,040,499 bytes），包括512条完整 native capture、逐rank原结果、额外控制、日志和真实八worker wait/release记录。

复用已发布 outer-backup-v2 的 stdlib fileproof/metadata；source-inventory.stdin.py 在只读 Node Python stdin 执行，source-metadata.stdin.py 在 Python -c 模式读取固定清单 JSON。清单与完整问题/轨迹均留在私有工作区。

该 BF16 phase 已闭合，外层零基线 controller 可以继续运行 FP16。复制不枚举其他phase、FP16合并缓存或模型文件，不获取训练锁，不等待parent退出，不占用GPU，不改Node，不删除源或现存副本。entry记录启动前状态，实际传输和验证结果由私有执行receipt记录。
