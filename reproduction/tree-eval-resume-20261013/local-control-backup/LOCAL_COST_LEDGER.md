# 纯本地逐轮成本账本工具

调用文件：append_nested_local_costs.py。仅Python标准库，不含SSH、subprocess、模型或tensor加载、训练控制。它不会读写STATUS、REPORT，也不启动任何watcher。

实际FULL19发布后，主线可执行（本次发布没有执行此调用）：

    python3 -B /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/work/tree-speed-repair-20261008/nested-local-cost-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup/append_nested_local_costs.py \
      --base /Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2/outputs/tree-nested-branch-handoff-20261013 \
      --receipt FULL_OUTER19_PROGRESS_RECEIPT.json

后续只替换receipt轮次。缺少receipt时返回PENDING且不写账本。输入仅为base内明确命名的小型JSON；它核对FULL绑定的immutable poll、native readback和resource SHA/bytes，再从本轮step/candidate_outer读取事实。不会用latest metric替代本轮。

已完成的1–18轮块文件保持原字节；首次追加从19开始新块。后续工具创建的active块可追加，接近100000字节则开新块，不丢标量。保留v3索引、原字段名、完整scalar、来源/config/data/model身份、真实有效ancestry和历史失败成本分栏。新块使用原字段名映射，旧vector仍按其原fieldsets解码。既有两个小型本地镜像沿用账本mirrors配置。

输出为PER_OUTER_METRICS_COMPACT.json/md、新scalar块，以及CUMULATIVE_COST_THROUGHN.json/md（实际文件名例如CUMULATIVE_COST_THROUGH19.json）。历史累计文件不覆盖。累计若超过100000字节，会把完整证据字段存入cumulative_cost_blocks的小型分片，主JSON保留每片SHA、字节、字段名和顺序，字段不被删除。每个JSON块限100000字节，单条异常超限会报错，不静默截断。

未知dev、每rank机会字段、candidate outcome分摊、padding分摊均保留null；成功写入、empty-minibatch skip、完整metric机会、独立rank机会分别记录，不互相推算。native writer累计计数已含祖先，不重新相加。历史失败2/6、eb6 bootstrap与整服务CPU范围不加入有效轮分阶段成本。

完整native200存在而200的full metric缺失时，可以记录原生完成证据，但full_metric_status保持PENDING_CURRENT_OUTER_FULL_METRIC，缺失的评分/方差/完整outer耗时仍null，累计对应total为null并保留known_sum与缺失轮次；不得借199轮指标或dev。若已有当前metric却step或原receipt不一致，明确报错，不把坏记录降成pending。

8×已知完整outer墙钟只列物理共享容量上限。GPU实际活跃/独占账单/能耗仍null；stage、正常dev、整服务与失败子阶段不重复相加。

同一轮同一receipt SHA再次调用为幂等；不同receipt覆盖已计轮次会拒绝，需另行明确修正凭据。非终轮要求下一轮已进入；目标200不要求启动201，但native driver完整next_step仍按实际metadata绑定。所有新原始证据为固定receipt/poll/native/resource哈希，动态CURRENT/STATUS不是固定证据。

本次发布仅AST语法检查，没有运行工具、没有成本账本重写、没有第19轮信用，也没有review/测试/训练节点调用。执行权限和实际训练验收仍由主线掌握。

