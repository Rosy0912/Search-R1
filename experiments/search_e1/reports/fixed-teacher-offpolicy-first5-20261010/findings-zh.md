# 固定普通teacher轨迹：前5轮初期验收

固定source100 teacher用普通提示生成训练轨迹；student从同一source100权重fresh Adam开始。保留BF16、R随机选五、W正确性权重、M=.1特权teacher+.9冻结outer-old student、LR1e-6和原reverse KL/cap10。q/p/old在同一被选D历史上计算，未比较不同自主历史的token分布。

前5个outer完整结束，共51次真实Adam更新、1508个配对曝光；第6轮已生成2560条新teacher轨迹并进入训练。初期验收通过，200轮总目标尚未完成；同一流水线继续，每10轮顺序独占8卡固定512，不额外评第5轮。

| 轮次 | 实际配对 | 累计Adam | 有效policy token |
|---|---:|---:|---:|
| 1 | 327 | 11 | 62314 |
| 2 | 302 | 21 | 56097 |
| 3 | 294 | 31 | 54495 |
| 4 | 291 | 41 | 52606 |
| 5 | 294 | 51 | 54169 |

原生采样和成功训练阶段实际wait0并确认自有GPU后代释放。8rank adapter及optimizer digest一致；训练更新记录的loss标量有限。这里是原生收据与小型标量读回，没有额外独立反序列化全量Adam或重算logits。

每轮保留完整teacher普通训练轨迹、检索和实际preupdate逐token loss；另存1个同题student普通自主轨迹与teacher特权自主轨迹，无训练/评测分母贡献。HTTP诊断中的历史source100标记只作为记录器标签，模型角色以同目录role及student_checkpoint为准。

训练采样EM衡量固定teacher在变化TRAIN cohort上的表现，不是student训练收益。当前没有新的完整512效果结果，不根据这5轮健康验收宣称收益。

首次部署附带Mac元数据的0Adam失败已保留；首次CP0后旧目录名称清理保护失败也已保留。独立恢复只将清理保护绑定到精确新目录，原训练数学、precision、采样与配置未热改。复用已完成采样/CP0继续，没有重复生成或新Adam。

实际目录：`/data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010`。配置SHA：`1a14d19772893e7ceb325de976125e92fba31c50b12afe430ecc7a476ed98d9f`。科学源码：`cf19f0fc0c58356766eb7d2c8e63f9ae3fe8955d`；目录保护恢复：`f5f0bed2b73b8c9582dd7f583df9f00643638294`。Mac只保存代码、配置和小摘要，权重/optimizer/原始轨迹在HKUST。
