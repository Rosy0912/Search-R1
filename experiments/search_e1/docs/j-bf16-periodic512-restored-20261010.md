# BF16 RWM：恢复每10轮固定512评测

用户最新纠正：不要重复已做过的第5轮评测；正常每10轮评测并未取消。旧“no512”解释废止。

旧续训owner3304656及其已确认同UID/run后代已按人工授权停止，实际GPU检查只剩获准oracle2239529。已完成到18轮的后10轮输出保存到HKUST1原目录 preserved/rollback-after10-1791610140，没有删除。当前从已完整保存的第10轮CP100及全部Adam/RNG/cursor恢复，不从初始权重重训。

新独立入口 opsd_j_bf16_periodic200_v1 持原训练锁，在已完成第10轮后先512，随后fresh第11–20轮、512，再按每10轮直到200。训练和评测不并发占卡。原训练数学、BF16、R/W/M、科学config以及已发布guard200模块不变。评测普通提示、原冻结SearchAgent、HFSDPA BF16base+FP32未mergeadapter、greedy/rep1.05、8192/500/5turn4search/top3/480、原512八rank分片seed42。没有额外第5轮、零权重或hook补测。完整8worker实际wait0、资源释放及唯一512覆盖后才写summary。

实际新owner3484455/UID1005/tick940184230，tmux e1-j-bf16-periodic512-200-20261010，Node日志 logs/periodic512-200.log。实际第10轮输出 periodic512/eval-010；启动后的连续只读实测12→16→25题，尚未完整512，不能宣布EM验收。每rank完成64才算正式512。源权重与实际加载504FP32adapter digest核对保持。

总预算200，5轮仍只是初期验收。自动跟进prompt及本地task-progress记忆已纠正，后续不再执行旧no512安排。Mac仅源码/config/小摘要，所有CP与预测在HKUST。不得仅初始化或PID存在称已完成评测。
