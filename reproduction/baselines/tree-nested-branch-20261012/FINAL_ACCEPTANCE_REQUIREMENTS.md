# 完整目标验收证据清单

新作业启动前：旧3360实际200outer、51713完整结果和训练终态；旧checkpoint/原源码保留。新源码提交推送与SHA核验、独立无上下文review及必要最少测试；独立source/config/远端output路径；官方fresh8rank初始化optimizer空/scheduler0；同机8GPU检索/回环/资源租约门禁。

新作业运行中：每outer20机会全部处理且至少1成功write才credit；直接rank每轮8条、200credit共1600条；scheduler57warmup按creditedouter推进；partial与skip独立。每次停止回合前至少两次实测，首次5–10完整迭代的正常推进沿同一实例确认，不额外重启短训。

采样实现：每题A/A1/B/B1/B2恰5唯一叶；A随机有效位置；B和B1每非初始有效位置最多1candidate，首次异奖接受、深度0一次fallback保留；B2 parent=B1及完整实际快照；prefix模型/检索/mu ledger继承；2/3leaf质量及原归约；失败candidate不进PPO但计成本；没有答案污染生成prompt；不筛题。

每outer证据：reward/raw/intra/totaladvvariance、根内同奖/混奖、五叶全对/全错/混奖、root均奖差、nested共享前缀重数及原信用（数学审查要求）、candidate每次reward原因位置parent耗时、初始fallback比例、生成token/检索/评分/优化总数。语义selection-bias限制明确。

终态：200实际outer、各rank成功write及skip、梯度参数有限及真实非零更新；40次dev512实际NQ256Hotpot256；HF/native5/50/100/150/200与rolling、source/config/data身份；最终七集计数3610+11313+14267+7405+12576+2417+125=51713；实际HF/tokenizer冷重载/全参数有限/身份/短forward；GPU/CPU及额外candidate成本，历史3360成本分列；资源背景保留/仅自身清理；独立最终review。

公平比较须同硬件后端split及完整训练测试GRPO控制，预算差异单列。缺少控制不伪比较、不操作他人任务。任何有限搜索均不保证探索成功，效果退化和额外成本如实交付。
