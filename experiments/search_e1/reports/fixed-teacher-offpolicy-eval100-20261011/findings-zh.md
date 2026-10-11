# 固定teacher普通轨迹 BF16 RWM：第100轮完整512

228/512（44.53125%），NQ108/256，Hotpot120/256。八worker实际wait均0，完整释放accepted。

10至100轮正确数243/236/233/223/227/226/230/220/227/228。比90轮多1题，比source100同协议239少11题（-2.1484375个百分点）。前100轮未显示持续收益，未计算本次配对置信区间，不做单机制归因。

raw CP1006/outer99 manifest SHA6872e869450d12988cb077115a9f97028555c9a567310d7bdb3107e7964b6666。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-100/summary.json SHA f8037b5a00009bd0136c719d571225bbef155da6b2d17e433720cbd3afeb192a。

实读1791711008：108轮/1086Adam已完成，第109轮采样，无报错。200轮目标继续，不因100轮或初期验收终止。
