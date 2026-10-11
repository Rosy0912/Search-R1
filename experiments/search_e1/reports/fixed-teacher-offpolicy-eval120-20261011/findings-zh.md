# 固定teacher普通轨迹 BF16 RWM：第120轮完整512

231/512（45.1171875%），NQ109/256，Hotpot122/256。八worker实际wait均0，完整释放accepted。

比110轮多2题，比source100同协议239少8题（-1.5625个百分点）。近期回升，尚未回到基线；未计算本次配对置信区间，不做单机制归因。

raw CP1206/outer119 manifest SHA3ac06afd51b3396a63ba3c6ff526b12c4471fbd8d79b175b5c4ff33718f1f3ad。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-120/summary.json SHA1462cc06e8f226582b54842db739335bd76b507fef8a0c51b30eb775a0af9f97。

实读1791721958：122轮/1227Adam完成，轮次边界短暂Zombie admission HOLD，后续实读确认，不热改或重启。
