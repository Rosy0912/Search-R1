# 固定teacher普通轨迹 BF16 RWM：第110轮完整512

229/512（44.7265625%），NQ108/256，Hotpot121/256。八worker实际wait均0，完整释放accepted。

比100轮多1题，比source100同协议239少10题（-1.953125个百分点）。近期小幅回升，尚未回到基线；未计算本次配对置信区间，不做单机制归因。

raw CP1106/outer109 manifest SHA0b786313758edae07b62c7377fce0becd2c1657f9856528045e4118158ccb54a。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-110/summary.json SHAf7263f19746dd6bb518c93cd10af16e877fcb6513bf310fcce5734c693074674。

实读1791714518：110轮/1106Adam完成，第111轮正常训练，无报错。继续200轮及每10轮评测。
