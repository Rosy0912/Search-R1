# 固定teacher普通轨迹 BF16 RWM：第80轮完整512

220/512（42.96875%），NQ100/256，Hotpot120/256。8worker实际wait均0，完整释放accepted。

10至80轮正确数243/236/233/223/227/226/230/220。相比70轮少10题，比source100同协议239少19题（-3.7109375个百分点）；是目前保存点评测最低值，未显示持续收益。未计算本次配对置信区间，不归因单一机制。

raw CP805/outer79 manifest SHA4f2a39e0d1db27cbf28cd99e8bfa3517404dc1ff39d68153378f635d6507ad29。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-080/summary.json SHAa310a92dd16498ddd59122912556d9c0cd5cffc36aedbfa1b6393b652f825485。

实读1791696608：已完成86轮/865Adam，第87轮采样，无报错。按授权继续200轮及每10轮评测，未改参数。
