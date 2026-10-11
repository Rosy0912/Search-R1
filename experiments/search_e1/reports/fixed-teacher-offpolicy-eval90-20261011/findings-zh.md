# 固定teacher普通轨迹 BF16 RWM：第90轮完整512

227/512（44.3359375%），NQ102/256，Hotpot125/256。八worker实际wait均0，完整释放accepted。

10至90轮正确数243/236/233/223/227/226/230/220/227。相比80轮回升7题，比source100同协议239少12题（-2.34375个百分点）；尚未显示持续提升。未计算本次配对置信区间，不做单机制归因。

raw CP906/outer89 manifest SHAc6391838ba372c71211124924459ea91804c75fee361d94c9d3e8658b7594fbf。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-090/summary.json SHA1507a84a23ad53adc9d2cb2d7c5cf70e59be263988676261813a5b5d135dc8c6。

实读1791703476：96轮/967Adam完整完成；采样边界短暂Zombie admission HOLD，需后续实读，不将其认定模型故障或擅自重启。
