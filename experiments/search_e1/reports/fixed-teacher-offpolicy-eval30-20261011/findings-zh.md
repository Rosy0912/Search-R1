# 固定teacher普通轨迹 BF16 RWM：第30轮完整512

233/512（45.5078125%），NQ106/256，Hotpot127/256。8worker实际wait全部0，完整释放accepted。

10/20/30轮正确数243/236/233，source100同协议239；30轮相比基线少6题（-1.171875个百分点），相比10轮少10题（-1.953125个百分点）。当前三个固定评测点未显示稳定提升；未计算本次配对置信区间，不能断言统计确定下降或单机制原因。

真实raw CP303/outer29 manifest SHA c0348f8eeaa97fc79e42a69ef1310e3bc409139279fd32540b915fbee48d88ed。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-030/summary.json SHA83b4921ac1a73a41f3fcb8ed1b8f831074c245fcb15c4272032f1f884fb75ce4。

实读1791663428：同owner967452/run5dd5f0d68c4249ae9cb273bed38dabe3已完成37轮、374Adam，第38轮训练，无报错。继续200轮与每10轮正常512。
