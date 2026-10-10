# 固定teacher普通轨迹 BF16 RWM：第40轮完整512

223/512（43.5546875%），NQ102/256，Hotpot121/256。8worker实际wait全部0，完整释放accepted。

10/20/30/40轮正确数243/236/233/223，source100同协议239；40轮相比基线少16题（-3.125个百分点），相比10轮少20题（-3.90625个百分点）。四个保存点评测呈下降，固定teacher轨迹未显示持续收益。未计算本次逐题配对置信区间，不断言统计确定下降或单机制原因。

真实raw CP405/outer39 manifest SHA9ec74b236b2b042913276970fe98bb4886a1d53b0c041d2cb67bdcd66179d43e。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-040/summary.json SHA490565348ac1956eae9dcb1231228f7a639756f308d249af17ca7b17026e79ae。

实读1791667185：同owner967452完成40轮/405Adam，第41轮已实际生成14387433字节rollout，无报错。继续既定200轮及正常每10轮512，未调整参数或重启。
