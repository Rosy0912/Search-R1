# 固定teacher普通轨迹 BF16 RWM：第70轮完整512

230/512（44.921875%），NQ105/256，Hotpot125/256。八worker实际wait均0，完整释放accepted。

10至70轮正确数243/236/233/223/227/226/230。比60轮回升4题，比source100同协议239少9题（-1.7578125个百分点）。尚未显示持续提升；未计算本次配对置信区间，不归因单一机制。

raw CP707/outer69 manifest SHA f992b359c23814fed0981f8178acab17794d4de2f868cd7852ffafdd548e0559。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-070/summary.json SHA5b33fa53ee466068c58a48663f0bd2ae19055b3099004e139cea93d6665e9f78。

实读1791689325：已完成75轮/755Adam，第76轮采样，无报错。继续200轮及每10轮正常512。
