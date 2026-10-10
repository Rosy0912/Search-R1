# 固定teacher普通轨迹 BF16 RWM：第50轮完整512

227/512（44.3359375%），NQ106/256，Hotpot121/256。八worker实际wait均0，完整释放accepted。

10/20/30/40/50轮正确数243/236/233/223/227。相比40轮回升4题，但比source100同协议239少12题（-2.34375个百分点），尚未显示持续提升。未计算本次配对置信区间，不做单机制归因。

raw CP507/outer49 manifest SHA9848a9e240d42701324507c0df04fd212c4ffc797feac8976be6105d1569d504。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-050/summary.json SHA359ea3e8ffdd6dc6a03a81484583253798c7e58230c3899bd998430e0d921de2。

实读1791674440：同owner967452完成52轮/527Adam，第53轮采样，无报错。继续既定200轮及每10轮评测，无参数更改。
