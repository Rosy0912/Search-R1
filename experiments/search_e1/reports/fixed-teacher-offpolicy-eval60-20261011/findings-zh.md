# 固定teacher普通轨迹 BF16 RWM：第60轮完整512

226/512（44.140625%），NQ105/256，Hotpot121/256。8worker实际wait均0，完整释放accepted。

10/20/30/40/50/60轮正确数243/236/233/223/227/226。60轮比source100同协议239少13题（-2.5390625个百分点），与50轮相差-1题，未显示持续提升。未计算本次配对置信区间，不归因单一机制。

raw CP608/outer59 manifest SHAe6d12fb6159d565edb9a6097264438d97572de03deacb1c26eed4c0a7a7c5924。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-060/summary.json SHA26eded3ffff6ce0abb396983cc4e56d00478afe328d04882fe5ce2630704047e。

实读1791681792：同owner967452完成63轮/637Adam，第64轮采样，无报错，继续200轮及每10轮评测。
