# 固定teacher普通轨迹 BF16 RWM：第20轮512评测

完整512为236/512（46.09375%）：NQ104/256，Hotpot132/256。8worker实际wait均0，full release accepted。

第10轮243，第20轮236，净少7题（-1.3671875个百分点）。同协议source100既有239，当前净少3题（-0.5859375个百分点）。目前不能认定固定teacher采样产生稳定提升，也不能仅凭两个点归因为单一机制；未计算本次逐题配对置信区间。

真实raw checkpoint201/outer19 manifest SHA5b3575bbfdf220cc4f34c00dc3335d09d306b0d074f84d46eef6e397298bc055。
远端summary /data/shaoshitong/search-e1-opsd-fixed-teacher-bf16-rwm200-20261010/periodic512/eval-020/summary.json SHA299192a6da1df2026fab57e87fef6b09bd56229c1281dc06a3d0ec7911c523b0。

实读1791656108：同owner967452已完成27轮/272Adam，第28轮采样，无报错，继续200轮及正常每10轮评测。
