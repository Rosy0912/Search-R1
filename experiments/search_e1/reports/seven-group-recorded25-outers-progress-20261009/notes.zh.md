# 25个闭合训练外轮的记录曲线

该快照于2026-10-09 10:26:28 CST采集：C0十轮、C1十轮、M五轮，25/70完成；251次真实Adam、7538条pair曝光、1528283个有效policy token。旧23轮事实（包括绑定的源文件证明）逐项相同，旧报告完整保留。M尚未完成十轮，也没有M固定512结果。

CSV保留70个显式槽，未闭合槽留空；图中M虚线表示部分组。每outer采用新的512个TRAIN问题并采样2560轨迹，TRAIN strict reward不能当固定heldout准确率。正式评测覆盖仍为2048/5632槽，C1为236/512，共同BF16起点为239/512。

M累计1503条pair、316110个有效token、50次Adam。五轮K=327/308/293/284/291，Adam增量11/10/10/9/10，实际尾批7/20/5/28/3，对应1574/4242/1029/4760/529个有效token。目标按每条轨迹的有效token均值，再除实际轨迹数；跨outer均值按实际K加权，不除权重和或更新数。

M五轮混合目标loss为0.006661280478，teacher KL为0.405164738074，outer-old KL为0.000765205401，1503条轨迹标量完整覆盖且有限。各轮native clipped vocabulary components为0，committed overflow/retry均0。outer4尾批三条轨迹的preclip norm为0.1534846574，有限；不能把尾批标量变化直接称为不稳定或因果证据。clip mass/vocabulary fraction缺失，保持NA。

teacher/outer-old/mixed-target KL分别指向不同分布，不能直接以不同目标的绝对loss判组间收益，也不代表梯度贡献。这次仅读取committed标量，并未读取或生成自主teacher轨迹、重新评分、扫描token或重读模型/Adam tensor。M效果仍等待固定512评测。

本次复用已发布CPU collector和formatter，实际采集、派生均rc0，直连观察成功；没有采用临时转发、重启生产或修改全局网络。公开summary移除私有路径及执行细节；CSV/PNG/SVG与成功派生产物逐字节一致。该记录验收不表示70轮、11模型或整个study终态已完成。
