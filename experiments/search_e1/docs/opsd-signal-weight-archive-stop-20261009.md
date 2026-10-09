# HKUST4旧archive人工替换停止与四卡释放

2026-10-09 17:05:42 CST 独立终态验收通过。本次人工“那个search-e1已经崩溃可以停掉了”授权停止旧archive，以便在四张48G卡上安排新的加权消融；仅替换该旧run，不暂停整个Search-E1目标，不停止K1七组、公共检索、两个监测进程或GPU0–3他人任务。原人工消息精确时间未由provider提供，授权记录观察时间为2026-10-09T08:36:04.039269+00:00，二者不混同。较旧heartbeat要求K4继续的文字不能自动重启已被本次人工明确停止的对象。

## 实际停止对象与结果

- HKUST4=`xielab5`、UID1001；根目录 `/weka_3/shaoshitong/search-e1-config-study-20261004`。
- 旧run `f04fe593830c4831918ede872812c016`，包括三个已核验supervisor及其确切归属子进程，共188个历史身份。每个PID均绑定effective UID/starttick/run/实际argv；只使用pidfd，不使用进程组／名称杀进程。
- 188个身份均为ABSENT_OR_ZOMBIE，没有活跃旧任务或自动后续阶段。只停止自己旧run，其他进程不在范围。
- GPU4–6各used12MiB/directfree48749MiB。GPU7 used766MiB/directfree47995MiB，唯一compute为原检索1749337、738MiB。`memory.free`是实际读值，另记录每卡379MiB driver保留量。
- 检索1749337及observer2062794、1751713前后effective UID/starttick/argv一致且存活。GPU0–3他人compute未动。

## 原件与副本

旧日志最后为199真实GRPO更新完成、200已进入；reward/finish为0是性能崩塌，不能说此前进程已故障退出。actor检查点50/100/150共36文件原地保留；停机前作了完整文件SHA、safetensors header/shape/dtype/EOF/index覆盖核验，终态只再核名字／size／mtime连续性。终态不冒充重新读取所有模型payload、tensor finite、或完整199 Adam/scheduler/RNG续训状态。

保留目录：`/weka_3/shaoshitong/search-e1-config-study-20261004/preserved/user-stop-archive-20261009-057b905a73ae4ca2a28c220466d5608a`。source/runtime/opsd-runtime/logs/receipts/train.log副本4449文件、91891952bytes已独立fresh SHA验证；原文件和检查点未删除。原始result/preservation/quiesced字节也已复制Mac并与Node逐SHA核验。

## 保留技术失败与独立终态

- V1仅预检HOLD：Node Python构建未暴露pidfd API，零信号。V2在原Ray子进程单卡CUDA mask预检HOLD，零目标信号。两个版本源码、测试及部署保留。
- 发布V3原helper SHA `79f760c6cfed229cf2aad34347dcf62acbce580f6b8650df571fce6f0e582a1f`，49CPU fixtures通过。Linux x86_64内核原生434/424 pidfd适配不退回数字PID信号；合法4–7子集仅用于确切旧run归属。
- V3完成保存及564个STOP/TERM/CONT信号后，在driver退出瞬间读到argv变化，留下原rc2/HOLD。原状态完整保留，不能写成训练OOM／崩溃或假称脚本当时rc0。
- 独立只读verifier SHA `b6977cdecdf62a8213281f894e17751e415ef06923cc185ed7ad23365c832db4`；Mac/Node各8CPU fixtures通过，实际Node执行rc0／accepted=true。它不发送新信号、不恢复旧run；前后检查全部历史身份、保护进程和GPU，并fresh SHA复制证据。原HOLD没有被覆盖。

Node独立终态receipt：`/weka_3/shaoshitong/search-e1-config-study-20261004/receipts/human-stop-archive-f04fe593-20261009-4921e2a715fe472d8adf0cbe0d955483.json`。
SHA256：`6fd6a4af1e3c44e902f515e833a4e99d221946633a370c6f48579390b78d8eea`；Mac对应原字节在 `work/search-e1-analysis-output/opsd-signal-weight-plan-20261009/archive-independent-terminal.stdout.private.json`。完整私有原始轨迹／授权不进入Git。

## 新实验状态

四种加权数学定义、代码改动清单、每组25 online outer及每5轮同协议512评估已写入 [计划](opsd-gap-entropy-four-weight-ablation-plan-20261009.md)。数学参考14CPU tests通过。新训练未启动：指定线程实际命名C0/C1/M/R/W/P/J，无E；R=240/512是候选，但需绑定用户所指E。R比共同起点239仅多1题，不是可靠增益的证明。新生产四rank版本尚待按确定基线实施；本次没有改正跑K1作业或共享环境，也未通知原七算法线程。
