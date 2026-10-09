# 已完成检查点压缩备份的 USTAR 边界修复

本修复只针对 Mac 上自己的 checkpoint backup queue。训练、GPU评测、Node原件和科学源码保持原样，训练与备份继续并行。

## 实际问题

CP61/outer5两次远端SSH实际rc0、stderr空；31个文件、602628560成员字节身份相同。成员header与各512块填充总S=602654208，S mod10240=9728。原compact采用S+10240上限；Python tarfile USTAR先写1024字节结束块，再对齐到10240字节record。真实长度ceil((S+1024)/10240)×10240=602664960，恰比旧上限多512。正常尾填充被本地错误拒绝，未发现训练或源文件动态异常。失败partial和小票据保留。

## 最小修复

独立外部overlay复用原compact4d2065e、legacy554c52c和cleanup275aa9b，只把raw_upper改为精确record长度；compress、XZReader、压缩writer上限和disk reserve继承同一界限。原stream_archive和归档字节不改变。复用原cleanup修复，旧accepted包保留原provenance、压缩字节和成员SHA证明，不重压。新包额外记录bound-recovery源码和发布身份。

延用同70包goal、destination和原Mac queue锁；旧source不热改。前态必须有实际PID3547/UID501/launch/argv/queueidentity/status、真实终态rc及自有后代释放证明，不能仅靠PID消失推断成功。在当前传输自然结束、队列无live child的技术恢复边界切换；第三次旧界限失败也保留actualrc和partial。新工具先提交推送并远程逐文件读回，再执行。

## 验证与完成范围

必要CPU回归只验证20种512字节record余数及实际CP61 inventory长度，用标准tarfile内存小包，不重跑模型或旧科学fixtures。备份成功仍要求实际新传输rc0、固定成员SHA、无额外或遗漏成员、独立disk解压与whole-USTAR SHA。数学回归或初始化不代表CP61备份已完成。

备份不阻塞训练、续训或每组10轮后的512题评测。
