# 在线 OPSD 状态、备份与终态证据

本次保持同一在线 run `1d1d67b60f0e46a2afc0536a36ab69b1`、配置 `d0c30df9b035e73d87bb5402bbcd32c16b2314a4180c451a543522ca5b321b17` 和实际 CPU-merge recovery `51eb6c574dd726b929dffcd6c0abfdc7b441a5d4`。上限10、outer200、LR5e-6 constant、原矿选、固定 teacher、轨迹内 token 均值后实际 pair 均值及所有科学设置未改。

## 实际状态审计

2026-10-08 20:35 CST 只读快照绑定原 controller 的 PID/UID/starttick/完整 argv，实际核验305文件冻结 runtime、23个原助手文件、两个恢复源码及出版链、source100全部12文件与 TRAIN/native512 数据的新 SHA。没有修改节点文件、生产 RNG 或模型，也没有启动或停止任务；只请求既有18004的 health。

该快照记录17个完整 outer、169次已完成轮次的 Adam 和5153对累计曝光，第18个 outer（索引17）训练中、最新已提交更新177。连续完成票据、outer/checkpoint manifest 与最新完整 CP169 的文件 SHA 对账通过；这不等于反序列化了177次 Adam 的全部内容。八卡当前最少实际余量15003MiB，observer新鲜、历史低于门槛计数0；既有 oracle 身份未变，没有外来 GPU 进程。本次读取 /data 实际可用1545887328KiB。所有数字是对应快照，不自动作为更晚进度。

独立 CPU 读取此前明确固定的 CP149/outer14，manifest `5079cabf70e7cb3c13eb7d93f675efa7dcff1d8af14f2c37237ad41224dc7d01`：八 rank 的真实 Adam 均为149，完整 moments/groups 逐项一致有限；504个 adapter 张量全部有限并相对保留 CP5 改变，最大绝对变化0.0010421955958008766。该 outer 的10次更新与296条 token 文件清单对账通过；末批8条轨迹的全部1510个有效 token、原 mask、轨迹均值及实际尾批分母8通过。没有重算 logits、分词器或更早所有 token。真实执行 exit0，私有独立审计 SHA `9a96dae7dde0229f99550e7d2c84c6b62b00d722363fcf83d20db567bda390a3`。

HKUST4 原 archive 同次只读记录174/200 GRPO已完成、175进行中，174的 reward0.525/gradnorm0.662有限；最新验证与模型检查点仍为150，不能将其称为174的验证。五个正确映射的669文件冻结源码目录均逐 SHA 一致，自有 GPU4–7/原控制与检索进程保持。未停止或改变 archive。

## 持久副本验收

原保护 CP0/1/5/11、关联输入及紧凑验收记录已完成本地备份：58个文件，成员总字节6239760772。传输实际 exit0，逐成员完整 SHA 与节点 inventory 一致，安全解包后的完整副本存在；原节点数据继续保留。完整票据 SHA 为 `0169e6b4f9be5429e3fb0d39211cec16cf6a234e208eb32f6af6ad9a6718db65`，归档 SHA 为 `0646f2da8b338108f753f880b30859a96831681cc99767cc6c47a50da93e9dd1`。

root另核文件清单、覆盖、字节数、SHA声明一致与本地实际存在；没有再进行第二次6GB逐字节重算或将全部成员反序列化。以后最终状态和评估原始证据仍须备份；这次完成不代表整个200轮的资产已经全部备份。

## 后续终态与效果边界

原200轮训练完成后没有自动 native512 评估，继续按已发布完整验收与效果计划执行。预选 source100、CP89 与最终 outer199 raw adapter；不能以 outer199 的采样 cache（outer198权重）代替最终模型。真实训练完成、终态 rc/release、完整科学状态对账和固定权重效果是分别验收的证据。

为保留真实退出证据，新增辅助入口仅核 live→terminal 的同一 tmux server/socket/pane、单前台 Bash 命令、原 PID/UID/starttick/argv和已发布源码链。节点既有 pane 已启用 remain-on-exit；助手不修改 tmux 选项。只有该同一保留 pane 的实际 dead status0、同一 controller的 COMPLETE200、所有自有后代释放，才可确认这个 controller 的 exit0。历史失败 attempt 不由该状态推断为成功。读取不确定、进程消失或源码兼容问题是 UNKNOWN，需要另查，不能当作训练失败或成功。

没有产生新的 native512 EM；随机 TRAIN reward、loss和状态验收均不表示效果提升。不会恢复旧 M3/M4、旧 GRPO或旧八卡七基准作业，也不会自动追加训练消融。
