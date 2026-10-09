# R/W/P/J 保护状态的独立只读备份

这是已批准七组目标的 Mac 侧薄入口，补齐外层结束队列不包含的 initial、update1、update5。不修改已有 M 入口、V2/compact 队列、训练、评测、配置或 Node 资产；不取训练锁，不发信号，不用 Torch、CUDA、HTTP 或模型推理。当前正跑源保持原路径和版本，本次实现只新增此目录，未在 Node 执行或进行真实传输。

## 设计与范围

`backup.py` 单次处理一组。固定原七组授权 goal 的 path/SHA，选 R/W/P/J 中一组的原 config/run；Node 对同一 goal 和 config 再 fresh SHA。首先只读 pin 真正 committed 的 u0/u1/u5 manifest，保存精确 path、SHA、实际 outer 后才读取文件。缺任何一个状态（包括 CP5 尚未产生）返回 `PENDING`，不伪造缺失状态。未提交 `.pending-*` 目录不参与。后续调用使用完全相同 scope 原字节及 SHA，禁止 mutable latest。

当前七组实际 schema 是 `opsd-online-checkpoint-v1` 和 `shared_exact_eight_rank_optimizer_v1`。CP0 在 outer0/cursor0，包含初始空 Adam 的 `optimizer-shared.pt`（已存在 M 实际文件为3296B）、八份 rank RNG/cursor/scaler 文件及三个 PEFT 文件。入口不反序列化 Adam，也不要求初始状态具有非空 moments；复制全部原字节。每 CP 的核心集合恰为13文件：manifest、shared optimizer、rank0–7、adapter README/config/safetensors。u1/u5 取最早实际 committed outer；有效 K0 可使它们晚于 outer0，不假定均在第一轮。当前 producer 不支持 outer=-1，此处明确拒绝。重复更新计数的后续 K0 状态不替换已固定的原保护状态。完全不要求 `outer_end`，不等待十轮结束。

每包39个核心成员，另加原 config、run-identity、goal 三个小身份文件，共42成员。来源 config/runtime/model/common-initial 合同与 manifest 对应；这些只是声明和文件身份检查，不是八 rank 数值或科学验收。恢复所需四个 training inputs、outer-old/common adapter 及冻结 source100/runtime 的其他闭包由原独立 raw/checkpoint/source 备份保留，本包不宣称单独包含全部外部训练依赖。

复用原已发布 V2 `backup.py` 的固定句柄 SHA、每路径组件拒 symlink、USTAR stream、安全逐成员解包和第二次完整本地磁盘 SHA。依赖 exact SHA 为 `d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57`；以已验源码裸字节 compile 执行，不 import/runpy 或读取 timestamp-valid pyc。Node 使用既有该依赖路径，不部署新源码，传输的新 operation 是经本目录 publication/readback 绑定的精确 `python -B -c` 源码。

流程为 pin → 冻结 scope → fresh 42文件内容 SHA/metadata → 原 USTAR 固定句柄 stream/再次逐文件 SHA → source-after（精确三核心集合、manifest SHA、所有源 metadata 稳定）→ Mac 安全解包/成员 SHA → 独立第二次本地文件全 SHA → complete。source-after 是 fresh metadata/manifest 检查，不谎称再次扫描所有源内容。每次实际 SSH rc 持久化，进程消失不算成功。特殊/重复/逃逸/额外/缺失/隐藏扩展成员及尾部非零数据拒绝，原失败 partial、日志与 Node 原资产保留。

## 调用

root 先提交推送、FETCH/逐文件 readback，并提供本目录 source publication（保留四个或全部七个原 run_ids）。下面 `NEW_ATTEMPT` 必须是新的绝对目录，父目录已经存在；重试用新目录，不覆盖既有 PENDING/BACKLOG/partial/complete。

```sh
python3 -B /ABS/PUBLISHED/opsd_ablation_protected_backup_v1/backup.py \
  --goal /ABS/PRIVATE/original-goal.json --goal-sha256 ORIGINAL_GOAL_SHA \
  --remote-goal /data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/receipts/outer-backup-v2-554c52c-20261009/goal.json \
  --variant R --destination /ABS/PRIVATE/NEW_ATTEMPT \
  --source-helper /ABS/PUBLISHED/opsd_ablation_outer_backup_v2/backup.py \
  --operation-sha256 PUBLISHED_OPERATION_SHA \
  --source-manifest-sha256 PUBLISHED_MANIFEST_SHA \
  --publication /ABS/PRIVATE/publication.json --publication-sha256 PUBLICATION_SHA \
  --source-commit FULL40_REMOTE_READBACK_COMMIT
```

W/P/J 只换 `--variant` 和独立 destination。推荐 root 在每组 CP5 实际 committed 后异步调用；也可更早调用得到明确 PENDING。入口一次运行不常驻轮询、不启动后台训练。直接路由固定 `/usr/bin/ssh -B en0`、ProxyCommand/ProxyJump/ControlMaster 禁用，目的仅 hkust1。Node CVD 空、UID1005、无本 run marker。scope 以严格 base64 原字节和 SHA 通过命令参数传入，JSON inventory 单独 stdin，不依赖不存在的 Node Mac 文件。

新 attempt/extraction 根0700。真实 inventory 决定空间门禁：原 tar 物理上界＋全部解包字节＋5GiB 余量；不足则保存 `BACKLOG` 并不发 stream。未来队列和 raw 占盘仍需各自 fresh space gate；此检查不是所有并行写入的全局空间预留。绝不自动删旧备份、Node 状态或独特资产。空间不足只影响备份，不能成为 GPU hold 或停止训练的依据。

## 验证边界

四个必要 CPU fixtures：身份拒绝/CP5 PENDING；实际 CP0 13成员与 later-outer u1/u5、outer=-1拒绝；固定 manifest/payload 异变与 source-after拒绝；空间 BACKLOG。使用微型不透明字节检查传输合同，未加载 Torch，不将 toy 当真实 Adam 或实际 Node 传输验收。原 M 文件格式通过已有实际 manifest 与发布 trainer 源接口核对。

```sh
python3 -B -m unittest discover -s /ABS/PUBLISHED/opsd_ablation_protected_backup_v1 -p 'test_*.py' -v
```

最终 `opsd-ablation-protected-local-backup-v1` 仅表示42原文件的传输身份、实际 rc0、源稳定和完整本地 SHA 通过；数值 Adam/梯度、训练进度和评测效果另行验收。本候选没有真实备份执行结果。
