# V2 interface repair

Independent copy of the frozen V1. Only the transfer and verifier inventory consumer changes: retain the original full inventory object; call `validate_inventory` for validation without replacing that object with its member mapping. Original scope, source-operation, USTAR, gzip, member SHA and Node assets are unchanged. V1 and its failed execution remain preserved.

# 闭合 native512 评测的通用只读备份

仅适用于已授权原七组 study `reverse-seven-a1dacc9d8c77451c8b04140e8e355777` 和固定 eval72 / driver8b 输出根。复用已发布 C1 备份的固定句柄 SHA、USTAR、gzip、footer、全部成员 SHA；不修改 C1 或其他已发表入口，不部署/写 Node、不取训练锁、不发信号、不用 Torch/GPU/HTTP、不重新推理。CPU备份不会成为训练或评测的依赖。

## Scope

新的不可变 JSON：`schema=opsd-ablation-closed-native-scope-v1`，`accepted=true`，精确 `study_id`、`variant`、**eval** `run_id`（不是训练run）、`versions`、`parent_identity={pid,uid:1005,starttick}`。

`pins` 必须含 config / summary / controller / accepted / parent_exit / parent_release 六个原 Node `{path,sha256}` 引用，可带 bytes。`phases` 按 version 各含 phase / terminal 两个原文件引用。全部SHA来自原字节或原嵌入 fileproof，不使用重格式化的本地JSON冒充原源SHA。任一闭合证明尚未产生则 scope 不可准入；缺文件/不闭合只是备份 HOLD，不是训练失败。

固定根（脚本由 variant 推导，scope 不得改根）：

- native：`N/eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851/<variant>/runs`
- parent：`N/eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs/<index>-<variant>/evaluation`
- config：parent 的上级 `evaluation-config.json`。

N 是本study主目录 `/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008`。index 为 C0/C1/M/R/W/P/J 对应00–06。pins 的固定文件名分别为 summary.json、controller-complete.json、result-accepted.json、actual-child-exit.json、actual-child-release.json；每 phase 为 phase-complete.json 和 actual-terminal.json。

C0/C1/M/R/W 是 `versions=[raw10_bf16]`，512正式槽；P/J 是 `[raw10_bf16,raw10_fp16]`，1024正式槽。不包含共同zero评测；额外控制生成不加入正式分母。

先核原 config/schema/group/helper72 和完整父进程 exactargv/configSHA、真实 parent wait0/非signal及原 accepted release。controller与原 summary/config引用相符；每阶段严格八个唯一 PID/tick/run/父链、真实wait0，worker complete64/rank/GPUUUID/config/source/runtime引用与原summary一致。原results固定SHA及已有每行身份验证512唯一题/固定logical分片；双阶段须同一题ID/semanticSHA/dataset。strict hits只核**原记录的** protocol_em 与原summary相符，没有再次调用QA、模型或语义评分。

## 覆盖和排除

复制闭合 native tree、该组闭合 parent evaluation tree和原 eval config，保留完整原worker结果、private traces、HTTP、生成IDs、退出/释放/CPUmerge元数据。唯一已知模型缓存排除为 P/J `raw10_fp16/cpu-merge/model`，是原controller实际CPU部署产物路径；其元数据仍备份。其他无法解释的 model/models/cache/checkpoints/merged-model-cache 子树 HOLD，禁止静默夹入模型或扩大排除范围。

活动study driver、全局mutable角色/observer、训练raw/CP、模型/环境、其他group/zero不会枚举。包内闭合receipt引用的release是**当时**证明，不当作当前资源门禁，不意味着整个study已结束。所有Node原资产保留。

## 使用

root 先发布本目录并完成Git/FETCH逐字节readback；调用的 operation SHA 必须是已发表版本，下面源和 scope 路径都是Mac路径。Node不需要Mac scope文件：原 scope 字节 strict base64和SHA随 `-c` 参数传送，inventory JSON另走stdin，避免源码/JSON流混合。

1. 将固定 scope 原字节保存到新的私有文件并取 SHA。一次只读 inventory 调用使用同一已发布 source-operation.py 原字节：

```text
/usr/bin/ssh -B en0 -oProxyCommand=none -oProxyJump=none -oControlMaster=no -S none -oBatchMode=yes -oConnectTimeout=10 hkust1
  env -u E1_RUN_ID CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 ORIGINAL_ENV_PYTHON -B -c EXACT_OPERATION_SOURCE
  inventory --scope-base64 EXACT_BASE64 --scope-sha256 SCOPE_SHA
```

实际 argv 由安全 shlex quoting 构造；不把上述多行展示直接粘成未转义shell命令。可对同入口 `after` 使用原 inventory JSON stdin，保存真实stdout/返回码。

2. Mac独立传输（新绝对destination；失败partial不覆盖）：

```sh
python3 -B /ABS/PUBLISHED/opsd_ablation_closed_native_backup_v2/transfer-gzip.py \
  --scope /ABS/PRIVATE/scope.json --scope-sha256 SCOPE_SHA \
  --inventory /ABS/PRIVATE/inventory.json --destination /ABS/PRIVATE/NEW_ATTEMPT \
  --source-operation /ABS/PUBLISHED/opsd_ablation_closed_native_backup_v2/source-operation.py \
  --operation-sha256 OPERATION_SHA --source-helper /ABS/PUBLISHED/opsd_ablation_outer_backup_v2/backup.py
```

3. stream真实rc0后，原入口 `after` 核scope、closure、全成员集合和源metadata稳定。随后Mac全压缩文件SHA、解压每member完整SHA、连续regular USTAR header和gzip/footer/exactset验收：

```sh
python3 -B /ABS/PUBLISHED/opsd_ablation_closed_native_backup_v2/verify-archive.py \
  --scope /ABS/PRIVATE/scope.json --scope-sha256 SCOPE_SHA \
  --inventory /ABS/PRIVATE/inventory.json --after-metadata /ABS/PRIVATE/after.json \
  --destination /ABS/PRIVATE/NEW_ATTEMPT \
  --source-operation /ABS/PUBLISHED/opsd_ablation_closed_native_backup_v2/source-operation.py \
  --operation-sha256 OPERATION_SHA --source-helper /ABS/PUBLISHED/opsd_ablation_outer_backup_v2/backup.py
```

最终 schema=`opsd-ablation-closed-native-compressed-local-backup-v1`；保存原scopeSHA/closure/source-after/真实SSH返回码。压缩包新根0700，无第二份裸解包。容量使用真实inventory的tar上界＋1%＋5GiB；不足保留库存并排 BACKLOG，不干预GPU。失败保留partial、stderr和原资产，无删除或复活作业。

V2依赖 exactSHA `d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57`，fresh原源码裸字节 compile，不读timestamp-valid pyc。Source-after是fresh小receipt/manifest/result引用及metadata检查，不谎称所有trace再全内容hash；stream原算法逐fixedhandle再次SHA验证原内容。

## 候选验证

仅两项微型CPU fixtures：合法单/双原producer形状闭合；错run/真实parent或worker非零拒绝。没有重新运行模型、数学或大归档测试，没有Node或真实传输结果。原archive算法未改；内部member padding不宣称独立逐字节核验。

```sh
python3 -B -m unittest discover -s /ABS/PUBLISHED/opsd_ablation_closed_native_backup_v2 -p 'test_*.py' -v
```
