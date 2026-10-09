# 七组保护点副本的本机无损压缩与重复件回收

本入口只接受显式、已验收的 R/W/P/J `u0/u1/u5` 42成员备份。当前 R/W 已有完整备份；P/J 必须等各自真实 accepted receipt，不能用占位符运行。没有自动发现、远端访问、模型/训练操作或 GPU 门禁。备份和磁盘整理不阻塞科学队列；既有 compact queue 不变。

只加载下列原已发布源码的精确 SHA 字节，绕过 pyc，不调用其固定旧实验 main：

- `opsd_local_archive_recompress_v1/recompress.py`：`9bfb249e78957e7f34c38b1d8beacf0d75cadae68b15106793b91dce212ac921`，复用 `compress_exact` / `verify_disk_xz`。
- `opsd_backup_duplicate_reclaim_v1/reclaim.py`：`4c4d2518c9fc44f8ef75ef0ef31c3fe51d7cb00f2de0242932587c26f2bd3e6c`，复用路径/owner/nlink/metadata/SHA、`lsof`、独占写入/fsync 等原语。
- `opsd_local_single_tar_reclaim_v1/reclaim.py`：`70ea28a46393db506f3992010e7558c94781adb565dbfe14556ecc0b7ca5aa7e`，唯一删除操作复用 `unlink_one` 的 exact dirfd/stat + durable pending 算法。

新目录需先提交、推送、逐字节回读，运行时显式传入本目录清单 SHA、publication 的 path/SHA 和 commit。publication 必须绑定对应固定 run。源码候选批准不等于实际压缩或回收已经完成。

## 固定输入与三阶段

每个组一个小输入，不复制实际权重内容：

```json
{"schema":"opsd-ablation-protected-compact-input-v1","variant":"R","backup":{"path":"ABSOLUTE_ACCEPTED_BACKUP/complete-sha-verified.json","sha256":"ACTUAL_ACCEPTED_RECEIPT_SHA"}}
```

输入仅允许本实验 analysis 根下的原完整保护档。验证固定 run/config/goal、原42成员映射、0/1/5 scope、source-after、原 stream/read-only 操作真实返回0、原 V1 source publication/readback。不读取 Node，不修改原 receipt。

三次命令都带公共参数：

```text
env PYTHONDONTWRITEBYTECODE=1 ACTUAL_SYS_EXECUTABLE compact.py MODE
  --source-manifest-sha256 SHA
  --publication ABSOLUTE_PUBLICATION_JSON --publication-sha256 SHA
  --source-commit COMMIT
```

1. `compress --input PATH --input-sha256 SHA --output NEW_XZ_DIR`：Mac UID501、本地独占 lease、空间至少原 tar 的1.01倍+1MiB+5GiB、无打开的原 tar。写 XZ partial，独立磁盘解码，严格单流 CRC64、完整原 tar SHA/bytes 与稳定源/句柄核验。原 tar 的所有 header/padding 字节也必须相同。成功才 rename 为 `archive.tar.xz` 并写 `complete-xz-verified.json`。失败 partial 保留，不删任何原资产。
2. Root 等上一步**实际进程 wait0**，保存下面 parent terminal；然后 `prepare --input PATH --input-sha256 SHA --completion PATH --completion-sha256 SHA --parent-terminal PATH --parent-terminal-sha256 SHA --output NEW_PLAN_DIR`。缺 parent proof 或非0直接拒绝；不由 completion、child、PID消失推断 parent0。仅此阶段 fresh hash42展开文件，核对原成员 SHA；保存完整 inode/device/size/mtime/ctime/mode/UID/nlink。检查整个 `verified-files` 恰好42文件和 no-open，产不可变 `plan.json`。
3. `execute --plan ABSOLUTE_PLAN_JSON --plan-sha256 SHA`：重核全部小链与 XZ 当前 metadata，核原 source/pub 不变。只按43个 exact targets 回收42 loose、最后原 tar。每文件 durable pending/done、立即 dirfd identity/no-open 检查；有既定 pending 才可识别崩溃后 absence，明确不把它称为本次实际 unlink。相同 plan 可以继续，不能从别的 plan 认领已缺失资产。保留空目录，XZ、全部 receipt/source/inventory、旧失败 partial、Node originals 永不删。

parent terminal（Root 在真实 wait 后生成）字段合同：

```json
{"schema":"opsd-protected-compact-actual-parent-terminal-v1","actual_pid":123,"uid":501,"actual_returncode":0,"status_not_inferred_from_absence":true,"argv":["EXACT_START_ARGV"],"start":{"path":"ABSOLUTE/start.json","sha256":"SHA"},"completion":{"path":"ABSOLUTE/complete-xz-verified.json","sha256":"SHA"},"actual_wait_evidence":{"path":"ABSOLUTE_WRAPPER_EXECUTION_JSON","sha256":"SHA"}}
```

`actual_wait_evidence` 必须是原实际执行小 JSON，含相同 `actual_pid` 和 `actual_returncode:0`。该 compression CLI 不启动自己的 child；它自身就是 Root 启动并实际等待的独立进程。所有原始 argv/stdout/stderr/真实返回码留在私有执行目录。生产 Mac Python3.9.6 没有 `sys.orig_argv`，因此 Root 的 subprocess 必须使用精确 `[sys.executable, ABSOLUTE_compact_py, ...args]`，不加 `-B` 或其它解释器参数；由环境 `PYTHONDONTWRITEBYTECODE=1` 和模块自身 `sys.dont_write_bytecode=True` 禁写 pyc。这样 start 的 `[sys.executable,*sys.argv]` 与真实 parent argv 完全一致，不通过猜测重构解释器参数。

## 证据与限制

本工具不再逐成员扫描大 tar：原42成员 SHA 验收通过独立解码的**完整 tar 字节完全一致**继承，明确 `fresh_member_hashes_repeated:false`；42 loose 的 SHA 在 prepare 实际重读。prepare/execute 不重新大读已验收 XZ，只要求其完整 metadata 连续。新的压缩档与历史证明共同构成持久表示，旧历史 receipt 仍描述原验收时点；回收后不能声称它引用的已删除路径仍存在。

每组逻辑回收量是原 tar+42 loose 字节，净空间收益还需减去新 XZ 字节；APFS 实际 free gain 未测量，不预报压缩率。R/W 两组当前原件合计约3.36GB，实际净收益待压缩后才能确定，未来P/J复用同显式合同。无 tensor/Adam/效果重新验收。

唯一 tiny CPU fixture 只在临时目录验证新多目标编排：metadata变化/open不确定时一个也不删；成功只删指定 regular files；同 exact journal 可恢复；保留代表和目录。fixture 中仅 `lsof` 检查被 mock，生产没有 bypass。不重新测试原压缩/删除引擎，不触碰真实大资产。

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s ABSOLUTE_HELPER_DIR -p 'test_compact.py' -v
```
