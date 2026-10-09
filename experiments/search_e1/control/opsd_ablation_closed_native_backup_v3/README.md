# V3 normal SSH alias transport

Independent copy of published `opsd_ablation_closed_native_backup_v2` (manifest SHA `8b7c5dd63748f2132bcebdd7fc45f52e3e82f5cc8fd65668e104993f53170fd1`). V2 and all prior receipts remain unchanged. The only executable delta is the transfer SSH prefix:

```text
/usr/bin/ssh -oBatchMode=yes -oConnectTimeout=10 hkust1
```

This honors the user’s latest normal-alias routing instruction. The remote command/env/old Python, inline scope, source operation, inventory, USTAR/gzip stream, actual SSH wait code, partial preservation, source-after and full member verifier are unchanged. `source-operation.py`, `verify-archive.py` and `test_scope.py` are byte-for-byte V2 copies. No new fixtures were run; AST parsing only. Root must publish/read back the exact V3 files before actual execution.

Use the original V2 CLI options, with the independent V3 transfer path:

```text
python3 -B /ABS/PUBLISHED/opsd_ablation_closed_native_backup_v3/transfer-gzip.py --scope ABS_SCOPE --scope-sha256 ACTUAL_SHA --inventory ABS_INVENTORY --destination ABS_NEW_DEST --source-operation /ABS/PUBLISHED/opsd_ablation_closed_native_backup_v3/source-operation.py --operation-sha256 086c15e619a19ba646232b5aa62653a43b316252b1862b1ba7cd4291086d5311 --source-helper /ABS/PUBLISHED/opsd_ablation_outer_backup_v2/backup.py
```

Inventory and after use that same original source-operation via normal `ssh hkust1`; verification uses V3 `verify-archive.py` with the unchanged V2 options. See the frozen V2 README for the exact closed-scope contract and full CLI. Supported groups remain C0/C1/M/R/W/P/J under the original fixed study; no mutable study/observer tree, training model/cache or new inference. Every actual closure SHA must be supplied; a missing terminal is pending backup evidence.

No Node files are deployed or modified. No GPU/model work, HTTP, process signals, locks or numerical tests are performed by this repair. Backup is independent of continued W/P/J training and evaluation.
