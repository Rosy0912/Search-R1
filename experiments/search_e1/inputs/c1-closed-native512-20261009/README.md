# C1 closed native512 original-byte backup

Fixed scope: accepted C1 raw10 BF16 evaluation run `8d74046ea72b409d9fdf6682f5637bd8`, evaluation configuration SHA `804e7b13849cecda959fc820e872a8809960279d5ffb77578e2e86583c1592ee`. This is distinct from the C1 training run/config. Existing native result is236/512 (NQ109/256, Hotpot127/256), all eight actual worker waits0, native evaluator parent wait0 and accepted releases. These are existing result identity checks; this helper does not generate or rescore anything.

Payload is all regular original files under:

- `/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/eval-stats-v2-72e1e12-8b641cb51b4c478684f596f75c603851/C1/runs/`.
- `/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs/01-C1/evaluation/`.
- The immutable `01-C1/evaluation-config.json`.

The common archive root is the main study directory, retaining original relative paths. Included native worker results/traces/HTTP/IDs and closed controller/phase/parent receipts are complete original bytes. Active study driver/global observer/M files, training raw/checkpoints/models and other evaluation scopes are excluded. A release embedded inside an immutable closed receipt is historical evidence, not a fresh current resource gate or claim the whole study ended.

Source pins come from the original nine-JSON concatenation saved in private Goal21 `transition-c1-m-093937-3ea760/batch.stdout.private.bin`. Original JSON byte boundaries were recovered; summary83202B/SHA25ddf5…, controller72464B/SHA23888a… and phase80727B/SHAf1906a… also match their independently embedded file references. The `.original.private.json` files now contain the exact original byte slices; earlier reformatted copies were retained separately as `.decoded.private.json`. Full source pins are explicit in source-operation and entry. No private result body is included in this input directory.

Reuse the already accepted fixed-handle SHA/metadata/USTAR, gzip level1/mtime0 and full-member readback implementation; no new backup framework or fixture suite. Publish/FETCH/byte-verify the five files first. Root then performs:

1. Remote mode `inventory` using the exact published `source-operation.stdin.py` bytes under original Python `-B -c`, `env -u E1_RUN_ID CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1`; preserve actual SSHrc/output in a separate private control directory.
2. Mac `transfer-gzip.py --inventory INVENTORY --destination NEW_UNUSED_DESTINATION --source-operation SOURCE --operation-sha256 PUBLISHED_SOURCE_SHA`. Direct route is `/usr/bin/ssh -B en0 -oProxyCommand=none -oProxyJump=none -oControlMaster=no -S none` with BatchMode/ConnectTimeout10. Only local destination is created0700; source Node remains read-only.
3. Remote mode `after`, exact saved inventory bytes on stdin, after actual transport completion; preserve actual SSH rc and fresh metadata/file-set/receipt identity result.
4. Mac `verify-archive.py --inventory INVENTORY --after-metadata AFTER --destination DESTINATION --source-helper PUBLISHED_LOCAL_V2_BACKUP_PY`. Its fixed primitive SHA is `d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57`.

Destination must not exist. Capacity checks the exact USTAR upper bound plus1% and5GiB. Actual transfer wait0, exact regular member set, every decompressed member SHA, whole compressed-file SHA, CRC/footer/EOF and atomic rename/fsync are required before exclusive `complete-sha-verified.json`. Partial is `closed-native512.tar.gz.partial`; final is `closed-native512.tar.gz`. No bare extraction or source deletion. Failure preserves local partial and all Node originals; existing destinations are not overwritten.

Preparation verification: three Python AST checks only; no Node execution, transfer, model/GPU/HTTP, training lock, signal or new test run. Backup remains independent of ongoing M training.
