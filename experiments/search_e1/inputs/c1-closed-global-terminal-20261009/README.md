# C1 closed global training evidence

This fixed input supplements the already accepted C1 outer0–9 raw backup (4087 files). It does not copy those raw payloads again. It binds C1 run `a1a56efd32e5456d8c6813454975e38f`, original configuration SHA `13fdbd51cc290bc6d23b0577c67bf8160a17292fab6f315fbfb07ea3f6ed3310`, TRAIN_COMPLETE10/101, final CP101/outer9 and the original actual foreground exit0 and release references. These are identity/closure checks, not another Adam or numerical audit.

The archive source root is `/data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008`. Exact payload scope:

- `group-C1/runs/phase-exits/`: all regular closed actual-phase receipts, including any historical attempts. Only successful parent-bound receipts are described as rc0.
- `group-C1/runs/{child-history.jsonl,status.json,run-identity.json}`.
- `eval-study-driver-stats-v2-8b641cb51b4c478684f596f75c603851/runs/01-C1/training/`: closed group training start, foreground log, actual exit/release, terminal and associated regular files.
- `group-C1/receipts/dtype-recovery/b26ef0a9c11d4acc2b08453c8248b2c532eeb8bde0956245354f42a78e4e381d/` and the exact small repair manifest under `receipts/dtype-recovery-538ac99-20261009/`.

For each completed outer, inventory derives external terminal references from the original sampling, teacher, training and CPU-merge parents. Those parents are already in the accepted outer raw archives. Teacher NA and fresh BF16 outer0 without a merge child remain explicitly NA. Referenced phase receipts must have the exact run/config, nested child identities, actual rc0 and accepted owned-release proof. Additional historical phase files are preserved without inventing successful exit codes.

The active study driver's global files, scoped CPU-role records, memory observer, heldout evaluation partial outputs, model/checkpoint payloads and inference caches are excluded. A historical driver identity embedded inside the closed group terminal does not claim that driver has terminated. File counts and bytes are produced only by actual inventory; this preparation contains no transfer success claim.

Publish and FETCH/readback-verify all five fixed input files before use. Reuse the existing direct SSH route (`/usr/bin/ssh -B en0 -oProxyCommand=none -oProxyJump=none -oControlMaster=no -S none`, BatchMode and ConnectTimeout10). The remote Python invocation is the original environment, `-B -c` with the exact published source-operation bytes, `env -u E1_RUN_ID CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1`. No Node file is written.

Execution order:

1. Remote `source-operation.stdin.py` mode `inventory`; preserve stdout/actual SSH rc in a private control directory separate from the future archive destination.
2. Mac `transfer-gzip.py --inventory INVENTORY --destination NEW_UNUSED_DESTINATION --source-operation SOURCE_OPERATION --operation-sha256 PUBLISHED_SHA`. It uses the fixed inventory as stdin to remote mode `stream`. The destination must not exist; it is created0700. Capacity is the exact USTAR upper bound plus1% and5GiB, independent of assumed compression.
3. Remote mode `after` with the exact saved inventory bytes as stdin; preserve its stdout and actual SSH rc. It rechecks closed identities, path set and original metadata. Do not reuse a pre-transfer after-proof.
4. Mac `verify-archive.py --inventory INVENTORY --after-metadata AFTER --destination DESTINATION --source-helper PUBLISHED_LOCAL_V2_BACKUP_PY`. The pinned V2 helper SHA is `d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57`.

Original fixed-handle SHA and safe USTAR streaming, gzip level1/mtime0, actual transport wait code, exact regular member set, every decompressed member SHA, whole compressed-file SHA, CRC/footer/EOF and atomic rename/fsync are reused. No bare extraction is created. `closed-global.tar.gz.partial` becomes `closed-global.tar.gz` only after all checks; `complete-sha-verified.json` is then written exclusively. Failures preserve partials and all Node originals. A second destination is not a retry that overwrites earlier evidence. No GPU, model import, HTTP, signal, training lock or training dependency is added.

Preparation verification is static AST only: no new fixture suite, Node execution or transfer has been performed for this scope. Root controls publication and actual execution.
