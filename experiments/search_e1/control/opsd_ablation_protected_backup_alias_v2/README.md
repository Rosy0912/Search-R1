# Protected u0/u1/u5 backup through the existing HKUST SSH alias

This independent Mac-only revision preserves the original protected backup protocol and all Node assets. It is separate from the published v1 and does not modify a live queue, training, evaluation, or any Node file. Backup remains independent of GPU progress.

The only transport change in `backup.py` is the SSH prefix: `/usr/bin/ssh -oBatchMode=yes -oConnectTimeout=10 hkust1`. It honors the existing alias configuration and does not force an interface, proxy, jump, or control-socket override. Successful alias transport is not a claim about its underlying route. The publication prefix changes only to bind this independent source directory honestly.

`source-operation.py` is byte-identical to v1 (SHA256 `b6e7274ed585b2404e4a561421c2aad5f91a1bba486702b6d4d4bb1015341316`). All pin, immutable scope, 42-member inventory, fixed-handle USTAR stream, fresh source-after metadata, safe extraction, and independent full local member SHA code is unchanged. The source-only primitive dependency remains `opsd_ablation_outer_backup_v2/backup.py`, SHA256 `d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57`.

The input goal, exact R/W/P/J config/run, actual committed CP0/1/5 manifests, and original shared optimizer/rank RNG/scaler/three-adapter-byte contract remain unchanged. CP0 is the actual outer0 initial state with the empty shared Adam file; no numerical Adam or model evaluation is performed. No outer-end requirement is added. Missing states produce PENDING, insufficient archive+extraction+5GiB space produces BACKLOG. Neither state authorizes holding or restarting GPU work.

Root must first commit, push, and perform exact remote byte readback. Use a new destination and this revision's own manifest/publication/commit references; preserve all previous attempts. The original CLI is unchanged:

```sh
python3 -B /ABS/PUBLISHED/opsd_ablation_protected_backup_alias_v2/backup.py \
  --goal /ABS/PRIVATE/original-goal.json --goal-sha256 ORIGINAL_GOAL_SHA \
  --remote-goal /ABS/NODE/original-goal.json --variant W \
  --destination /ABS/PRIVATE/NEW_ATTEMPT \
  --source-helper /ABS/PUBLISHED/opsd_ablation_outer_backup_v2/backup.py \
  --operation-sha256 b6e7274ed585b2404e4a561421c2aad5f91a1bba486702b6d4d4bb1015341316 \
  --source-manifest-sha256 THIS_MANIFEST_SHA \
  --publication /ABS/PRIVATE/THIS_PUBLICATION.json --publication-sha256 THIS_PUBLICATION_SHA \
  --source-commit THIS_REMOTE_READBACK_COMMIT
```

A complete backup still requires actual transport rc0, unchanged pinned source metadata after streaming, all 42 members, and complete independent local SHA readback. Failures and partial files are retained; no asset is deleted. This revision has not run a real transfer. No new fixture was added or executed: the two Python files were AST checked, the operation bytes were compared, and normalizing only the SSH/publication constants reproduces the original complete source exactly.
