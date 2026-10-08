# CP89 verified-prefix resume, independent CPU candidate

This fresh namespace copies the reviewed pinned-backup stdlib fixed-handle USTAR/inventory/safe-extract/readback engine. It imports no old science/backup Python or pyc and does not change the old helper, original partial/failed records, CP374 transfer, weights or node files. Only the exact original CP89 run/update89/outer8 and manifest SHA325c9f41f5a44d5e16a32578c5fa03dfaecec92bbdbd87a4f5e289c81c9bf75f are supported. Scope is the already authorized failed backup, not training restart or training rc0. The only accepted failed prefix is 1741258752 bytes/SHA44859dd3b9e4bfbe61dccfbaa3986e63f0489cdd1bc181b768a5ef1e9f9dd58f; those are explicit required CLI arguments checked against constants, not mutable-latest guesses.

Mac resume mode reads the original inventory JSON at explicit SHA and fresh-hashes the original partial at exact byte count/SHA. It requests a fresh full node inventory over stdin (the original raw JSON bytes are SHA-checked, not reserialized and falsely claimed identical). Every original file/stat/primary source-manifest identity must match. The old helper publication identity stays in the original inventory; the fresh inventory independently binds this new published/readback helper. No original source bytes or unknown inventory can be concatenated.

A deterministic USTAR stream is generated from the actual unchanged files on the node. PrefixWriter hashes and discards precisely offset bytes, including offsets inside headers/data/padding: there is no rounding to512 or token alignment. Only after the digest matches does it emit suffix bytes. A mismatching or incomplete prefix emits no suffix; offset beyond total archive length fails. A source change after emission also causes nonzero transport and prevents acceptance. The node creates no tar/cache/output file. Suffix stdout is binary only; failures use stderr.

Mac creates a new0700 private destination, copies the old prefix via fixed handle/full hash/fsync, then appends the streamed suffix. It records actual SSH wait rc immediately even if communication later fails, preserves all new partial/error records, and never signals the child. Actual rc0 is necessary but insufficient: original strict USTAR member checks, exact18 members, full memberSHA, independent second local diskSHA, old-prefix/inventory stable-stat checks must pass before complete-sha-verified.json and final archive naming. Original files and prior failed backup remain untouched. This accepts file transport only; it does not prove numerical Adam equality, training returncode, benchmark benefit or completed science.

## Explicit local interface

After root publication, sourcepush/readback and independent node-copy verification, run backup.py resume with:

```
--config NODE_ORIGINAL_CONFIG --config-sha256 CONFIG_SHA
--checkpoint NODE_CP89_MANIFEST --checkpoint-sha256 CP89_SHA
--original-inventory LOCAL_ORIGINAL_NODE_INVENTORY --original-inventory-sha256 ORIGINAL_JSON_SHA
--original-partial LOCAL_FAILED_CHECKPOINT_TAR_PARTIAL
--prefix-bytes 1741258752 --prefix-sha256 44859dd3b9e4bfbe61dccfbaa3986e63f0489cdd1bc181b768a5ef1e9f9dd58f
--source-manifest LOCAL_NEW_HELPER/source-manifest.json --source-manifest-sha256 NEW_MANIFEST_SHA
--publication LOCAL_NEW_PUBLICATION --publication-sha256 NEW_PUBLICATION_SHA --source-commit NEW_COMMIT
--remote-helper-root NODE_NEW_HELPER --remote-publication NODE_NEW_PUBLICATION
--destination NEW_PRIVATE_MAC_DIR
```

Only SSH alias hkust1 is allowed. The read-only remote inventory/suffix entries require real/effective UID1005, exact original nodebase config/CP path, the same explicit config/CP/sourcepublication/prefix/inputSHA flags, and original raw inventory JSON on stdin. Remote output is JSON only for inventory and USTAR suffix only for suffix. CUDA_VISIBLE_DEVICES is empty; no Torch/model/GPU/RNG/HTTP/weights evaluation is performed. Fresh source hashes still read full checkpoint bytes; network transmits only the suffix, and node disk reads include discarded prefix generation. Space admission conservatively reserves full archive+extraction+5GiB even when resuming.

## Local synthetic checks

```
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_backup_resume_v1 -p 'test_*.py' -v
```

Eight necessary fixtures pass with no skipped tests: wrong-prefix/no-output, beyondEOF/interrupted prefix, arbitrary exact-byte offsets including complete archive, successful strict reassembly, changed source inventory/stream, safe independent copy, unsafe trailing archive rejection, simulated rc0 diskreadback and simulated255 partial preservation. Fixtures patch pin constants only around synthetic local files. No real SSH, transfer, source QA, model, GPU, node action or old-directory mutation has occurred.
