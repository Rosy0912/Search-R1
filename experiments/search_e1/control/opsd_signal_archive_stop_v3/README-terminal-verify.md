# Independent read-only stop terminal observation

`terminal_verify.py` is restricted to the published v3 stop and its original raw
Node result SHA `1e58cb985f68e8ff5e772d58f2d0be64c2bd6b1e9f137399aa0858d7954e3c2b`.
That stop sent 188 STOP, 188 TERM, and 188 CONT signals, preserved evidence, then
returned HOLD while the exiting driver PID 1434651 briefly had an empty argv.
The original HOLD and exit code 2 remain historical facts. This new observer
cannot send signals, retry the stop, or overwrite any original file.

The observer binds the exact v3 helper, original authorization, input snapshot,
result, preservation, and quiesced bytes. It reuses the reviewed v3 read-only
process/protected/GPU checks. All 188 captured identities must now be absent,
zombie, or demonstrably replaced with a different positive start tick; an opaque
identity is HOLD. It checks the three protected processes before and after,
GPU4–7 UUIDs, only the admitted retriever on GPU7, and direct `memory.free`.

The original 36 checkpoint members are checked by exact names, size, and mtime
continuity only. The original recorded payload SHA remains bound evidence;
this observer never opens or rehashes those model payloads, checks tensor finite
values, or captures Adam/RNG. It freshly hashes each copied source/log/receipt
file and checks literal symlink targets. Active original observer receipts are
not required to equal their earlier preserved snapshots.

Run after publication, as the original UID on HKUST4, with a real deployed file:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B terminal_verify.py \
  --helper /published/v3/archive_stop.py \
  --result /weka_3/shaoshitong/search-e1-config-study-20261004/preserved/user-stop-archive-20261009-057b905a73ae4ca2a28c220466d5608a/result.private.json \
  --result-sha256 1e58cb985f68e8ff5e772d58f2d0be64c2bd6b1e9f137399aa0858d7954e3c2b \
  --preservation-sha256 a7c99df18d0e4f83e88caa0bcb07c2d9324c46bd452c46b20c13b715e952a592 \
  --quiesced-sha256 15243436aba89cdd79074b98e4b9bb6d68ae539124ebacbfdebee5d5704746cf \
  --authorization-sha256 b1b362ede782745c6b37ec7ea9dbebbffabb46223981c5a2d588bfe29e9d59f6
```

Stdout is a new independent JSON observation; redirect it outside the Node using
the caller. No Node output path or signal option exists. Failure returns 2 with
`HOLD_INDEPENDENT_TERMINAL_AUDIT`. This is not a full-state resume acceptance.

The eight local CPU fixtures exercise receipt scope, exact original signals,
protected exclusions, preservation equality, model-metadata-only checks,
copied-file corruption, reparented/opaque historical identities, and GPU release.
They run only local temporary files and mocked proc/GPU observations.
