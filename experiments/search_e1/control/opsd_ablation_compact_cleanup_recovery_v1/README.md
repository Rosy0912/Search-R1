# External compact backup cleanup recovery

This Mac-only launcher fixes one directory cleanup error in the published compact
backup V1 (`4d2065e81e3350abc69a6a5861b33be64803a458`, manifest
`63e1c9aa3d6fd50a8c80e23143e4375360da658bfa5170069855ede1e5f2f13c`).
It never edits that source, the published V2 source, checkpoint format or accepted
compact receipts. All original Node assets remain retained.

V1's bottom-up traversal cached child-directory names. After removing a child,
the parent iteration attempted to check that now-absent child and failed. A
compact package could therefore be independently accepted and its planned
tar/payload duplicates reclaimed while empty extraction directories remained.
This is a local cleanup failure, not a failed checkpoint, training or XZ audit.

The new launcher SHA-checks and source-compiles the exact original queue/core/V2
bytes. Its bootstrap supplies the original module with only `reclaim_legacy`
overridden. That function is compiled from the original source AST, changing
only its final empty-directory block. The replacement constructs the allowed
directory set from pinned original member parents, validates the entire existing
tree, then removes its empty directories in reverse order. Planned directories
already absent after a durable deletion plan are allowed. Foreign files, foreign
empty directories, symlinks and new residual entries are refused and retained.
Every original plan, payload identity, independent full-XZ verification,
reclamation receipt and byte format check remains in place.

The original queue entry/arguments, same V2 queue lock, seventy immutable pins,
queue identity and compact-source publication remain unchanged. Existing COMPLETE
packages are reused; this recovery does not recompress them. Only the original
queue's canonical status pointer advances. Existing compact/legacy/history JSON
files are never rewritten. Each recovered package gets a new independent
`cleanup-recovery-COMMIT.json` provenance receipt. A prior matching cleanup
provenance can be reused after its immutable launch receipt is verified.

## Startup binding

Root retires only the idle original Mac queue separately. This launcher sends no
signal and does not stop/restart training, evaluators or transport processes.
The first recovery binds the actual old queue PID45060/UID501, original source
commit, same goal/queue identity, actual exit code and absence after retirement.
PID absence alone is never an exit-code proof. Reused/live PID45060 or uncertain
process observation rejects startup. Its separate publication must carry accepted
search-e1 remote byte readback, exact recovery commit/manifest and the existing
seven authorized run IDs. It records the actual new launcher argv/PID/source and
old terminal proof in an exclusive new receipt. This receipt admits the source;
the original queue still must acquire its unchanged lease before package work.
No new Node deployment is required. Use the existing direct SSH shim in PATH.

```text
python3 -B /published/opsd_ablation_compact_cleanup_recovery_v1/recover.py
  --original-helper-root /published/opsd_ablation_compact_backup_v1
  --recovery-manifest-sha256 NEW_MANIFEST_SHA
  --recovery-publication /private/recovery-publication.json
  --recovery-publication-sha256 NEW_PUB_SHA --recovery-source-commit NEW_COMMIT
  --previous-queue-terminal /private/old-queue-terminal.json
  --previous-queue-terminal-sha256 OLD_TERMINAL_SHA
  --recovery-launch-receipt /private/exclusive-new-launch.json
  -- queue [the exact original compact queue flags, including --reclaim-legacy]
```

The previous terminal JSON schema is `opsd-ablation-compact-queue-terminal-v1`:
`accepted:true`, `pid:45060`, `uid:501`, actual integer `actual_exit_code`,
`source_commit` equal to the original 4d commit, `goal_sha256`, the original
`queue_identity`, `queue_status:{path,sha256}` (preserved immutable status),
`launch:{path,sha256}`, `no_live_transfer:true`, `owned_children_terminal:true`.
Root supplies actual observations and persistent proofs; templates must not
invent them. A later new-owner failure needs a new terminal boundary and separate
review before another recovery; the old45060 proof is not general permission to
duplicate owners. The original lock also rejects concurrent queue ownership.

## CPU scope

```text
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_ablation_compact_cleanup_recovery_v1 -p 'test_*.py' -v
```

Three focused regressions exercise normal cleanup and durable-plan resume after
payload deletion, refusal of a foreign file, and refusal of a foreign empty
directory. They compile the actual frozen V1 reclamation source, with the already
accepted archive-verification boundary mocked to isolate this cleanup bug. They
do not claim a real checkpoint decompression, transfer, deletion, source
publication, retirement, launch or numerical acceptance. No production smoke is
executed by source preparation.
