# Ablation150 storage metadata and candidate plan

Independent CPU/standard-library reader. No training import, tensor load, network,
lock, signal, archive, or deletion. Live mode is strictly bound to
`/dataset/shaoshitong/recursive-ablation150-20261014`, the seven fresh groups, and
an explicit frozen scientific source commit. Run on the already owned Linux
node, through the existing task controller:

```sh
python3 -m release.recursive_bran_a.ablation150_storage_v1.inspect \
  --source-commit 233b0dc95de3f7b797d5b424d3b7dcaafca5328f
```

It prints small metadata plus a separate dry-run plan and actual receipt
(`deletion_executed=false`, empty deleted list). The CLI has no apply mode. A
saved `metadata` object may be replayed with `--metadata-json`; replay never
authorizes live cleanup. Keep only this small output locally.
It also measures the task filesystem and the existing same-cluster
`/mnt/shared/workspace/training_data_node_5` mount from Linux mountinfo/statvfs.
This neither creates a mount nor copies any checkpoint.

Candidate analysis is stricter than minimum redundancy: the whole group must
have accepted150/fourdev, exact entry-exit0, exact parent-wait0, accepted owned
GPU release, absent current owner PID, no open writable descriptor in the group,
all four actual protected post-evaluation manifests, complete eight-rank and
driver member presence, bound config/source/data/model/step, and same-key HF
inventory. Missing payloads, symlinks, incomplete state, or any mismatched state
refuse the entire group's cleanup candidates. All protected post-evaluation
native payloads and all HF exports stay. Pre-evaluation is superseded by the
completed post-evaluation boundary; their RNG states need not be byte-identical.

Only the nine exact pre-evaluation worker/driver payloads per key and the nine
obsolete native-rolling payloads per completed group are candidates. Never
delete a whole checkpoint directory: manifests, acceptance evidence, logs,
failures, partial attempts, and other files remain. A/foreign runs, cache, model
assets, retrieval indices, credentials, and arbitrary large output are outside
scope. A missing protected payload can never be treated as a successful prune.

The fresh-space guard is measured `statvfs.f_bavail` versus an actual next
keypoint write burst: two complete native states + one HF export +16GiB
headroom. Existing rolling states already occupy disk; a normal rolling write
needs only one additional full native state before old rolling cleanup. If
there is enough next-write space, the returned review candidate list is empty.
The seven-run final/peak forecast separately indicates possible archival need;
the forecast alone does not trigger cleanup.

Metadata does **not** verify stored payload SHA256 or tensor contents. Before
any separately implemented cleanup, freshly repeat the complete metadata scan,
recheck no active load/save/recovery reference, and hash every canonical native
and HF file against its recorded digest. Persist exact planned and actual
receipts separately. If pre payloads are removed, retain their manifest but add
an explicit payload-pruned receipt/tombstone with file list and canonical link;
never report that retained pre manifest as a fresh complete native state.

Forecast uses maximum measured native/HF sizes. It retains four post native +
four HF/group; sequential peak after completed-group pruning adds current
group four pre native + two rolling native + headroom. It excludes uncounted
logs, partial/failure/debug files and caches; therefore it is a lower bound on
capacity needed. If too large, archive only completed canonical own states to
verified existing same-cluster storage with per-file SHA readback and a durable
receipt. Reassess actual source and destination capacity; do not widen cleanup.

```sh
python3 -m unittest release.recursive_bran_a.ablation150_storage_v1.test_inspect
```
