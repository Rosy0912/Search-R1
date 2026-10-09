# P/J recorded precision history from independent HKUST copies — transport v3

This is a separate reporting entry for the latest storage rule: large raw data
stays on HKUST. It reads the accepted independent inode copies made by the exact
published `opsd_ablation_remote_raw_snapshot_v1/snapshot.py` (`40e05812069c118b14df6605128204bd1a013286674c4cacab768a91ec48c211`).
It does not alter v1 or v2, create a pretend gzip proof, train, generate, evaluate,
construct an optimizer, import Torch, use CUDA/network/HTTP, acquire a training
lock or signal a process. Execution requires original HKUST UID1005 and empty
`CUDA_VISIBLE_DEVICES`. New output must be under the authorized experiment's
`analysis` directory. Keep private inputs/details and payloads on HKUST; only
small public summaries/source/configuration may be retained on Mac.

The v1 `scaler`, `precision`, `coverage` and `analyze` functions, producer SHA pins
and closed-outer science/counter checks are retained. The replacement boundary
is `prepare_direct`: it accepts only independent direct copies whose native ten
outers have genuinely ended. Either P, J or both can be supplied; a P-only result
is not all seven groups or the whole study.

## Immutable input

Use exact absolute path, SHA256 and byte count for each transport reference.
Keep `ready:false` until real closure, actual-copy wait and source readback exist.
The manifest schema is `opsd-ablation-recorded-precision-history-remote-inputs-v2`.

```json
{
  "schema":"opsd-ablation-recorded-precision-history-remote-inputs-v2",
  "ready":false,
  "sources":{
    "aggregate":{"path":"/HKUST/aggregate_clipmass.py","sha256":"b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a"},
    "trainer":{"path":"/HKUST/871/trainer.py","sha256":"4dc352f5091594ab5bcae71ecc17516afd8607601b5932216cbda66befc5409c"},
    "precision":{"path":"/HKUST/871/precision.py","sha256":"7dd23d6378c91cb33c730241fe5cd63618768aa95eea6136bf571a7cb64358bc"},
    "retry_state":{"path":"/HKUST/871/retry_state.py","sha256":"f3b4abe482b1dbc430959d4a189f282cdfeeca7372e57b78a9492c14e836c8fe"},
    "dtype_launcher":{"path":"/HKUST/538/trainer_launcher.py","sha256":"5b8bfe675ee5b2186aee45a598e56991b2738becb15b140cf8d0cdbb89cbeb7a"}
  },
  "facts":{"path":"/HKUST/facts.private.json","sha256":"PENDING"},
  "groups":[{
    "variant":"P",
    "config":{"path":"/HKUST/original-authorized-config.json","sha256":"PENDING"},
    "backups":[{"kind":"direct",
      "source":{"path":"/HKUST/copy-output/P/source-inventory.json","sha256":"PENDING","bytes":0},
      "local":{"path":"/HKUST/copy-output/P/copy-accepted.json","sha256":"PENDING","bytes":0}}],
    "remote_copy":{
      "deployment":{"path":"/HKUST/snapshot-deployment.json","sha256":"PENDING","bytes":0},
      "launch":{"path":"/HKUST/snapshot-launch.json","sha256":"PENDING","bytes":0},
      "actual_wait":{"path":"/HKUST/snapshot-actual-wait.json","sha256":"PENDING","bytes":0},
      "complete":{"path":"/HKUST/copy-output/complete.json","sha256":"PENDING","bytes":0}
    },
    "dtype_repair_receipts":[]
  }]
}
```

The producer pins remain exact old science bytes, not the latest documentation
commit. `facts` must be the accepted native collector output, with the original
variant/run/config identity and all ten unique closed outers. Source config and
closed native outer/checkpoint references are compared path/SHA/bytes, cumulative
Adam and pair exposure. No counter table is substituted.

### Actual copy provenance

`deployment` is the accepted captured source deployment/readback document. It
must name the original `snapshot.py` SHA above and `source-operation.py` SHA
`34857f4a6d0cfea4da2e2c4b71a5af662aca9a49e7ee7134ab97cd3c88185222`,
with actual `Git_FETCH_files`, real deployment SSH0 and unchanged science. The
current producer bytes are re-read against those pins. A published immutable
source deployment may be reused for a later independent output; the source root
is not incorrectly equated with the new copy destination.

`launch` preserves exact SSH argv, `run_id`, `output`, `started_unix` and whichever
execution identity was actually captured. `actual_wait` preserves the matching
`run_id`/`output`, actual `actual_ssh_returncode:0` and `ended_unix`. If a local SSH
PID was captured, both must carry the same positive `actual_ssh_pid`. If direct
tool execution captured an actual unified session while SSH PID was unknown,
both instead carry the same positive `actual_unified_session`; the SSH PID may be
null/absent. Unknown PID is not fabricated, and PID absence alone is never proof
of success. Known identities cannot disagree.

Transport v3 preserves the same immutable v2 input schema and all original
receipts. The controller-host launch/wait times and HKUST copy start/end times
are each required to be finite, positive and ordered within their own host.
The two hosts' clocks were not calibrated, so v3 does not compare their absolute
values or add a tolerance. It records all four raw clock values, marks
`cross_host_clock_calibration_unverified:true` and
`absolute_order_not_claimed:true` in each transport and public summary. This does
not replace actual wait0: exact captured execution identity, run, argv, output,
source pins, independent source/destination SHA inventories and native closure
checks remain required unchanged.

The exact remote command must be:

`env E1_RUN_ID=<run> CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 <original-env>/python -B <pinned-snapshot-path> --groups <ordered-subset> --source-operation <pinned-scope-source-path> --output <actual-copy-output>`

The global completion must be the captured output's `complete.json`, accepted
with no scientific/source deletion or Mac raw payload copy, and its ordered
groups must match the actual launch. Each group's completion, source inventory
and copy inventory are resolved from that accepted output, not guessed mutable
paths. The embedded direct backup specification must equal the caller's input.
The scope remains original full closed10 noncache raw plus fixed native global
closure/phase exit/dtype-recovery files. Only original derived model-cache
subtrees are excluded. Checkpoint payloads are outside this raw copy; its native
checkpoint manifest references are identity/closure evidence, not an independent
optimizer payload audit.

Every accepted source/destination member must have equal SHA/byte count, exact
relative path and independent `(device,inode)`. All source and destination current
metadata are checked at preparation; the exact whole-copy SHA acceptance remains
the original snapshot's recorded verification, not a new whole-payload SHA claim.
Native final status, controller terminal, actual parent wait0 and owned-release
JSON are freshly re-read from the independent copy. Existing `FixedBackups`
then freshly hashes each selected precision/update/retry/history JSON around its
actual read. Input/provenance documents and pinned source bytes are re-read at
completion. No gzip or archive CRC evidence is claimed for direct copies.

## Run only after publication/readback

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /original/env/bin/python -B /HKUST/derive.py \
  --manifest /HKUST/inputs.private.json --manifest-sha256 ACTUAL_SHA \
  --output /data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/analysis/new-independent-output
```

An optional `--plan-only` reads provenance/inventories and fresh native closure
JSON but does not read precision observations; it is not required as a separate
extra test. Actual normal execution reads one selected-JSON visit per group,
writing `read-plan.private.json`, `details.private.json`, `summary.json` and
`derivation-manifest.json` in a fresh output directory. The caller must capture
real process wait/status; this script's summary does not independently establish
its own exit code. CPU reporting never gates continuing GPU training/evaluation.

## Scope and limits retained from v1

- All attempt directories found in the exact raw inventory are included, not
  only successful committed paths. Missing ranks/source load, original invocation
  mismatches, nonzero original training invocation terminals and orphan retry
  capsule refs produce `PARTIAL`; contradiction raises an error. Zero preserved
  gradient-overflow is claimed only with complete recorded coverage.
- All installed module dtype counters and capped4096 detailed events per
  rank/attempt remain reconciled. These are recorded boundary observations, not
  every kernel/activation tensor. Master/model storage must be FP32; observed
  FP16/FP32 outputs, forward/recompute stages and scaler states are separate.
- Committed update scaler identities match all eight ranks. Retry manifests
  retain failure kind, cursor, outer-old, checkpoint/encoding/order joins and
  eight PT references. Same-invocation retry counts match receipts; preserved
  cross-invocation CLI selections must choose their exact capsule/checkpoint.
  PT contents, failure scale-before and actual RNG/Adam/CUDA restore are not read
  or independently proved. Existing final numeric Adam audits remain separate.
- Original trainer emits no attempt-UUID/PID join. No fabricated per-attempt PID
  association is created. Optional `dtype_repair_receipts` are explicit SHA-bound
  references checked against original run/rank/outer and pinned repair source;
  an empty list is not zero repairs and is not a complete external inventory.
- Private output retains source/CP/log/attempt paths, process identities and
  original refs on HKUST. Public summaries contain only labels, counters, dtypes,
  scaler states and limitations. No raw question/response/token/tensor payload
  or full per-question prediction data is exported to Mac or Git.
