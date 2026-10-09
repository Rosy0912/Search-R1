# P/J recorded precision and failure history

This local stdlib reporter closes a reporting gap in the committed-update curves.
Those curves do not inventory failed attempts, precision observations or overflow
capsules. This entry does not train, generate, evaluate, run an admission gate or
access Node, HTTP, Torch, CUDA, locks, signals or the active production tree.

Only an explicit subset of **P/J with all ten native outers already closed** is
accepted. A subset result is not the full 70-outer/5632-prediction study delivery.
No archive is read until the original config, closed facts, exact accepted backup
inventory and pinned sources pass the small-input preparation. Output must be new;
failed/partial output remains evidence and is never overwritten.

## Inputs

The SHA-bound input JSON has this shape (replace every pending value with an actual
immutable reference; an unresolved template must keep `ready:false`):

```json
{
  "schema":"opsd-ablation-recorded-precision-history-inputs-v1",
  "ready":false,
  "sources":{
    "aggregate":{"path":"/absolute/aggregate_clipmass.py","sha256":"b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a"},
    "trainer":{"path":"/absolute/871/trainer.py","sha256":"4dc352f5091594ab5bcae71ecc17516afd8607601b5932216cbda66befc5409c"},
    "precision":{"path":"/absolute/871/precision.py","sha256":"7dd23d6378c91cb33c730241fe5cd63618768aa95eea6136bf571a7cb64358bc"},
    "retry_state":{"path":"/absolute/871/retry_state.py","sha256":"f3b4abe482b1dbc430959d4a189f282cdfeeca7372e57b78a9492c14e836c8fe"},
    "dtype_launcher":{"path":"/absolute/538/trainer_launcher.py","sha256":"5b8bfe675ee5b2186aee45a598e56991b2738becb15b140cf8d0cdbb89cbeb7a"}
  },
  "facts":{"path":"/absolute/accepted-outer-facts.json","sha256":"PENDING"},
  "groups":[{
    "variant":"P",
    "config":{"path":"/absolute/original-config-copy.json","sha256":"PENDING"},
    "backups":[{"kind":"gzip",
      "inventory":{"path":"/absolute/original-inventory.json","sha256":"PENDING"},
      "complete":{"path":"/absolute/reader-compatible-proof.json","sha256":"PENDING"}}],
    "dtype_repair_receipts":[]
  }]
}
```

`facts` is original `opsd_ablation_outer_report_v1/collector.py` output, not a
replacement counter table. Each group uses one full closed10 generic gzip, with
the already published `opsd_ablation_closed_raw_reader_adapter_v1/adapt.py`
compatibility proof (exact adapter SHA is pinned). It retains original scope,
inventory, source-after, transfer/verifier return codes and accepted gzip/member
proofs. The inherited `FixedBackups` implementation checks current archive
metadata, exact members and fresh selected JSON SHA; it does not rehash the whole
archive. A traversal still decompresses the gzip through CRC EOF. No token file,
log body, NPY or capsule PT payload is parsed or independently hashed here.

Optional `dtype_repair_receipts` are explicit local copies with SHA, from the
538 launcher's external `receipts/dtype-recovery/<repair-manifest-SHA>/` directory.
Their actual fields are checked against run/config/rank/outer-start and the fixed
repair source. This list is **not** an independently collected complete external
inventory. Empty means not reviewed, not zero repairs. Original repair receipts
do not emit the training attempt UUID, so that association is not invented.

## Run after publication/readback

First stage the exact small-input/read plan without opening the large archive:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python3 -B /absolute/derive.py \
  --manifest /absolute/inputs.private.json --manifest-sha256 ACTUAL_SHA \
  --output /absolute/new-plan-directory --plan-only
```

Then use the same immutable input in a separate new output directory, omitting
`--plan-only`. This performs one sequential selected-JSON visit per group.
It records `read-plan.private.json`, `details.private.json`, safe `summary.json`
and `derivation-manifest.json`. Root's actual process invocation/return-code
receipt remains the execution authority; `cli_arguments` preserves script args.
Publication and the real P/J closure are prerequisites to the eventual reporting
execution, never prerequisites to continuing the GPU study.

The reader includes every attempt directory found anywhere in the exact accepted
closed inventory; it does not select only successful attempt paths. It reads all
precision JSON and update/retry manifests plus original phase-exit JSON and child
history. Committed update receipts are bound to the accepted facts; uncommitted
receipts remain separately indexed. Same-invocation retry IDs must reproduce
the committed receipt's retry count. Recorded cross-invocation retry CLI flags
must select the precise preserved manifest and its accepted checkpoint reference.
Eight capsule rank refs are checked against the already accepted member inventory.

`COMPLETE_RECORDED_ATTEMPT_COVERAGE` means the closed inventory, attempt precision
files and original invocation records are accounted for in this restricted scope.
Missing ranks/source-load records, unmatched attempt/invocation counts, failed
invocations or orphan capsule refs produce **PARTIAL**, never an assumed zero.
Invalid SHA/run/config/cursor or contradictory evidence raises an error.
`zero_preserved_gradient_overflow_proven` is only allowed with complete coverage;
it describes preserved history, not proof that no unrecorded physical event ever
occurred. Missing earlier attempts in an unclosed or incomplete backup cannot be
accepted through this interface.

## Interpretation and limits

- Producer precision JSON has no schema/run/rank fields: those identities come
  from the exact original path, source-load record, committed input and inventory.
  Original trainer emits no UUID-to-PID mapping. Count/identity reconciliation is
  explicit; no false per-attempt PID mapping is created.
- `dtype_counts` includes all installed-module observations, grouped by original
  role and forward/backward-recompute stage. Detailed `events` stop at4096 per
  rank/attempt; `events_dropped` is preserved and reconciled. These are module
  boundary observations, not every kernel or every activation tensor.
- Recorded master/model parameter storage must be FP32 for P/J. FP16/FP32 module
  I/O counts, end scalers, committed-step scalers and preserved retry-after scales
  are reported separately. No compute dtype is inferred from configuration alone.
- Retry JSON keeps failure kind, checkpoint/cursor, old identity, scale-after and
  eight PT refs. The failure's scale-before and actual Adam/RNG state live inside
  PT payloads and are not read. Reference joins do not establish actual CUDA/NCCL
  recovery, optimizer equality, unchanged weights or successful RNG restoration.
- Public output contains only labels, counters, dtypes, scaler states and limits.
  Absolute source/CP/log/attempt paths, child identities and original refs remain
  private. No question, question ID, response, token sequence or environment is
  copied into public output. This is reporting, not new scientific validation.

## Preparation check

One tiny CPU reporting-boundary method checks that a missing rank or omitted
invocation cannot be counted as complete zero-overflow coverage. It is not a
model, loss, numerical or production-recovery test:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s /absolute/opsd_ablation_recorded_precision_history_v1 -p 'test_*.py' -v
```
