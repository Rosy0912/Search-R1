# HKUST shared full-vocabulary closure

This independent stdlib wrapper executes the already published physical collector
and final-ledger consumer without changing either source. It runs only on HKUST1
as UID1005, after all seven original ten-outer writers have an actual foreground
wait0 and clean release. It does not wait for, start or interfere with native512
evaluation. It acquires no training lock and uses no model, Torch/NumPy, GPU,
HTTP, SSH, signals, RNG restore, reservation refund or cleanup.

All private inputs, physical inventories, copied closure documents, original NPY
payloads and independent raw copies remain on HKUST. Only `summary.json` and
small execution references may be returned to Mac. The frozen consumer's term
"local file I/O" means the filesystem of the processing host, **HKUST1** here;
this is not a claim that no Node filesystem was read. Its original numerical and
ledger rules are unchanged.

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B close_remote.py \
  --inputs /data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/analysis/NEW/input.json \
  --inputs-sha256 ACTUAL_INPUT_SHA \
  --source-sha256 ACTUAL_PUBLISHED_WRAPPER_SHA \
  --output /data/shaoshitong/search-e1-opsd-ablation-archive100-8gpu-20261008/analysis/ANOTHER_NEW_OUTPUT
```

Source, input and every reference must have been published or actually read back
before this invocation. Output must not already exist. This README is not a ready
input or an actual-run claim. Root assembles a new real input only when the
requirements below exist.

## Input schema

`opsd-ablation-remote-final-vocab-inputs-v1`, `ready:true`, with:

- `study_id` from the original seven configs.
- `sources` with exact `{path,sha256,bytes}` references to the frozen `collector`,
  `consumer`, `aggregate`, `producer` and `producer_manifest`. SHA pins are in
  `close_remote.py`; producer is read as bytes, never executed/imported.
- `facts`: actual accepted all70 original `opsd-ablation-outer-facts-v1` capture,
  ten unique complete rows per group. It is read on HKUST, not transferred to Mac.
- `copy_batches`: actual independently completed raw-copy batches, for example
  the completed six-group batch and completed J-only batch. Each item has
  `complete` and `actual_wait` references. The complete file must be the actual
  `opsd-ablation-remote-raw-copy-v1` output, and captured wait must have real
  `actual_ssh_returncode:0` plus the exact output directory. Together they must
  cover seven distinct groups. This wrapper does not fabricate a combined wait.
- `groups`: ordered C0,C1,M,R,W,P,J. Each has `variant`, original `config`, the
  actual bound `evaluation_config`, and `coverage`. Coverage is the existing
  recorded `kind`, `config_sha256`, `publication`, `details`, `derived_manifest`
  plus accepted `backups` direct source/local specs. Backups must exactly equal
  those of that group's actual completed copy batch. All seven may reuse one
  new all7 coverage audit, or reference accepted separate audits if their run,
  config, artifacts and NPY checks match. No old deleted Mac path is used.
- Optional `classifications` retain the unchanged consumer's explicit
  failed/retry/residual interpretation and small evidence references. They do
  not turn partial/reserved/uncommitted arrays into committed complete arrays.

The wrapper resolves each terminal exclusively through the **actual bound
`evaluation_config.training.terminal`** and verifies its config/run. In particular,
the reused C0 original writer is never assigned a guessed replacement terminal
directory. Its status path and ordered completion references come from the
fixed original output root and the accepted all70 facts; actual writer closure
is then checked by both unchanged readers.

## Actual execution and scope

The physical request has `accepted_source_inventories:[]`: every physical NPY in
all seven original `outers` trees receives a fresh SHA, including retry, failed,
partial and uncommitted paths. The collector verifies two scans, unchanged
source metadata and the exact shared ledger before/after. Expected array counts
are not hard-coded.

Collector stdin and stdout are exact bytes on HKUST, and its actual child PID,
wait code, command and stdout SHA are recorded. Only after actual wait0 and
accepted collection does the wrapper copy every embedded small original closure
document and the original ledger to new independent inodes. Each copy verifies
fresh original bytes/metadata before and after, plus destination SHA. Ledger
copy starts after collector closure acceptance. These proofs build the genuine
unchanged consumer input; the physical inventory and its real execution receipt
are bound together. No `accepted:true` field replaces a child wait.

The unchanged consumer checks all70 counters/updates, config/source identity,
seven closures, deterministic committed position coverage, accepted finite
proofs, direct-copy persistence, physical original identity, and shared
24GiB accounting. It does not reread NPY values, normalize distributions,
recompute KL/gradients or test GPU resume. The prior recorded coverage checks
supply NPY finite proofs.

Normal outputs are `collection.stdout.private.json`, actual execution receipts,
private byte copies and consumer inputs, `ledger-audit/audit.private.json`,
`complete.private.json`, and public `summary.json`. Consumer actual wait0 means
this narrow artifact join is complete. Actual wait2 means `PARTIAL`, with gaps or
residual reservations preserved; it is not a training failure. An earlier wrapper
exception preserves whatever actual stdout/stderr/execution files exist and
must never be called complete. The 5632 formal predictions, state-tensor checks,
final analysis and complete goal acceptance are separate requirements.
