# Independent committed-token audit

This stdlib-only helper reads one explicitly pinned four-card signal-weight
checkpoint and its own `outer_update_receipts`. It never glob-selects failed
attempts, trains, generates, acquires a resource lock, signals a process, imports
Torch, restores RNG, deserializes Adam, reads full-vocabulary arrays, calls HTTP,
or changes original training/config/source bytes.

After publishing and verifying this new namespace, run on HKUST4:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 ORIGINAL_ENV_PYTHON -B audit.py
  --config ORIGINAL_IMMUTABLE_CONFIG
  --config-sha256 EXACT_CONFIG_SHA
  --checkpoint ACCEPTED_COMMITTED_MANIFEST
  --checkpoint-sha256 EXACT_MANIFEST_SHA
  --expected-updates ACTUAL_ADAM_COUNT
  --expected-outer ZERO_BASED_OUTER
  --original-helper-manifest-sha256 af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314
```

For the first same-run checks, pin actual update1/update5 and outer0 manifests,
not mutable `latest`, initialized processes or adapter exports. The helper can
also read a later explicit complete/partial outer checkpoint with a committed
update inventory. Zero committed updates do not establish a token audit.

It reads and freshly hashes every file in the exact original00fec71 helper
inventory. It compiles only the original stdlib function ASTs:
`trainer.verify_committed_receipts`, `sibling_scores.weights`,
`sibling_scores.compare_selected_training_metrics`, and `checkpoint.science_contract`.
Their original whole-file SHA remain fixed. No copied mathematical rewrite or
model import substitutes for those functions.

The audit independently checks committed cursor/order/update counts and exact
four-rank shuffled pair assignment; complete header sampling identity identical
to the original outer and checkpoint; source/config/teacher/outer-old/preupdate
adapter identities; original selected D/reference text SHA, pair SHA, sibling
row SHA and encoding identities; effective recorded mask/token IDs/offsets and
actual global pair denominator. It repeats the original scorer comparison and
normalization, including all-five mean1/range.1–1.9, sorted unique answer indices
within valid policy positions, G-arm missing-answer gap neutral positions and
explicit all-five encoding fallback. H-arm entropy weights remain dynamic when
answer gap is missing and entropy is available. An encoding fallback is
reported `checked=false`/NA and never counted as a logits match.

Every saved valid-token scalar must be finite. The audit reconstructs original
raw/post loss and drift means, clip counts/mass metadata, trajectory contributions,
weighted trajectory means and actual-count batch contributions. It divides by
actual pairs, including a tail; never by summed weights. The scorer/actual-forward
comparison uses the original absolute/relative2e-5 budget, and output preserves
maximum observed errors. These are recombinations of saved *actual preupdate
forward* records, not newly computed logits or reconstructed historical generation
scores. Header encoding SHA/mask/token identities are bound records; this helper
does not retokenize original text or independently reproduce all-vocabulary KL.

Stdout schema is `opsd-signal-committed-token-audit-v1`. Successful output includes
accepted, explicit config/checkpoint refs, run/variant/counters, committed update
and token/pair counts, per-update selected-weight mean and reconstructed weighted
loss, checked/fallback/gap-missing counts, top32/full-vocabulary metadata counts,
file inventory SHA and scalar max-error summaries. It emits no original question,
response, token ID/text, tensor or raw probability array. Exit0 means this pinned
record audit accepted; exit2 preserves a compact reason. It is separate from
numeric Adam readback, byte backup, five-update execution acceptance and EM effects.
Capture only its small stdout on Mac; original logs and arrays remain at HKUST.

CPU fixtures use tiny synthetic JSON/JSONL and actual original function bytes:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover
  -s THIS_HELPER -p test_audit.py -v
```

`OPSD_SIGNAL_TOKEN_TEST_ORIGINAL` identifies the actual published original helper
when deployed separately. Fixtures reject wrong full/tail denominator, weight-sum
normalization, changed signal/weight, invalid/duplicate mask positions, mismatched
token IDs/preupdate model, unreported NA and nonfinite/generated-score metadata.
A complete pinned seven-pair tail runs the full reader, excludes an unreferenced
failed attempt and verifies compact output without Torch or private case text.
