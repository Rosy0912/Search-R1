# Generic closed-raw proof adapter (local CPU only)

This new entry bridges the accepted generic gzip backup to the unchanged published clipmass `FixedBackups` reader, and consequently its debug-coverage consumer. It does not modify the original generic helper, C1/C0 proofs, reader source, Node, training, backup queue, or payloads. Root must publish/read back this source before real use. Preparation does not certify any real group backup or analysis.

The original generic proof uses `opsd-ablation-closed-raw-compressed-local-backup-v1`. The reader is schema-name independent but requires three fields absent from that original proof. The output explicitly uses `opsd-ablation-closed-raw-reader-adapter-v1`, retains every original field and the exact original complete SHA, and adds only evidence-derived fields:

| Field | Required original evidence |
| --- | --- |
| `contiguous_regular_tar_members_verified` | Actual verifier rc0, original argv and program SHA, exact original complete SHA. Frozen `verify_members` requires regular exact members, no PAX, `offset == end`, `offset_data == end + 512`, each payload SHA/bytes and gzip CRC/footer/terminal padding. |
| `source_stream_fixed_handle_SHA` | Actual transfer rc0/error-null, same inline scope and inventory, exact source-operation SHA. That operation invokes the pinned V2 `stream_archive`, whose fixed file handles check pre/post fstat, streamed payload SHA, and source-path stat after each member and all members. |
| `source_noncache_file_set_stable_after` | Original after proof is accepted and binds the exact scope/inventory SHA, closures, source/run/count, all pinned metadata and the noncache file set. Frozen `after` re-enumerates paths/exclusions and checks all source stats. |

The entire generic five-payload source manifest SHA is pinned to `5ffb33951d3d9a8863c4a3fa097d9c2f3328c488ad18443e027a2ae338949228`; the original V2 dependency SHA is `d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57`. Every small source/proof is freshly SHA-read with opened-handle/path metadata continuity. Actual source is compiled from verified bytes only, never timestamp-valid pyc. No reader, model, tokenizer, Torch or GPU code is loaded in production.

## Immutable inputs

Use `opsd-ablation-closed-raw-reader-adapter-inputs-v1` with these explicit `{path, sha256}` refs (bytes can also be supplied):

```json
{
  "schema": "opsd-ablation-closed-raw-reader-adapter-inputs-v1",
  "source_manifest": {"path": "/ABS/generic/source-manifest.json", "sha256": "5ffb33951d3d9a8863c4a3fa097d9c2f3328c488ad18443e027a2ae338949228"},
  "transport_dependency": {"path": "/ABS/V2/backup.py", "sha256": "d2a3a3dd87dfed4df36a9ec27356073a0a663bfea6da3d1e84dc5e241e916f57"},
  "scope": {"path": "/ABS/scope.json", "sha256": "ACTUAL"},
  "inventory": {"path": "/ABS/inventory.json", "sha256": "ACTUAL"},
  "complete": {"path": "/ABS/backup/complete-sha-verified.json", "sha256": "ACTUAL"},
  "after_metadata": {"path": "/ABS/after.json", "sha256": "ACTUAL"},
  "transfer_launch": {"path": "/ABS/backup/transfer-launch.json", "sha256": "ACTUAL"},
  "transfer_terminal": {"path": "/ABS/backup/transfer-terminal.json", "sha256": "ACTUAL"},
  "verifier_execution": {"path": "/ABS/actual-verifier-execution.json", "sha256": "ACTUAL"}
}
```

The small `verifier_execution` receipt records an actual execution, not a guessed absence or a rerun:

```text
schema: opsd-ablation-closed-raw-verifier-execution-v1
actual_returncode: 0
program: {path, sha256, bytes} of the frozen verify-archive.py
argv: [absolute_python, "-B", exact_program_path,
       "--scope", exact_scope_path, "--scope-sha256", exact_scope_SHA,
       "--inventory", exact_inventory_path, "--after-metadata", exact_after_path,
       "--destination", original_complete_parent,
       "--source-helper", exact_V2_path,
       "--source-operation", exact_generic_operation_path,
       "--operation-sha256", exact_operation_SHA]
complete: {path, sha256, bytes} of the original complete-sha-verified.json
```

Only an execution with all these exact inputs can support the missing contiguous-check fact. Root may bind its captured original argv/return code/source refs and complete ref into this small receipt. Missing execution, nonzero rc, wrong program/invocation, changed inputs, wrong member identities or changed archive metadata reject/HOLD. Do not manufacture this receipt from `accepted: true` alone.

## CLI and unchanged downstream interface

```text
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -B adapt.py \
  --manifest /ABS/input.json --manifest-sha256 ACTUAL_SHA \
  --output /ABS/NEW/reader-compatible-proof.json
```

The output parent must already exist and the output file must not exist. Nothing is overwritten. In the original clipmass input's selected group `backups[]`, keep `kind: gzip` and the original inventory ref; use the new adapter proof ref for `complete`. Retain the original complete separately. The existing full-ten facts/config and SHA-fixed clipmass reader remain unchanged. The existing debug-coverage input still points to its prior accepted clipmass results and the exact published reader SHA. The adapter does not run either analysis or relax their token/count/selection checks.

This pass does not fresh hash/decompress the large archive or members. It reuses the original independent disk/member acceptance, bound actual execution and source-after evidence plus current archive metadata continuity. It does not assert current Node freshness, rerun strict QA, reconstruct logits/gradients, claim shared-ledger coverage, or perform numerical/training/effect acceptance. It retains the original `internal_member_padding_independently_checked: false` limitation. It does not archive native512 or mutable study ledger data.

Four focused CPU fixtures exercise the actual frozen local verifier on a tiny artificial USTAR and the unchanged `FixedBackups` constructor, nonzero/missing/mismatched execution, wrong stable-set/stream identity, wrong member and changed archive. They use no SSH, production payload, scientific state or Node completion. Run from publisher:

```text
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_ablation_closed_raw_reader_adapter_v1 -p 'test_*.py' -v
```
