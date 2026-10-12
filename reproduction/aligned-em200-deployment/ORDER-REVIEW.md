# Independent CPU baseline order capture review

Reviewer: `alignment_generation_review`. Date: 2026-10-09 (Asia/Shanghai). The author confirmed `capture_baseline_order.py`, `reference/`, tests and documentation frozen before this executable review. No implementation edits, remote operations, GRPO training, model generation, or GPU allocation were performed.

## Result

**PASS for the reviewed CPU reference-tool source and fixture scope.** No blocking source defect was found. This is not an actual order artifact, historical169103 parity receipt, runtime training approval, or proof of target-host execution. Deployment still must run the tool against the pinned common169028 inputs and official local tokenizer and bind the resulting immutable file before production admission.

## Archive and real-loader evidence

The reviewer directly read `/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-publish`, computed file hashes and extracted `RayPPOTrainer._create_dataloader` with Python AST. The bundled method equals the actual archive AST source segment plus its trailing newline, byte for byte. The original trainer SHA is `c46280f4bc753187471c9f31b983291270f17990915de0afa87ebd3040646964`. Archive dataset and torch-helper snapshots match exact bytes; archive model/fs dependency hashes match the provenance runtime requirements (`6b6def87d02f60692f175d3a4f8e8c2a07e914b896881c1ab3bf5e83287e359a` and `a922d349ce08eb4e23d878a33eba415e478ca467562b2082708fcc6f409ecb7e`).

The tool executes that actual method, original dataset/tokenization helpers and collate function. Its dataset subclass only records successful original `__getitem__` positions and returns the original row unchanged. The original DataLoader provides native shuffle, its iterator seed draw, zero workers, and drop-last. Every actual collated `index` must equal the stable question ID at each observed position. No synthetic permutation replaces those observations. The archive's prompt filter remains commented out; oversize prompts fail actual truncation rather than being silently filtered.

Capture validates the original user-pinned common manifest SHA `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24` through the source holdout validator, requires169028 training rows, and compares real archived dataset IDs to actual prepared-file identities. Historical169103 is explicitly excluded by policy/scope labels. Offline tokenizer loading disables remote code. Actual corpus/tokenizer assets are not available in this local review, so that full capture path is source-reviewed rather than asserted as executed.

CPU RNG is explicitly set/restored and captured immediately before `iter(loader)`, including full uint8 state and digest. Default seed42 is labeled a new common169028 CPU reference boundary, not historical model-initialized RNG reconstruction. Supplied RNG files require integer bytes0..255 and are SHA-bound. Reusing the captured boundary reproduces observed order despite caller RNG drift; another boundary changes order.

## Output and SOURCE080 consumption

The output uses the production `baseline-aligned-order-v1` schema with ordered original IDs, batch512, one complete epoch of observed indices, drop-last and zero workers, plus actual per-batch indices/IDs, RNG, source/tokenizer/config/common identities and tool hash. Canonical `order_sha256` covers all extended fields. Actual `baseline_alignment_data.verify_order_manifest` from the production alignment checkout accepted the captured full fixture. Source inspection confirms `RecordedOrderSampler` consumes these `epochs` and the saved hash;330 complete batches cover requested outer1..200 without recording only a partial200-batch epoch. The required totals are330 batches,168960 emitted questions,68 dropped. Exclusive atomic publication refuses existing output, with fsync and read-only mode.

## Independently executed verification

From the deployment-tools worktree:

```text
PYTHONPATH=../bapo-baseline-alignment-20261009 ../fresh200-local-test-env/bin/python -m pytest -q reproduction/aligned-em200-deployment/test_capture_baseline_order.py
5 passed, 1 warning in 8.00s
```

Tests execute real archived dataset and DataLoader over synthetic169028 parquet rows through all330 batches with a small CPU tokenizer fixture, check observed positions and collated IDs, consumer schema/hash validation and tamper rejection, RNG replay/drift, actual archive truncation failure, reference snapshot tamper rejection, exclusive no-overwrite output, and unpinned common-manifest rejection before model/data access. The warning is the existing local urllib3/LibreSSL compatibility notice. These fixtures are not the actual user cohort or official tokenizer. Runtime model/fs guards and full actual loader paths remain hard failures; they must not be bypassed to obtain an order file.

## Frozen source SHA256

| Reviewed file | SHA256 |
| --- | --- |
| `capture_baseline_order.py` | `f17fd803d378c8bfdf9107360779e067dc52f5755891f01222694798f0f93994` |
| `reference/create_dataloader.py` | `ada6e4261c0dcde17734f660d30ca8b2024004bbfc3c5bd935bb4e9e2334842c` |
| `reference/provenance.json` | `0dd5f3ee3913bc78c59201911b2300a74b28f86eb8ec924f7ffbd1383e0aca6b` |
| `reference/rl_dataset.py` | `e95b34a590956ed37e69631c87712c9dacbddc0fae632ca1e5403ed19e409815` |
| `reference/torch_functional.py` | `e6f0578b0f52b530d52e9fc69968108d67badeb8374f18f39258e3550ac35454` |
| `test_capture_baseline_order.py` | `5ba30e11c3f0b733a7131121c64f16d6aaf5bc0c43b553ae63a4819f04fa4392` |
| `ORDER-CAPTURE.md` | `efbd576e21192bdceff604de6553364b02520d54f3fd45650d865fd60e00552d` |

Signed: `alignment_generation_review`, independent source/CPU reviewer. Signature scope is the exact source hashes and tests above; actual deployment evidence remains outstanding.
