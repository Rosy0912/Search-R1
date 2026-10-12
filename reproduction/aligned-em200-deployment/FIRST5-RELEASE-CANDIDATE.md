# External first-five release publication candidate

This tool is not an independent reviewer or PASS generator. It publishes the **exact bytes of an externally authored and reviewed release receipt**, after the actual SOURCE080 `fresh200_gate.validate_release` accepts those bytes. It has not processed an actual five-outer checkpoint, executed on target Linux, or published a real release. Source080 remains unchanged.

## Design and deployment

`publish_first5_release.py` has two stages. `prepare` validates all inputs and saves an exclusive `release-payload.json` byte copy plus `release-plan.json` under a new directory directly below the fixed ROOT/evidence. A reviewer can inspect both files and freeze their SHA values. `publish` requires the explicit reviewed plan/payload SHAs, repeats validation, and publishes at the actual resolved config's gate path using a fully written/fsynced temporary file plus atomic no-replace hardlink. Consumers never observe a partial JSON file. A differing existing receipt is never overwritten; identical existing bytes produce a no-write result. No retry, job submission, process signaling, service change, or training resume is performed.

Append the publisher and its tests to the existing, canonical `ROOT/tools/deployment/first5-cpu-v1` directory. Do not overwrite the three already deployed CPU helper/provenance files. The production location guard requires this exact directory and its actual colocated extractor. This replaces the earlier separate first5-release-v1 suggestion. Root owns source publication and any installation; this candidate does not deploy itself.

Fixed ROOT: `/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009`. Fixed algorithm source: `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`. Actual job/token/RUN are derived only from the original `scheduler-head/formal-submit-receipt.json`. Old jobs3315/3345, hardware submission scope, historical roots/sources, another token/RUN, dirty checkout, another interpreter, optimized Python, or visible CUDA are rejected. Production must invoke the new ROOT/env interpreter with `CUDA_VISIBLE_DEVICES=''`.

## Externally supplied receipt

The external receipt must contain the complete existing SOURCE080 aligned release schema. In particular its status must be `PASS_INDEPENDENT_BASELINE_ALIGNED_FRESH200_STEP5_REVIEW`, with the actual nonce/config/checkpoint/profile/accounting/prefix/evidence/independent_review fields required by `validate_release`. The publisher never adds this status, copies identity from a manifest into a PASS receipt, or converts CPU collection success into independent acceptance.

The independent author additionally supplies:

- `publication_schema`: `aligned-first5-external-release-publication-v1`.
- `formal_submission`: `{path, sha256, bytes}` binding the original dynamic formal submit receipt.
- `cpu_evidence`: `{path, sha256, bytes}` binding the complete actual first5 CPU evidence JSON.
- `driver_birth`: the extractor's original driver process fields `pid`, `start_ticks`, `command`, `uids`, `executable`, `selected_env`; volatile process state is excluded.
- `independent_review`: the original externally written review report path/SHA, already required by the source gate. The CLI also requires its independently selected SHA.

The CPU extraction must bind the same submit receipt, source, common/config, nonce, immutable prefix, original manifest members and eight native GPU-produced reports, original driver plus eight rank process births, all twelve fifth HF files, complete official initial model manifest/file inventory, and actual deployed helper/formal sources. Its official CPU HF result must report actual finite parameter/load/forward checks, preserved tied storage, exact official parameter count and a nonzero finite dtype-aligned trained-weight difference. Tiny CPU fixtures, incomplete CPU output and unavailable official CPU HF results are not accepted. CPU native inspection is not promoted into a GPU reload; native reload/EM/stage/ledger/config semantics remain checked by the actual source gate and the independent reviewer.

The publisher hashes the fifth HF export and all original member inputs, and binds the pinned original official-model manifest. It does not rehash the giant static official model assets that the CPU extractor fully scanned and the independent report reviewed. That static-asset claim comes from the external actual CPU evidence/report; the publisher is not a second official-model scanner.

After slow gate validation it closes the original manifest, signed evidence, immutable prefix, CPU/report/submission, helper/HF/native-member metadata, rehashes the small CPU/report/submission bindings, and rechecks no RELEASED marker/exactly five attempts plus original process births. It cannot lock the training processes; a process may die after the last observation, in which case a receipt cannot restart it. Publication does not claim continuation or later training success.

## Reviewable usage

The following commands are templates only. Fill SHAs from the independently reviewed originals; no arbitrary job/RUN/source overrides exist.

```bash
ROOT=/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009
CUDA_VISIBLE_DEVICES='' "$ROOT/env/bin/python" -B "$ROOT/tools/deployment/first5-cpu-v1/publish_first5_release.py" prepare \
  --receipt "$ROOT/evidence/EXTERNAL-INDEPENDENT-RELEASE.json" \
  --receipt-sha256 REVIEWED_RECEIPT_SHA \
  --report-sha256 REVIEWED_REPORT_SHA \
  --cpu-evidence-sha256 REVIEWED_ACTUAL_CPU_EVIDENCE_SHA \
  --output "$ROOT/evidence/NEW-UNUSED-RELEASE-PLAN"
```

Review the exact payload and plan first. Only then, under the actual first-five release authorization, use `publish --plan <release-plan.json> --plan-sha256 <reviewed SHA> --payload-sha256 <reviewed external receipt SHA>` with the same interpreter/location/CPU environment. Reviewable plan preparation grants no release permission.

## Local verification scope

Focused tests: `PYTHONPATH=../bapo-baseline-alignment-20261009 ../fresh200-local-test-env/bin/python -m pytest -q reproduction/aligned-em200-deployment/test_publish_first5_release.py` — **17 passed**. Every test publication uses a temporary fixture root; none writes the actual ROOT gate.

Tests cover reviewed exact-byte prepare/publish, no overwrite/partial file, same-receipt no-write, symlink rejection, old jobs, altered report/CPU/HF/source, incomplete official CPU evidence, PID reuse before/after slow validation, manifest/HF mutation during validation, live release/sixth-attempt changes, and explicit plan/payload hash tampering. One integration rejection directly calls the actual SOURCE080 gate implementation for incomplete GPU fixture evidence. The positive publication fixture uses a gate spy to verify exact receipt/manifest/repo/original PID forwarding; it is not an actual complete source-gate positive acceptance, official Qwen CPU load, Linux /proc test, GPU reload or independent review. Existing source-gate tests separately cover complete simulated first-five ledger/native/EM/stage behavior. No actual first-five PASS or release receipt has been signed by the tool author.
