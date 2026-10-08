# SOURCE080 aligned first-five CPU evidence candidate

This helper is an unexecuted candidate for the new formal job, not an acceptance report or release signer. It does not submit jobs, change services, mutate the training checkout, create release receipts, resume training, or claim a GPU reload. Its fixed source is `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`; its fixed new NFS root is `/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009`.

The actual job, token and RUN must come from the new root's original `scheduler-head/formal-submit-receipt.json`. Jobs3315 and3345, historical roots, old sources and synthetic receipts are refused. The receipt's formal-tool hashes are checked against deployed bytes. The helper uses SOURCE080's actual aligned manifest-v3 validators directly, without manufacturing an independent review or calling the release signer.

## Future invocation (not executed)

After independent review and deployment, use the isolated root environment and a new output directory under `ROOT/evidence`. GPUs must be hidden before this Python process starts:

```sh
CUDA_VISIBLE_DEVICES='' /mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/env/bin/python /mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/tools/deployment/aligned_first5_cpu_evidence.py --output /mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009/evidence/first5-cpu-NEW-UNUSED-NAME
```

No job or RUN override exists. Outputs use exclusive creation and retain incomplete evidence with the exception traceback. A rerun needs a new unused directory. The script prints factual collection status, `acceptance:false` and `release_receipt:false`; the independent reviewer must read the saved evidence rather than treating process exit as approval.

## What is checked

* The actual source080 checkout is clean, Linux environment is the fixed isolated environment, the copied generic math module matches its extraction provenance, and every original manifest member's bytes/hash stays stable throughout extraction. The original manifest itself is bound before parsing, matched to the original pause marker, and rehashed after CPU work along with all three helper/provenance files; changes prevent collected status.
* Five accepted same-driver-PID attempts, no skip, five20-write rank ledgers (100 writes per rank), complete-outer scheduler progress5 with horizon200/warmup57, pending-stage cleared, stage64 proof and final commit records, EM audit/provenance, fixed common0b input/order/retrieval bindings: the actual source080 gate performs the shared validations. The helper independently binds original driver and eight rank reload sidecars, fifth ledger, source/config, pause nonce and an immutable original train-log prefix. Original processes remain alive with the same start ticks, owner UID, executable, RUN and job throughout extraction. These checks observe an existing fifth pause; they cannot cause one.
* The official logical model path must be exactly the deployment's reviewed borrowed symlink, resolving to `/dataset/shaoshitong/search-r1-rl/bapo-fresh200-safety-20261006/assets/qwen2.5-3b-instruct-aa8e725`. Its pinned official manifest and all12 files are fully hashed; the symlink target and manifest are rechecked after extraction. The fifth HF export's complete12 files are independently hashed before and after CPU work.
* Offline real HF/tokenizer load, complete tokenizer metadata and chat-tokenization comparison, Qwen2.5-3B dimensions/parameter count, actual tied embedding storage, all loaded CPU parameters finite, and an actual finite CPU forward. Every official/export safetensor is scanned, including tensors after the first changed tensor. Official weights are cast to the export dtype before measuring all weight changes, so rounding alone cannot pass. Full scan reports changed elements, maximum/L1 difference and precision pairs.
* GPU hiding, unavailable CUDA and uninitialized CUDA are required before driver deserialization, HF work and each native inspection, and after those operations including failed native loads. Any CUDA initialization invalidates the extraction. Original GPU producer exact-roundtrip reports are archived/hash-bound. CPU deserialization of DTensor/native archives is only inspection; unsupported CPU deserialization is recorded with a traceback and is never represented as a GPU reload. There is no process-group initialization or GPU device operation in this helper.
* Wall time, CPU time, process peak RSS and runtime versions are recorded. These are extraction costs only, not invented GPU allocation hours or monetary cost.

## Local validation scope

Run from the tools checkout:

```sh
PYTHONPATH=../bapo-baseline-alignment-20261009 ../fresh200-local-test-env/bin/python -m pytest -q reproduction/aligned-em200-deployment/test_aligned_first5_cpu_evidence.py
```

The20 CPU fixture tests cover historical-identity refusal, exclusive writes/symlinks, postwork mutation refusal for the manifest and each of the three helper files, CUDA visibility/availability and failure-after-native-load guards, pinned borrowed-directory identity, full precision-aware weight scans (including later nonfinite tensors), an actual tiny Qwen2 HF/tokenizer/tied/finite-forward load, native-unavailable recording, old schema refusal, and original driver/manifest plumbing through the real five-attempt validator. The plumbing fixture's EM/native/source validator is a spy, its native reports are explicitly synthetic, and live Linux process validation is disabled there. Thus these tests do not certify actual source080 assets, the real official3B CPU load, actual formal submission, EM audits, native DTensor reloads, Slurm, target hardware, process identities or training success.

Generic functions were extracted as exact AST source segments from the earlier helper, with original/extracted SHA in `aligned-first5-math-provenance.json`. Historical ROOT/source/job/production21 policy, raw-native sampler assumptions and old release logic were not imported. The current PPO denominator remains frozen HF-old; this evidence helper imposes no native-behavior denominator requirement.
