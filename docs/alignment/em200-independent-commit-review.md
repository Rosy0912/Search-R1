# EM200 independent stage-commit review

Verdict: **ACCEPTED FOR SOURCE AND CPU TEST SCOPE; no remaining must-fix findings.** This document does not approve a GPU run, establish training results, or issue a first-five production release receipt.

Reviewer: `/root/alignment_protocol_data_review`. Reviewed the frozen working-tree implementation based on `597d227c2249883c937613b1a15c3f778fc400f5`. The source hashes below identify the reviewed bytes; the base commit alone does not identify these uncommitted changes.

## Independence and scope

The independent scope consists of `aligned_outer_commit.py`; the stage-commit integration and native fault-state fields in `fsdp_workers.py` and `ray_trainer.py`; `fresh200_gate.py`; the newly added production/common-input binding in `baseline_alignment.py`; the production YAML and launch entries. Source inspection also followed their contract/progress and checkpoint call sites.

The reviewer authored `training_em.py`, its Controller instrumentation, and the earlier protocol/data alignment utilities. Those implementations are explicitly excluded from this independent approval. Their interface with the gate was inspected for matching run, source, PID, attempt, snapshot, order, retrieval and common-input bindings; this is not a claim of independent review of the EM implementation itself.

## Findings and resolved checks

- All eight ranks must complete twenty optimizer writes with no skips before the worker prepares an outer. `WRITES_PREPARED` does not advance the scheduler or the driver's accepted-outer counter. A pending outer blocks a new update.
- The driver persists its provisional ledger, runs the prescribed complete stage64 evaluation, and binds its finite score in `[0,1]`, source, original process, input identities and question-order manifest. Only then does it call the collective scheduler commit. The final rank event is `STAGE_COMMITTED`; driver acceptance follows validation of all eight final reports.
- Stage failures, short/nonfinite evaluation, proof-file failures, periodic evaluation failures after writes, scheduler admission/disagreement, scheduler failure and final-ledger persistence failure exit through fault handling. Native state carries the factual latest rank event and pending commit state. Retained optimizer writes and possible scheduler side effects remain distinguishable from accepted outers; no automatic rollback or retry is introduced. A later scheduled checkpoint failure preserves an already committed outer as factual completion rather than relabeling its completed writes.
- Ordinary successful outers write small prepared/stage/final JSON records. They do not trigger a new model/Adam checkpoint on every outer. Existing scheduled checkpoint boundaries and the fifth-outer full-state roundtrip remain the costly persistence points.
- First-five release independently requires exactly five complete aligned attempts, one hundred writes per rank, no skips, immutable prepared/stage/final records, and the provisional-to-committed transition. Stage coverage, finite score, exact scheduler epochs and warmup57 learning rates are checked against the actual metrics. HF-old denominator and diagnostic-only raw-sampler roles remain explicit; aligned release does not require the legacy native-behavior denominator proof.
- The release binds the original driver PID, run root, pause nonce, source and native rank process identities. It verifies fifth-state roundtrip reports and the current train-log prefix; it permits continuation of the original process to outer6 rather than authorizing a new job. The launcher claim PID is intentionally distinct from the training driver PID.
- `run-production.sh` selects the whitelisted production profile. `run.sh` requires clean source, an exclusive new-run claim, a source marker created with exclusive open, and the resource supervisor/preflight path. The production profile requires EM, stage commit and the first-five gate, while the original common manifest SHA is pinned to `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`; actual train/dev/final/stage loader paths must match its bound files.

Earlier review findings were resolved before this verdict: missing order fields in the EM gate binding; absence of per-outer stage/provisional/final evidence checks; missing finite-score and exact-LR receipt validation; and a postwrite periodic-evaluation fault window. The final small gate follow-up also requires the native report to contain `aligned_pending_outer` explicitly with value `None`: an omitted key or a remaining pending outer refuses release. The worker already emits this field from its saved native payload; the check is conditional on the aligned profile and leaves legacy admission unchanged. No reviewed production-source blocker remains.

## Independently executed verification

The reviewer ran the workspace interpreter with `PYTHONPATH=.` against:

```text
tests/baselines/test_aligned_outer_commit.py
tests/baselines/test_baseline_aligned_first5_gate.py
tests/baselines/test_baseline_alignment_common_goal.py
tests/baselines/test_fresh200_driver_fit.py
tests/baselines/test_baseline_alignment_contract.py
tests/baselines/test_fresh200_contract.py
tests/baselines/test_fresh200_skip_contract.py
```

Result: **69 passed**, 2 dependency warnings, 5.02 seconds. Interpreter: `../fresh200-local-test-env/bin/python`. Both `reproduction/baseline-aligned/run.sh` and `run-production.sh` passed `bash -n`. After the explicit pending-state gate follow-up, the reviewer reran `test_baseline_aligned_first5_gate.py` and `test_fresh200_gate_skip.py`: **18 passed** in 2.91 seconds, including missing/nonempty pending-state rejection and legacy release coverage. Only the gate source hash changed from the initially reviewed nine-file snapshot; its final SHA is recorded below.

The stage-commit tests perform real CPU Adam and scheduler updates and invoke the actual driver completion/fault methods. Distributed collectives are simulated in these tests. Actual-fit tests exercise the real trainer loop with mocked GPU RPCs; receipt fixtures simulate files/process evidence. The gate suite includes actual EM producer-to-verifier interface coverage, which is an integration check against reviewer-owned code, not an independent EM review. No live eight-rank FSDP/NCCL stage-commit run, target-GPU resource check, retrieval service execution, first-five training evidence, or EM200 completion was performed by this review. The parent-reported combined 253-test pass is separate from the 69 tests independently executed here.

## Reviewed source identities

The table is generated from the frozen source bytes after the independent checks.

| File | SHA256 |
| --- | --- |
| `search_r1/baselines/aligned_outer_commit.py` | `baa21cab2007371789719408c60a9cb22dbafafead63094d100a5301a9aad872` |
| `verl/workers/fsdp_workers.py` | `cecb2714927c258703b47701c9a5583c9e1305791698fef4831f25bda3a91729` |
| `verl/trainer/ppo/ray_trainer.py` | `a0382e19a98c49efd246c21c6aa49c7669c189d6c3879fc44ac90ef3d9ec34bc` |
| `search_r1/baselines/fresh200_gate.py` | `746db2f6772903b488ef553d4996a58c75299b251005e6fead280413521dfff7` |
| `search_r1/baselines/baseline_alignment.py` | `1cf26f68a725935d0dcb06438b3b53bb72f03fdb68bddbc12c1468a465aeeeef` |
| `search_r1/baselines/fresh200_contract.py` | `2eb13c37890830323022557a4fa149dced60aa63d9e70c2cd75a6e9313b88903` |
| `verl/trainer/config/ppo_bapo_baseline_aligned_production.yaml` | `2d8bb4e6c145cf1bab3cd55575a7381ad7eb48cea193857393985e1079d4d0cb` |
| `reproduction/baseline-aligned/run.sh` | `4e6c60196e6519b71b48bd862496556431498ff37f520a96f63454f8a502b95b` |
| `reproduction/baseline-aligned/run-production.sh` | `a664fd90dfe6e9fa990a57767cec36958f2becca70882b7e2962578089bcfa2b` |
