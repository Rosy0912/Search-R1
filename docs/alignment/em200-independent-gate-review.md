# Independent EM200 first-five gate review

Date: 2026-10-09. Base HEAD: `597d227c2249883c937613b1a15c3f778fc400f5` plus the reviewed working-tree changes. This is a source/CPU evidence review, not a deployed GPU first-five approval or permission receipt.

## Scope and conclusion

Reviewed the independently authored `fresh200_gate.py`, `training_em.py`, controller EM integration, and trainer first-five evidence collection. The reviewer authored the outer-commit helper earlier; this report does **not** claim independent review of that implementation. Its evidence schema was checked only as an interface consumed by the gate.

**Current result: no outstanding must-fix in this reviewed gate/EM scope.** The native pending-outer admission gap identified during review was fixed in the current bytes and independently rechecked. Real target-Linux/GPU states, baseline observed order, loaded retriever identities/golden probes, and an actual independent release receipt remain prerequisites for production acceptance.

## Must-fix finding resolved before acceptance

The earlier gate verified native step5/scheduler5, exact reload digests and the latest committed rank event, but did not require `aligned_pending_outer` to be present and empty. A report could therefore omit the pending-state field while still satisfying these checks. This was a missing explicit native state admission condition, even though the real producer cleared its pending state after commit.

The current aligned-only guard requires both `'aligned_pending_outer' in report` and `report['aligned_pending_outer'] is None` on **each of eight** native rank reports. The native producer exposes the value from the saved/reloaded payload rather than supplying a gate-side default. Legacy admission remains unchanged.

The missing-key and non-null counterexamples now reject at this guard. Independently exercised the actual `validate_release` with the existing synthetic CPU fixture, refreshing the changed report, manifest and release evidence hashes so rejection did not depend on a stale hash. Both cases raised the specific pending-outer error. These fixtures simulate native state; they do not prove real SHARDED/DTensor GPU reload.

## Verified interface and evidence checks

- The aligned protocol is version3, exactly five accepted attempts,20 actual writes per rank per outer,100 cumulative writes per rank, zero skips and no mini-batch switch. Legacy sampler-denominator proof is not invoked for aligned HF-old PPO.
- Immutable source/config/input/order/retrieval bindings and the original new-run claim are collected. Release checks the live immutable log prefix, exact driver PID, next step6, actual completed count5 and driver sampler/RNG/BAPO reload digest.
- Eight native reports bind state/report paths and hashes, step5/scheduler5, model/optimizer state, original process identities, native version3, exact state and generation-RNG reload digests, the latest final rank ledger, and the explicit empty pending state.
- Each of five stage proofs binds source hashes, original run/PID, common and stage inputs, order manifest, exact integer64 coverage and finite EM score in[0,1]. Gate recomputes the canonical stage SHA, checks actual metrics and compares provisional/final rank records. Final records must be `STAGE_COMMITTED`, accepted, scheduler before outer-1/after outer, with the prescribed warmup LR. Prepared records preserve the twenty-write facts without claiming accepted completion.
- EM collection uses the original repeated QA batch before actor update, records all512 questions×5 siblings before resampling and after retention, preserves sibling/source identity through replacements, and binds the policy snapshot, actual outer/attempt, original targets and question IDs. EM is calculated from masked authored model spans and the legal terminal answer, independently of the BAPO reward.
- EM replay verifies authored span content/hash/boundary, terminal answer and EM, ordered initial/resample/retained provenance, exact coverage and sibling identity, original ID/target digests, retained identity and both response-mean and any-of-five aggregates. Gate compares these recomputed metrics with the actual attempt summaries and source/run/input/order bindings. Audit files publish atomically and refuse overwrite.
- Trainer first-five collection includes five EM journals, prepared/stage proofs and native/driver states; aligned collection avoids the legacy native-sampling denominator artifact. Release waits in the original process and grants continuation from outer6; the receipt is not evidence that all200 have completed.

## Validation and limits

`tests/baselines/test_baseline_aligned_first5_gate.py` and `tests/baselines/test_training_em.py`: **22 passed** on the local CPU test interpreter. This includes actual EM producer/replay→gate integration at512×5 over five outers, controller resampling integration, and the new pending-state negatives. The parent reported253 tests passing before this focused review; that broader result was not independently rerun here.

No source edits, remote operations, training operations, release publication or commits were performed by this review. Only this requested Markdown report was written.

## Reviewed byte identities

| File | SHA256 |
| --- | --- |
| `search_r1/baselines/fresh200_gate.py` | `746db2f6772903b488ef553d4996a58c75299b251005e6fead280413521dfff7` |
| `search_r1/baselines/training_em.py` | `8c3659081c49ce85430bbf54360800eac245aa7db4addbd63544e1fa62c02fb5` |
| `search_r1/baselines/controller.py` | `b8cce07c151b89bfa847c9808c6cb79ef444b91b493874a4f350f9d31f572cd2` |
| `verl/trainer/ppo/ray_trainer.py` | `a0382e19a98c49efd246c21c6aa49c7669c189d6c3879fc44ac90ef3d9ec34bc` |
| `verl/workers/fsdp_workers.py` | `cecb2714927c258703b47701c9a5583c9e1305791698fef4831f25bda3a91729` |
| `tests/baselines/test_baseline_aligned_first5_gate.py` | `c7d547a50688f1d640dc2e5adbca9bd23fc97a4e81b3473639ff2eff6255e8f4` |
