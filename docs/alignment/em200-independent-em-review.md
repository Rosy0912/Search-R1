# Independent EM200 source review

Date: 2026-10-09 (Asia/Shanghai). Reviewer: `alignment_generation_review`, independent of the authors of the EM journal/controller instrumentation and the two-phase outer commit implementation. This review did not modify implementation files, launch training, contact a remote machine, or produce an experiment release approval.

## Outcome and reviewed snapshot

No blocking defect was found in the reviewed EM instrumentation or production stage-before-acceptance flow. The conclusion applies to the working-tree source bytes listed below, based on source inspection and the independently executed local tests. The repository HEAD was `597d227c2249883c937613b1a15c3f778fc400f5`; the reviewed production additions were not yet represented by that HEAD. Hashes bind this review to their actual contents rather than treating the old commit as the new implementation.

| File | SHA256 |
| --- | --- |
| `search_r1/baselines/training_em.py` | `8c3659081c49ce85430bbf54360800eac245aa7db4addbd63544e1fa62c02fb5` |
| `search_r1/baselines/controller.py` | `b8cce07c151b89bfa847c9808c6cb79ef444b91b493874a4f350f9d31f572cd2` |
| `search_r1/baselines/aligned_outer_commit.py` | `baa21cab2007371789719408c60a9cb22dbafafead63094d100a5301a9aad872` |
| `verl/trainer/ppo/ray_trainer.py` | `a0382e19a98c49efd246c21c6aa49c7669c189d6c3879fc44ac90ef3d9ec34bc` |
| `verl/workers/fsdp_workers.py` | `cecb2714927c258703b47701c9a5583c9e1305791698fef4831f25bda3a91729` |
| `search_r1/qa_protocol.py` | `9a825a7b222b5a3fd2c1de7f63ed38e157258fcf858e773ff64c963a4b6daddd` |
| `verl/utils/reward_score/qa_em.py` | `fa5e8f9386b0db2411dafa7612bc633a1ced362232c81f2495fe78c7ed1b4997` |

The earlier first-five gate implementation was authored by this reviewer and is outside the independent signoff here. Its integration appears only as a boundary to the reviewed producer; this document does not independently approve that gate.

## EM provenance, masks, and scoring

`Controller.generate` enables the journal through `baselines.alignment.training_em_enabled`, after initial rollout generation and before actor updates. It captures the original repeated QA batch IDs and targets, outer/attempt/snapshot/source/run/driver identities, shared data and retriever manifests, order manifest, and common-input binding. The initial capture precedes resampling; each replacement uses `rows[int(replacement_source)]` to translate local replacement positions back to the original repeated batch. Re-composed retained states keep the original source index. The full-size controller test verifies initial and retained counts of 512 original questions and 2560 responses, including changed retained scores after resampling.

`TrainingEMJournal.capture` uses the response suffix of `info_mask AND attention_mask`. `decode_model_spans` decodes contiguous authored regions separately; it does not join across retrieval observations. `extract_final_answer` accepts only the final authored span with one terminal, non-nested answer action. An earlier answer, a final search, nested answer/search markup, and an observation containing answer markup do not supply EM. Response token and authored-mask hashes are recorded after removing unattended padding. EM uses the existing normalized exact match function (case, punctuation, articles, whitespace), against any supplied target alias. It is distinct from BAPO F1/IDK rewards: a legal IDK answer can have EM1 when the reference actually matches it.

The journal stores initial, replacement, and retained phases with generation ordinals and stable question/sibling identity. Retained capture must match the latest generated answer, spans, score, and trajectory; the verifier additionally requires equality of all retained row fields except phase, including token/mask digests. Replay checks phase ordering, replacement generation monotonicity, original target/ID digest, exact sibling ordinals and coverage, terminal extraction, individual EM, and every reported aggregate. Initial and retained response EM, any-of-five question EM, exact correct counts, valid-answer counts, denominators, and resampled response counts remain separately named metrics.

Publication uses a temporary file, file fsync, exclusive atomic hardlink, read-only permission, and directory fsync. An existing audit filename cannot be overwritten. A rollout audit is explicitly labeled pre-update and is not accepted-outer evidence. Instrumentation-on/off controller testing checks unchanged generated tensors, source mapping, rollout cost metrics, and generation call counts.

## Twenty writes, stage, scheduler, and accepted outer

In the production actor worker, full write ledgers are persisted as `.prepared.json` with `WRITES_PREPARED`, `accepted_complete_outer=False`, and unchanged scheduler epoch. `prepare` requires twenty complete writes on every rank and records pending outer/attempt/write-ledger identity. The next update refuses unresolved pending state.

The trainer persists the prepared all-rank record, evaluates the configured independent stage, validates exact integer coverage64 and finite score within [0,1], and persists a canonical hashed stage proof binding source/run/driver/common/stage/order inputs. Only then does it invoke the all-rank `commit_aligned_outer` RPC. Collective admission checks pending outer/attempt, scheduler epoch, identical stage proof, and rank agreement before stepping the scheduler. The final rank artifacts are published as `STAGE_COMMITTED`; a duplicate commit is rejected. Driver validation precedes controller outer-count advancement and accepted-attempt recording. Scheduled checkpoints occur afterward.

Stage failure, incomplete/nonfinite stage, prepared/stage persistence failure, or RPC failure invokes fault backup and cannot record an accepted outer. Actual Adam writes remain factual and are not rolled back. A scheduler or final-ledger failure after a side effect remains fatal partial state with pending fault identity, rather than allowing retry as completed work. A checkpoint failure after accepted completion preserves the factual completed count and records the later checkpoint infrastructure fault. These distinctions are exercised by the tests below.

## Independently executed tests

Executed in the reviewed repository using the existing local dependency environment:

```text
PYTHONPATH=. ../fresh200-local-test-env/bin/python -m pytest tests/baselines/test_training_em.py tests/baselines/test_aligned_outer_commit.py -q
30 passed, 2 warnings in 2.14s
```

The five EM tests exercise actual journal publication/replay, resample provenance, terminal and observation masks, normalized EM and independent IDK scoring, tamper rejection, no-overwrite behavior, and actual controller instrumentation for512x5. The25 outer-commit cases exercise real CPU Adam/scheduler writes, actual driver completion/fit methods, five successful outers with100 writes, collective epoch disagreement, duplicate/wrong-attempt rejection, stage/admission type checks, and persistence/evaluation/scheduler/checkpoint faults. Collective/GPU workers and generation are controlled fixtures where hardware is required. The two warnings were pre-existing local urllib3/LibreSSL and Ray deprecated import notices; neither was a test failure. The parent-reported253-test result was not independently rerun as part of this focused review.

## Additional first-five native-state boundary finding — resolved

At the parent's explicit request, this reviewer checked whether the first-five gate rejects a native pending-outer marker. The original reviewed gate did not require it. Matching serialization digests and final `STAGE_COMMITTED` events do not themselves prove absence of unresolved pending state, so a non-None marker could otherwise survive exact roundtrip alongside a final ledger. The worker already includes `aligned_pending_outer` in native payload, state digest, live roundtrip comparison, and JSON report; successful commit clears it to `None`.

The parent subsequently added an aligned-only admission check requiring `aligned_pending_outer` present and `is None` on every bound native report. The check follows report-file/hash/exact-content binding and leaves legacy admission unchanged. This reviewer independently inspected the parent's change and executed the aligned first-five suite plus the legacy skip regression: **18 passed in 2.96s**. The new two counterexamples replace a bound report with a missing field or non-None pending object, rehash the report/manifest/receipt consistently, and still fail specifically on the pending-outer check. Thus the result is not merely generic tamper rejection. Existing valid-None positive coverage also passes. The identified admission gap is resolved for these exact source bytes:

| Supplementary reviewed file | SHA256 |
| --- | --- |
| `search_r1/baselines/fresh200_gate.py` | `746db2f6772903b488ef553d4996a58c75299b251005e6fead280413521dfff7` |
| `tests/baselines/test_baseline_aligned_first5_gate.py` | `c7d547a50688f1d640dc2e5adbca9bd23fc97a4e81b3473639ff2eff6255e8f4` |
| `verl/workers/fsdp_workers.py` (unchanged native producer) | `cecb2714927c258703b47701c9a5583c9e1305791698fef4831f25bda3a91729` |

Supplementary test command:

```text
PYTHONPATH=. ../fresh200-local-test-env/bin/python -m pytest tests/baselines/test_baseline_aligned_first5_gate.py tests/baselines/test_fresh200_gate_skip.py -q
18 passed in 2.96s
```

This supplementary review covers the parent's pending-marker fix; it does not replace a wholly independent review of the earlier gate authored by this reviewer or establish actual GPU/full-state release evidence. No implementation files were changed by this review.

## Scope limits

This is a local source/CPU review, not evidence of target Linux execution, actual eight-GPU collectives, real vLLM responses, baseline hardware retrieval parity, model quality, first-five release, or completion of200 accepted outers. The journal replay verifies the captured authored strings and bound hashes; it cannot independently reconstruct token decoding from hashes alone or substitute for original loader/runtime/source receipts. Actual input bindings, sampler order export, retriever identity/golden proofs, clean published source, original-process native/full-state roundtrips, prescribed first-five independent release, and final full evaluation remain runtime requirements. No old-run approvals or state are admitted by this review.
