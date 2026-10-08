# Newly authorized aligned fresh200 production preparation

This source extends597d227 with perattempt initial and retained training EM. It does not resume old3315, inherit its optimizer/scheduler/RNG, or count its updates toward the new200 goal. The old job received an explicit user cancellation; its original driver is absent and NVML reports no compute processes in the saved task snapshot.

The production profile is `ppo_bapo_baseline_aligned_production`. It enables immutable EM journals, the newsource fifthouter sameprocess release, and stage-before-completion accounting. The original common-data manifest0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24 is validated by the existing full rawsource/parquet/provenance checker and its paths must exactly match the actual loaders. The derived alignment manifest remains separately pinned. Training count is169028, independentstage64, debug512 and final51713; there is no resplit.

Each EM journal has original stable question identities and sibling ordinals, preactor initial responses, every replacement generation, and final retained responses. It records modelauthored spans and separate terminal answer EM, exact correct response/question counts, denominators, uniquequestion count and replacement provenance. All scalar metrics are distinct from rewards and fixed debug evaluation. Journal publication uses exclusive atomic hardlink and fsync; a separately callable verifier checks its SHA, sharedterminal extraction and aggregate replay. A failed attempt may have a real rollout EM journal; it is explicitly not proof of an accepted complete outer.

Production completion must follow20 actual writes plus successful fullstage64 before scheduler/acceptedouter advance. Failed or incomplete work retains factual writes and evidence; it is not a completed outer or an automatic rollback.

## Execution prerequisites still outstanding

- Sourcefixes/tests/independentcrossreview/commit/push; sourcefixed targetLinux and actual8GPU acceptance.
- A new canonical ROOT/source/tools/UUID supervisor binding and durable single-instance submittransaction. Existing source60/3315 approvals are not authorization for newsource.
- An actual baseline DataLoader order export for this common169028 cohort and independently captured baseline localGPU retrieval golden/runtime/assets. The archived existing169103 experiment and the old14query CPU/GPU overlap receipt cannot prove these identities. No historical parity is claimed.
- The newrun claim must happen before bootstrap stage files are written inside RUN; source/config/runtime evidence is written only after successful newrun admission.
- First5 native/fullHF and independentrelease on the original newdriver, then continue6..200; checkpoint5/50/100/150/200 and final51713; terminal fullhash/CPUHF/cost/resource and independentfinal review.

The implementation goal remains active. CPU source checks do not establish actual training, targetGPU acceptance or experiment effectiveness.

## Actual local validation

Combined source/CPU suite: 253 passed, eight dependency warnings, 43.50 seconds (`em200-validation.log`). An earlier combined run exposed two minimal CPU-fixture config-compatibility failures; the configuration getter was corrected, and the final combined suite includes those device-boundary tests. The earlier failed log is retained. Production Hydra composition also verifies mini128, micro16, token_mean, actor KL0.001, no Dual-Clip, no finite-update skip, training EM and stage commit enabled. Compile and shell syntax checks pass. These results do not substitute for target Linux or actual eight-GPU execution.

Independent review additionally found that the aligned release must reject a missing or non-null native `aligned_pending_outer` marker even when separate final ledgers say committed. The gate now requires an explicit null marker on every rank; two negative fixtures and legacy regression pass (18 focused tests). This final change is separately re-reviewed and does not change the update budget.
