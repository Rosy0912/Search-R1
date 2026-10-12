# Hardware3345 independent numeric review

Decision: **PASS_ACTUAL_HARDWARE3345_SYNTHETIC_GPU_NUMERIC_REVIEW_NOT_TRAINING.** Reviewer `/root/alignment_actor_review`, independent of runtime fixture author and separate from protocol/resource/retrieval reviewer. UTC 2026-10-09T01:59:30.742413+00:00. No must-fix in the reviewed numeric evidence. This review reads actual archived terminal records and independently recomputes scalar FP64 oracles locally; it does not rerun GPU work or perform remote operations/production changes.

## Binding and actual result

Actual job3345 token `13e07285c97044fd97974d94f53fea1b`, source `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`, fixed ROOT `/mnt/shared/workspace/shaoshitong/search-r1-rl/bapo-baseline-aligned-em200-20261009`, RUN `ROOT/runs/hardware-aligned-em200-13e07285c97044fd97974d94f53fea1b-3345`. Raw archive file-manifest SHA256 is `a398cda90ae7b7d956e9b26286d68b55200d075dc3ca17cf86dbeeab7f460a00`, matching supplied identity. The numeric tool matches Git927035d byte-for-byte and original hardware-submit receipt SHA pin `adb451471ccf7d75b74ab78a30a01d074e460c69dd1cbcd4ebad7eced86c4028`. All8 rank JSONs, numeric log, script and submit receipt match original inventory bytes/SHA.

Every rank0..7 reports PASS, world8, Torch2.4.0+cu121, exact new environment prefix and source080/core_algos/gradient_norm module paths, with exactly23 expected ordered cases (184 rank-cases total). Numeric log contains exactly one REAL_FSDP_NUMERIC_FIXTURE_PASS for each rank, with no numeric traceback. The bound script rejects optimized Python, validates exact clean source and resource guard, uses CUDA/NCCL world8, and only emits rank PASS after all value/gradient/collective assertions. It performs synthetic backward/clipping, no Adam step or real actor update.

## Independent mathematical checks

| Case family | Actual evidence and independent oracle |
| --- | --- |
| Four orig/offload combinations, native and full-vector norm | 71.55417527999327 = sqrt(256×2²+256×4²), including tied replicated weight counted once. |
| Finite2e20 FP32 gradients whose squares overflow FP32 | 4.525483490288562e21; independent oracle rounds2e20 to actual FP32 then multiplies sqrt512; recorded norm matches1e-12. Script separately compares all clipped gradients. |
| Empty rank0 sharded gradients | 21.908902300206645 = sqrt(256×7/8+256); every combination matches1e-12. |
| Pure NO_SHARD all-empty | Exact0 on both orig modes. |
| One-rank replicated NaN / NO_SHARD Inf | Bound executed script requires caught collective count8 before recording rejection; all rank outputs contain each expected rejection case. |
| Low-var-KL saturated GPU FP64 value/backward |408 points, including extreme±10000 and89/90; max value error9.85796649644044e-7, max gradient error6.296325025800797e-7 on every rank. Script compares complete arrays to bounded FP64 saturated oracle, checks finite gradients and refuses3 nonfinite inputs. |
| Ordinary PPO value/gradient, clip0.2 | Independently recomputed masked token-mean loss1526639225934.9673 and log-ratio30 negative-adv gradient exp30/7=1526639225932.0662. Actual FP32 values both1526639230976.0 are within2e-6 relative tolerance. This is much larger than DualClip or an exp20 clamp branch and proves that finite branch remains ordinary PPO. Bound executed script compares every backward coordinate and rejects NaN/Inf/log-ratio89 overflow. |

All numeric JSON scalar fields are finite. Case identities, orig/offload combinations, point/nonfinite counts and overflow_nonfinite_rejected boolean are exact. Summary error values alone are not substituted for the assertions: the original executed script SHA and full numeric log bind the complete array-gradient comparisons/collective refusal checks. Individual full gradient arrays were not archived; this review cannot reconstruct every CUDA/FSDP coordinate from the JSON summaries alone.

## Scope boundary

This is actual8rank synthetic GPU mathematical acceptance for3345, not CODE_ONLY. It contributes only the numeric component of hardware review. Slurm terminal authority, H100 UUID allocation/leases, resource supervision, retrieval golden/live parity, own cleanup and job exit are independently reviewed by the protocol reviewer; the manifest itself remains TERMINAL_RAW_ARCHIVED_NOT_ACCEPTED. This document does not create a combined hardware receipt, authorize formal submission, claim official3B actor/native checkpoint loading, count any formal outer/write, or accept final200 training. Root must combine independently bound components before formal hardware admission.

## Exact original SHA256

| Archived file | SHA256 |
| --- | --- |
| file-manifest.json | `a398cda90ae7b7d956e9b26286d68b55200d075dc3ca17cf86dbeeab7f460a00` |
| scheduler-head/hardware-submit-receipt.json | `68b5f9937c5387cc38cc2ecfca5b59549245f7479a14fd54c996095ed38e909f` |
| tools/numerics_gpu_fixture.py | `adb451471ccf7d75b74ab78a30a01d074e460c69dd1cbcd4ebad7eced86c4028` |
| run/numeric-fixture.log | `9aaae59914a06c401a351f8fdb42c62d13182f2ea18693b21a1ced8392543b0e` |
| run/numeric-fixture/rank-0.json | `afab397b9a8c89fff5276a596e3207d9d38e2c63a88bc45c19d31ccb206cddff` |
| run/numeric-fixture/rank-1.json | `1cd992f1947ebcfc630e1fc76569cbc979d22a0ef60fc749dd23c85408755a2b` |
| run/numeric-fixture/rank-2.json | `de9b20743da6e5bced9b8ef1e6ffa4489bb6ea7fc653be9ce626cac6e7d872fe` |
| run/numeric-fixture/rank-3.json | `4ce501f435db6a5121e3da7bec64a01976dc8579b237b362cfa7e269681553e1` |
| run/numeric-fixture/rank-4.json | `8c100707bf44aa50601d2db3ded93f8b87d61f09125fb09282364c734572e112` |
| run/numeric-fixture/rank-5.json | `bc27af81bc10bbacab6e3410d0736d1b687300f86c9bc64a437b9a57991cba2a` |
| run/numeric-fixture/rank-6.json | `48fa679dcb05dcc5074121ca814c9fad9b95872436d34025186de02f7270b4aa` |
| run/numeric-fixture/rank-7.json | `a30eaf5ca21852d29a6f269be34859f76d6db6094db7654ac64b48eb433868dd` |
