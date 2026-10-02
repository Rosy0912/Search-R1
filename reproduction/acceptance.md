# Reviewed acceptance record

Recorded 2026-10-02. Algorithm/runtime code revision: `b251ba6`; upstream parent: `598e61bd1d36895726d28a8d06b3a15bed19f5d3`.

Two independent reviewers examined paper logic and server/runtime adaptations, then reviewed the corrections. Their final reports identify no remaining publication blocker: [logic review](review/logic-review.md), [runtime review](review/runtime-review.md). This is bounded review evidence, not a guarantee that all bugs are absent.

## Completed checks

- **40 CPU regression tests passed, zero skips**, in the real target training environment using real DataProto and Linux process identities. [Test output](evidence/cpu-tests.txt).
- **30/30 official assets passed content-hash verification**. [Asset record](evidence/asset-verification.json).
- A separate clean environment was rebuilt with the pinned bootstrap. Imports and declared dependencies passed; the documented cuBLAS override is the only accepted dependency exception. The import record's `gpu_integration_tested=false` denotes that check's scope; separate subsequent eight-GPU NCCL/BF16 and real Qwen2.5-3B generation checks passed. [Environment record](evidence/environment-acceptance.json), [dependency freeze](evidence/environment-freeze.txt), [GPU output](evidence/clean-gpu-checks.txt).
- A real eight-GPU mixed NQ/HotpotQA smoke completed two update calls, saved both checkpoints, and evaluated all nine validation samples including the final one-sample batch. The second checkpoint was fresh and changed relative to the base policy: 4,192,688 changed elements in the inspected query-projection tensor, maximum absolute delta `4.131346940994263e-06`. [Parameter check](evidence/checkpoint-update.json).

The first update follows upstream's zero-LR warm-up. On the second update all sampled group rewards were zero, so its policy-gradient term was zero; its KL gradient still changed the policy. The smoke demonstrates pipeline integration, not learned benchmark quality. Its nine-example EM was 1/9 and is not a formal result.

The smoke process imported generation code before the metric-only action-counter correction: its reported maximum of five counted a non-action. Regression tests verify four actual worker calls with the paper budget, versus five for legacy forced-final mode. The fresh formal process uses the corrected counter. Likewise, its final-metrics JSON marker is present in the formal process; the earlier smoke retained the upstream human-readable final print.

## Full experiment still pending

The corrected **GRPO variant** launched on 2026-10-02 at 15:22:41 Asia/Shanghai, from the original base model, with all seven validation datasets. [Stage snapshot](evidence/pipeline-stage.txt), [dataset audit](evidence/dataset-audit.json).

Training uses 169,615 examples and evaluation covers 51,713 examples. Completion requires 500 update calls, a fresh complete step-500 checkpoint, and final EM with the exact seven dataset counts. These results are not yet available. Eight H20 GPUs and CPU FAISS FP32 are disclosed departures from the paper/example hardware and index precision; this is not the default PPO main-table reproduction and no paper-score equivalence is claimed.
