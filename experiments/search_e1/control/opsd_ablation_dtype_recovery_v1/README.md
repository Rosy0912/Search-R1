# Independent outer-old dtype repair

This namespace overlays only the published `871dc46` trainer path. It leaves all seven original immutable configs, helper sources, scientific settings, locks, process scanner, actual-child waits, sampling, mining, teacher diagnostics, checkpoints, RNG and recovery rules unchanged. C0 invokes the original Controller with `--resume`; the other six invoke its fresh mode. This is recovery from a pre-Adam adapter-load failure, not a claim to have restored GRPO Adam or completed five updates.

The original PEFT 0.13.2 `add_adapter('outer_old')` inherits BF16 base storage. Its bound original adapter is FP32. The launcher promotes only the 504 named outer-old parameter storage tensors **before** the original PEFT setter writes the original FP32 file, then checks exact name/dtype/shape/bytes digest and unchanged current adapter. The original SharedTargets digest/freeze and current-only optimizer guards still execute. P/J keep original FP32 master and FP16 compute contract. No forward, seed, target, prompt, mining, or gradient operation is replaced.

Publish, push, read back, and verify this source manifest before deployment or execution. The root executor supplies a separate immutable repair manifest:

```json
{
  "schema": "opsd-ablation-dtype-repair-manifest-v1",
  "accepted": true,
  "study_id": "original study id",
  "original_configs": [{"path": "each of the seven unchanged configs", "sha256": "actual SHA"}],
  "original_helper_manifest": {"path": "original source-manifest.json", "sha256": "8b22e616d470d79022cc6a0b07dcb0774df21b8e35ebd7dbcaa9b19150236a41"},
  "helper_root": "/absolute/published/repair/helper",
  "source_manifest": {"path": "/absolute/repair/source-manifest.json", "sha256": "actual SHA"},
  "source_commit": "40-character published repair commit",
  "publication": {"path": "/absolute/repair/publication.json", "sha256": "actual SHA"}
}
```

Publication must be accepted on `search-e1`, bind the exact repair manifest SHA, and have equal source commit/remote readback SHA. All original config references are re-read and SHA checked; each remains bound to original commit `871dc46f9ae436e5b4115dd33e51715b858ee57d`, original helper manifest, and the same study. The original helper's complete source inventory is checked. Original helper Python modules are compiled from verified bytes rather than cached `.pyc`.

Controller entry uses the unchanged original config/run marker and CPU-only parent:

```text
CUDA_VISIBLE_DEVICES='' E1_RUN_ID=<original group run> <original env python> -B <repair helper>/controller.py --repair-manifest <repair.json> --repair-manifest-sha256 <SHA> --config <original config> --config-sha256 <original SHA> --run-id <original group run> [--resume only C0]
```

Only an exact original eight-rank torchrun command in `training-running` is rewritten, at its trainer script slot. Original Python, world eight, all CLI values and original extra environment are passed unchanged. The overlay adds `E1_DTYPE_REPAIR_MANIFEST` and `E1_DTYPE_REPAIR_MANIFEST_SHA256` for the launcher; its original trainer main remains responsible for all rank collectives and accepted state semantics. Independent unique controller/outer/rank receipts are written below original group `receipts/dtype-recovery/<repair manifest SHA>/`. They record the actual original config/outer/old identity, adapter dtype/digest and exact path substitution, not raw question or token text.

Focused CPU regressions, with no CUDA, generation, retrieval or production spawn:

```text
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_ablation_dtype_recovery_v1 -p 'test_*.py' -v
```

Neither a CPU fixture pass nor a launcher-start receipt proves actual updates or effect. The executor must observe actual repaired 504-tensor and eight-rank state receipts, retain the failed original trainer logs, and continue the single immutable run.
