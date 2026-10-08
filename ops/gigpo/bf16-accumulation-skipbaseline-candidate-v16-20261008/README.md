# GiGPO BF16 accumulation + skip-baseline v16 candidate backup

Scientific source remains `2ddbb0a1619ce7cde28b4c5be110ffe5bd11c963`. Original frozen local runtime: `work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v16-bf16accum1-skipbaseline1`. All 44 runtime source, entry-point, launch and existing test files are preserved exactly. Earlier v15 candidate and publication remain intact in their separate namespace.

The latest human direction is “跳过训练前基线，复用之前结果，然后速度启动”. This v16 candidate changes only `calibration_driver_v1.py` and `calibration_run_v1.py` relative to v15 to skip a repeated pre-training baseline and retain hash-bound reuse of prior baseline evidence. The other 42 runtime files are byte-identical to v15, including the BF16 accumulation flag and its checks. This publication does not run any candidate or test.

Status at this publication: **UNEXECUTED FROZEN v16 CANDIDATE; exact v16 two-file delta and separate adapters/binding/derivation received independent clean-context nonauthor `PASS_STATIC`; no numerical acceptance or deployment is asserted.** No numerical acceptance is asserted. Existing job3332 raw probability admission remains BLOCK; this backup does not change its live flag, frozen runtime or job.

Author source manifest and diff are immutable preparation snapshots. The final publication receipt is kept outside this payload manifest, avoiding a circular self-hash. Frozen v16 delta review, adapters, binding, derivation, author SHA inventory and design are recorded in this namespace. The exact final adapter/binding review is included in this namespace. Each payload receives exact byte size/SHA256 and independent Git download verification before the publication receipt reports success. Publication and static PASS are distinct from a stop/submit action or probability acceptance.

This is a persistent backup of the exact frozen v16 snapshot. Any later protocol change requires its own exact bytes, review and publication proof. This publication has performed no stop, submit, test, SSH to a training node, live-process edit or deployment.
