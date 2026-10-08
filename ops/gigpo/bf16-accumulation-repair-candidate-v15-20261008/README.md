# GiGPO BF16 accumulation v15 candidate backup

Scientific source: `2ddbb0a1619ce7cde28b4c5be110ffe5bd11c963`. Original local runtime: `work/gigpo-e1-quality-resume-runtime/safety-calibration-fresh200-v15-bf16accum1`; this directory preserves all 44 implementation, entry-point, launch and test source files exactly. This backup does not execute its copied runtime or tests.

Status at this publication: **UNEXECUTED LOCAL CANDIDATE; exact frozen bytes received independent clean-context nonauthor `PASS_STATIC`; no execution binding or approval created; no deployment, job, GPU use or numerical acceptance.** The candidate proposes setting `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` at actor_rollout/ref construction and binding/readback checks. Its effectiveness and root cause are unproven. Existing job3332 and frozen v14 have not been modified by this publication; its probability admission remains BLOCK.

The copied design and source manifest are immutable author snapshots recorded before publication. Their historical `committed=false` / `pushed=false` statements describe that earlier snapshot; the publication/readback receipt records the later Git backup outcome separately. File hashes are unchanged. The four changed files and 40 unchanged files relative to v14 are documented by `BF16-ACCUMULATION-REPAIR-FILE-MANIFEST-v1.json`; the exact patch is `BF16-ACCUMULATION-REPAIR-DIFF-v1.patch`.

`BF16-ACCUMULATION-REPAIR-INDEPENDENT-REVIEW-v1.json` records the independent review at 2026-10-08T04:25:28Z. Its PASS_STATIC covers the exact four changed-file hashes and both tree manifests; it does not approve execution, a future binding or probability acceptance. Any changed bytes require a new review.

`MANIFEST.json` lists SHA256 and byte size for every payload file in this directory, excluding itself. Backup is not execution authorization. Before any separately authorized execution, the exact candidate and a fresh immutable binding require the stated independent review. No live-process flag setting, frozen-runtime edits, tolerance changes or new validation job are authorized by this backup.
