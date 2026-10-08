# Independent phase2 Linux fixture compatibility review

Decision: **CODE_CPU_ONLY — no must-fix in this test/documentation diff.** Reviewer `/root/alignment_actor_review`; UTC 2026-10-08T21:33:33.609032+00:00. Reviewed actual diff against Git66d30a6, exact frozen bytes and original Linux failure receipt. No suite rerun, main invocation, remote operation or large asset scan occurred during this review.

Only the test/documentation changed. Production runtime SHA remains `c7dc6c94896b292cd066481d76fa2eed071d46a3de83df7498d7f4d6595b77f3`; Git comparison shows no runtime diff. The test still writes identical bytes, then explicitly advances mtime by2 seconds with os.utime and asserts mtime changed before expecting fingerprint rejection. It tests the implemented changed-metadata boundary without relying on clock-tick granularity or sleeping. No skip, acceptance-value change or production check relaxation was introduced.

The documentation correctly removes the earlier unconditional claim that a same-byte rewrite must alter ctime/mtime and explains that identical metadata is not cryptographic proof of identical bytes. Metadata-only hash-to-HF continuation can miss content changes that retain its observed fingerprint. The prior independent review's continuation claims remain bounded to implemented metadata checks; they are not final-byte acceptance.

Original target receipt `work/evidence/aligned-em200-final-phase2-target-linux.actual.json` remains unchanged:15PASS/1FAIL3.90s, exit1, failing solely the same-tick fingerprint assumption. Receipt SHA256 `0d35df2d6e7901c4165ceaa7e232cecae7520d8242722b89c135ab7c1c68d644`. Author reports local16PASS2.09s after correction; this review did not rerun that suite and does not claim target revalidation. Root plans append-only linux_v2 test+compat documentation installation, preserving original runtime/helpers/test and receipt. Transport/install requests are outside this diff review and independently reviewed elsewhere.

## Required final byte closure

Root has required a separate real post-CPU-HF execution of the existing hash CLI, fully hashing the official manifest/all12 assets and every file in all HF/native-driver/algorithm groups at5/50/100/150/200. Independently compare the first and post-HF inventories by exact original source/job/RUN, complete directory membership, path, byte count and SHA256 for every file; no witness/chunk/metadata-only shortcut. Bind both reports and CPU evidence and record the second scan's actual separate wall/CPU/RSS cost. If any byte/member differs, final acceptance must be refused or the evidence recollected; metadata continuity cannot substitute for that comparison. This is a future final-acceptance requirement, not an already executed scan or a new runtime framework.

Source080/newROOT remain fixed. NOT_REAL_OFFICIAL3B, NOFAULT_ONLY and pending state200/native semantics, full resources/cleanup, fault-chain retained writes and independent final review remain unchanged. This signature grants no node mutation, GPU/training permission, actual200 acceptance or release.

## Exact SHA256

| File | SHA256 |
| --- | --- |
| aligned_final_phase2.py | `c7dc6c94896b292cd066481d76fa2eed071d46a3de83df7498d7f4d6595b77f3` |
| test_aligned_final_phase2.py | `f14dd2bee0fa8e42464112a25d4bff9c7dc9928558bf70684c1c63a4e2bb30a4` |
| FINAL-EVIDENCE-PHASE2-CANDIDATE.md | `1965ccde8a7e7dc198382a7e70502d776b226684cadea1cf545f8886b6a92c20` |
