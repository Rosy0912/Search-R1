# OPSD empty loss chunk optimization

The production distillation file now matches the exact external candidate tested in job2990: SHA256 `2131315b6770779f0c7579bf2a6139e742cf75de03b7ee9f44662f8c32a870cf`. Its parent implementation is commit `e43fbfa73088ed70ed3efef79ba561507c0afc33`.

The 13 added lines skip teacher projection and distribution loss only when every row in a chunk has a zero loss mask. A single compact host read determines that schedule. Nonempty chunk shapes, order, contexts, denominator, full-vocabulary FKL, temperature1, component clip.05 and all precision settings remain unchanged. Student forward/backward and its connected zero remain present for a locally empty rank; globally empty minibatches retain the existing collective skip.

Job2990 independently passed four real eight-H100 groups: matched-identity e43/e43 control, e43/A nonempty, rank0-local-empty and globally-empty. Full initial masters including padding, buffers, RNG, modes and teacher counters were restored between paired legs; the same FSDP and Parameter objects remained alive, with a new empty Adam. Two nonempty updates compare every loss, all3,085,938,688 raw gradient elements, original FSDP clipping, clipped gradients, updated parameters and Adam state exactly. Empty cases retain their required collective behavior. See [actual independent review](REAL_JOB2990_PAIRED_REVIEW.md).

Job2987 remains failed: its independent-process second clipping norm differed by one FP32 ULP. Matched-identity paired success does not retrospectively prove that mismatch's cause or imply deterministic clipping across arbitrary independent processes. No tolerance or production clipping operation was changed.

The local production CPU regression completed with286 passed/1 platform skip. Independently checked production diff contains only these13 added lines; actor and other production files are unchanged. The candidate's11 tiny-HF equivalence cases and comparator checks, plus7 paired restoration/observer checks, are separate diagnostic evidence.

This adoption does not claim a measured production speedup, larger safe microbatch, improved validation scores, fresh five-update acceptance or formal500 completion. New source must pass worst-width capacity and actual five-update/HF acceptance before formal use. Validation remains64 questions; E5, GPU FAISS and teacher compute remain on the same H100 node.
