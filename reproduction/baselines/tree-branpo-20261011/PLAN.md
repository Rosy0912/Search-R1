# Tree-GRPO with BranPO-inspired tail branching

User objective: fresh official Qwen2.5-3B-Instruct aa8e725 run, 200 credited outer updates, original common split, final51713 evaluation. Previous3341 remains historical evidence, never a new run's initialization.

## Sampling change and mathematical boundary

Preserve exactly512 questions, two initial roots per question and leaf quotas[3,2], total2560 training leaves per outer. Capture token-preserving states immediately before each actual model action. Spend remaining three leaf continuations near the tail first. A reward-equal continuation can cause the next available continuation to move to an earlier real action state, with an explicit finite budget and no retry-until-correct. All sampled leaves stay in training; no reward-selected deletion or extra hidden candidates. Ground truth only enters the existing scoring path, never the generation prompt.

BranPO motivates tail-first recursive branching and separate prefix/suffix credit; here the explicit user requirement retains original Tree-GRPO inter/intra advantages and reduction. Therefore this run is a distinct sampling ablation, not a full BranPO loss reproduction. Conditional state selection changes the sampling design; do not claim an unbiased estimator of independent root trajectories. Five finite leaves cannot guarantee contrasting rewards, especially when all continuations fail or succeed.

## Execution

1. Independent mathematical review of quota, weight, shared-prefix credit, reward selection bias, zero-variance cases and statistics.
2. Isolated implementation with actual final-action state inheritance, bounded backward branching, and separate immutable source/config identity.
3. Necessary CPU checks delegated to implementation/review agents; independent context-free code review before runtime use. Commit and push final implementation and runtime scripts.
4. Check real retained18 storage, resource lease and queue; use same-node E5/FAISS/data/weights, loopback Ray/HTTP. No cross-cluster assets or Mac tensors. Do not modify historical runs or foreign jobs.
5. Submit one fresh200 run from official model/reference only, without old checkpoint/optimizer/scheduler; no repeated initial baseline or short0–5 stop. Follow at least five complete initial iterations and the next iteration start, with two actual progress measurements and failure/exit checks.
6. Continue same run through200, final51713, HF/tokenizer reload, native finite/nonzero and lineage checks, costs and independent final review, then own resource cleanup.

## Per-outer evidence

Record actual training outer and attempt separately; successful/skipped Adam per rank,20 opportunities/rank, credited outer only complete opportunities with at least one successful update. Scheduler/warmup only credited updates. For each question/root save structure, inherited prefix hash, branch depth, leaf reward and original leaf/token weights remotely.

Small local summaries: raw reward population variance per root and five-leaf group; tree-local centered advantage variance; inter-root advantage variance; combined actor-token advantage variance with explicit masks/weights; same-reward trees versus five-leaf same-reward questions; all-correct/all-wrong/mixed counts and zero-adv coverage. Include population definition and denominators. Normalized variance alone cannot demonstrate improved exploration. Record token/tool/retrieval/update costs including discarded or failed work (none silently omitted).

## Unchanged protocol

train169028/common SHA0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24; dev512 NQ256/Hotpot256; final51713 seven datasets. mini128/epochs1, global actor micro16 and old/ref32,8GPU; LR1e-6/warm57, BF16 forward/FP32master+Adam, entropy0/referenceKL0/low_var, PPOclip.2/gradclip1, DualClip3/samplingpolicyKL.2/negativeadv ratio tail.01/expbound30/finite checks and sampling-mu correction. Four searches, lengths2048/4096/500/500,no forced answer. save5/50/100/150/200 plus original rolling boundaries. Fixed leaf count does not hide changed branching depth or sampling cost.

Current progress: mathematical reviewer and isolated implementation agent are active. No new training submitted. Current remote poll confirms historical3341 exit0 and final51713 F1.5295794609225355/EM.45984568677121807; full historical independent acceptance remains separate.

Latest retained18 read-only capacity probe epoch1791656272.8641663: available5,780,740,136,960B; free6,548,906,778,624B; node queue empty. Official model exists at realpath /mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/instruct300/assets/qwen2.5-3b-instruct-aa8e725. This supersedes historical zero-available warnings for launch capacity planning; it does not attribute who freed space or count this task as having performed cleanup. Launch still needs actual GPU/lease/resource checks and independent code review.
