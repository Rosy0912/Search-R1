# Recursive Bran experiment B on HKUST4

The authoritative algorithm is the user's DOCX, SHA256 ff2555ceffd175ef581bc3f6de8cf1e6018554c87c94b0694ce4377129ed7db1. The new Bran B goal is separate from the canceled OPSD signal-weight run and its deleted heartbeat. No OPSD restoration or automatic sending task is created.

B samples an eligible accepted parent with probability proportional to 1/(generation+1). The common executor, lower-generation action+observation LCP boundary, uniform legal start, fixed descending backoff, total ten-attempt budget and same-reward failure pool follow the document. The single initial root does not consume a continuation attempt. Both reward directions are contrast accepts. All final records are unique; insufficient groups are reported without extra rollout.

The loss preserves intra-root plus intra-question advantage (twice the same standardized reward for one root), PPO clip .2, equal questions, equal final trajectories per question, and per-trajectory effective model token means. Document section 3 excludes a new DualClip term; the inherited configured safety value is diagnostic only for this objective. Native behavior-mu correction and the remaining numerical safety gates remain explicit.

The shared integration was imported from experiment A commit 007662e26dd28ef857ac5f65113d977c7bfcdf8f without modifying its checkout. Shared changes and B selector changes are tracked in this branch. The data preparer is copied byte-for-byte from A and reads only existing HKUST assets. Official model revision is aa8e72537993ba99e69dfaafa59ed015b17504d1, not a Search-R1/GRPO checkpoint.

Target hardware is HKUST4 physical GPUs 4–7 (four 49140MiB cards); UID1001 and host xielab5. GPU7 retains the existing own retrieval process 1749337/tick641133475, <=1024MiB. Any other selected compute process causes refusal. Foreign GPUs 0–3 are untouched. All output, checkpoint, event/snapshot/trajectory files and caches are remote under /weka_3/shaoshitong/recursive-bran-b-4gpu-20261011. Mac stores only source/config/docs and small receipts.

Actor global micro16/local4, old/ref global32/local8, mini128, epochs1, fixed twenty nominal opportunities; scheduler advances once per credited complete outer. Total200, LR1e-6/57 credited outer warmup then constant, BF16 forward and FP32 master/Adam, dev512 each5, protected5/50/100/150/200 plus rolling eachouter and final51713. Ten healthy completed outers and subsequent progress are the launch acceptance; backend continues200.

Incomplete groups use temporary forward transport slots discarded immediately, and zero-policy actor slots after real advantages and weights are formed. Dummy slots do not generate, enter any group, or count as selected trajectories. Weight rescaling cancels the fixed capacity row denominator. Real training data remains separate for metrics/checkpoint evidence.

Status at publication: CPU implementation; no B GPU run has started. GPU6 is occupied by another UID1016 task, so launch must wait for a fresh four-card admission. The launcher never sends signals to that task. CPU preparation can continue.
