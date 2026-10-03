# Search-G1 A800 experiment — preparation only

This branch preserves Search-G1 probe-shaped GRPO. It does not load the Search-R1 policy checkpoint or substitute Search-R1 EM-only rewards. Initial policy/reference is the official Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

Requested schedule: 300 real GRPO updates, complete fixed64 validation before training/at5/every50/final300, inference-weight checkpoints at5/every50/final300. Threshold301 corresponds to updates1–300. Checkpoints do not contain Adam/RNG/sampler state. The initial two readouts must be fitted on actual intervention labels before GRPO. At50/100/150/200/250 the same trainer waits for a newly fitted encoder/readout version before the next rollout.

Adaptations: shared question-disjoint NQ+Hotpot training data; reserve150 unique questions from each source, remove all equivalent questions from training and fixed64. Source calibration uses256 generated tokens/top_p1; training500/top_p.95. Source `max_turns=4` permits four search-capable rounds plus a final answer round (up to five generations), unlike the R1 four-total-action setting. Authored search tags are reward action counts, not proof of actual retrieval calls. Calibration closed-book labels use exact deployed EM instead of the supplied code's substring matching. Boundary fitting weights unique questions rather than repeated trajectories. Scheduled calibration is synchronous every50 rather than paper's asynchronous25, matching the requested checkpoint interval; report this difference in results.

Cross-process evaluator locking serializes calibration against probe inference/reload. It does not unload the old encoder or prove memory fit. Intended allocation is training0–3, FAISS5–6, evaluator/calibration6, contingent on measured peak memory. GPUs4/7 belong to other projects. Never run global Ray shutdown or name-based process cleanup.

Independent review fixes include shared Instruct chat templates/raw question identity, shared punctuation/Unicode EM, model-authored answer provenance, immutable aligned finite probe responses, atomic paired-readout adoption, question-group CV, failure-safe attempt directories and preservation of the original CF service contract. Step5 acceptance blocks update6 until real1–5 metrics, expected evaluator version, exact per-source64 counts, canonical finite checkpoint and changed initial parameters pass.

Deployment: the isolated pipeline is now live waiting for verified R1 update300 save/stop and idle authorized GPUs. Prepared code and CPU checks do not admit GRPO directly. The same pipeline must actually pass GPU integration, calibration/readouts, frozen evaluator release/reload/HTTP and resource gates before training, then verify five real updates and the full300 terminal result. Never invoke train.sh to bypass these gates.

Closed-book intervention prompts also use the official Instruct chat template, with the source direct-answer/no-search content and24-token greedy budget retained. This repairs untemplated Instruct generation without changing label normalization or adding search. Source fingerprints include YAML configurations as well as Python/shell. Actual retrieval counts are logged separately from authored search tags.

Numerical corrections preserve the masked PPO objective and existing clipped low-var KL: exclude mask0 values before ratios, exclude them with where in reduction, and bound KL log-difference to[-20,20] before exp only where the existing output clip is already saturated. Collective fixed-microbatch finite checks reject bad forward/gradients before optimizer writes, including an empty-gradient rank. All24 relevant target CPU checks passed, including real two-rank Gloo and comparison with the original float64 clipped KL output/gradient. Actual GPU/NCCL/FSDP checks remain required. These risk corrections do not establish the cause of R1's observed collapse.

The public model downloader writes logs/assets-status.json. Input admission consumes that actual status and recomputes every official LFS SHA/Git blob before publishing a run manifest; no separate inferred model-ready filename is used.

### Initial calibrated evaluator admission

`probe_admission.py` requires actual intervention labels for all 300 reserved unique questions, both nondegenerate grouped-CV classifiers, complete frozen encoder/readout hashes and identity-aligned finite HTTP responses before admitting GRPO. HTTP smoke proves interface behavior only. Training rechecks all nine model assets, input/source hashes and current endpoint version; every actual reward response must match the expected evaluator before any update. Synchronous refit returns the version confirmed by health, which is adopted before the next update. Four admission regressions passed on A800, including executing the actual reward client against stale/correct versions. Real GPU calibration, resource admission, and five GRPO updates remain unverified. The previous prelaunch source fingerprint must be re-bound transparently before GPU work because these changes were made before launch.

### GPU and terminal checks prepared

`gpu_preflight.py` restricts physical GPUs0–3 before CUDA initialization and exercises four-rank NCCL, BF16, real CUDA JIT, collective nonfinite/empty-rank rejection before Adam writes, and a valid Adam update. It is not yet GPU-executed. `verify_full.py` requires actual updates1–300 with finite training metrics, each exact admitted evaluator version, immutable original checkpoint300 hashes before independent tensor validation, saved steps5/50/100/150/200/250/300, five-update evidence and exactly64 final validation items with bounded per-source EM. Five A800 CPU regressions passed; neither test fixtures nor prepared checks are evidence of G1 training.

### Independent pipeline and cleanup

`run_pipeline.py --source-commit COMMIT --wait-for-r1` verifies live bound R1 handles while waiting; it takes no GPU resources until R1 actual101–300, saved hashes and owned-runtime terminal state are verified and authorized GPUs are idle. Source rebinding records prior prelaunch fingerprints before any G1 GPU work. It then runs real four-card admission, FAISS5–6, genuine dual-head calibration0, frozen probe HTTP/release/resume admission and GRPO1–300 with final proof. Before each scheduled refit, the old probe encoder is explicitly unloaded and its remaining CUDA allocation must be below256MiB; the unchanged-round watcher does not silently reload it. The new calibrated version loads after publication. The initial resource marker is a two-second sampled occupancy check, not proof of every instantaneous peak.

Each component has a durable PID/start-ticks/session/UID ledger. Ray process-title changes are audited, SIGTERM/SIGINT requests are deferred until spawn registration is complete, and completed sessions are not scanned again using stale IDs. `--cleanup-owned` recovers the same-run ledger solely to clean up; it cannot start duplicate jobs. Final status requires cleanup and GPU release checks; unfamiliar occupants are reported without being stopped. Five actual Linux process/signal tests, three evaluator-release control tests and two R1 handoff tests passed. Actual GPU admission/probes/five GRPO updates remain pending until the deployed pipeline executes them.

## Terminal-answer and intervention protocol audit

The shared terminal-action parser rejects nested answer tags in searches, trailing
text after an answer and reuse of an earlier answer after another model action.
Calibration now uses the same parser before submitting a search: malformed actions
terminate without an empty-query retrieval or a later rescued answer. These changes
apply to the new G1 run; historical live R1 metrics were not recomputed.

Evidence deletion preserves the source's operational normalized-answer-change
label: failure to produce a usable counterfactual answer also counts as a flip.
The prefilled answer prompt and the source's budget forced-close convention remain.
NPZ and per-trajectory records include counterfactual validity; the calibration proof
reports invalid counts for both deletion variants and this policy explicitly. These
are proxy labels, not evidence that two valid answers differ or that reasoning
causally depended on evidence. No malformed nested answer is extracted as a usable
counterfactual answer. Closed-book labels remain no-retrieval raw-answer exact match.

Target CPU tests exercise actual DataProto reward managers for both branches and
G1 shaped rewards, plus the actual calibration label and rollout helper bodies.
They do not establish probe accuracy, GPU admission or GRPO acceptance.
