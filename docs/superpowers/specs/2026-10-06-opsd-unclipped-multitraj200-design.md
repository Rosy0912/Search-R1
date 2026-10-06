# OPSD unclipped multi-response 200 design

The human has explicitly authorized this configuration and continuous execution. Read the task's `work/e1-unclipped-multitraj200/GOAL.md` for the full contract. This design implements that approved scope without adding Search-G1 algorithm behavior.

## Configuration and algorithm

Add an isolated `multitraj200` profile; retain old author200 and E1-300 contracts. Fresh official revision, common physical manifest, 512x5 rollout, global mini256/micro32/log-prob64, eight ranks, LR1e-6 and 57 outer warmup, global gradient norm1, pure OPSD and component_clip null. The human follow-up selected temperature=1/top_p=1/top_k disabled. Ten optimizer calls per rank are observed rather than inferred from a configuration. Full KL uses teacher q on the actual student history with current verified teacher user context. Empty supervised mini batches cannot count as completed optimizer calls; report and fail the required update gate.

## Trajectory diagnostics

Archive all 2560 student trajectories before balancing or updating, with stable source/row/question/group/response identities. Preserve student prompt IDs, response IDs, attention/info/loss masks, actual tool actions and observations including pre-truncation text when available. Save in immutable per-outer records with SHA256/count manifest; no overwritten files or summary-only evidence.

Generate one short fixed-teacher answer per unique training question (512), separately from the distillation forward pass. Use the same official fixed teacher weights and explicitly recorded teacher-only conditioning, a bounded short-answer diagnostic decoding budget, and paired question identities. Keep generated short answers out of q targets and all student/evaluation inputs. Isolate RNG and generation caches; diagnostic timing and tokens are separately charged. Record generation conditions so privileged answers are not mistaken for an unprivileged accuracy comparison. Do not collect teacher trajectories by merely renaming gold answers or greedy argmaxes from teacher-forced student histories.

## Capacity and failures

The prior student forward allocated logits for padded prompts as well as responses. Local micro4 is not yet accepted. Prefer an exact supervised-position/chunked LM-head path that preserves full-vocabulary distributions and gradients if capacity requires it; prove equality and test actual FSDP eight-rank execution. Never silently reduce the requested microbatch, vocabulary or lengths, or move retrieval to another cluster. Resource ownership checks must distinguish a process that exited between NVML and /proc reads from a live foreign process, preserve original observations and fail closed on unresolved ownership.

## Training continuity and acceptance

Keep the established signed first5 gate, with new contract identities and update counts, before the next data fetch. Save full model/Adam/scheduler/RNG/data/OPSD state and HF; independent stored-content and live-process checks precede release of the same process. Initial/dev512 every50, stage64 at5/every50, final51713. Save required 5/50/100/150/200. No 201 and no historical count inclusion. Freeze source and tooling before submission, bind actual job IDs only after scheduler submission, and preserve authentic terminal/resource evidence.

## Implementation plan

1. Add profile and generalized acceptance expectations with regression tests preserving old profiles.
2. Add paired raw trajectory archive and RNG-isolated fixed-teacher diagnostic generation, without altering the OPSD loss target.
3. Independently review normalization, memory, checkpoint and recorder contracts; repair demonstrable defects and test exact gradients/state behavior.
4. Build and freeze an isolated runtime bundle, verify common physical artifacts and actual scheduler/node availability, then run real eight-GPU preflight and fresh training.
5. Independently accept real first5 and release the same process; monitor 6..200 and complete genuine final acceptance.

Engineering compatibility: actor micro remains local4/global32. The frozen teacher forward uses the complete corresponding four-row microbatch in its existing native full-batch branch (`teacher_forward_micro_batch: 4`); no teacher row segmentation is used. The student uses its native full-position/full-vocabulary head, followed by the established predecessor-aligned causal response slice in the loss. Full-vocabulary KL mathematics and global valid-token weighting are unchanged. Actual worst-width teacher4 memory capacity is not established by CPU tests and still requires all eight owned GPUs. Warmup follows the common LambdaLR clock: outer1 uses LR0, post-outer1 is 1e-6/57; outer57 ends at full LR, outer58 first uses full LR. Used LR and post-outer LR are both recorded.

The owned eight-rank GPU probe repeats the same official-model short synthetic native full-head/full-teacher4 path twice, comparing losses, FP32 teacher probabilities and every FSDP gradient shard without an Adam call. This is same-path repeatability, not teacher4 versus teacher2 equivalence and not approval of teacher segmentation. Global gradient norms aggregate all eight ranks; declared BF16 engineering tolerances remain relative L2 ≤0.02, cosine ≥0.9999 and relative loss ≤0.002, with teacher mean probability L1 ≤1e-4. All raw differences and per-shard statistics are retained; zero/nonfinite gradients cannot pass. These tolerances do not claim bitwise native GPU repeatability. The probe then clears gradients and performs two separate worst-width micro4 synthetic Adam calls with resident optimizer states and 15% conservative memory headroom. None count toward the 200 actual RL outers or feed the fresh training state. The earlier source667c/job3299 combined full-head/teacher4 versus response-head/teacher2 comparison and sourceb611/job3302 single-factor teacher4 versus teacher2 comparison both failed their cosine gates before any RL outer or Adam probe update. The latter has no same-path repeated GPU baseline; neither result establishes a unique cause of the historical training collapse. Frozen sources and original failure evidence remain unchanged. Keep the prior pure OPSD safeguards; add no PG, Search-G1 dual clipping or new skip/brake logic.
