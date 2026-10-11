# Sparse recording and generation overhead change

The latest human schedule is evaluation, state saving, and detailed tree recording at outer 5, 50, 100, 150, and 200 only. It replaces the earlier every-five evaluation and every-outer rolling storage for experiment A.

Published code now skips an unused per-action HF actor scoring pass when native tree emissions are enabled and the sampler/actor consistency diagnostic is disabled. The complete assembled trajectory is still scored by HF in the trainer, and actual emitted behavior probabilities remain the PPO denominator. Recursive selection, rewards, inherited prefix state, generation budget, optimizer, precision, and policy safety mathematics are unchanged.

Detailed recursive events, full state snapshots, trajectory logs, and TreeRankAudit microbatch evidence are gated by the sparse schedule. Internal pre-action snapshots and raw probability ledgers required by the algorithm are retained in memory. Essential errors, optimizer counters, safety decisions, and scalar training logs remain available each outer. This does not promise exact recovery of an unsaved outer after a failure.

The first cold continuation refused before any new update because its full configuration comparison did not account for native `_create_dataloader` filling the critic scheduler total to 201. The second entry recognizes only the existing native scheduler injection in its initialized comparison; the immutable user configuration remains the same. Original logs, complete fourth checkpoint, and failed initialization output are preserved. No failed initialization is counted as training progress.

The actual fourth checkpoint is `4d781f6920ce8ea8d56bff7c488ea960aa5b60e7b99a7635d19961bbfa06816f`, with all eight ranks at Adam 80. The continuation must demonstrate actual restored state and new updates before it is accepted. No speedup number is claimed before measuring the new round. Generation-RPC wall time is recorded separately from total recursive rollout wall time; it includes weight synchronization and backend work, not pure CUDA decoding.
