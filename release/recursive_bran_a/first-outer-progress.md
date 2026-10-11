# Experiment A: first complete outer, acceptance still pending

Production source: 5045430e67860102c304ef8098e4862ddcdf6e3e.
Run: 52f0bd4ef1044a15bd4ef4d366843f46, HKUST1 eight GPUs.
Config SHA: 622e3c083cd8e2db5f6ef3ba126cd6d4f71babd85c97feb46f839a20fe593d03.

The first native outer completed with 512 questions and 2560 actual selected trajectories. There were 4896 actual continuation attempts, 580 contrast accepts and 1468 uniform failure-pool fills. All 512 groups were complete; 91 reached the accepted target and 421 exhausted the budget. Generation histogram among new complete candidates: 4280/456/153/7 at generations 1/2/3/4. No recorded execution error, identity duplicate or budget violation.

All eight native worker receipts show Adam step20, scheduler1, finite full worker state, and zero refused or partially committed attempts. The same-world manifest is COMPLETE_SAME_WORLD_STATE_SAVED, SHA076325d32b71f41e56eef63e88d9e670745bcdfe4d0f210c11ec03ba85f99d28. The native manifest/file sizes/live next-generation log were checked; this is not an independent deserialization of all optimizer tensors. Round2 has actually begun generation.

The scheduled first warmup round used LR0, advancing to 1e-6/57 afterward; these are real Adam calls but do not imply a weight change in round1. Reported policy loss0.00021611387783195823; pre/post gradient norms3.1569819209031325/0.9999996878537043. KL and entropy coefficients remain zero despite their diagnostic values being recorded. Single-root intra plus inter normalized advantages give actual population variance1.1874952696833432 versus component0.29687381742083579, consistent with retaining both original components.

First outer timing: generation2076.4s, actor update782.8s, checkpoint412.5s, total step3523.4s. Measured direct GPU free temporarily reached70MiB; no OOM occurred, but this is a tight memory margin. Full checkpoint is about60GB including FP32 master/Adam/grad/RNG and HF policy files, kept on HKUST. Monitor remote disk and reuse safe rolling retention; no large data is copied to Mac.

Three pretraining infrastructure repairs were published: actual same-host retriever address, short dedicated Ray socket path, and the required per-outer full-state rolling save. Previous zero-Adam partial generation and exact owned-process stop proofs remain preserved. No existing non-owned process or retriever was stopped.

The requested ten-outer acceptance is NOT complete. Continue the same run through ten complete healthy outers and actual round11 progress, while preserving the 200-round training budget, dev512 every5 and final51713 protocol. No automation message task has been recreated.
