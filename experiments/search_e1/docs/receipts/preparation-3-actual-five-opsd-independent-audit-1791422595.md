# Independent Search-E1 OPSD runtime and five-update audit

Actual remote runtime read: 2026-10-08T09:23:14.923183+08:00 to 2026-10-08T09:23:36.980056+08:00. CPU saved-state check: 2026-10-08T09:25:49.997989+08:00 to 2026-10-08T09:26:00.763396+08:00. The local connection resumed after a clock jump; this is the actual Oct8 audit, not a claim that an Oct7 21:45 sample was collected.

Run 914bfff786c84ede8d82414dff65f60e on HKUST1 /data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007 has passed the requested source100 to actual OPSD updates1-5 gate. The external receipt records acceptance at 2026-10-08T06:05:51.818327+08:00. This uses source archive GRPO100 weights and a fresh OPSD Adam optimizer; it does not claim recovery or continuation of the GRPO Adam state.

- Full fresh preparation is complete: 169103 questions and 845515 trajectories, with every complete JSON/source-position/five-sibling order valid. The final per-worker90/85-row tails are expected final partial64-question batches. All original generation workers and their GPU process entries were released normally.
- Native launcher3588780 and ranks0-7 at PIDs3588873-3588880 have UID1005, expected source/pairs/output argv and run identity. Eight native workers have WORLD_SIZE8; launcher rank fields are unset by design.
- Adapter updates1 and5 each contain504 finite tensors. All504 differ between updates1 and5. The baseline best_adapter remains finite with all252 LoRA-B tensors zero; all504 update1 tensors differ from the baseline.
- All16 saved update1/update5 rank bundles were freshly hashed and read on CPU. Actual Adam steps are1 and5 respectively. Every rank has504 optimizer states; step, exp_avg, exp_avg_sq and parameter groups are exactly equal to rank0 for each update, with finite tensors and complete Torch/CUDA RNG fields. No CUDA context was initialized by this audit.
- External direct actual-free sample is 2.983s after the latest update5 state file; all eight actual memory.free values were at least 15077MiB. Historical minimum3099MiB remains intact. Current runtime actual free is at least 13545MiB. GPU7 retriever peak654MiB and its identity were unchanged. No new readable dmesg Xid/NVRM/CUDA match was found.
- Host MemAvailable is 886495832KiB; exact owned-run process PSS totals 91163508KiB. No unrelated job memory contents were inspected.

Native config has usable_pairs100974, skipped_pairs0, epochs1, global trajectory batch32 and accumulation4. Log sequence1-3156 is exact and non-repeating; ceil(100974/32)=3156 includes the final14-pair update. Latest native live allgather reports all eight actual Adam step3156, finite and changed weights. The saved tensors independently re-read above are u1/u5; this does not claim a latest3156 optimizer checkpoint exists.

The single native training epoch is finished. The last epoch log is pairs100974; the control-flow next step is post-epoch native512 validation, and the eight native workers remain active with no history.json or printed baseline/validation EM number yet. Current phase still names TRAIN_AND_BASELINE_VALIDATION; it is not a per-validation-progress counter. No EM number is inferred from earlier GRPO evaluation.

All305 actual frozen source hashes, controller/helper/config hashes and full train/validation/test identities matched. Source100 large-weight exact inode/size/mtime_ns/ctime_ns stayed equal to the Oct7 20:45 baseline. Small model/tokenizer files were freshly hashed. Three large model files were not rehashed: accepted transfer/startup content hashes remain bound through unchanged metadata, rather than being described as a fresh13GB hash.

First runtime JSON retains one false checker finding: its argv filter included the torchrun launcher and required WORLD_SIZE8 on all nine entries. Separate interpretation verifies eight ranks and one launcher without altering raw evidence or any remote program. Every other original check and every actual saved-state check passed.

Five-update numerical/memory acceptance is complete. Native512 result, merge, seven-benchmark final evaluation and effect attribution are pending. Old preparation2 CUDA/Xid31 evidence remains intact; this audit does not establish its root cause was repaired.

Artifacts:

- Raw runtime JSON: preparation-3-hourly-runtime-readonly-audit-1791422595.json, SHA256 1d62cd9c5e1aaf0b4964a087a8d848d59830e70dc9a39718490f5cbc3cb4755e.
- CPU actual-state JSON: preparation-3-actual-five-opsd-cpu-state-readonly-1791422595.json, SHA256 9837cade78596daf587e26b14a7b63357102b5299a52732ace2aaf31e2cd683c.
- Native-rank interpretation: preparation-3-hourly-runtime-readonly-audit-1791422595-native-rank-scope-interpretation.json, SHA256 cf25dbcf844d9650752d9aaf0126e5d229afec09cd32c0b689ed1bdca91c5ab0.
- Official external5 SHA256 96509ab16a51dc038501153b7a41d7fb1a60d2ca62761fa793a1db635e2bb2e3; native5 SHA256 e311b200bec03918e3dcc4eda3f58233b7042dbc3ae9daadf5aeacb1e52cd6aa.

Remote inspection was read-only. No job, process, scientific parameter, runtime, route, environment, affinity or publisher document changed. No GPU or retrieval workload was added.
