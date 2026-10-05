# Real production cross-process GPU proof

Use a separately approved new job, same production main_ppo/init_workers/Qwen3B/FSDP/offload/mesh/8 GPUs/config. Bind actual allocation/source/new run in a new recovery manifest and independent `PASS_INDEPENDENT_BAPO_RESUME25_GPU_VERIFY_PERMISSION` (code review evidence only). Set trainer.resume_full_state, resume_permission, resume_witness_dir and resume_verify_only=true. The launcher must wait for this actual-job permission before calling main; source fails closed if absent. No submission is done by this directory.

init_workers validates complete historical25 input hashes/config/source/model/data identity and permit before creating GPU workers. Every real actor loads native DTensor model/Adam/scheduler/RNG into a new process. fit restores real driver and sampler after iterator construction, emits 8 rank+driver exact state digests, and returns without initial evaluation, generation, optimizer updates, sampler consumption or training loop. Run validate_gpu_witness.py on actual raw artifacts; an independent reviewer must additionally inspect raw Slurm/resource/import/source/manifest and finite states. This helper never signs a permit.

A different actual training job must have its own manifest and independent `PASS_INDEPENDENT_BAPO_RESUME25_REVIEW` bound to that actual new job/run/source, containing reviewed completed verify-only GPU witness bindings. It repeats actual fresh-process loading before starting outer26 and keeps total301/warmup114/actual300. Do not transplant the verify-only job identity or historical first5 release to the new PID. Old25 lacks rollout generation RNG; this boundary is explicitly reconstructed, not bitwise exact. Subsequent v3 native checkpoints record sharding-manager generation RNG and an atomic full checkpoint completion manifest.

CPU test artifacts validate guards and content tests only. Target Linux full suite and actual production 8-GPU verify-only remain required; nothing in this candidate is an actual GPU PASS claim.

## Production permission proof schema after independent review fixes

A formal continuation permit must contain `gpu_validation` with `job_id`, `run_root=NEWROOT/runs/bapo-instruct300-hardware-JOB`, `manifest` binding (path/bytes/SHA), approved `tools_file_count`, and `raw_artifacts`. The proof job differs from the formal continuation job, but its source and every historical25 input binding must match. `gpu_witnesses` must be exactly `recovery-witness/rank_0.json` through `rank_7.json` plus `driver.json` inside that proof run. All nine JSON records are parsed; no arbitrary JSON or text file is accepted as GPU evidence.

Raw keys are exactly slurm_terminal, source_start, resolved_config, tools_manifest, cpu_gate, cpu_log, cpu_source, launch_approval, guard_summary, guard_helper, resource_watcher, resource_watch, cleanup, train_log. Raw terminal must bind COMPLETED/ExitCode0:0/DerivedExitCode0:0/Restarts0/h100-5. Target CPU must bind the full reviewed dynamic expected count (minimum327), zero failed/skipped, target env/source marker/log hashes and actual reviewed tool manifest/file count. The actual training log must contain zero training/five-gate markers and exactly eight native plus one driver recovery JSON matching the standalone producer files. Guard summary is actually recalculated by the exact independently reviewed SHA-pinned guard_journal.py using SHA-pinned resource_watch.py, then compared as a complete dictionary including raw/code identities. These requirements do not replace the independent reviewer auditing actual Slurm/allocation/foreign-safety and raw runtime/model proof.

Native producer records now include actual NCCL backend, real Qwen model class/config and DTensor count, original and new process identity/startticks/environment, runtime module origin/SHA, model/Adam/scheduler/RNG content digests and true LR25/114. Driver producer records include saved/restored actual sampler/RNG digests and a no-generation/no-update flag only in the real verify-only return branch. Adam step must be a finite zero-dimensional floating scalar with exact value500; truncating500.5 is forbidden.

Production verify-only additionally calls the real worker native v3 save/load once
at step 25. Eight original `.pt` files and reports remain in the fresh owned
`RUN/recovery-native-v3-roundtrip` directory. Reports contain the full-state
before/after digest, generation RNG before/after digest, actual byte-tensor
shape/dtype, and actual vLLM DP8/TP1 device mesh. This is a real production-model
persistence proof without training or generation. The formal continuation branch
never invokes this extra test. It does not generalize the legacy-only v2 loader.

The GPU receipt's `gpu_validation.native_v3_roundtrip` is an exact eight-item list
of `{rank, state: {path, bytes, sha256}, report: {path, bytes, sha256}}`. The strict
source kernel validates actual rank/PID/step25/native topology, unchanged full and
generation RNG digests, report/state hashes, and exact eight
`BAPO_RECOVERY_V3_RANK_JSON` log records. Actual distributed tensor inspection is
still an independent target-GPU acceptance requirement; CPU schema tests use
explicit synthetic bytes and do not establish a GPU result. Keeping the extra
full native checkpoint costs approximately another full-state save/hash/reload,
which must be counted in hardware acceptance cost.
