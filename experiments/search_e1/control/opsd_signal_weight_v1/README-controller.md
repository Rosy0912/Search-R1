# Seven fresh reverse-KL online ablations

This independent helper namespace implements only C0/C1/M/R/W/P/J, each with ten online TRAIN outers. A group starts from the bound common zero-B 504-FP32-tensor LoRA template and source100, with fresh Adam/RNG; no extra GRPO or historical Adam is restored. The complete group remains one run. A first-five-update check does not end or restart it.

`prepare.template(base, variant, study_id, study_base)` returns `opsd-ablation-config-v1`. C0 has lr5e-6; all others lr1e-6. P/J use `precision.mode=fp32_master_fp16`; the other groups use `bf16`. Component upper clip10, trajectory mean, global32/micro1, rank16/alpha32/dropout0, frozen source100 teacher, original optimizer and generation budget are explicit. The trainer owns the new reverse-KL objective and consumes the mandatory `contract` with direction, target version, variant, precision and mixture alpha. Original online manifest/checkpoint/update schema names are retained with that additional scientific identity; they are not old forward-KL runs.

Required config references are `source_model_manifest`, `runtime_manifest` (exact unchanged305), `helper_source_manifest`, `train_identity`, `validation_identity`, `common_cohort_manifest`, and `common_initial_adapter_manifest`. The common initial JSON has `payload_files` (all adapter files), `adapter_model` (the safetensors fileproof), and `adapter_tensor_sha256`; config repeats the actual tensor digest as `common_initial_adapter_tensor_sha256`. CPU preparation binds bytes; the trainer and CPU merge child inspect actual finite FP32 tensors and zero B. See README-mining.md for the one common TRAIN shuffle and versioned selection/weights.

The four execution permission refs are under `gates`: `publication`, `storage`, `resource_transition`, `retriever_admission`. Publication/storage/transition must additionally contain a new-goal scope:

```json
{"scope":{"study_id":"NEW_STUDY","variant":"C0","run_id":"FIXED_32HEX","output_root":"/new/group-C0/runs"}}
```

A study-wide receipt may instead name the exact seven variants in order and `run_ids`/`output_roots` dictionaries for all seven. The old 200-outer local-storage permission does not satisfy this scope. Missing storage/publication/transition remains HOLD, with no CUDA spawn. Retriever scientific evidence may bind the existing admitted18004 GPU7 service; no public routing change is made.

`gates.memory_binding_path` is the planned group-specific `receipts/memory-binding.json` pathname. This avoids a config/observer SHA cycle. Root starts the independent observer against the immutable config; its binding, actual PID/UID/tick/source and independent run marker, config SHA, latest direct memory.free sample (at most30s old), and eight UUIDs are verified on each admission. The observer is outside the controller's descendants and output tree.

`gates.scoped_cpu_processes_path` is the planned `receipts/scoped-cpu-processes.json`. Its `opsd-ablation-scoped-cpu-processes-v1` body binds the exact live study driver and independent memory observer to this group run/config SHA. Each role records actual PID/UID/tick/PPID/PGID/argv/independent marker, canonical JSON `argv_sha256`, source fileproof, helper manifest/publication and actual input-config ref. The driver must execute the published evaluator `study_runner.py --study-manifest`; the observer must execute this training helper's `memory_observer.py --config`. Fresh source/publication/input/actual process checks and direct GPU inventory reject a CPU role that holds any CUDA context. Normal GPU admission and resource release require exactly both roles.

`require_launch_without_memory(config, run_id, owner, known_children=[])` exists only to start the independent observer: it requires the same publication/storage/transition/oracle and empty-eight-GPU20GiB startup gate, admits a driver-only role registry, and reports `observer-startup`. It is not a skip-memory option for sampling/training. The study runner then writes both roles and waits for the real fresh memory binding before starting the training controller. `require_external_study_release` additionally requires the exact validated driver owner and uses known phase controllers/children as ancestry anchors; normal training release still anchors its actual controller. Exact CPU roles do not override the scanner's known/run/argv/owned-child-descendant blockers or opaque-process rejection.

The controller holds its group `online-experiment.lock`, the four original HKUST1 experiment leases, the old online200 lease, and `study_base/ablation-gpu-experiment.lock` throughout the run, including uncertain admission/evidence/release. After acquiring locks it repeats admission. Each startup requires all eight configured UUIDs, no old owned work, free>=20GiB, and only the exact admitted oracle on GPU7. Running worker cards require free>=1GiB; unstarted/unowned cards retain20GiB. An unexpected CUDA context in the CPU controller is rejected. Full same-UID ancestry/run/argv/opaque-process release uses the separately published unchanged V2 scanner. No signal, kill, tmux change or automatic foreign-process recovery is implemented.

Each outer samples all512 frozen TRAIN questions ×5 with eight physical workers. Native generation/executor/scoring is loaded by SHA-bound source-only compilation of the exact305 tree, without timestamp-valid pyc. The wrapper preserves native generated objects, request args/kwargs/timeouts, response objects and exceptions. Private records retain raw generation IDs, full HTTP queries/documents/scores/content, authored-span protocol diagnostics and strict native rewards. Diagnostics never redefine reward.

After mining and before training, `teacher_diag.py` runs a separate privileged source100 HF greedy diagnostic child on physical GPU0. It uses the full real successful B with the original `core.prompt(question, reference=B)` and native SearchAgent budget (500/8192, five turns, four searches, top3/480 observations and final forcing). SHA256(seed42, outer, qid) ranks each mutually exclusive self-pair/CW/CC pool, then a stable round-robin covers the available pools, at most8 pairs per outer and at most560 across the seven ten-outer groups. Empty eligible pairs are explicit NA with no invented B or GPU child; absent strata remain explicit. The saved selection, pairs, source bytes, actual child rc0 and full resource release are bound before a diagnostic is accepted or reused. Partial diagnostic output is preserved as HOLD for independent recovery review.

This unified diagnostic is source100 BF16/SDPA for **all** seven groups, including P/J, and records `diagnostic_precision_common_bf16=true`, `diagnostic_teacher_autoregressive=true`, `original_training_objective=false`. P/J training q retains its FP16 scientific contract. The independent seed42 process consumes no student sampling/trainer RNG. Complete teacher raw IDs, actual contexts/observations, HTTP queries/JSON/documents and strict native scores are private debug artifacts; teacher self-generated history is not the original D-history training q. These cases never enter training pairs, token-loss records or the heldout512 denominator.

C0/C1/M/R/W first-outer sampling may use the exact source100 weights, with the bound common zero-B identity. Later sampling uses an isolated CPU child to perform the original BF16 PEFT merge of the latest raw adapter. P/J use the CPU FP32 base and FP32 adapter merge, followed by one explicit FP16 cast; the actual vLLM constructor dtype is FP16. The parent never imports Torch/model or creates a CUDA context. CPU-child actual rc0, request/readback, parameter CPU/dtype and pre/post CUDAfalse proofs plus full release bind every accepted cache. Partial/unproven caches are preserved as HOLD, not reused or overwritten.

Actual child return codes are durably saved before release/admission validation. A transient running admission observation does not turn a real rc0 into worker failure. Unknown process identity, nonzero child exit, persistent admission failure or evidence I/O failure cannot advance a CUDA phase; the lease remains held while uncertainty is resolved. A fast child exit before its `/proc` identity is readable is recorded as unknown with its actual Popen return code, not an infinite wait for the dead PID to return and not inferred rc0.

Resume requires the same immutable config/run. It revalidates the original common cohort, strict rollout scores, new miner/weights, original outer-start `outer_old_adapter`, sampling completion and real exit proofs. The latest committed checkpoint is selected only from exact `update-########-outer-#####` directories, never pending directories. Same-outer overflow recovery consumes Root's `retry_state.select_retry_capsule` bound to that accepted checkpoint and passes its explicit path/SHA to the trainer; a subsequent outer does not load an old retry capsule or refresh p_old mid-outer. A completed outer lacking actual terminal evidence requires independent recovery review.

After ten accepted outers, a final fresh full resource release produces `TRAIN_COMPLETE`. This is training/process completion; heldout512 accuracy, paired flips, precision diagnostics and final actual-state numerical audit remain separate evaluator work. Full vocabulary logs use the shared `logging.full_vocab_study_root=study_base/runs/full-vocabulary` and trainer's fixed24GiB study cap. Every scalar valid-token loss remains uncapped; selected full-vocabulary logp rows are a small diagnostic sample, not a population estimate.

Retention defaults to `keep_all`, with no checkpoint/raw-debug deletion and no unapproved merged-cache deletion. Rough capacity is about1.4TB for seven groups' approximately100 two-GB checkpoints each, plus merge caches/debug. The fresh100GiB disk floor admits initial progress, not a claim that the full seven-group footprint fits; storage/caching policy must be resolved explicitly by Root.

CPU configuration preparation (always not launch-ready until the real refs are bound):

```sh
python -B prepare.py --base /new/group-C0 --variant C0 --study-id NEW_STUDY --study-base /new/study --output /new/group-C0/config-candidate.json
```

After publication, resource/storage acceptance and Root's actual source readback, the fixed run entry is:

```sh
E1_RUN_ID=FIXED_32HEX CUDA_VISIBLE_DEVICES='' python -B controller.py --config /new/group-C0/config.json --config-sha256 ACTUAL_SHA --run-id FIXED_32HEX
```

`--resume` is only the same run/config. CPU fixture entry from publisher root:

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_ablation_v1 -p 'test_controller_v1.py' -v
```

These fixtures verify scope/matrix, real rc vs transient observations, durable terminal evidence, evidence HOLD, fast-child registration, per-card memory floors, parent-CUDA rejection and HTTP null-effect behavior. They do not claim live GPU, precision, oracle, effect or seven-group acceptance.
