import json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
ROOT='/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006'
RUN=ROOT+'/runs/bapo-instruct300-hardware-3263'
SOURCE='b7119791b86951727cea36a1881e8a525c932494'
A=B.parent/'evidence/freeze-scalability-hardware3263-raw-evidence'
I=A/'runs/bapo-instruct300-hardware-3263/actual-fullhash-inventory.json'
i=json.loads(I.read_text()); assert i['job_id']==3263 and i['source_commit']==SOURCE and i['run_root']==RUN
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ev=i['evidence']; assert len(ev)==372 and len({v['path']for v in ev})==372
bound={v['path']:v for v in ev}
bind=lambda p:dict(bound[p])
report=B/'HARDWARE3263-FINAL-INDEPENDENT-REVIEW.md'
report.write_text('''# Hardware3263 final independent acceptance

Decision: **PASS_INDEPENDENT_E1_HARDWARE_REVIEW**, limited to the actual hardware verification allocation. This is not a training permission or an algorithm-effectiveness conclusion. No new must-fix was found in the delivered evidence. The target worker must still execute its frozen complete receipt gate before any subsequent independent production permission.

## Actual identity and evidence closure

Job **3263**, single node **h100-5**, source **b7119791b86951727cea36a1881e8a525c932494**. Root `/dataset/shaoshitong/search-r1-rl/bapo-e1-instruct300-freeze-scalability-20261006`; run `runs/bapo-instruct300-hardware-3263`. The original Slurm record independently establishes COMPLETED, ExitCode=0:0, DerivedExitCode=0:0, Restarts=0, 2026-10-06 03:52:16 to 04:15:31 Shanghai. Its 1638 original bytes have SHA256 `d12eec48826586a1ed3645a2b333c99a025edf48ed66552ce29ea39a883b3c84`. Six head-ledger originals were independently decoded and hash-checked, including job ID/intent, approved original submit and sbatch, and byte-identical terminal record. This was one verification allocation, not a fresh training run.

The downloaded archive SHA256 is `4255049463833b05e9d1e23f2228582df5c0d0040cf427e7b70cd2c5ad5c4c4a`. I independently rehashed all **364** archived bindings in the actual 372-file inventory. The eight remaining large native-v3 files, total **38,278,182,288 bytes**, remain on the worker and are bound by the actual full-stream SHA inventory and each producer's actual save/reload report. They were not downloaded or locally loaded. The receipt binds all 372 original files and the inventory itself, **373 evidence bindings**. No historical receipt/hash was substituted for a new output.

The actual manifest SHA is `c9e755820bad8fc9bf361f51c871d6f1e36cb438328c2078c21ae0927f710841`. Its old complete25 inputs, all eight state/report identities, driver, algorithm, sampler, original resolved configuration, shared inputs, HF files, historical first5 provenance and prefix match the previously accepted complete25 recovery inputs. No partial50 or nonexistent complete26 checkpoint is accepted.

## Source, protocol and mathematics

The actual source-start marker, runtime module hashes, new process/module identities, actual 29-file tools manifest and CODE+actualCPU launch approval match this source and root. The tools manifest SHA is `4eae35dbb37cff24286d2a75720f17537d22f1ed83f15bf1946e90779e95b1cb`; launch approval SHA is `00e13e918fbe706aac632830222f5038e4087697da82cedba40b4e2c4e4e1d75`. The actual target-Linux CPU record (`8a32af877fdc8393c8d13378756d606e2e3b89cb04b416c8025a7c011e300095`) establishes 395 passes, zero failures and zero skips, correct source/import origin and new-root environment. Those CPU results are preconditions, not GPU acceptance.

The actual resolved verification config has TP1/world8, warmup114, total_training_steps301, target300, original169028 train split, stage64, mini128, global micro16/logprob32 and original model/data/tool budgets. Its protocol identity agrees with the original complete25 resolved config using the actual config_identity function. Verification mode explicitly prohibits generation and optimizer training updates.

I checked eight rank-level native actor records: true production bf16/fp32 sequence_mean/full-mini normalization and microbatch invariance, empty cases, finite update behavior and 20-update clocks. All eight native-state fixtures check real FSDP native v3 save/reload profiles and the exact next-update oracle rather than only file presence. Each rank's 22 numeric records cover native/offload gradients, large finite FP64 norms, empty shards, nonfinite rejection and actual CUDA KL value/gradient comparisons (408 points; error limits 2e-5). Actual flash-attention finite backward/oracle and official Instruct vLLM CUDA token output are bound. These are implementation and restoration checks, not a claim that finite gradients ensure stable long-run learning.

## Real production25 recovery and native-v3 round trip

All **nine** actual witnesses independently bind this job/run/source: eight distinct NCCL/FSDP actor processes and one distinct verification-only driver. All eight restore the genuine Qwen architecture/config, 435 model DTensor entries, 37 optimizer-state entries, old state SHA/digests and exact Adam/scheduler contents from complete25. The scheduler clock is25 and learning rate is 1e-6*25/114; next outer is26. The driver restores actual_outer_updates25, epoch0, sampler cursor12800, BAPO state, sampler/RNG digests, and directly binds all eight actor witnesses. Runtime module hashes prove the new source origin.

Each actual rank then saves/reloads a new native-v3 checkpoint with exact state digest and generation-RNG digest, TP mesh [8,1], genuine CUDA/RNG records, actual size/hash and rank identity. All eight `.pt` bindings agree with their actual reports and full-stream inventory. Raw train markers agree with the nine witnesses and eight v3 reports. There is no BAPO_TRAIN_METRICS_JSON production-training event in this verification run.

The explicit continuity condition remains **LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT**. Saving RNG correctly in new v3 checkpoints does not retroactively manufacture the missing legacy-generation RNG state.

## Freeze scalability, resource isolation and guard replay

I ran the exact frozen new validator function (verifier SHA `2016ff3dcd38c31ea69226d29754123e8af686a02cbb2969f287a2466cbf8c5d`) against the actual raw fixture files with read-only artifact-I/O relocation; source/job/run strings remain the actual remote strings. This is independent raw recomputation, not a claimed worker execution. All 43 fixture raw files were archived. The strict 45-binding component proof, raw topology, certificates, PID/starttick/UID/job/run/source/UUID leases, parent epochs and exact STOP/stable/CONT journal agree.

There are **310 own processes**: one phase, 301 CPU descendants and eight CUDA/NCCL ranks. Actual freeze time **1.1056974288076162 seconds** is within the unchanged five-second budget. The stable stopped epochs have no host-counter progression; all eight counters124 advance to at least127 after resume, with finite genuine NCCL allreduce records and required local transport. The foreign sentinel exits naturally with exact status0; no foreign signal was sent. The fixture's forced-private journal is an intentional own-process test, not an invented production-unsafe GPU observation. Host SIGSTOP and counter proof do not establish that already queued asynchronous CUDA kernels instantly quiesce.

I independently reran the exact bound guard helper SHA `2a714082c22675b975eff07dd060b13ca8eb9da9394d9b6f8bea605ffff46960` with watcher SHA `e56781fefcd42b3d97f06af824312ac69fd318eb480093a66c4194d1f0fcaf74`. It recomputes all raw resource decisions using the bound assessor, and its full dictionary equals both actual guard-summary and inventory summary: **69 safe samples, zero recovery samples, zero pauses, zero aborts, no unfinished recovery**. Unknown observations were not converted to zero. Maximum observed/upper-bound external SM is **7.0%** and memory **1.3756912174009%**, within the unchanged thresholds. External LTX PIDs2210497 through2210504 and their epochs remain stable and are absent from owned cleanup signals. Own cleanup has `remaining=[]`. This actual safe run tests observation and cleanup; it does not claim an actual naturally unsafe production recovery occurred.

## Retrieval and cost

Actual local CUDA E5 and same-node sharded GPU FAISS use the same verified assets/index (768 dimensions, 21,015,324 entries). CPU retrieval here is only a hardware parity oracle, not the production deployment. Across the same14 queries, top3 rank order matches14/14, top50 set overlap is1.0 for all14 while exact top50 order matches5/14 (ties/order differences); maximum matched score difference is0.00042307376861572266. No bitwise score/order equivalence is claimed. Production remains same-node GPU retrieval and loopback, without A800/cross-node retrieval.

Actual Slurm reserved wall cost is **1395 seconds**, with eight allocated GPUs: **3.1 reserved GPU-hours**. The scale fixture itself reports13.432698369026184 seconds; this is included, not added again. Post-terminal full-stream hash sealing costs32.36367356590927 seconds wall,32.39219887 CPU seconds and33840KiB peak RSS. Guard recovery cost is0 because no recovery occurred. Cost boundaries are stated explicitly; they are not an algorithm speed benchmark.

## Scope and release boundary

All required hardware/raw/state/mathematical checks pass. This acceptance permits only the existing frozen actual hardware receipt gate to establish its prerequisite; it does not sign a new production-run permission, submit a job, alter a tool/source, change any budget or signal an external process. The subsequent real allocated production run requires its own identity-bound independent recovery permission. No repeated GPU/CPU/HF benchmark was performed during this review.

Reproducible local review: `check-hardware3263-final.py` and `work/evidence/freeze-scalability-hardware3263-independent-final-check.jsonlog`. Its two statuses explicitly distinguish independent raw recomputation from the required target worker gate.
''')
rep={'path':ROOT+'/logs/'+report.name,'sha256':h(report),'bytes':report.stat().st_size}
raw_paths={'slurm_terminal':RUN+'/slurm-terminal.txt','source_start':RUN+'/source-commit.txt','resolved_config':RUN+'/resolved-config.yaml','tools_manifest':RUN+'/tools-manifest.json','cpu_gate':RUN+'/cpu-gate.json','cpu_log':ROOT+'/logs/cpu-gate.log','cpu_source':RUN+'/cpu-gate-source.txt','launch_approval':RUN+'/launch-approval.json','guard_summary':RUN+'/guard-summary.json','guard_helper':ROOT+'/tools/guard_journal.py','resource_watcher':ROOT+'/tools/resource_watch.py','resource_watch':RUN+'/resource-watch.jsonl','cleanup':RUN+'/own-process-cleanup.json','train_log':RUN+'/train.log'}
r={'status':'PASS_INDEPENDENT_E1_HARDWARE_REVIEW','source_commit':SOURCE,'root':ROOT,'job_id':3263,'run_root':RUN,'common_manifest_sha256':'0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24',**{k:True for k in ('native_actor_independent_review_passed','native_state_independent_review_passed','same_node_gpu_retrieval_verified','real_production_restore_independent_review_passed','production_native_v3_roundtrip_independent_review_passed')},'evidence':ev+[{'path':RUN+'/actual-fullhash-inventory.json','sha256':h(I),'bytes':I.stat().st_size}],'independent_review':rep,'gpu_witnesses':[bind(RUN+f'/recovery-witness/rank_{n}.json')for n in range(8)]+[bind(RUN+'/recovery-witness/driver.json')],'gpu_validation':{'job_id':3263,'run_root':RUN,'tools_file_count':29,'manifest':bind(RUN+'/recovery-manifest.json'),'raw_artifacts':{k:bind(p)for k,p in raw_paths.items()},'native_v3_roundtrip':[{'rank':n,'state':bind(RUN+f'/recovery-native-v3-roundtrip/rank_{n}.pt'),'report':bind(RUN+f'/recovery-native-v3-roundtrip/rank_{n}.json')}for n in range(8)]},'actual_gpu_accepted':True,'production_training_permission_signed':False,'cost':{'slurm_reserved_wall_seconds':1395,'slurm_reserved_gpu_hours':3.1,'fullhash_seal_wall_seconds':i['wall_seconds'],'fullhash_seal_cpu_seconds':i['process_cpu_seconds'],'guard_recovery_wall_seconds':0,'scale_fixture_wall_seconds':13.432698369026184,'scale_freeze_seconds':1.1056974288076162},'review_limits':{'large_v3_files_not_downloaded':True,'target_hardware_receipt_gate_required':True,'historical_generation_rng_exact_continuity':False,'forced_scale_fixture_not_actual_unsafe_observation':True,'host_stop_does_not_prove_instant_cuda_kernel_quiescence':True},'freeze_scalability_independent_review_passed':True,'freeze_scalability_proof':{'validator_sha256':'2016ff3dcd38c31ea69226d29754123e8af686a02cbb2969f287a2466cbf8c5d','own_certificates':310,'cuda_nccl_ranks':8,'freeze_seconds':1.1056974288076162,'foreign_signals':0,'actual_unsafe_gpu_observation':False,'raw_file_count':43},'sampling_continuity':'LEGACY25_GENERATION_RNG_RECONSTRUCTED_NOT_BITWISE_EXACT','local_independent_check_sha256':h(B/'check-hardware3263-final.py'),'local_raw_recomputation_record_sha256':h(B.parent/'evidence/freeze-scalability-hardware3263-independent-final-check.jsonlog')}
p=B/'hardware3263-final-receipt.json';p.write_text(json.dumps(r,indent=2)+'\n')
assert len(r['evidence'])==373 and len(r['gpu_validation']['raw_artifacts'])==14
print(json.dumps({'report':str(report),'report_sha256':h(report),'report_bytes':report.stat().st_size,'receipt':str(p),'receipt_sha256':h(p),'receipt_bytes':p.stat().st_size,'evidence':len(r['evidence'])}))
