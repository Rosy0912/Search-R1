# Eight-card OPSD from archive GRPO checkpoint100

Latest human instruction on2026-10-07 supersedes the older first-GRPO200 requirement for the replacement eight-card experiment. Stop only the owned degraded aligned8 run after preserving complete eight-rank weights, Adam, scheduler and RNG. Keep archive4 training and every older stopped run unchanged. The exact human message timestamp is unavailable; remote authorization receipt records the live-turn receipt time and original text.

Use archive4 global_step_100 model weights, verify every transferred file SHA256, and start OPSD directly. No additional GRPO and no GRPO optimizer restoration. The display100→105 means GRPO source checkpoint100 plus five real OPSD updates; source and OPSD counters remain separate.

OPSD already computes valid-token mean within each trajectory then global trajectory mean. Its original pair_loss and encoded_pair functions are unchanged in the new runtime. Preserve lr1e-5, rank16, alpha32, clip10, chunk32, max_context8192, seed42, epochs1 and global trajectory batch32 (eight ranks × accumulation4). GRPO entropy, micro64, KL coefficient and scheduler do not apply to OPSD. Do not add a GRPO entropy term to imply preservation.

Full training data169103 questions, five fresh rollouts each, original mining,512 heldout validation and seven-benchmark final evaluation. Full generation and mining precede training; initialization, synthetic updates, shortened data or baseline evaluation are not acceptance. No extra GRPO is included after this OPSD stage without new human authorization.

New HKUST1 root: /data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007. Source: HKUST4 /weka_3/shaoshitong/search-e1-config-study-20261004/runs/archive/round-1-grpo/actor/global_step_100. New tmux e1-opsd-from-archive100-20261007; logs/pipeline.log, logs/rollouts.log, logs/opsd.log; receipts/pipeline-stage.json, receipts/archive100-transfer-accepted.json, receipts/opsd-runtime.sha256.json. Separate memory observer e1-opsd100-memory-20261007.

Instrumentation observes each real Adam update, finite trainable parameters and gradients, changed LoRA_B weights, actual per-rank Adam counters and GPU allocator peaks. Save actual adapter and optimizer/RNG state at OPSD update1 and5 without stopping or resetting the one continuous run. Acceptance requires real updates1–5, same source model, finite changed weights, matching Adam counters on all8 ranks and healthy observed GPU usage. Do not label GRPO metrics or source checkpoint as new OPSD updates.

Current setup status: source checkpoint transferred and12 files hash-verified. Old state capture queued behind active update; new pipeline waits for full-complete.json before the narrow stop. Five OPSD updates have not yet been executed; no acceptance claimed. Original frozen runtime retained; instrumentation modifies only independent runtime/opsd copy. Hourly automation updated to new human scope; never reactivate replaced aligned8 GRPO.

## 2026-10-07 12:32 CST: old run preserved and replacement generating real trajectories

All8 old ranks saved scheduler80/Adam1600 and full FSDP weights/Adam/RNG at preserved/replacement-state-b8b93546fb314915b3e72f355f22f291/full-rank-0..7.pt,4628970258 bytes each. Native driver reward rows had reached79 before replacement; the preserved optimizer counter80 is distinct from an emitted80 reward row. Narrow SIGTERM stop left only3 verified own NCCL workers143287/143289/143290; after rechecking original UID/private PGID/E1_RUN_ID and saved full state, those residual PIDs were killed, releasing all8 cards. No foreign GPU/process or archive job touched.

New controller2196101 advanced toOPSD100_FULL_FRESH_TRAJECTORIES at12:32:17 CST, full169103 questions×5,8 disjoint worker shards. Actual8 GPU worker processes loaded the approved source model. Generation is a prerequisite; no OPSD Adam updates or five-step acceptance yet. The new continuous run does not include extra GRPO, test-then-reset or source optimizer restoration.

## 2026-10-07 14:24 CST: live generation and stronger external verification

Controller2196101 and all8 generation workers remain live. The eight shard files contain53120/845515 actual trajectories; OPSD has not started and acceptance is pending. The305-file OPSD runtime still matches its frozen hashes. No synthetic updates or shortened training data are used.

The external CPU acceptance observer was replaced only after verifying the original owned observer PID2204061 and its exact script path. The original script and log remain preserved. New observer PID2231928 runs source/watch_acceptance_v2.py in tmux e1-opsd100-acceptance-v2-20261007, writing logs/acceptance-v2.log. Its transition identity is recorded in receipts/acceptance-observer-v2-transition.json. Training/controller/GPU workers and frozen runtime were not restarted or edited.

The observer now retries incomplete JSON writes, checks every adapter tensor at updates1 and5, and compares all eight saved Adam states and parameter groups exactly on CPU, including step/exp_avg/exp_avg_sq. Memory evidence must cover all8 cards and be fresh after the actual fifth-update record. Any failed verification writes FAILED_ACCEPTANCE rather than acceptance. These checks do not change the scientific loss or optimizer settings.

Read-only speed diagnosis found substantial shared retrieval queueing:16 observed requests averaged11.15s service plus74.34s wait. The full64.6GB FP32 index is re-transferred for each request, including small tail requests. This is a speed bottleneck, not evidence of OOM. Existing generation and the shared K4 oracle remain unchanged. A separate candidate may share each original index tile across independent requests while retaining original per-request encoding boundaries, GEMM shapes and top-k operations; no candidate is deployed or accepted yet. K1 index/encoder/corpus identity checks match K4; any future migration needs actual retrieval equivalence and memory evidence, and must preserve the current attempt.


## 2026-10-07 14:46 CST: complete asset identity and unadmitted retrieval candidate

The original continuous preparation has64000/845515 actual trajectories and zero OPSD updates. The source100 model, generation seeds42–49, question shards, full five-rollout budget and305-file frozen runtime remain unchanged. Original pipeline/controller and8 GPU workers are live. No five-update acceptance file exists.

An independent candidate runs on HKUST1 at /data/shaoshitong/search-e1-retriever-segmented-20261007, private port18004, server PID2239529. It is CPU-loaded with the full21015324-vector FP32 index and original CPU FP32 E5 encoder; only private GPU comparison probes are enabled. Production /retrieve is still unadmitted. The original HKUST1:18001 relay and HKUST4:18003 service remain unchanged.

All22 identity comparisons now pass actual SHA256 reads: full64,559,075,373-byte index, full14,393,579,520-byte corpus,18 encoder files and both original encoder/scanner sources. K4 index was actually read completely, with unchanged stat during the read; the earlier manifest-only partial proof remains preserved. Separate actual CPU encoding tests use770 queries across7 request sizes and591360 FP32 values; both hosts return byte-exact embeddings. Different request batch boundaries can change FP32 embeddings, so the candidate preserves original independent128-query encoding boundaries and each request's64-query GPU micro shapes, tail shapes, top-k operations and order.

A first real one-query full-index GPU comparison returns exact scores, IDs and actual matrix shapes. This single-query probe is preliminary, not candidate admission. Its52.45s duration includes original and candidate full scans and trace overhead, not training speed. A full candidate-only operator/stress gate and one sequential320-query K4 HTTP equality check are now running. No bulk stress requests are sent to the shared K4 oracle.

The separate continuous candidate observer samples actual process usage and whole GPU7 reserve, binding PID, GPU UUID and source digest. The observed initial candidate peak is574MiB and whole-card minimum reserve3531MiB while sharing our own existing generator. These sampled numbers are provisional until the complete gate is covered. Scientific training parameters and loss remain unchanged. Any future production use requires the complete equality/memory gate and a preserved preparation attempt; no rollout or OPSD update is fabricated.


## 2026-10-07 14:53 CST: actual free memory correction

Independent review found both legacy observers called total-minus-used free memory. Driver reservations make this computed reserve about352–354MiB larger than directly queried memory.free on these cards. That calculation is preserved as historical evidence and is no longer accepted as actual free memory. There was no OOM: the actual sampled minimum is3157MiB at this point.

The candidate V1 gate's single320-query K4 HTTP integration returned complete exact JSON. Its private gate client and V1 observer alone were stopped after ownership/path verification, and their logs/evidence preserved. The V1 gate is explicitly not admissible because its memory interval used computed reserve. No generator, OPSD process, server, relay or archive process was stopped. An external V2 monitor now directly queries memory.free and memory.reserved; a new V2 full gate requires that explicit metric and sampling coverage before/through completion. Actual candidate worker thread counts are Torch16/FAISS144, distinct from startup FAISS32; recorded HTTP equality is the numerical evidence.

The whole-eight-card observer likewise writes a separate memory-continuous-v2.json with directly sampled actual free values, UUIDs, interval timestamps and preserved V1 computed snapshot. The five-update verifier is now source/watch_acceptance_v3.py, tmux e1-opsd100-acceptance-v3-20261007, logs/acceptance-v3.log. It requires actual free memory>=1024MiB, complete8GPU coverage, observation beginning before OPSD config creation and a fresh sample after real update5. It never falls back to the old computed reserve. The original and V2 verifier sources/logs remain preserved. No OPSD update or five-update acceptance has occurred yet.
