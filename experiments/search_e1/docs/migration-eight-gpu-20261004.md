# Search-E1: move the common retriever and expand the R1 configuration to eight GPUs

## Authorized outcome

The user approved moving retrieval away from HKUST1 and explicitly selected an eight-GPU run starting again from the common initial weights. Preserve the previous four-GPU run and its logs/receipts; do not call this optimizer-state resume. Keep the archive configuration running on HKUST4 GPUs4–7. Do not stop other users' processes or send messages to the seven algorithm threads.

Both arms retain train_batch_size512 and five agent trajectories per question. Preserve each original configuration's other logical parameters, including archive mini128 and R1 mini256, entropy, LR/warmup, step budget, reward protocol, reference weights, and loss reduction. Batch equality does not mean replacing the requested two hyperparameter configurations with identical configurations.

## Resource and service transition

At the first inspection, HKUST2 GPUs1/2/3/6 were free and passed actual CUDA allocations. GPUs0/4/5 were foreign and excluded. The target had5TB disk free and707GiB available host RAM. Replicate the verified full FP32 index and compressed corpus over HKUST LAN to a new task directory; verify original manifest hashes and the existing encoder before decompression. No credentials or other-cluster assets are transferred.

During copying all remaining HKUST2 GPUs became occupied by foreign processes. The idle-admission check correctly rejected the four-shard startup. Copies completed and verified, but no replacement GPU service was started. HKUST5 remains an unhealthy alternative (GPU4 N/A, fresh tiny CUDA probe on5/6 failed).

Resource fallback: test a bounded-memory exact FP32 streaming backend on the archive run's OWN GPU7 on HKUST4, authorized by the user's GPU-sharing question. Keep all vectors in host RAM; transfer32768-row FP32 tiles (~96MiB) and reuse each tile for all query blocks of64. Maintain exact full-index dot products and merge top-k across all tiles, with no quantization/ANN/corpus change, TF32=false. Keep CPU FP32 E5 and request serialization. Test CPU/golden-GPU retrieval outputs before use with original absolute score tolerance1e-5 and report rank differences inside that numerical tie boundary separately; reject non-tie rank changes. Measure real full-index/concurrent512 performance and require PyTorch reserved<=512MiB, service device footprint<=1GiB, and aggregate archive peak maintaining6GiB reserve during actual retrieval/actor-update overlap, plus a conservative bound from the sum of separately measured historical training and service maxima; record these separately, not merely idle measurements. Reject this fallback if it changes non-tie retrieval results, violates headroom, or is too slow to provide the intended eight-GPU speedup. GPU admission must verify every existing occupant is the exact owned archive training worker UUID/PID, rather than allowing foreign co-tenancy. Running archive runtime bytes remain unchanged.

If this fallback passes, deploy it at10.123.4.11:18002 and point the old HKUST1 HTTP entry point to that backend; HKUST4's original relay and existing training URLs remain stable. Future archive-stage admission must explicitly permit only the verified small owned service PID/footprint onGPU7, with all other original UUID/occupancy checks retained. Failure cleanup stops only the owned streaming service; the original resident oracle stays until replacement acceptance.

Start the same four-shard FP32 oracle at10.120.17.102:18002. Its only initial source difference is HTTP binding; preserve source-manifest identity, query encoder precision, TF32=false, exact FlatIP, top3, query_micro64, and request serialization. Before switching, compare top-k/documents/scores with the original live backend, exercise two concurrent512-query requests, and verify device headroom. Acceptance hash may change because timings/device receipts change; record both, and verify actual outputs rather than pretending hashes remain identical.

The preceding four-shard route is retained as the original attempted plan; the bounded-memory fallback supersedes it while HKUST2 has no free GPUs.

Retain the original HTTP entry points for running clients. On HKUST1, replace only this task's original GPU oracle with a CPU-only relay to the accepted replacement backend after new-backend acceptance, no active old requests, and an owned-PID/session check. The initial replacement was HKUST2; the sharing fallback is HKUST4:18002. HKUST4's original18000 relay continues targeting the same HKUST1 URL. Service relocation does not require stopping either training run. Only the later four-to-eight-GPU training transition needs to stop the owned R1 run.

## Eight-GPU mathematical execution

Changing GPUS4 to8 while keeping per-GPU micro16 doubles the global micro64 to128. Keeping global micro64 instead gives local micro8, which also changes masked-token normalization unless handled explicitly. Preserve the original four logical DP workers using pairs of physical ranks.

Balance the batch into the original four logical partitions. Within each partition, stripe every16-row logical microbatch into two8-row halves, concatenating the even halves for physical rank2v and odd halves for2v+1. Global mini256 gives each physical rank32 rows and four local micro8 calls, matching the original logical mini64/local micro16/four accumulation calls. Each original mini-batch retains the same trajectory membership after the shared four-partition balance.

For a pair's logical microbatch, sum the two local masked-token counts. A physical piece contributes loss times2*piece_tokens/pair_tokens, divided by the unchanged four accumulation calls. FSDP's mean over eight ranks then equals the original mean over four logical workers. Use the same factor for loss metrics. A pair with zero total masked tokens must reject; a zero-token half still executes collectives with zero loss weight and a safe temporary loss mask. Create all four paired groups in the same global order, once. Leave archive and default four-GPU behavior unchanged.

Preserve global score micro128 (R1 original4*32), batch shuffle seed42, five-rollout collection, original optimizer/scheduler calls, and the existing physical/logit/optimizer memory adapters. New vLLM workers can change stochastic samples; do not claim bitwise identical training trajectories. The aim is the same logical loss/minibatch reduction, within measured floating-point tolerance.

## Isolated run and acceptance

Create a new R1-eight-GPU base/runtime/receipt set; reference existing verified immutable assets without modifying them. Preserve the complete old run, stopped at a completed iteration boundary where possible. Start the new run from the exact common initial model, with the full500-step schedule. Never reuse the old run UUID/output directory or imply its Adam state was resumed.

Before stopping the old training, verify the row mapping and original-source loss/gradient/Adam-step equivalence on unequal masks, entropy/KL terms, zero-token half, and an all-zero pair. Then perform the actual eight-GPU two-update memory gate, inspect finite/changed checkpoint tensors, and require6GiB measured headroom. Tune only physical execution/cache parameters; retain logical parameters. Start formal training only after the gate passes, verify five complete normal formal iterations and baseline evaluation, and measure actual iteration timings before claiming acceleration. Continue both full GRPO/OPSD pipelines, unified evaluation, and effect attribution.

Hourly monitoring must cover the new base after transition while preserving old evidence. Stage failures require diagnosis and owned recovery; no full scientific acceptance until the complete pipeline/evaluation is done.

## Work checklist

- [x] Inspect live resources, code, and checkpoint limitations.
- [x] Explain sharing/CPU alternatives and confirm eight-GPU fresh-start choice.
- [x] Start independent verified asset replication without stopping training.
- [x] Reviewed twice (Approved); implement and numerically verify isolated execution adapter.
- [x] Accept full-index streaming backend, switch original URLs, free owned oracle GPUs.
- [ ] Preserve old four-GPU training, accept eight-GPU gate and formal updates.
- [ ] Finish configured full pipelines and scientific comparison.

## 19:07 CST implementation checks

User explicitly chose fresh eight-GPU restart from common initial weights. Original B still runs; original A reached18 formal updates. New B base is `/data/shaoshitong/search-e1-config-study-8gpu-20261004`. Immutable assets/env are referenced; original base remains untouched. The new frozen runtime contains305 real source files after excluding macOS resource-fork metadata. A preliminary imported runtime containing those metadata files is preserved separately; original301 archived real files and scientific source remain unchanged. Seven overlay checks pass; Hydra confirms batch512/n_agent5/mini256/micro64/score128/entropy0.001/eight GPUs/full501 scheduled steps.

Actual archived PPO code on eight CPU Gloo ranks matches the original four-rank gradients, Adam states and updated weights to1e-12, including unequal masks/zero-token half; world-mean metrics agree1e-7. All ranks synchronously reject an all-zero virtual pair. OPSD world8 retains global32 pairs per optimizer update by changing local accumulation8 to4. No running frozen runtime was edited.

Streaming full-index oracle on owned HKUST4 GPU7 passed CPU top4 checks (1e-5 absolute score tolerance), identical two concurrent512-query outputs,44.47s/41.01s request service times,144703488-byte allocator reserve peak and610MiB sampled process footprint. Live old/new document comparison and overlap-peak acceptance precede relocation. The old resident oracle remains available until these checks pass.

Full original/global2560-row partition verification passes: all rows occur exactly once, all ten original optimizer minibatches retain their trajectory membership, and every paired8+8 microbatch reconstructs the original16-row microbatch. Live backend probes return identical documents and scores within1e-5; measured16-query wall times resident0.149s versus streaming11.587s. Streaming trades retrieval throughput for four additional policy GPUs. The expected end-to-end speedup must therefore be established by actual eight-GPU update timings, not inferred solely from card count. Continuous512-query overlap requests so far took65.92s,53.68s,59.41s under active archive actor-update load; actual shared GPU7 peak30847MiB, service610MiB. Original archive GPU7 historical peak39574MiB plus610MiB bounds combined observed component peaks at40184MiB, leaving8956MiB (8.75GiB); actual peak-phase monitoring continues before relocation.

## 19:17 CST relocation completed

Seven continuous512-query requests overlapped the actual archive actor-update phase; shared peak30847MiB, service610MiB. Conservative separate-component sum39574+610=40184MiB leaves8956MiB above measured component maxima. This is explicitly an upper-bound calculation, not a claim that the old39574MiB training peak was simultaneously observed during calibration. Ended only the owned synthetic load client; the shared-memory observer remains active.

Stopped only owned original retriever PID2440107 after source/environment verification and no active old connections. New CPU relay PID3421872 listens on the same HKUST1:18001 endpoint and targets HKUST4:18002. Its health and real retrieval query passed; original4–7 each now use4MiB. Archive18000 relay is preserved. Owned r1 old run continues0–3 until completed step7; e1-eight-restart-controller-20261004 will then preserve its evidence and terminate only its verified private process group before launching new eight-GPU gates. Hourly automation now covers the new base, old preserved run, shared retriever, and full pipeline/assessment.

## 19:38 CST transition recovery

Old four-GPU B completed step7 normally. Initial transition guard rejected nine Ray dashboard spawn children whose environment was intentionally scrubbed by Ray; no training process was stopped on that failed attempt. Inspected UID1005, same private session/group2986860, exact old environment Python source, and direct parent2987152 with the verified old run UUID and private dashboard source. Added narrowly scoped parent/session/source verification for these children, preserved the rejected controller/log, and restarted the controller. No foreign processes qualify for this exception.

## 19:41 CST eight-GPU gate active

Private process group2986860 (185 verified owned processes) exited completely. Original B steps1–7 and full old base remain preserved; partial next rollout was discarded, no optimizer resume claimed. New eight-GPU gate UUID8d708a8556084021ae3069b93335109d, supervisor3448662, main_task3451483, actor workers3452460/3452777/3452778/3452779/3452780/3452781/3452782/3452783. Execution receipt records E1_GPUS8 and E1_VIRTUAL_DP_WORLD4, all original logical parameters, physical1/score1/JIT1/logit128/CPU chunk states1, vLLM0.25. All eight cards passed idle/UUID admission. Initial load sampled peak13521MiB, first full512-question/five-rollout stage executing; no completed new update is claimed yet.
