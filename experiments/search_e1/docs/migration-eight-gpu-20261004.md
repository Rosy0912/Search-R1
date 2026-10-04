# Search-E1: move the common retriever and expand the R1 configuration to eight GPUs

## Authorized outcome

The user approved moving retrieval away from HKUST1 and explicitly selected an eight-GPU run starting again from the common initial weights. Preserve the previous four-GPU run and its logs/receipts; do not call this optimizer-state resume. Keep the archive configuration running on HKUST4 GPUs4–7. Do not stop other users' processes or send messages to the seven algorithm threads.

Both arms retain train_batch_size512 and five agent trajectories per question. Preserve each original configuration's other logical parameters, including archive mini128 and R1 mini256, entropy, LR/warmup, step budget, reward protocol, reference weights, and loss reduction. Batch equality does not mean replacing the requested two hyperparameter configurations with identical configurations.

## Resource and service transition

At the first inspection, HKUST2 GPUs1/2/3/6 were free and passed actual CUDA allocations. GPUs0/4/5 were foreign and excluded. The target had5TB disk free and707GiB available host RAM. Replicate the verified full FP32 index and compressed corpus over HKUST LAN to a new task directory; verify original manifest hashes and the existing encoder before decompression. No credentials or other-cluster assets are transferred.

During copying all remaining HKUST2 GPUs became occupied by foreign processes. The idle-admission check correctly rejected the four-shard startup. Copies completed and verified, but no replacement GPU service was started. HKUST5 remains an unhealthy alternative (GPU4 N/A, fresh tiny CUDA probe on5/6 failed).

Resource fallback: test a bounded-memory exact FP32 streaming backend on the archive run's OWN GPU7 on HKUST4, authorized by the user's GPU-sharing question. Keep all vectors in host RAM; transfer32768-row FP32 tiles (~96MiB) and reuse each tile for all query blocks of64. Maintain exact full-index dot products and merge top-k across all tiles, with no quantization/ANN/corpus change. Keep CPU FP32 E5 and request serialization. Test CPU/golden-GPU retrieval outputs before use. Measure real full-index/concurrent512 performance and require PyTorch reserved<=512MiB, service device footprint<=1GiB, and aggregate archive peak maintaining6GiB reserve. Reject this fallback if it changes non-tie retrieval results, violates headroom, or is too slow to provide the intended eight-GPU speedup. GPU admission must verify every existing occupant is the exact owned archive training worker UUID/PID, rather than allowing foreign co-tenancy. Running archive runtime bytes remain unchanged.

If this fallback passes, deploy it at10.123.4.11:18002 and point the old HKUST1 HTTP entry point to that backend; HKUST4's original relay and existing training URLs remain stable. Future archive-stage admission must explicitly permit only the verified small owned service PID/footprint onGPU7, with all other original UUID/occupancy checks retained. Failure cleanup stops only the owned streaming service; the original resident oracle stays until replacement acceptance.

Start the same four-shard FP32 oracle at10.120.17.102:18002. Its only initial source difference is HTTP binding; preserve source-manifest identity, query encoder precision, TF32=false, exact FlatIP, top3, query_micro64, and request serialization. Before switching, compare top-k/documents/scores with the original live backend, exercise two concurrent512-query requests, and verify device headroom. Acceptance hash may change because timings/device receipts change; record both, and verify actual outputs rather than pretending hashes remain identical.

The preceding four-shard route is retained as the original attempted plan; the bounded-memory fallback supersedes it while HKUST2 has no free GPUs.

Retain the original HTTP entry points for running clients. On HKUST1, replace only this task's original GPU oracle with a CPU-only relay to HKUST2 after new-backend acceptance, no active old requests, and an owned-PID/session check. HKUST4's relay continues targeting the same HKUST1 URL. Service relocation does not require stopping either training run. Only the later four-to-eight-GPU training transition needs to stop the owned R1 run.

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
- [ ] Review this focused design and implement the isolated execution adapter.
- [ ] Accept new GPU backend, switch original URLs, free owned oracle GPUs.
- [ ] Preserve old four-GPU training, accept eight-GPU gate and formal updates.
- [ ] Finish configured full pipelines and scientific comparison.
