# Eight-GPU FP32 streaming candidate

Backend import expected by the independent HTTP entry:

```python
from eight_gpu_streaming import EightGpuStreamingFlatIP
scan = EightGpuStreamingFlatIP(original)
scan.search_many(list_queries, list_ks, trace=None)
scan.search(queries, k)
```

This source prepares a candidate. It does not start GPUs, replace the original
18004 oracle, change the original route, or claim numerical/performance admission.
Root owns the independent 18006 service and its full-index HTTP oracle gate.

All 642 original 32768-row tiles (last tile 11036 rows) are distributed round
robin over GPUs0–7: 81/81/80/80/80/80/80/80 tiles. Each vector appears exactly once;
each GPU transfers about one eighth of the corpus. No shard boundary cuts a tile.
Round robin is necessary for bounded buffering and parallel progress: contiguous
80-tile shards followed by ordered merging would stall GPUs1–7 behind GPU0.

Each daemon worker owns a separate compute stream, a copy stream, two pinned CPU
slots and two device slots, reused across HTTP scans. Local buffer slot parity is
based on that worker's local tile counter, not its global round-robin tile ID.
The producer preserves each request and query64 micro-batch GEMM shape and sorted
local top-k. Two ready tiles per GPU is the bounded CPU result queue. Workers
transfer packed local scores/IDs unchanged to CPU, never a shard-wide aggregate.

The coordinator on GPU7 consumes tile0,1,...,641 and executes the original
`cat(best, local) -> topk(sorted=True) -> gather(ids)` per request/micro, including
the original k/local-k merge width and global ID offset. This avoids introducing
a different global tie-selection reduction. GPU local top-k and GEMM still run
on different devices, so **actual bitwise score/ID equality must be measured**;
CPU tests do not establish CUDA equality.

Memory budget at dimension768 and shared 40 queries, top3:

- CPU full index: original64,559,075,328B, one read-only shared host matrix among
  all worker threads. No per-GPU full corpus CPU/GPU copy.
- Per GPU: 201,326,592B (192MiB) double device vector buffers, at most8MiB for one
  query64 distance matrix, query matrix replicated on eachGPU (122880B for40Q).
- Pinned CPU: 192MiB per worker, 1,610,612,736B (1.5GiB) for eight workers.
- Two queued top3 tile results per GPU: <=23,040B for40Q across eight GPUs, plus <=11,520B for one pending producer tile per GPU and Python objects.
- GPU7 owns global best-score/ID tensors and small merge scratch in addition.
- CUDA contexts, allocator reserves, cublas workspace, host index pages and
  concurrent oracle/training are excluded. These are planned, not actual peaks.

`last_scan_profile.actual_worker_shapes` also keeps per-worker actual tensor shapes/counts, tile-start coverage and native threadID when an ordinary HTTP call supplies no trace callback. This is worker-side operand instrumentation; it does not claim that a main-thread TorchDispatchMode observed worker kernels.

The existing HTTP collection queue preserves independently encoded requests and
limits one shared scan at a time; the backend adds a `scan_lock`. Query lists are
never combined into a new GEMM shape. On worker failure the coordinator sets a
shared cancellation event, bounded producers unblock, and both worker streams
finish before the caller receives failure. Code faults do not yield partial IDs.

`trace` records actual worker GEMM shapes, dtype, GPU and native threadID at the
operator point. A main-thread-only matmul recorder cannot certify these worker
shapes. Full source/PID/UUID/memory, actual thread operator shapes, exact score-ID
and full HTTP JSON gates remain separate actual deployment requirements.

CPU check:

```sh
PYTHONPATH=. python -m unittest release.recursive_bran_a.retriever_overlap_candidate.test_eight_gpu_cpu -v
```

Four tests passed: original whole-tile coverage/balance, exact same CPU original
merge even with tied scores and reverse worker arrival, correct local double
buffer parity, and bounded-queue cancellation. No GPU speed or HTTP claim.
