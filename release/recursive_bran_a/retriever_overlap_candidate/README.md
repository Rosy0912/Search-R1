# Independent pinned-copy overlap candidate

This is preparation, not a deployed retrieval service or a measured acceleration.
The original GPU7 oracle, its endpoints and ongoing training remain untouched.

`PinnedOverlapFlatIP(original)` uses the original read-only CPU FP32 flat matrix.
It does **not** place the 64.56 GB matrix permanently on GPU. Two pinned CPU
slots and two GPU slots allow a later tile H2D to overlap an earlier tile's GEMM
and top-k. Before reusing a CPU slot, its previous DMA ready event is synchronized;
before overwriting a GPU slot, the copy stream waits for the previous compute
consumed event. Exceptions synchronize both streams before buffers are freed.

Unchanged arithmetic contract: 32768 original vector tile boundaries, 64 query
micro-batches separately within every request, FP32 vectors/queries/matmul/top-k,
TF32 disabled and CUDA autocast refused, tile -> request -> query-micro operator order, sorted local and
merged top-k, global IDs, original tail tile. No concatenated request GEMM.
Exact floating score/ID equality is still a **GPU benchmark requirement**, not
proved by the CPU tests. Allocator addresses and CUDA scheduling change.

For 21,015,324 × 768:

- Every shared scan still uploads **64,559,075,328 bytes**.
- 642 tiles, final original tile 11,036 rows.
- Two pinned CPU slots: 201,326,592 bytes (192 MiB).
- Two GPU slots: 201,326,592 bytes (192 MiB).
- At most one 64 × 32768 distance matrix at a time: 8,388,608 bytes (8 MiB).
- These numbers exclude queries/results, CUDA context, allocator reserves and
  the existing oracle/training. They are **not** an observed total-memory peak.

CPU tests (no CUDA or performance claim):

```sh
PYTHONPATH=. python -m unittest release.recursive_bran_a.retriever_overlap_candidate.test_lifecycle_cpu -v
```

Six tests check exact original tile/tail dimensions, unchanged copied bytes,
prefetch before compute, CPU/GPU slot reuse event dependencies, unsafe overwrite
refusal, exception cleanup, and rejecting CUDA autocast before copies. They do not test real CUDA event timing.

## GPU profile, only after independently verified idle GPU7

`benchmark_when_idle.py` is a prepared command, not launched. It requires explicit
admission and checks GPU7 memory.free >= 20480 MiB and no compute process other
than approved oracle PID 2239529. Changed oracle identity still requires the
external fresh source/PID/tick gate; the helper does not replace that gate.

```sh
CUDA_VISIBLE_DEVICES=7 python benchmark_when_idle.py \
  --oracle-source /data/shaoshitong/search-e1-retriever-segmented-20261007/source \
  --index ACTUAL_EXISTING_FLAT_INDEX_PATH \
  --profile-tiles 8 --repeats 3 --authorize-idle-gpu7-bench
```

Default is seven complete original tiles plus a seven-row test tail, eight tiles
in total. It compares original and candidate exact scores/IDs and all per-request
GEMM/top-k trace shapes before wall-clock timings, and reports actual Torch peaks
and direct free before/after. It exercises 5/19/40/65-query requests and an
independent shared request list; no E5 encode or HTTP correctness claim.

**Important:** `faiss.read_index` still allocates the full CPU flat index even for
this small GPU profile; about 64.6 GB extra host RAM must be admitted. If CPU RAM
is unavailable, root should provide an already accepted smaller independent
matrix fixture, not launch this full-index loader during training. The original
oracle's index is not copied from or mutated inside its process.

Only after the small test is accepted, `--full-scan` requests a real full-index
kernel gate on identical saved FP32 query embeddings. That gate and the HTTP
JSON oracle gate remain separate and require root's resource admission. No
service deployment/restart is performed by these scripts.

Pinned copying cannot reduce the mandatory 64.56 GB PCIe traffic. Improvement
is limited by measured H2D bandwidth, CPU staging, compute overlap and launch
costs; the current 9.7-second scan cannot be promised to shrink to any value
before this actual benchmark.
