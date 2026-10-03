# Independent CPU BLAS candidate review

Candidate is worth a dedicated full-index parity gate. It is not approved for production switching. All probes were isolated CUDA_VISIBLE_DEVICES='' children, default and preload serially, OMP/OpenBLAS threads8, nice19, timeout120. No current server/worker/venv/global environment was modified. Synthetic index only: 50k vectors ×768 float32, fixed RNG42, true top3 search14/128/512. Full production index was NOT loaded.

| batch | default wall | preload wall | observed ratio | default CPU | preload CPU |
|---|---:|---:|---:|---:|---:|
|14|0.016035s|0.016180s|0.991x|0.120596s|0.111191s|
|128|0.326806s|0.080615s|4.054x|1.866278s|1.196549s|
|512|1.247163s|0.297909s|4.186x|2.789582s|3.896805s|

All synthetic top3 IDs/scores exactly equal. One sample each, sequential runs on a shared host: ratios are indicative, not stable throughput guarantees. 512 CPU time INCREASED despite wall time improvement. 14 batch likely stays below BLAS threshold and saw no benefit. Default uses FAISS embedded BLAS with get_num_threads1; actual LD_DEBUG SGEMM relocation binds FAISS itself. Preload actual SGEMM relocation binds ownroot libopenblas.so.0. Direct faiss CDLL symbol lookup still resolves embedded numthreads1, so that API alone misidentifies the active SGEMM backend. Separate global symbol query proves ownlibrary OpenBLAS0.3.20, pthread runtime8, reported core Prescott (not an assumed AVX kernel), while FAISS OMP8. Process thread totals15/22 include other libraries, not an exclusive SGEMM team count.

First preload attempt produced no completed result (SSH broken pipe during excessive loader output, return255). Failed raw logs retained; own PID confirmed gone. Only preload retried with remote loader-output filter and succeeded. LD_DEBUG lookup messages marked fatal for unavailable optional ctypes symbols are dlerror diagnostics handled by getattr, not an exited successful process; return codes and stdout must be checked. Evidence: isolated-microbenchmark-results.json, SMALL_BENCHMARK_SUMMARY.json, default-*/preload-retry-*, preload-thread-identity.json. Package SHA12f1a06f966803bc4c03b5ef5e587da182944a6727178d8e7ce62d58586127b4 is parent-provided extraction provenance; this review did not redownload/reinstall it.

Future full gate: reserve sufficient CPU/RAM (at least64.56GB index plus runtime overhead) without active retrieval contention. Transfer this standalone script to an own review directory only after scheduling that separate CPU task. Fixed existing CPU embeddings come from ownroot runs/retrieval-2854/cpu-embeddings.npz and cpu.json; script verifies query/embedding/indexSHA and expected21,015,324×768 IndexFlatIP. It uses14 originals and repeats them to512; this is not 512 unique real queries or encoder throughput. Two repeats enforce stability, top3 and top50 are independently queried. Both child measurement JSONs include all IDs/scores, cost/RSS/thread/symbol identities; comparison emits DIFFERENCE_REPORT_ONLY and never starts service/training or acceptance.

For each child use the same absolute ownenv Python and threads8. Clear PYTHONPATH/LD_LIBRARY_PATH/LD_PRELOAD, CUDA_VISIBLE_DEVICES=''. Preload child alone adds the own OpenBLAS path. Use LD_DEBUG=bindings with remote filtering if proving actual relocation; dladdr report by itself is lookup evidence only. The full64GB file hashing is included in tool total costs but not search timing.

Example arguments (not executed):

```text
full-index-parity.py measure --allow-large-index --label default
  --index /dataset/shaoshitong/search-r1-rl/tree-grpo/assets-root/assets/wiki/e5_Flat.index
  --expected-index-sha256 69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166
  --embeddings /dataset/shaoshitong/search-r1-rl/tree-grpo/runs/retrieval-2854/cpu-embeddings.npz
  --reference-cpu /dataset/shaoshitong/search-r1-rl/tree-grpo/runs/retrieval-2854/cpu.json
  --threads 8 --repeats 2 --output OWN_REVIEW_DIR/default.json
```

Repeat in a fresh child with label preload and local LD_PRELOAD, then use `compare --default ... --preload ... --output ...`. Human review must examine top3/top50 rank/content identity inference, tiny/tied score differences, latency and total CPU/RAM costs before any service change. CLI help and small synthetic comparison fixture (1052 comparisons) pass; missing512 batch refused. No full-index benchmark or production deployment was launched by this reviewer.
