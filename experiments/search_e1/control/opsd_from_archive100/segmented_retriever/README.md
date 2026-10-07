# Independent segmented retriever candidate

This directory is preparation only. Nothing here deploys, stops a process,
changes a relay, edits the running generation, or authorizes GPU sharing.

## Preserved operations

`segmented_scan.py` subclasses the existing `torch_streaming_flat.py` scanner.
For every queued HTTP request it retains the original FP32 embedding array,
query micro-batch boundaries of 64 (including 1/63/65/129 tails), GEMM shape,
vector tile boundary of 32768, local top-k, concatenate order, and merged top-k
order. The only loop change shares one CPU-to-GPU vector tile among up to eight
requests. Requests are never concatenated into one query array. Whole index
coverage remains 21,015,324 vectors of dimension 768.

The single worker encodes each request separately using the same CPU FP32 E5
calls with batch size 128 and the configured 16 Torch threads. Each request keeps its original
padding/batch boundaries. The scanner retains TF32=false. Top-k 1..4 and
`return_scores` keep the existing JSON document/score representation.
The service records actual `torch.get_num_threads()` and
`faiss.omp_get_max_threads()` after CPU loading and on the encoding worker.
Torch/FAISS may share OpenMP state: configuring Torch=16 and then FAISS=32 can
change the actual thread count. Configuration instructions alone are not proof
of numerical equality; owner inspection and exact live comparisons are required.

The FIFO queue collects for a fixed 25 ms window, selects at most eight queued
requests (each <=2560 queries, total <=20480), and has only one encode/scan
consumer. Errors fail each affected future with HTTP 503; they do not turn into
empty documents or a lower-precision/index fallback. Queue wait, independent
encoding, shared scan and document assembly times are recorded separately.

## Owner-controlled startup and admission

Use an independent `E1_BASE` on HKUST1 and private port 18004. Root must confirm
the local index/corpus/encoder SHA identities against the existing accepted
oracle before starting. Suggested assets are:

- `/data/shaoshitong/search-e1-retriever-gpu-20261004/assets/index/e5_Flat.index`
- `/data/shaoshitong/search-e1-retriever-gpu-20261004/assets/corpus/wiki-18.jsonl`
- `/data/shaoshitong/search-e1-retriever-gpu-20261004/assets/encoder`

Use the preserved execution runtime's CPU encoder overlay in `PYTHONPATH`, plus
the original control directory containing `torch_streaming_flat.py`, plus this
candidate source directory. The candidate refuses an unpatched upstream GPU
encoder before loading it. Set `CUDA_VISIBLE_DEVICES=7`,
`NVIDIA_TF32_OVERRIDE=0`, `E1_ENCODER_DEVICE=cpu`,
`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and a private random
`E1_SEGMENTED_PROBE_TOKEN` of >=32 characters. Pass the existing verified
asset manifest with `--asset-manifest`.

CPU startup loads the full host index/corpus and CPU encoder and records
`receipts/source-identity.json`; it must report CUDA uninitialized. It does
not accept retrieval requests or allocate GPU buffers yet. Root alone may write
`receipts/enable-gpu.json` with the current process `pid` and
`source_identity_sha256`. This enables the private token-protected
`/probe/retrieve` and `/probe/compare` endpoints. Public `/retrieve` continues
to return 503. Restarted processes require fresh PID-bound controls.

After complete full-index GPU/HTTP equality and shared-card memory review,
root may write `receipts/admit-requests.json` containing:

```json
{
  "pid": "actual integer PID",
  "source_identity_sha256": "source-identity digest",
  "accepted": true,
  "full_gpu_http_gate_path": "/absolute/path/to/accepted-gate.json",
  "full_gpu_http_gate_sha256": "SHA256 of that exact gate receipt"
}
```

The gate must bind the same PID/source identity and scope
`full GPU and HTTP oracle equality`. Admission alone does not switch any relay.
Root must separately choose and verify any operational routing after review.

## Verification already performed versus still required

`cpu_fixture.py` compared original and segmented operators for request sizes
1/63/64/65/128/129/320, top-k 3/4, full 32768-row tile plus a 3-row tail,
duplicate vectors and all tied scores. It captured actual `aten.mm` shapes and
required exact scores and IDs, including tie ordering. The fixture also checks
the FIFO eight-request/20480-query limit and empty search outputs. When local
FAISS is unavailable it uses only a flat-pointer stand-in, explicitly recorded
in `cpu-fixture-results.json`. **This is not live GPU acceptance.**

`full_live_gate.py` is for the owner to run only after GPU activation and live
memory observation. `/probe/compare` executes segmented search and then the
unchanged original full GPU search with the same actual E5 embeddings, on the
candidate's single scanner thread. It requires exact scores, IDs and actual
GEMM shapes for 1/63/64/65/128/129/320/2560, both score modes, with mixed
top-k 3/4. These are same-GPU operator comparisons on the complete index.

There is **exactly one** sequential HTTP integration request to the existing
shared HKUST4 oracle: 320 actual questions, top-k=3, `return_scores=true`.
The candidate response must match its complete JSON, including documents,
scores and ordering. This is the only cross-machine HTTP evidence; the gate
does not claim every shape/top-k/score mode has been compared over HKUST4 HTTP.
It never sends parallel test requests or 2560-query requests to the shared
oracle, protecting the running training/generation queue.

`--stress-max` adds eight concurrent 2560-query requests **only on the
candidate**. Admission requires evidence that they actually entered one
eight-request/20480-query shared scan; merely launching eight HTTP calls is
insufficient. This is a pressure gate, not the real rollout budget: the current
eight generation workers send at most 320 queries per individual request.
No stress/probe duration may be presented as real training speed.

The gate combines the passed CPU shape/ties fixture and its exact original/
segmented scanner SHA with the live source identity and owner-verified SHA
equality of the full index, corpus, encoder assets, original scanner source and
original encoder module. Pass `--source-identity` pointing to the candidate's
receipt, `--cpu-fixture-evidence`, and `--asset-identity-evidence`. The latter
uses this schema (each encoder asset file should have its own `encoder` row):

```json
{
  "accepted": true,
  "comparisons": [
    {"kind":"index","path":"index/e5_Flat.index","k1_sha256":"actual SHA256","k4_sha256":"same SHA256"},
    {"kind":"corpus","path":"corpus/wiki-18.jsonl","k1_sha256":"actual SHA256","k4_sha256":"same SHA256"},
    {"kind":"encoder","path":"encoder/model.safetensors","k1_sha256":"actual SHA256","k4_sha256":"same SHA256"},
    {"kind":"scanner_source","path":"torch_streaming_flat.py","k1_sha256":"actual SHA256","k4_sha256":"same SHA256"},
    {"kind":"encoder_source","path":"retrieval_server.py","k1_sha256":"actual SHA256","k4_sha256":"same SHA256"}
  ]
}
```

Known verified identities supplied by the owner:

- index: `69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166`
- corpus: `85a787a692e73fdd657e86411bfd1ac810ed193b1eea4a478de449d4871c06b9`
- original encoder module: `fc528a2df9d2927f7dbf21a92d4671405259cba90e8e02dca661a4ad784c1a97`

Any score/ID/shape difference is a failed strict gate; it does not silently
relax equality to a tolerance or admit an incomplete diagnostic run.

Root's continuous actual GPU observation must bind the candidate PID and
source digest and prove retriever process peak <=1024 MiB, whole physical
GPU7 minimum free >=1024 MiB, and coverage before/through the gate. The memory
JSON must provide numeric `sample_start_unix`, `sample_end_unix`,
`updated_at_unix`, and `samples`, in addition to `candidate_pid`,
`source_identity_sha256`, `max_process_mib`, and `min_free_mib`. Gate checks
actual start <= gate start, actual end >= gate end, update >= actual end,
freshness and sample count. Boolean claims alone are rejected. Keep those
samples and all gate logs. Existing HKUST4 shared retriever/archive margins
remain their own >=6 GiB criterion and must continue to be monitored.

Gate/probe timing includes extra oracle scans and queue waiting and does not
prove real training acceleration. Only later unchanged-budget live generation
timings can support that conclusion.
