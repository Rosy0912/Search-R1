# B phase-scoped FP32 retrieval candidate

The scanner retains the original 32768-row tiles, query micro64, FP32/TF32-off
GEMMs and original global tile order when merging top-k. The backend supports
an explicit four/eight-device count. Current HKUST5 authorization is **only
GPU0–3**: `phase_http_server.py` pins four devices and merges on GPU3. GPU4 is
not exposed, queried or probed. The original K4 18003 oracle is unchanged.

`phase_http_server.py` keeps CPU FP32 E5 (mean pooling, max256, batch128) and
corpus alive. A private spawned process keeps four GPU scan contexts during
sampling and actually exits before HF/update/model loading/checkpoint. CPU
parent never initializes CUDA. Readonly memmap layout is derived from the
same-runtime tiny FAISS serialization. Full transfer checksums run once via
`verify_assets.py`, never on every retrieval request.

`gate_four_http.py` measures full-index requests with 1 and 65 queries against
the immutable oracle, compares the complete score/ID/document JSON, records
all 642 original tile shapes and per-device allocator memory, releases the
child, recreates it for a second phase and checks its real exit. Training
traffic remains disabled until this real GPU gate succeeds. CPU tests are not
numerical GPU or training speed acceptance.

The large index/corpus move directly between HKUST nodes using the bounded
private range server; no heavy assets are staged on Mac. Transfer source,
checksums, receiver entry and partial files are preserved. Exact local paths
and tokens are runtime receipts, never account credentials or Git secrets.

Current status: four-device backend and HTTP/lifecycle wiring published;
asset transfer in progress; full-index GPU equality/speed and new training
iteration still pending. Eight-device admission is not claimed.
