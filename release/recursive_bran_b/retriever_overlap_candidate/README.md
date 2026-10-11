# B independent eight-GPU phase candidate

Adapted from A's immutable FP32 streaming, double-buffer, whole32768-tile code.
The CPU parent uses metadata rather than constructing a CUDA scanner. The
spawned child owns all eight CUDA contexts and is reused within a phase, then
actually joined before HF. Memmap layout comes from same-runtime tiny FAISS
serialization probes. No CPU/GPU claim is based on guessed vector offsets.

This directory is a candidate, not the active B retriever. The original K4
18003 oracle remains unchanged. K5 GPU4 had a foreign workload at inspection;
no eight-GPU admission/activation is permitted until all eight UUID leases and
fresh process/free-memory checks succeed. The current B isolation exposes
only0–3. Do not weaken it for this candidate or run a copied A fixed-PID script.

Before activation: bind actual K4 oracle source/process/health, all eight K5
UUIDs, candidate source and complete assets. Then run full-index1+65 query
score/ID/complete HTTP equality, real memory/throughput, lifecycle exit checks.
The current adapted gate checks remote health; it still requires an explicit
oracle identity and resource-start entry before it can be used for admission.
No full-index GPU speed or memory acceptance has yet been obtained.
