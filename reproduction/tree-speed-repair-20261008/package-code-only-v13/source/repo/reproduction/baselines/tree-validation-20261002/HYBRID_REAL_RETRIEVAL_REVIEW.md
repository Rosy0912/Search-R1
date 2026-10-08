# Hybrid real retrieval independent review (2876)

**Recommendation: accept this fixed hybrid recipe as a limited Tree actual-top3 retrieval sanity gate for attempting the two-actual-update smoke.** This review does not accept five updates, formal500, checkpoint quality, or Search agentic-RL effects. Training actual updates remain0 at the review boundary. No acceptance marker or dispatch was written by the reviewer.

H100 uses the original CPU FP16 E5 mean-pooling/query256/batch512 model and local corpus/API; only flat-IP index search goes through binary float32 transport to fixed A800 GPU4, one FP32 IndexFlatIP and512MiB scratch. This removes the previous changed/slow A800 CPU encoder from the candidate. The old A800 CPU candidate timeout/embedding differences remain failed/limited evidence and are not reused as encoder acceptance.

| Real path | Query rows | Actual k | Same ID rank | Same content rank | Same sets | Two-repeat stable | Maximum matched score delta | Wall seconds (two runs) |
|---|---:|---:|---:|---:|---:|---|---:|---|
| binary | 14 | 3 | 14/14 | not loaded | 14/14 | yes | 5.96046448e-07 | 2.755980, 2.593625 |
| binary | 14 | 50 | 11/14 | not loaded | 14/14 | yes | 5.96046448e-07 | 3.498028, 3.052841 |
| binary | 512 | 3 | 512/512 | not loaded | 512/512 | yes | 8.94069672e-07 | 17.370938, 14.641263 |
| binary | 512 | 50 | 402/512 | not loaded | 512/512 | yes | 1.1920929e-06 | 17.860574, 18.455814 |
| http | 14 | 3 | 14/14 | 14/14 | 14/14 | yes | 5.96046448e-07 | 2.477276, 2.758509 |
| http | 14 | 50 | 11/14 | 14/14 | 14/14 | yes | 5.96046448e-07 | 3.367495, 3.656503 |
| http | 512 | 3 | 512/512 | 512/512 | 512/512 | yes | 8.94069672e-07 | 24.274449, 22.916256 |

Every summary and matched score delta was independently recomputed from IDs/scores/content hashes in both repeats against2854. All per-ID common-content hashes match. HTTPk50 changes only three query ID orders; raw Wiki content hashes remain identical in each swapped position because these are duplicate-content documents. It is not ID/ranking equivalence for CoSearch/top50, and actualk3 must be called rather than slicingk50.

| Query | Swapped ranks | Baseline IDs | Actual IDs | Content hashes at swapped positions |
|---|---|---|---|---|
| 3: Where is Mount Everest located? | [7, 8] | [15622197, 3199837] | [3199837, 15622197] | identical baseline and actual |
| 7: When did the Apollo 11 mission land on the Moon? | [42, 43] | [11729919, 9602779] | [9602779, 11729919] | identical baseline and actual |
| 11: What is the tallest building in the world? | [1, 2] | [16864745, 7009180] | [7009180, 16864745] | identical baseline and actual |

The fixed model/index/corpus identity and sources in actual HTTP provenance match reviewed2854/source files and the real binary instance. A CPU-only independent read of original saved2854 NPZ has fileSHA `0f72b397ddf8c4dadb0f77e26f0e6c7c86ba314be0023168fe4f3cbe017fefb9`. Its recomputed14 and repeated512 raw vector SHA and canonical NPY frame SHA match **all six** real H100 encoder/proxy records exactly:
- 14 raw vectors `fbb2788e0f57349dcde765a9cb94ce42756b3641fca5e372c233e7faf6c314c5`; frame `ddcea3a4f14d351cda50a781fa9cdd106b9f4ffeec8e5d84f5627665711f99f0`.
- 512 raw vectors `c7bc1b4a084a169e6b6f197958d462094a743d34347842bd57eb249c308da7de`; frame `ff23fb49f7b7937411c9b09b2b186b47295a433519594647ea86845017cf85c8`.

Real encoding wall times14: 0.148224, 0.143319, 0.142980, 0.140366s; repeated512: 4.605376, 4.688498s. This proves the measured queries retained H100 CPU vectors rather than merely two logs agreeing with each other. It does not prove all future query embeddings or latency. The512 rows repeat only14 unique questions, with short fixed texts; they are not512-unique throughput.

CPU Slurm2876 body is175s,8allocatedCPUs,0training/CPU-gate GPUs,exit0 (0.388889allocatedCPU-hours), excluding scheduler prolog/epilog. A800 index service remains alive on one selected GPU; its full loading/idle/lifetime GPU cost and current sampled memory require closure/archive and are separate from this0GPU CPU-gate declaration and any whole-platform reservation. The earlier single-index probe peak is a separate measurement, not the new service instantaneous peak.

Final gate source review: checker mandates eight distinct evidence roles/full SHA/source/assets/report stable coverage, and permits only a fresh stage CPU PID/run change with full startup provenance SHA. Remote index instance remains identical. Retired gate PID is rejected, poison503/identity changes/timeout and incomplete/unstable evidence are rejected. Head verified raw receiptSHA now propagates unchanged through srun into startup/periodic checker. Launcher validates own local CPU PID/run, retains190sTERM+5sKILL bounded checks, independent GPU ownership guard and frozenf4/audit/cache settings; own retriever cleanup is bounded15sTERM→KILL. No auxiliary remote service kill exists.33 checker+6monitor/HTTP toy fixtures pass.

New CPU cost sampler records only the local own-run uvicorn PID, validates exactjobsegment and starttime, counts utime/stime ticks and RSS/HWM units correctly, stops on PID reuse.3 actual-source fakeproc fixtures pass for ownjob42, foreign43 and prefixcollision420. Launcher starts/stops only its own sampler. It observes cumulative retriever process CPU and main-process memory, not simultaneous corpus-worker totals or allnodeCPU costs; sampling starts after loading though cumulativeCPU/HWM includes that earlier lifetime. Sampler failure is instrumentation loss, not a proof of failed mathematical training.

Remaining next-stage requirements: new reviewed immutable retrieval receipt; actual two-optimizer-update run; each step both rollout and policy audit completions/SHA bindings; true optimizer count, finite metrics, checkpoint/algorithm-state agreement and CPU reload/nonzero parameter delta. Existing validator must use --require-tree-audit. A successful retrieval gate alone cannot satisfy any of those training gates.

Evidence: `hybrid-real-independent-results.json`, `saved-2854-vector-hash-proof.json`, `hybrid-checker-fixtures.json`, `http-monitor-fixed-results.json`, `cpu-sampler-fixtures.json`, `HYBRID_GATE_SOURCE_MANIFEST.json`; raw files remain under outputs/evidence. Source identities are manifest-pinned. This reviewer executed no GPU/Slurm/large model service.
