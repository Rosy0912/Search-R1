# Real job 2860 — independent staged acceptance

## Current decision

**Final job2860 result: FAILED before training, zero outer updates. GPU/model preflight and retrieval/HTTP gates passed on real hardware; Ray CLI compatibility blocked trainer launch. Five-iteration/effectiveness/HF-reload acceptance remains NOT COMPLETE.** The reviewer performed read-only inspection and did not create `retrieval.accepted`, launch training, or modify services. Comparison SHA256: `076718989a6e516408cd06bffe5a965563935b41792ab5db2fcc8d476241b96e`. The parent may bind this SHA to Slurm job `2860` when creating its authorized gate marker.

Job2860 source commit from its evidence is `87c03c26dd1695e426d2b9e2821cc9090614874b`. Scheduler evidence shows RUNNING on exclusive h100-10, 8 H100, CPUs/Task96 (192 allocated), memory1T. GPU UUID inventory was initially zero-memory/zero-utilization with no compute applications. Actual NCCL all_reduce on eight ranks and BF16 matrix multiplication passed (`GPU_NCCL_BF16_OK 8`, Torch2.4.0+cu121). Real Flash bf16 forward/backward finiteness passed. Actual Qwen2.5-3B vLLM CUDA load used 5.7915GB weights and generated text beginning Paris; post-model GPU idle check passed. Enforce-eager async warning and missing vLLM version module warning did not abort this phase. Original evidence snapshot: `evidence/job2860_startup_snapshot.txt`.

## Query-specific retrieval inspection

CPU E5 fp16 and CUDA E5 fp16 used the same fixed assets and 21,015,324-vector, 768-dimensional index. GPU FAISS was actually sharded across 8 GPUs with float16. Encoder max absolute difference was0.000152587890625. The aggregate sorted-score max difference was0.0004788041114807129. The reviewer additionally joined scores and content hashes **by each concrete document ID**, checked all 50 IDs were unique in every row, checked each per-query set was equal, checked every matched-ID content hash was equal, and checked finite descending scores. This does not rely on the .95 screen or aggregate overlap alone.

All eight top3 ordered ID lists and corresponding content hashes are exact. All top50 ID/content sets are exact; no document/content is added or removed in this probe. Ordered top50 lists differ in every query through close-score changes below top3. The first moved rank is12, maximum displacement is2, maximum CPU-score separation of an inverted pair is0.0001233220100402832, and maximum same-ID score difference is0.000479280948638916. This magnitude and localization support accepting the CPU retrieval adaptation while reporting numerical/rank differences honestly. Eight fixed queries do not establish universal retrieval equivalence or equal downstream learning outcomes.

| Query | Exact top3 IDs | Top50 moved positions | First moved rank | Max displacement | Max reversed CPU score gap |
|---|---|---:|---:|---:|---:|
| 0: total number of death row inmates in the us | 12964449, 16754807, 6823959 | 4 | 17 | 1 | 6.58631325e-05 |
| 1: what is the meaning of by hook or cook | 17932575, 269947, 3539512 | 4 | 20 | 1 | 1.84178352e-05 |
| 2: who plays the army guy in pitch perfect 3 | 2244147, 6670881, 6680847 | 4 | 37 | 1 | 4.8995018e-05 |
| 3: Which magazine was started first Arthur's Magazine or First for Women? | 95403, 95404, 14417198 | 2 | 21 | 1 | 2.08616257e-05 |
| 4: Interpretations of Fight Club is a movie that starred what person know for their roles in low-budget arthouse and independent films? | 5674930, 20480681, 16538856 | 10 | 12 | 2 | 3.76105309e-05 |
| 5: Vietnam national cricket team will debut at what competitions at  Kinrara Oval | 7100828, 18379889, 9720087 | 8 | 32 | 2 | 0.00012332201 |
| 6: What is the capital of France? | 1039175, 9447685, 2543511 | 6 | 24 | 1 | 1.99079514e-05 |
| 7: Which river flows through London? | 18990009, 18990010, 2182608 | 2 | 44 | 1 | 8.28504562e-06 |

Concrete top50 moved ID/rank details (one-based ranks):

- Query 0: ID4634118 17→18; ID2739628 18→17; ID15460378 31→32; ID20217647 32→31.
- Query 1: ID18145899 20→21; ID1779007 21→20; ID17271032 34→35; ID6662564 35→34.
- Query 2: ID10215573 37→38; ID14351546 38→37; ID12333789 44→45; ID18947318 45→44.
- Query 3: ID4077806 21→22; ID8198272 22→21.
- Query 4: ID7233243 12→13; ID20480690 13→12; ID3040546 18→19; ID3040545 19→18; ID11003313 25→26; ID11483639 26→25; ID6750718 42→43; ID12979901 43→42; ID2281667 45→47; ID12585612 47→45.
- Query 5: ID8149979 32→33; ID3621777 33→32; ID56392 38→40; ID16983355 39→38; ID5936165 40→39; ID13027324 41→43; ID3621778 42→41; ID15091065 43→42.
- Query 6: ID8717833 24→25; ID16136916 25→24; ID6684948 47→48; ID15489871 48→47; ID10065990 49→50; ID14159814 50→49.
- Query 7: ID19595905 44→45; ID16723619 45→44.

The saved `evidence/job2860_retrieval_comparison.json` is the exact raw comparison file. `evidence/job2860_retrieval_independent_analysis.json` contains each moved ID, both ranks, both scores and the identical-content SHA, plus per-query score/inversion diagnostics. Timings in this single observation were CPU encoder1.256s, GPU encoder3.990s, CPU index load44.440s, CPU search8.359s, GPU clone11.726s and GPU search0.075s. Encoder timings include respective construction overhead; CPU/GPU search is a single eight-query probe and is not end-to-end service throughput.

## Required subsequent evidence

Require actual HTTP top3/top50 output to match CPU ordered content and scores before trainer launch. Then require five genuine outer updates with all8rank sample/supervision/log-prob/mask/loss/gradient/optimizer/frozen-teacher evidence, before/step5 model-only validation, checkpoint/save peak records and independent complete HF bf16/Flash CUDA reload in a fresh owned Slurm allocation. No part of this report accepts those stages prematurely.

The observed resource cgroup metrics point to `/sys/fs/cgroup/system.slice/slurmd.service`, with memory.max=`max`; these are service-scope metrics, not a dedicated job memory peak or its1T allocation limit. Exact-job PID records and GPU process membership were present. Report process RSS/HWM/ticks with deduplication and shared-page caveats, and retain the cgroup path rather than assigning all service memory to the OPSD job. NVML2s observations and per-rank allocator lifetime peaks remain different measures.

## Actual HTTP gate closure

**PASS:** downloaded the actual `retrieval-http.json` from job2860 and independently checked both returned arrays rather than relying on its booleans. Both calls contain exactly8queries and exactly3/50documents per row. Their ordered content SHA arrays are exactly the CPU comparison arrays. All scores are finite; maximum absolute deltas to the CPU reference are1.7881393432617188e-7 (top3) and2.384185791015625e-7 (top50). Observed HTTP durations are6.378656s and6.252410s for these8-query calls. Raw HTTP evidence SHA256 is `d2e42450412712a910da176f5919ef5f589524a7e5bec64936cbe20e9b174726`; files are `evidence/job2860_retrieval_http.json` and `evidence/job2860_http_independent_analysis.json`. The independent retrieval/HTTP gates are closed; training/five-step/reload acceptance remains pending.

## Actual trainer launch failure and recovery boundary

After `REAL_HTTP_CPU_RETRIEVAL_TOP3_TOP50_OK` and `CPU_RETRIEVER_OWN_PID_JOB_PORT_CONFIRMED`, the `ray start` CLI failed while importing `ray.scripts.scripts` and deep-copying command aliases: `ValueError: <object ...> is not a valid Sentinel`. The run `exit.txt` is1 and scheduler final state is FAILED/NonZeroExitCode, ExitCode1:0, runtime6m57s. No trainer log/outer update/checkpoint exists for this attempt. The job's inspected installed metadata is Ray2.40.0, Click8.5.0, Torch2.4.0, Transformers4.46.3.

The [official Ray issue56747](https://github.com/ray-project/ray/issues/56747) documents the same CLI deepcopy/Sentinel trace after a Click update and identifies8.2.1 as working. This supports a compatibility diagnosis, rather than a claim that retrieval or OPSD math failed. A limited repair is to pin only this task's Click to a compatible8.2.1/previous8.1.x and first verify the Ray CLI import/`ray --version`; preserve the training algorithm and Ray/Torch versions. The independent reviewer did not install packages, modify services or relaunch jobs.

A rerun must have a fresh Slurm job/run directory and retain job2860 raw evidence/real failed exit. Its comparison gate marker must bind its own job ID and current comparison SHA; old marker2860 is not authorization for a new job. Successful CLI import does not validate actor/FSDP training. Real five-outer/all-eight-rank/cost/checkpoint/reload acceptance remains outstanding. See `evidence/job2860_training_init_snapshot.txt` for the actual failure trace.
