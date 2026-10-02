# Real job2864 independent staged acceptance

**Final job2864 stage: FAILED before training, zero outer updates. Real GPU/model/retrieval/HTTP gates passed; Ray AF_UNIX path length blocked launch. Five iterations/HF reload NOT COMPLETE.** Fresh source commit `684b4671ce9dd6cf6c00f6552b8d9aa26c602d43`, exclusive h100-10 Slurm allocation8H100/96CPUsTask (192allocated)/1T. Job2860's failure/marker is not reused. The current raw comparison SHA256 is `fa004194874c162f5afd8c1d8ae481b080c555d29fbc6001e882cdab55e05ee6`. Reviewer did not write a marker or alter services.

All8fresh query top3IDs/content hashes/order are exactly equal between CPU and GPU. Every query contains50unique IDs on each side, with equal sets and equal content SHA for every concrete matching ID. AllCPU/GPU scores are finite and descending. Top50 differences are only close-score reordering, first appearing at rank12, maximum displacement2. Largest same-ID score difference is0.000479280948638916 and largest CPU score gap of an inverted pair0.0001233220100402832. This supports accepting the current retrieval adaptation probe, with exact top3 and quantified top50 ordering differences. This is not universal equivalence or downstream learning acceptance.

| Query index | Exact top3 IDs | Moved positions | First moved rank | Max displacement |
|---|---|---:|---:|---:|
| 0 | 12964449,16754807,6823959 | 4 | 17 | 1 |
| 1 | 17932575,269947,3539512 | 4 | 20 | 1 |
| 2 | 2244147,6670881,6680847 | 4 | 37 | 1 |
| 3 | 95403,95404,14417198 | 2 | 21 | 1 |
| 4 | 5674930,20480681,16538856 | 10 | 12 | 2 |
| 5 | 7100828,18379889,9720087 | 8 | 32 | 2 |
| 6 | 1039175,9447685,2543511 | 6 | 24 | 1 |
| 7 | 18990009,18990010,2182608 | 2 | 44 | 1 |

Every moved concrete ID/rank/score/content SHA and query text is preserved in `evidence/job2864_retrieval_independent_analysis.json`; the raw comparison is `evidence/job2864_retrieval_comparison.json`. Snapshot evidence is `evidence/job2864_startup_snapshot.txt`. The parent must bind its authorization to job2864 and this exact SHA. Subsequent actual HTTP must match CPU results, and subsequent five-step/all8rank/frozen teacher/student delta/resource/checkpoint/reload evidence must be independently reviewed. Service-scope cgroup memory metrics must retain their real path and cannot be attributed solely to this job.

## Actual HTTP closure and launch failure

The reviewer independently inspected this job's actual HTTP JSON: exactly8queries with3/50documents per row, ordered content hashes exact against its CPU comparison, finite scores with max deltas1.7881393432617188e-7 and2.384185791015625e-7. Durations6.526503s and6.625994s. Raw SHA `b703d564d65abe5406aa140f5a79666864364a706c23fdc710dc55a26e59c3c9`; evidence `job2864_retrieval_http.json` and `job2864_http_independent_analysis.json`.

After the actual HTTP and own PID/job/port markers, `ray start` reached Ray node construction but failed preparing its plasma-store socket: `AF_UNIX path length cannot exceed107bytes` for the141byte job-runtime path. The run real exit.txt is1 and srun reports exit1. No actual outer update ran. Full inspected stack is in `evidence/job2864_training_init_snapshot.txt`. This is a distinct infra failure after the Click fix, not an OPSD/retrieval mathematical failure. The separately reviewed shorter runtime remedy does not yet prove training runs. Preserve2864 evidence and use a new job-specific gate for retry.
