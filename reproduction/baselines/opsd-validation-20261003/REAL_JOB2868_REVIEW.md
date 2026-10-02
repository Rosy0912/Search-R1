# Real job 2868 — independent staged review

**Final job 2868 result: FAILED before training, zero outer updates. Preflight, current retrieval and actual HTTP passed. Ray component port overlap blocked launch. Five-step/checkpoint/HF reload acceptance remains NOT COMPLETE.** Source is `ec0808828d0c7ab02965895038bfb54ab7be6f16`. The exclusive h100-10 allocation has 8 H100, 192 CPUs available to the step and 1T memory, with a seven-day time cap. This job's actual log independently confirms fresh zero GPU occupancy, eight-rank NCCL and BF16 operations, Flash BF16 forward/backward, and real 3B vLLM CUDA generation. Teacher/retriever threads remain 8 and Ray CPU resources remain 80. Opening all already allocated CPUs to the step changes resource scheduling; this is not a learning outcome or a performance comparison to the earlier failed jobs.

Current comparison SHA256: `0d4ffa76ae414e95a090f808c520d3fe7ae990266c4130178ba5636a832e0faa`. All eight queries have exactly equal top3 IDs, content and order. Each query's top50 contains 50 unique IDs on each side with equal sets, and every matched-ID content SHA is equal. Scores are finite and sorted. Differences are only near-score reordering below top3, first appearing at rank 12, with a maximum displacement of two positions. The maximum same-ID score difference is 0.000479280948638916; the largest CPU score gap of an inverted pair is 0.0001233220100402832. This supports accepting this fixed-query retrieval probe, without asserting universal equivalence.

Actual raw data and joined per-ID/rank/score/content evidence are in `evidence/job2868_retrieval_comparison.json` and `job2868_retrieval_independent_analysis.json`. The reviewer did not write a marker. Authorization must bind job 2868 and this exact SHA.

| Query index | Exact top3 IDs | Top50 moved positions | First moved rank | Max displacement |
|---|---|---:|---:|---:|
| 0 | 12964449, 16754807, 6823959 | 4 | 17 | 1 |
| 1 | 17932575, 269947, 3539512 | 4 | 20 | 1 |
| 2 | 2244147, 6670881, 6680847 | 4 | 37 | 1 |
| 3 | 95403, 95404, 14417198 | 2 | 21 | 1 |
| 4 | 5674930, 20480681, 16538856 | 10 | 12 | 2 |
| 5 | 7100828, 18379889, 9720087 | 8 | 32 | 2 |
| 6 | 1039175, 9447685, 2543511 | 6 | 24 | 1 |
| 7 | 18990009, 18990010, 2182608 | 2 | 44 | 1 |

Actual HTTP must match this CPU reference before training. A successful Ray head start alone is not five-iteration acceptance. Require per-rank teacher freeze, optimizer, finite loss/gradients, masks, sample/token counts and cost evidence; actual outer updates 1–5; before/step5 validation; student change; checkpoint save; and independent HF BF16/Flash CUDA reload. Retain the service scope of memory-cgroup metrics, and archive actual `/tmp/o4-2868/ray` raw session/worker log contents rather than only the `run/runtime` symlink. Startup and scheduler evidence is in `evidence/job2868_startup_snapshot.txt`.

## Actual HTTP closure

This run's actual HTTP arrays were independently checked: exactly eight rows with exactly three/fifty documents each, per-position content SHA exactly matching this CPU reference, finite scores and maximum absolute deltas1.7881393432617188e-7 (top3) and2.384185791015625e-7 (top50). Raw SHA256 `a801b89f92a53660f0da6f2a9500d0faf2eefaee1836d1566cfbc4df177cea3b`; evidence `job2868_retrieval_http.json` and `job2868_http_independent_analysis.json`. Observed call durations were42.131734s/81.358799s, materially slower than earlier jobs' calls. These are this run's measured durations; do not substitute earlier6second results or attribute the slowdown to a specific cause without evidence.

## Actual Ray launch failure

The short runtime symlink is actually `/tmp/o4-2868`; the prior AF_UNIX failure was crossed. Ray failed before worker/trainer creation because selected GCS port16384 lies inside its default worker-port range10002–19999. This is an internal component-port overlap, not evidence of another user's process claiming the port. Actual stack: `Ray component worker_ports is trying to use a port number16384 that is used by other components`; the job run exit.txt is1 and srun exits1. No outer update or checkpoint exists. Full inspected trace is in `evidence/job2868_training_init_snapshot.txt`.

A minimal transport-only remedy is a supported explicit nonoverlapping worker range, for example30000–39999 via `--min-worker-port`/`--max-worker-port`, while retaining GCS16384/CPU retrieval18084 and algorithm settings. Both explicitly selected service ports then lie outside the worker pool. Merely moving GCS outside the default range leaves CPU retrieval18084 inside it. Actual new-run startup must still verify available ports/ownership and the fresh retrieval authorization; this reviewer did not change or start services. The real five-outer/all-rank/checkpoint/HF reload acceptance remains outstanding.
