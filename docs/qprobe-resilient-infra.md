# QProbe: resilient interactions and lean execution

This branch extends the A800 micro4 source f7d9f1488d3c1373098b9401463812b24fe94b63. It does not restart or hot-edit an existing run.

## Implemented

- Common8 treats max_prompt_length (4096) as the total actual sequence budget. The original 500-token action sampler is preserved. Before another search, the controller reserves one final500-token generation; long rows enter the existing no-search final stage early. Other rows may continue searching.
- Environment observations are clipped before use, up to the available budget and max_obs_length. A truncated information observation keeps its opening/closing wrapper. Model tokens and their actual sampled log-probabilities are never rewritten. The exact context remains traceable.
- The sampling/label contract and frozen-teacher cache bind the new context policy. The default teacher cache uses a new resilient4096-v3 directory; old caches remain intact. Existing question-only features can be reused.
- Recoverable guards snapshot only rank-local parameter shards, Adam state and RNG immediately before each write. A rejected post-update write is rolled back, including the initially empty Adam case. Earlier accepted writes remain; the rejected rollout is abandoned and fresh data is collected.
- A safe outer boundary can contain partial progress. Receipts explicitly report partial, actual accepted writes, physical calls and rollbacks. Only20 accepted writes constitute full_budget_complete. Warmup/scheduler and head advance once at a boundary with safe progress; zero-write attempts do not advance.
- Common8 use_kl_loss=true with coefficient0 skips the otherwise unused reference forward. The fixed reference identity remains bound; no synthetic reference probabilities are produced. With a nonzero coefficient the original reference path remains.
- Entropy coefficient0 skips full-vocabulary entropy computation. entropy_diagnostic_computed distinguishes an omitted diagnostic from a measured zero.
- FP32 full-vocabulary selected log-probabilities use a bounded-scratch first-order VJP. SEARCH_G1_LOGPROB_CHUNK_ROWS defaults128, capped by an80MiB scratch budget. This reduces peak intermediates; it is not a measured GPU speedup and does not remove the dense logits gradient.
- Head checkpoint receipts expose bytes and sha256 as required by the existing save consumer.

## Execution and compatibility

No new GPU admission, benchmark or acceptance job is added. Only small CPU checks and syntax checks were used. The local environment lacks tensordict, so the full DataProto controller integration check was not run locally. Existing source-delta metadata is refreshed from real Git objects instead of requiring another calibration job. No weights, raw trajectories, credentials or environment packages are committed.

This is a source revision, not a guarantee that every hardware exception is recoverable. Fatal device/communication errors still require checkpoint recovery. It does not deploy or stop another session's training.

## Measured baseline costs

Rosy public outer6–8 average56.53min on8 L40S48GB, FP32 padded execution: candidate selection plus2560 actual agentic trajectories5.84min; head fit0.195s; remaining50.69min includes probability/reward/update/records, not actor alone.

Our previous A800 four-card micro2 run: one full outer46.06min; rollout5.63min, reference5.75min, reward/advantage3.95min, actor29.09min, head0.388s. The micro4 successor has no measured full-outer timing in this audit.

These are observed runs with different devices/precision; no normalized GPU benchmark or acceleration multiplier is claimed. Further priorities are variable-length actor/ref forwards, question-prefix reward-probe deduplication, reusing a successful post-monitor as the next pre-monitor when its model/sample identity is unchanged, and asynchronous retrieval rollout. Global mini-batch and loss reduction semantics must remain explicit when physical micro-batch sizes change.

## Other sessions: measured ordinary training outers

TOP-D outers198/199 on4 HPC3 80GB cards take6.86/6.96min, including about182s generation/old probabilities,134s live32B teacher scoring/advantages and82.6s actor update. BF16 compute/FP32 master, varlen FlashAttention, micro4/rank and20 actual Adam writes/rank. The first measured micro has3902 actual tokens versus7836 padded slots. Teacher scoring is a batched vLLM TP4 prefill, not a cached teacher trajectory substitute. No separate RefPolicy forward or per-minibatch qprobe safety monitor is used.

CoSearch outers197–199 on4 HPC3 80GB cards average19.56min: rollout10.52min, main old probabilities27.73s, main20 updates103.64s, reranker old probabilities59.62s and reranker updates5.44min. Main and reranker updates are serial on the same cards. Its rollout agent loops are asynchronous and its BF16/FP32-master forwards use varlen FlashAttention. Reward scores are produced inside rollout, so the near-zero trainer reward timer only measures retrieval of existing scores.

These measurements exclude checkpoint/evaluation rounds. Both sessions completed200 training outers but failed in their subsequent final evaluation; their ordinary training timing is still observable. Exact CoSearch GPU product was not independently recovered;80GB memory is evidenced. Historical TOP-D hardware deployment names H100, while available stage-matched snapshots expose memory/counter data without an independent product-name readback.

The new question-head update is subsecond; it cannot explain actor29min versus83–104s. The larger differences are padding/variable-length handling, physical micro grouping, GPU generation and sampling precision, redundant reference/monitor forwards, and whole-vocabulary intermediates. Question diversity may also select longer/harder interactions, but that indirect effect has not been isolated. Do not claim a speed multiplier for any single fix without actual training logs.
