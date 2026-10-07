# Native512 comparison and recovery evidence boundaries

No real OPSD updates or native512 baseline/post scores exist at this review. Preparation3 continues; all findings below are code, dependency or CPU checks, not new GPU evaluation results. No live code, configuration or process was changed.

## Which validation comparison is valid

The native OPSD baseline and post-epoch evaluation use the same512 questions, same HF/SDPA BF16 model wrapper, same strict answer scoring, same segmented retrieval endpoint and the same greedy generation settings. Each rank evaluates64 questions. The baseline adapter has zero LoRA_B; the post-epoch adapter is trained. Generation retains full context until its8192-token limit, allows5 turns/4 searches,500 new tokens per turn, top3 retrieval, observation body480 tokens and a final-turn budget prompt. Both inherit source100 repetition penalty1.05. The retrieval encoder truncates each query to256 tokens, a distinct limit from the model context and per-turn generation budget.

Use the native post-epoch raw EM minus its native baseline EM to describe validation change. Also report the selected-best EM separately: best_adapter begins at baseline and is replaced only by a strictly better post-epoch EM. Selected validation EM cannot establish improvement by itself. An independent matched final-test comparison is still required. Actual evaluation-time service identities and score evidence remain to be recorded when those evaluations run.

The native JSONL and archive validation JSONL are byte-identical; a CPU read of actual parquet verifies all512 IDs, answers, sources and questions match, with256 NQ and256 HotpotQA. [Dataset identity evidence](receipts/native512-archive-dataset-identity-20261007.json).

## Why archive step100 EM is not the native baseline

The archive trainer requests do_sample=False, but the actual active-batch reconstruction drops meta_info. Both ordinary and final generation, with and without padding, therefore reach the vLLM default do_sample=True branch. A CPU probe with the actual frozen AST/DataProto verifies all four paths lose the flag. Existing initialization kwargs log temperature1.0/top_p1.0/max_tokens500, supporting the source-path inference; it is not a per-validation effective sampling-parameter trace. [CPU metadata evidence](receipts/archive-validation-cpu-meta-probe-20261007.json).

Archive and native also differ in inference backend, repetition penalty1.0 versus1.05, rolling/truncation behavior, observation formatting, final-turn prompt and invalid-action handling. Current strict scoring is present in both; the legacy diagnostic is not the current archive's main reward. The old archive229/512=44.7265625% is its GRPO validation under those conditions, not the pending native512 baseline or an OPSD result. Original compressed-package historical step100 scores describe yet another run.

[Full independent validation review](receipts/opsd-native512-validation-review-20261007.md), SHA256 `1bb1a00c09c898109cd87a658a3967c09ebe49b7dfe005cb48b7172c7314e90e`. Complete read-only source copies remain locally in `/Users/shaoshitong/Documents/Codex/2026-10-03/xia/analysis-output/opsd-validation-review-20261007/`.

## Actual checkpoint and possible future recovery

The frozen OPSD trainer has no native resume entry. At update1/5 it saves one shared adapter plus eight rank Adam states and Torch CPU/current-GPU RNG. It does not save Python RNG, data cursor/order digest, early baseline EM/history or all-rank loss accumulation. The multi-file save is not atomic. Optimizer.step precedes saving, so a partial save failure cannot be called zero actual updates. Epoch adapter alone contains no Adam.

If a real future failure occurs, preserve it first; select only a complete CPU-verified adapter/eight-state bundle, bind the same full pairs/order, source100 and baseline selection, and use a separately audited resume copy/output with explicit provenance. The present V3 is hard-bound to the original output and continuous_run=True; its receipt cannot certify a different resumed run. This design is not a performed recovery, a reason to restart the healthy current pipeline or five-update acceptance.

[Full recovery design](receipts/preparation-3-future-opsd-recovery-feasibility-readonly-1791367941.md), SHA256 `b3fb9c03c2e2c994cc9312539cad53e6565d677fc14839b22592404d7e21fdce`; [read-only provenance](receipts/preparation-3-future-opsd-recovery-provenance-1791367941.json), SHA256 `5f8163d22c256e3165b14cdfd2e3b31ee49f24f1aa57cb86be312c8813c627ee`.

## Actual split-content audit

The actual fixed train169103, selection validation512 and test51713 files match preparation3 configuration hashes. Exact `(data_source, SHA256(question as stored))` intersections are zero for every split pair. NFKC/casefold/whitespace normalization retaining punctuation, and supplementary source-independent text equality, also give zero. All512 selection questions are absent from the fixed test. This does not exclude semantic rewrites, repeated answer facts or pretraining contamination.

IDs are local `source:split:index` enumerations. Actual preparation holds256 questions from each NQ/Hotpot training pool for selection validation and uses test/dev splits for final testing. Exact duplicate question rows within train/validation/test are7/0/1204, normalized8/0/1207; the fixed benchmark retains them without deduplication or denominator changes. Selection exclusion is unnecessary for this version because matching test rows are zero. Actual final-test scores remain pending.

[Content evidence](receipts/prep3-dataset-overlap-20261007.json), SHA256 `478aa8e183c44b39eefcefd12b18a8f3a2909b580d01e75b04c2369c3e6aa580`; [independent report](receipts/prep3-dataset-overlap-review-20261007.md), SHA256 `fd084bff8ce161fc283a68e729d8abf8e03505e8ec8f827c0e4f8be19bc7c671`. Original source/report copies stay in `/Users/shaoshitong/Documents/Codex/2026-10-03/xia/analysis-output/prep3-dataset-overlap-20261007/`. No full questions/answers were published; no GPU/HTTP inference, data edit or live-run change occurred.
