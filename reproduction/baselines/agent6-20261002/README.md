# Agent 6: CoSearch production acceptance

Frozen source: 0723dca; plan: 1620c2e. Reviewed common deployment source: 297e1c140677dc621302f1d49fefdb31023fd2b7, cherry-picked as cd86c9d.

Candidate host h100-16; own persistent root /dataset/shaoshitong/search-r1-rl/cosearch; retrieval localhost:18086; Ray localhost:16386. A short Slurm allocation probe succeeded as job 2843 with eight visible devices and then exited. This is not a continuing reservation or GPU algorithm acceptance. Original LTX-2 low-memory processes remained untouched.

Agent-specific changes reject fewer/more than 50 retrieval candidates or malformed document content; record sampled legal/fallback ranking counts, maximum prompt length, ranker CPU/wall times, driver process peak RSS, informative-group ranking diversity, gradient norm and bounded parameter-sample delta; append costs after validation including the terminal evaluation. Diagnostics do not alter ranker reward, query grouping, sampling, clipping or optimizer settings. Explicit weight_decay=.01 preserves the previous implicit default. Parameter delta alone is not RL learning evidence: AdamW, old momentum and roundoff can cause changes. The controlled optimizer regression now uses distinct rankings and weight_decay=0, with a substantial gradient assertion.

Real-model generation probe: `python reproduction/baselines/cosearch_probe.py --model "$COSEARCH_RANKER" --train <98-percent-train.parquet> --url http://127.0.0.1:18086/retrieve --output <new-report.json>`. It keeps all 50 documents, max_new_tokens256, seed42 and CPU FP32. It reads no answers into ranking prompts and does not manufacture actor rewards or claim joint training. Context overflow stops without truncation.

Required sequence: assets/env hashes; reviewed CPU versus original retrieval comparison; Slurm eight-GPU communication/inference; real 50-document/base ranker capability; two actual joint outer updates with effective different ranking rewards; five-update save/load; formal 500 updates; 51,713-question final F1/EM/format metrics and costs compared with the new common GRPO. Use smoke_only=true only for short acceptance. No exact joint optimizer/RNG/dataloader resume claim.

Initial CPU source gate: 134 passed, 1 Linux procfs skip on Mac. After common deployment plus diagnostics: 140 passed, 1 skipped. CPU tests and independent controlled probes are not production effect verification. Real-model/real-retrieval/joint GPU acceptance is still pending while pinned assets and the isolated Linux environment are prepared.
