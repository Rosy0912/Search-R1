# Independent review of archive GPU retrieval golden tool

Reviewer: `alignment_generation_review`. Date:2026-10-09 (Asia/Shanghai). Scope: local source-only review of the unexecuted `capture_retrieval_golden.py`, bundled archive server/provenance and source080 `baseline_alignment_retrieval.validate_retriever_identity`. No deployment, GPU allocation, retrieval execution, service launch or training was performed.

## Result: source-review issues resolved

**PASS for the latest source-review scope only.** Latest independently reviewed collector SHA256: `67c888845d5c7708a239d508fcd7c697f27b8e4c4a495ab89027cd5b17b8b9fa`. No remaining blocking source defect was identified in this bounded review. The collector has not been deployed or executed on GPUs; this is not a runtime/live-service/training acceptance receipt.

Two additional must-fix issues reported during review are now resolved:

1. Mandatory guards originally used `assert` without excluding optimized execution. The latest `main` explicitly raises `RuntimeError` on `sys.flags.optimize` or a `PYTHONOPTIMIZE` environment setting before arguments, claims or GPU action. Thus optimization cannot remove admission checks silently.
2. Probe provenance originally trusted any512-row parquet at the common path. The latest collector pins the actual derived prepared-manifest file SHA `e5e47bdfbfdffcc9a5cf0feccc39e52f32aa093fa039f7464e9baa5422d21d15`, imports/locates the exact source080 data helper and compares actual `dataset_identity(validation.parquet)` to the complete frozen validation identity (file byte hash, row count, semantics, original IDs and prompt hashes). It requires twelve distinct nonempty queries, records each original question ID/row/query, the actual validation-file SHA and prepared-manifest SHA, and rechecks validation bytes after capture. This resolves the changed512-row parquet provenance gap without resplitting.

The fixes were independently reread in source. The earlier actual-input review established the pinned derived file's common169028 provenance. No execution authorization is supplied by this review; real capture and live validation remain separate requirements.

## Issues already corrected during review

The initial version computed all12 queries in one FP16 encoder batch while the production validator calls each probe as a singleton. Padding/kernel geometry can differ and invalidate exact ordered-content comparison. The latest version now calls archive `batch_search([query], num=3, return_score=True)` per probe and records `query_batch_size=1`, matching the consumer.

The initial version checked source080's git HEAD while Python imports could resolve another checkout/shared installed package. The latest version checks the new environment prefix, places the source080 checkout first on `sys.path` and verifies actual imported preflight/identity module files.

The initial version accepted any canonical child RUN and absent output. The latest version constrains the own `hardware-aligned-em200-...-jobid` run name, verifies the live same-run resource guard and inherited GPU leases, and exclusively writes `GPU-GOLDEN-CAPTURE-CLAIM.json` before importing/initializing GPU retrieval. The claim binds source/run/job/PID/start ticks/collector SHA and rejects a repeated capture in that RUN. No claim or receipt from another run is admitted. Optimized execution is now explicitly rejected, resolving that bypass.

## Actual archive and parameter parity

The bundled `reference/retrieval_server.py` exactly equals the user-referenced archive file `/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/search-e1-publish/experiments/search_e1/archive/upstream/Search-R1/search_r1/search/retrieval_server.py`. Independently computed SHA: `e85ab42b9032d66a35239a36e8cd73c6f3e3efde929541150fe7c5bfdb4936d9`. `retrieval-provenance.json` SHA: `20e8c73053014e729c92fe242e47eaf36f5e13f77658034a1d785bec434c8c7c`; its archive hash agrees with the collector's hard pin.

`runpy.run_path` loads archive definitions without running its `__main__` HTTP server. Config matches the actual archive CLI main: E5, topk3, mean pooling, query prefix `query: `, max query length256, fp16 encoder, batch limit512, GPU FAISS cloning with float16 and sharding. The original DenseRetriever performs CUDA encoder inference and FAISS search; there is no CPU fallback path. Exact CUDA availability, eight torch/FAISS visible GPUs, loaded encoder parameter device/dtype and all eight actual FAISS child devices are inspected. Source inspection confirmed archive clone flags are fixed true in the pinned unchanged implementation. The collector does not instantiate the production HTTP service.

## Manifest/consumer compatibility

The collector enumerates complete local encoder files plus corpus, index, production service source and production identity source; it hashes each asset with streaming SHA256 and checks identity stats before/after capture. Corpus/index must be on node-local ext4/xfs. The produced fields satisfy the production validator's schema/version/topk3/max_obs500/format/truncation requirements, its full required runtime key set and ordered contents probes. Production source hashes are pinned as assets, while archive source is separately bound in provenance; these serve different roles.

Passage formatting now follows the consumer exactly, including contents with no newline or multiple newlines. Observation wrapping, single-observation tokenizer call, no added special tokens and head500 IDs match production validation. The manifest pins ordered three document contents, exact formatted passage string and exact truncated observation IDs for each probe. It does not claim scores alone prove content parity. The consumer still must independently hash actual assets, query the live loopback `/identity`, verify loaded/current assets and actual CUDA/FAISS runtime, and retrieve each singleton probe to compare ordered contents and tokenized observations. Collector output cannot replace those live checks.

## Evidence and limits

Performed: direct byte equality and SHA comparison to actual archive, AST/syntax parsing of collector/reference, line-by-line archive main/encoder/FAISS/HTTP geometry comparison, manifest versus production consumer schema review, and preflight lease/resource-watch source review. Additionally, two dependency-free edge fixtures executing only the extracted collector/consumer formatting functions matched exactly for missing-newline, multiline, empty-title and trailing-newline contents. No collector main or GPU code was executed; no end-to-end mocked capture/consumer test suite was supplied at this snapshot. Earlier order-tool CPU tests do not test this collector.

Any future successful artifact is an **archive GPU golden capture**, not live production-service acceptance, training GPU acceptance, first-five approval or experiment completion. Actual deployment still requires independently bound final source/tool/runtime/assets/RUN/UUID lease/resource watch, real eight-GPU capture evidence, and separate live-service validator evidence. This review authorizes none of those actions.

Signed: `alignment_generation_review`, independent source reviewer; identified must-fix items resolved for the latest SHA above. No runtime acceptance is claimed.
