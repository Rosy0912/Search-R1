# Activation tools v5 independent review

2026-10-04. Frozen candidate manifest SHA256: a3a58a5967deac7aa045953849af0fbbdec67da53cf471338556e5227e9c3af3, directory work/reward-protocol-fix/activation-tools-v5. All listed source/test/evidence file hashes independently rechecked. No frozen candidate or production source edited, no GPU/scheduler action, no commit/push/upload by reviewer.

## Decision and validation scope

The two v4 software blockers are resolved in this candidate. Accept default-disabled software curation, with the separate exact-source native CPU validation acknowledged below. This is not approval to remove guards, deploy, accept actual500/HF500, or claim an effect gain. Original guards remain exit78 and future job/proofs remain UNFILLED.

Independent six-module CPU run: **108 passed in 5.48 s**, using existing CPU environment, empty CUDA_VISIBLE_DEVICES, OMP_NUM_THREADS=1, required real DataProto and reviewed repository PYTHONPATH. Source/protocol/process/schema tests retain their declared synthetic scope; launcher cleanup tests below use actual CPU subprocesses. No production base-model shard or GPU checkpoint was loaded or measured by this reviewer.

## Base model ancestry fix

The retained training-assets-8e.json is byte-for-byte equal to `git -C work/opsd show 8e60976d364ab7d7ae51714415813c1359dafe9a:reproduction/h20/assets.json`; independent SHA256 a033b9d6e5b4453374c25d8d9a33d70254c1a79ea0d71677f21afb083d94b3fd matches the embedded pin. This is a historical source anchor, not a new receipt's description of current contents. It identifies Qwen/Qwen2.5-3B revision3aab1f1954e9cc14eb9509a215f9e5ca08227a9b and exactly nine original files.

base_asset_ancestry verifies original manifest SHA, exact model/revision/path/file schema, exact original nine role entries, resolved per-file containment, original size and streamed full content. The two model shards are checked against original official SHA256; the seven small files against original Git blob SHA1 plus size, using the same content identity calculation as frozen8e download_assets.valid(). It additionally computes SHA256 for every current file and requires the layered receipt pins to match. read_receipt now calls this check for the base role after current complete file-set checks. Rotating current receipt pins cannot remove the original content requirement. CPU tests cover changed equal-size contents under both original hash mechanisms, size/missing/extra files and manifest tampering.

This establishes the software's original asset-content anchor. It does not establish that the actual training3007 process loaded those bytes: the actual frozen-source/model-path/runtime receipt remains required separately. No original accepted runtime receipt is fabricated by adding this historical source manifest.

## Launcher failure fix

stream_child records its original exception and signals a failure event; it immediately terminates only its own Popen child. wait_child observes that event before waiting indefinitely for normal completion. Cleanup stop_own_child uses bounded terminate then kill and bounded wait; reader join is bounded and daemonized. The run-level error path preserves original stream/capture/checker failure and separately records cleanup failure without changing the first exception. New immutable reload.log creation still prevents overwriting existing evidence.

Four actual CPU subprocess tests passed: stdout forwarding failure, existing-log failure with a full-pipe writer that ignores SIGTERM, termination escalation, and successful byte-for-byte raw output preservation including NUL/non-UTF8 bytes. The cleanup tests demonstrate the former pipe deadlock is avoided without signalling unrelated PIDs. Tests use short timeout overrides; production defaults remain bounded five-second waits. Permanently uninterruptible OS process state is outside what CPU fixtures can prove and must not be relabelled successful cleanup.

## Retained gates and remaining work

v5 updates its helper/dependency/source manifest pins to include the historical asset verifier and manifest. Numerical checkpoint checker remains immutable5586; original formal500x8/periodic64/oldfinal51713/normal terminal gates remain unchanged. Canonical approved activation transforms, exact dependency closure/import paths, prescription -> layered receipt -> final-stage ordering and explicit final stage/receipt/role bindings remain as reviewed in v4. Seven sources may reside in one combined parquet; no seven-file condition was introduced.

Actual independent final-bundle guard-removal review, normal training500 terminal/final structure, own-node single-GPU HF500 full435/three-shard loading and real Linux lock/process/resources, actual base asset read, native eight-worker callback/device UUIDs, local GPU retrieval, complete native scoring-view replay, actual zero-update dispatch counts, paired base/HF metrics and final terminal/cleanup still require real evidence. Native CPU and original base-content evidence acknowledged below do not replace these gates.

## Native CPU and actual base-content acknowledgment

Read the root's retained v5-native-cpu-retry-result.txt: exact candidate a3a58a5967deac7aa045953849af0fbbdec67da53cf471338556e5227e9c3af3 passed **108 in 8.77 s**, native raw log SHA256 34deb7930919a4b2dc54d8e3f7e8161aa6c817c60c4c90ecd59e1401803ecc7b. The earlier 89 passes plus 19 fixture setup errors are retained; the successful retry corrected only the isolated review's missing sibling-repository symlink, with no frozen source-byte change. These are Linux CPU tests, not GPU/Slurm acceptance.

Read and independently hashed the root's actual-base-original-content-pins.json: SHA256 105c9616c25c8b574608fc0a05d467ce05efd7110ffdbddf945a816f42e52d5c, exact nine files in the own original base directory, historical assets pin a033b9d6e5b4453374c25d8d9a33d70254c1a79ea0d71677f21afb083d94b3fd, official revision3aab1f1954e9cc14eb9509a215f9e5ca08227a9b; both actual shard SHA256 values match the fixed historical origin. The root reports actual nine-file streamed original-content verification passed and parent source8e remained unchanged. This reviewer acknowledges that retained CPU evidence; this reviewer did not reread remote multi-gigabyte shards. It proves original file-content ancestry at that check, not a fresh model load, actual training500/HF500 completion, native worker loading, or GPU evaluation. Future stage gates must reverify current files against the same immutable original anchor.

The native results support curation of the default78 software preview and independent review records. No GPU activation or final effect acceptance is granted.
