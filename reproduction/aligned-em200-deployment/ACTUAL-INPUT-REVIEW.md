# Independent review of actual baseline-aligned EM200 inputs

Reviewer: `alignment_generation_review`. Date:2026-10-09 (Asia/Shanghai). Scope: read-only review of actual target CPU order/prepared-data receipts, downloaded original manifest bytes, deployed tool/source bindings and the derived-manifest creation request. This review launched no training and made no remote or production-source changes.

## Conclusion

**PASS for the supplied actual input artifact lineage and the newly captured common169028 CPU reference order.** The order is backed by execution of the actual archived dataset/DataLoader and observed emitted indices plus collated IDs. It is not merely a seed declaration. The derived data manifest imports the existing common prepared files without resplitting or removing training rows, binds the original user-pinned common manifest and actual observed-order file, and is consistent with the shared169028 input goal. This conclusion does not assert equality to the old169103 historical experiment's permutation, model-initialized RNG timeline, or execution. It does not authorize a training launch or replace retrieval/native first-five/final release gates.

## Reviewed actual artifacts and source binding

| Artifact | Independently checked SHA256 |
| --- | --- |
| Actual order receipt `work/evidence/aligned-em200-actual-baseline-order.actual.json` | `43294c7cc0b59859cf7819a7892bcb212c75b08d23f820759082d67ff5755c52` |
| Actual prepared-input receipt `work/evidence/aligned-em200-prepared-inputs.actual.json` | `04babf7fa8adbc13dd1126cc6b4ed3a419110bc3dbfb2f13e080cce0f0f787c3` |
| Raw order `work/evidence/aligned-em200-inputs-local/baseline-observed-order.json` | `ff2d15592968ceaf7667489d61b9172c2c53f81d82cb7f4ff10fd7aeb5285bc8` |
| Raw derived data `work/evidence/aligned-em200-inputs-local/baseline-aligned-data.json` | `e5e47bdfbfdffcc9a5cf0feccc39e52f32aa093fa039f7464e9baa5422d21d15` |
| Reviewed capture tool | `f17fd803d378c8bfdf9107360779e067dc52f5755891f01222694798f0f93994` |
| Actual derived creation request `work/aligned-em200-deployment-20261009/prepare-input-manifest.py` | `40d5d58150b5117f254155a1d0a04b744e6f1243ee7e91275e252d3be1f59cac` |

The raw order is9643856 bytes and derived manifest19762920 bytes. Both were independently decompressed from `aligned-em200-read-inputs.actual.json`, rehashed and compared byte-for-byte with their local copies; they equal the target receipt hashes. The transport wrapper stdout reproduces both summarized actual receipts with exit0. Source-transfer evidence binds deployed source `080ebbc46b89d61567cd33e1a5e2f1b40afd8b84`. `install-tool-4.actual.json` binds the deployed capture tool to the exact reviewed SHA above. Installed archive reference hashes and manifest `reference_source`/`reference_provenance_sha256` agree with the independently reviewed archive snapshots in `ORDER-REVIEW.md`.

The actual capture command used the new target environment, deployed capture tool, source080 checkout, existing common-manifest path, official local tokenizer/model path and new output path. It completed exit0 at2026-10-08T18:53:55UTC. The log reports169028 unchanged training rows,512 validation rows and330 train batches; its SHA is `d2056cfb9d4d0acf4cae1128984aade0afad0f0a31d4ad98537754715de162e6`. The manifest itself records capture tool/source provenance, native torch `2.4.0+cu121`, actual effective shuffle/batch512/drop-last/zero-worker loader settings and complete pre-iterator CPU RNG bytes.

## Actual order and ID checks executed locally

Using the existing `work/fresh200-local-test-env/bin/python` with production alignment source on PYTHONPATH, the reviewer executed the real `verify_order_manifest` and `RecordedOrderSampler` against the downloaded target manifests. All checks passed:

- The complete ordered169028 unique training IDs equal the derived manifest's training IDs exactly, not just as a set. Training file path/hash and semantic SHA also agree between the independently produced order and data manifests.
- All330 batch ordinals are sequential. Each batch has exactly its512 epoch indices, and its actual collated question IDs equal the original IDs at those indices. There are168960 unique emitted positions and68 dropped positions. This is a complete drop-last epoch, not a partial200-batch capture.
- The canonical order hash verifies as `21b1e94f7f6eaf097202e86f83bf202c46aca8e944f59224504bdb9c26df9481`. The actual first-batch hash independently recomputes to `72a4e7207b7ea04e73e9c4cf4a36a04e6b8a6c0469b1481b2bd477fea8ab26fb`, matching the prepared receipt.
- Full recorded pre-iterator RNG bytes rehash to `5feb54a46230d321d888e9b43cf5a57492665f199e4377408d79ee460e36679c`. The consumer emitted the exact first102400 recorded positions for outer1..200 and advanced its cursor accordingly.

This uses the archived native shuffle iterator and successful dataset `__getitem__` observations established by the reviewed tool, with actual collated IDs checked during capture. Default seed42 specifies the **new** CPU reference's actual iterator boundary; no inference about historical trainer initialization is made. No local permutation was substituted for the target's recorded vector, and cross-version RNG replay was not used as a shortcut.

## Derived manifest and common-goal lineage

The target creation request imports `train.parquet`, `validation.parquet`, `final.parquet` and `stage.parquet` from the existing common-data directory through source080 `prepared_manifest`, asserting rows169028/512/51713/64, disjoint stage policy, and the original manifest SHA `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`. It verifies the actual order against the imported training ID sequence before exclusive output creation. No splitting, row sampling, filtering or training subtraction occurs in that request. The raw derived manifest records `preparation=import-existing-shared-prepared-files`, original common path/SHA, actual order path/fileSHA, source080 and explicit `NEW_COMMON169028_REFERENCE_NOT_HISTORICAL169103` scope.

The train parquet SHA is `5d40180ba54d35513bb780408caa068d7d60123214a49b0739f835a2ae47c8a1`; its semantic SHA is `873ebec9a86778213fc526c5068f3398a4e395fa2a7fa254d855ee25c83cad89`. All split ID/prompt arrays have their declared unique row counts. Local independent set checks establish stage64 ID and prompt-hash disjointness from train169028/dev512/final51713, and train/dev ID disjointness. The derived original-common path/SHA exactly matches the order capture binding. Source080 production startup separately revalidates the pinned original manifest, original raw/prepared provenance and exact actual loader paths before admission; that enforcement remains necessary.

## Limits

The reviewer inspected target readback evidence and executed local manifest-byte/ID/consumer checks; the reviewer did not independently rerun the target capture or download and decode every original raw/parquet/model file. Semantic/file identities are target-produced values bound to the reviewed validator/capture execution and receipts; their local agreement is not a second full parquet reconstruction. The actual capture validates the user-pinned common-manifest lineage, while derived preparation and production admission preserve and revalidate it. Any future byte/path/source/config change requires the applicable admission checks again. This is a new jointly used169028 reference order and leaves all claims about the old169103 historical experiment expressly unproven.

Signed: `alignment_generation_review`, independent actual-input artifact reviewer.
