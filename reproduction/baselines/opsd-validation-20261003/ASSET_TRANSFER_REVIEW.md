# Independent asset transfer review — 2026-10-03

## Conclusion

The H20 cache is a verified read-only source for all three missing fixed Wiki assets. Current full-file SHA256 measurements match the local `reproduction/h20/assets.json`, not merely an old success marker. Reusing these files does not reuse another job allocation or retrieval service. The reviewer did not modify H20 files/services, start a downloader, or alter original aria2 partials.

| H20 source | Observed bytes | Current SHA256 |
|---|---:|---|
| `/data/search-r1/assets/wiki/wiki-18.jsonl.gz` | 5,123,307,260 | `7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db` |
| `/data/search-r1/assets/wiki/part_aa` | 42,949,672,960 | `a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023` |
| `/data/search-r1/assets/wiki/part_ab` | 21,609,402,413 | `b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11` |

Index revision is `a4d31160a035f30764604f4827cd8f1d0315eb86`; Wiki revision is `69c1c00ffe7c5554c68d8548355cb22e46aabc51`. Remote manifest size/revision/SHA fields matched the local fixed manifest. The historical `/data/search-r1/logs/asset-verification.json` also reported 30 assets and 30 verified content hashes with no failures; current full hashes above are the stronger source identity evidence. See `evidence/source_asset_verification.json` and `evidence/source_index_sha256.txt`.

## Read-only route and bounded relay

Use the existing host alias and trust configuration, without credentials or configuration rewrites:

```sh
ssh -T -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes jump_h20-1 '<read-only command>'
```

For zero-based chunk number `k`, a 256 MiB source block uses `dd if=<source> bs=4M skip=<k*64> count=64 status=none`. Substitute the actual integer and only the manifest-approved absolute source path. Check SSH exit status and exact expected bytes (including the final shorter block); a truncated successful local read must not advance the relay cursor. Hash each local block and compare target appended bytes before advancing a durable cursor. A reconnect must verify the target cursor/file-size consistency to prevent duplicated or missing data. After all chunks, independently verify destination full size and manifest LFS SHA before atomic rename into the final asset path. A cursor or per-block agreement alone cannot replace full-file verification.

The parent has selected a separate target `.relay.partial` path and will preserve stopped original download partials. Stage only this task's small bounded blocks under `remote_codex_control/.transfer-staging`, verify all intended target copies, then remove only this task's verified local transit blocks. This follows the user download-transfer memory; failed/unverified targets require preserving the needed source or transit copy. Remote source hashes are already verified, so rereading source hashes for every block is unnecessary; the receiving block and final-file hashes remain required.

## Failure diagnosis and alternative cache inspection

Reported failures are curl HTTP/2 exit 92 and HTTP/1.1 exit 56 through the existing 3128 proxy, plus aria2 exit 1 after partial progress. These establish transport interruption, not a wrong expected SHA or a proven corrupt source. Long-lived proxy/CDN connections or ranged concurrency are plausible causes, but the captured evidence does not identify the precise failing hop. The original download wrapper discarded stderr. The current diagnostic uses captured subprocess output and does not expose it until process completion; its inspected log still only contained `START wiki/part_aa 42949672960`. Avoid claiming that this alone proves aria2 made no progress.

At initial inspection, `part_aa.partial` appeared 36G by logical file length but occupied approximately 3.1G, and `part_ab.partial` appeared 18G but occupied approximately 1.5G. Aria2 sparse allocation means logical length is not downloaded-byte progress. Gzip partial occupied approximately 1.4G. The verified H20 source therefore offers a concrete fallback independent of these unstable public transfers. No measured relay throughput was obtained by this reviewer, so speed improvement is a route-based expectation, not a benchmark.

The read-only h1008 check found an `assets-root` directory but did not expose a verified complete Wiki cache at the inspected shared paths. A second small H20 metadata SSH request timed out during banner exchange while the original hash connection continued and finished successfully; this is not evidence that the authenticated H20 source or VPN is unusable. No alternate allocation/service was used.

## Acceptance boundary

Source identity is **PASS** for the three files above. Destination transfer, assembled FAISS index, materialized corpus, actual retrieval comparison and authorized retrieval gate remain separate runtime checks. This report does not attest to any GPU launch, 3B execution, five iterations, validation improvement, checkpoint reload, or OPSD effectiveness. Those require the real production evidence described in the preflight/runtime reviews.

## Incremental sparse-resume fix review

**Static/CPU result: PASS; no deployment blocker found.** Reviewed the working-tree delta to `reproduction/h20/download_assets.py` and `tests/reproduction/test_runtime_adaptations.py`. The earlier pre-download full-logical-size branch could hash and quarantine an unfinished aria2 sparse file even while its progress bitmap existed. This is a reproducible resume defect; it should not be confused with the separately observed curl transport errors.

The fix chooses aria2 for assets strictly larger than 64 MiB and preserves the `.partial`/`.partial.aria2` pair when such an asset has a sidecar. Exactly 64 MiB still uses curl. The old completed, corrupt small curl partial with stale sidecar remains preserved under rejected names before a fresh attempt. Changing the size threshold changes transport parallelism only: fixed repo/revision/size/content hashes remain unchanged, and both completed subprocesses and preexisting final files must satisfy immutable content validation. File presence, sparse logical size, a bitmap, and return code zero do not authorize final promotion.

Independent regression run: **10 passed, 1 skipped** (`test_runtime_adaptations.py`, missing Linux `/proc` on the Mac). Additional reviewer probe exercised actual `truncate(65 MiB)` sparse files with **zero allocated bytes**, rather than a mocked file size. It confirmed: (1) interrupted aria2 retains the bitmap/file pair without `.rejected` moves; (2) return code zero with wrong upstream SHA does not promote; (3) correct full size and SHA permits promotion; (4) the exact 64 MiB threshold selects curl. These use mocked subprocess responses and therefore do not prove real aria2 parsing or remote transfer throughput. Evidence: `evidence/download_sparse_probe.py`, `evidence/download_sparse_probe.txt`, and `evidence/download_sparse_review_manifest.json`.

The new repository sparse regression checks retention on interruption; the independent extra probe covers the final promotion gate. Its bitmap is a fixture, so an invalid or stale real bitmap can still make aria2 fail until deliberately diagnosed; the final content hash prevents unsafe asset acceptance. Preserve both original progress files during this task's process restart and avoid concurrent writers to the same destination.

The parent subsequently identified diagnostic exit 7 as the intentional `--stop=600` window and observed roughly 5.9 MiB/s for the actual AA retry. This is parent runtime evidence, not a new independent throughput measurement; it supersedes interpreting that exit as another network failure. A remaining nonblocking existing log label reports all subprocess failures as `curl_exit`, including aria2. Read the selected command/actual diagnostic output when attributing errors. The repair does not modify OPSD math, gradients, retrieval content, or the real five-iteration acceptance requirements.
