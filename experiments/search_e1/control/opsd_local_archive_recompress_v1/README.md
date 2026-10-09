# Retained old protected tar → exact-byte XZ

The independent Mac operation fixes only the already accepted old online
protected0/1/5/11 archive: original run `1d1d67b60f0e46a2afc0536a36ab69b1`,
6,239,805,440 bytes, whole SHA
`0646f2da8b338108f753f880b30859a96831681cc99767cc6c47a50da93e9dd1`.
The existing accepted receipt SHA is `0169e6b4…`. Its58 GNU-format member proofs
remain historical evidence. Previously reclaimed loose duplicates are not
opened, restored or reclaimed again. CP89 is outside this version's scope.

This performs no deletion, extraction, Node operation, model/Torch import,
HTTP/GPU task, training lock, signal or Git operation. Backups never hold the
seven-group GPU pipeline. The original tar, receipt, inventory and all Node
assets remain. This does not change or recertify any Adam/model numerics.

## Minimal design

Python's installed standard `lzma` streams one XZ/CRC64 object with a128MiB
dictionary and preset6. It uses4MiB chunks. A fixed regular, same-owner,
single-link source handle is SHA-read during compression; full original
metadata is checked before/after. The writer has a hard bound of raw size +1%
+1MiB, with an additional fresh5GiB free-space margin. A lack of space is a
backup BACKLOG, never a training error. Source-read and metadata checks do not
prevent a different actor from mutating the same file between checks.

After the compressed file is closed/fsynced, a separate disk read hashes its
actual XZ bytes and independently streams decoded bytes into the whole-tar SHA
without writing a6GB recovery copy. Decode output is bounded by the original
byte count; incomplete CRC/footer, trailing bytes and concatenated streams are
rejected. Exact decoded original SHA/bytes preserve all original GNU headers,
padding and member bytes; the58 member hashes need not be repeated. This is
whole-byte recovery evidence, not a fresh archive member parser or numeric audit.

The parent records the real child return code and its stdout/stderr SHA. Only a
successful child, accepted disk-decode proof, and unchanged original metadata
allow the `.partial`→final XZ rename and immutable completion receipt. New0700
destinations are required. Failures retain partials and real terminal evidence;
there is no automatic retry or deletion. Repeated executions need new attempts.

`lzma` is preferred to a separate `xz` process for portability and bounded
proof collection. Native `xz` is installed locally but is not invoked here.
Compressing6GB may take minutes and should run in the background. Peak encoder
memory is larger than the128MiB dictionary (liblzma's encoder state); the decoder
memory limit is256MiB. No wall-time or compression-ratio promise is made.

## Published invocation

Root must first publish/read back all payloads and the source manifest. Small
source/publication files are checked against branch`search-e1`, exact commit,
manifest SHA and each remote byte-readback row before execution.

```text
python3 -B /PUBLISHED/opsd_local_archive_recompress_v1/recompress.py
  --source-manifest-sha256 EXACT_MANIFEST_SHA
  --publication /PRIVATE/publication.json --publication-sha256 EXACT_SHA
  --source-commit EXACT_PUBLISHED_40_HEX_COMMIT
  --output /NEW/PRIVATE/protected-gnu-exact-xz-attempt
```

Root preserves this top-level invocation's actual return code separately. The
operation also saves the real child code in `actual-child-terminal.json`, and
`complete-xz-verified.json` binds final compressed SHA/metadata, original
SHA/metadata, child proof, historical receipt and publication/source identities.
`--worker-only` is the identical internal child command; it is not a separate
production entry or a success receipt for deletion.

Only later, after accepted evidence and fresh exact identity/no-open checks,
Root may separately authorize reclaiming the one original proven duplicate tar.
This helper has no deletion API. It must not be mistaken for a compact queue
format conversion or a reason to re-transfer old checkpoints.

## One tiny CPU check

```text
python3 -B -m unittest discover -s /PUBLISHED/opsd_local_archive_recompress_v1 -p 'test_*.py' -v
```

The tiny arbitrary-byte roundtrip/truncation check proves the streaming codec
contract and failure preservation only. It does not read or compress the actual
6GB protected tar, run a model, or claim production compression acceptance.
