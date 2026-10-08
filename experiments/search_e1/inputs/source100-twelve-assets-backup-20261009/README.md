# Complete source100 asset copy

The source100 12-file manifest and model payload are distinct assets. This fixed
operation copies all 12 payload files from the already frozen HKUST1 source,
including the three full FP32 weight shards, to one lossless gzip USTAR archive.
The original 4797-byte manifest is retained separately with its exact SHA.
Existing local TRAIN/validation, runtime305 and common LoRA copies are reused.

The operation reuses published `fileproof`, `stable` and `stream_archive`:
inventory hashes must equal the original manifest; archive streaming hashes
fixed source handles; post-transfer metadata remains unchanged. A separate
Mac read checks the compressed archive, contiguous regular tar members, all
decompressed member hashes, terminal zero padding and gzip footer. USTAR is
the pinned producer format; individual member padding is not independently
checked. Rename continuity reuses the complete pre-rename archive hash without
a second large compressed read. Only then is the
partial archive renamed and a completion receipt written. No bare second copy
is made. Restoration needs at least the 13,604,379,671 member bytes plus margin,
an independent destination, and a fresh member SHA check after extraction.

There is no model/GPU/HTTP operation, Node write, signal, source deletion or
training wait. The original pipeline and protected assets remain running or
preserved. Source12 archival acceptance does not certify numerical Adam state,
future six groups, or the final 5632-result analysis.
