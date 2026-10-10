# Historical TRAIN strict-C preflight V2

V2 supersedes the **unexecuted** full-JSON V1 candidate; both candidates remain
preserved.  No remote execution or GPU/model forward has been performed by the
implementation author.  The root must publish/review these exact bytes before
executing the fully explicit remote CLI.

Only actual committed TRAIN updates are selected.  The accepted metadata gold
audit must bind every selected step, attempt, probe version, native-gate SHA and
full-trace SHA.  Original TRAIN file bytes must match the run manifest.  The
original index hash uses canonical UTF-8 JSON, exactly as `RLDataset` and
`prepare_identity` did.  If `extra_info.index` is absent the original default0 is
preserved; missing hash joins or multiple distinct ground-truth records HOLD.
Duplicate index records are permitted only when original ground-truth bytes are
identical.  Dataset gold records are read in8192-row CPU batches.

Each large canonical trace is authenticated by a complete4MiB-buffer SHA stream.
The actor rows are the first canonical top-level field.  A64KiB incremental JSON
decoder then yields one row at a time, with an8MiB hard JSON-buffer limit.  It
does not materialize or decode the much larger trailing token/sampling journal.
The full native gate authenticates the trace's original run/source/step/snapshot/
attempt.  All2560 actor IDs, sibling/group hashes, masks and token counts must
match.  The streamed canonical actor-array SHA must equal the native gate; file
size/mtime/inode/device are checked again after reconstruction.

Each row is scored using only the original small tokenizer assets and frozen
mask-aware `decode_segments` / `contexts` / strict `qa_em.em_check`.  The original
`hf_tokenizer` is loaded directly from its pure utility file; training main is
never imported.  Transformers has Torch/TF/Flax backends explicitly disabled;
actual Torch import is a HOLD.  Tokenizer class, library versions and default
decoder flags are retained.  Model-weight tensors are not read.  There is no
generation, retrieval/HTTP, model inference, optimizer or production RNG restore.
Balance-order replay is not needed for C; it is a separate future shaped-R replay
requirement and is not implemented here.

Every exact hits/2560 count must match the original native three-decimal TRAIN EM
within `.0005001`.  Passing an aggregate check does not prove an original online
per-row reward record existed.  Outputs are labeled independent strict-C
reconstruction, not preserved historical online scores.  This does not compute
shaped R or Pearson.

Outputs use an explicit, fresh, independently owned Linux diagnostic directory
outside the original experiment.  Per-row IDs/C/hash records remain there; only
counts, scalar comparisons, hashes and paths reach stdout.  Each completed
iteration has an immutable `TRAIN-<step>-summary.json`, allowing the root to
verify two actual increases in completed iterations before ending its turn.
Partial diagnostics are preserved on HOLD and never overwrite earlier output.

CPU fixtures include a complete synthetic512×5 gate/actor-array chain plus
mutated run/source/step/snapshot/attempt/SHA/IDs/group/token bindings and bounded
streaming.  Synthetic fixture success is not real historical-input acceptance.
