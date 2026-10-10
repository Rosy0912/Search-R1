# Historical TRAIN strict-C CPU preflight

Prepared independently; **not deployed or remotely executed**.  This reconstructs
the strict binary C for each committed TRAIN trajectory, not shaped R and not
Pearson.  Earlier iteration-mean/calibration coefficients remain withdrawn as an
answer to the full historical TRAIN correlation request.

Entry: `reconstruct_train_raw_c_preflight.py`.  All required paths and hashes are
explicit CLI arguments.  An accepted metadata gold-join audit is a prerequisite.
The root must publish/review the helper first and later bind the actual remote
Linux hostname, immutable run/config/audit bytes, and a fresh diagnostic directory.
The current implementation intentionally cannot run its production path on Mac.

Authentication sequence:

1. Verify original run/config, current scoring source hashes, original TRAIN file
   bytes versus run manifest, and original small tokenizer assets.  No model
   weight tensor is read.  Mapping uses exactly SHA(canonical `extra_info.index`)
   as `prepare_identity`/`RLDataset` did; absent keys or multiple distinct original
   ground-truth records cause HOLD.  Duplicate dataset rows with the same index
   and identical ground-truth bytes are allowed and counted.
2. Select only consecutive actual committed update-ledger rows, not skipped
   attempts, calibration rows, validation or current in-flight generation.
   Fresh full SHA of every selected native gate and raw trace must equal the
   committed ledger.  Verify gate accepted/2560 counts, run/source/step/snapshot/
   attempt identity, full actor-row SHA and unique parent/sibling identities.
3. Reproduce original prompt/response/mask slices; call the frozen
   `decode_segments`, `contexts`, `qa_em.em_check` and original `hf_tokenizer`.
   Transformers has every DL backend explicitly disabled; importing Torch is a
   hard HOLD.  No import of training main, model load, forward, generation,
   retriever/HTTP call, GPU or production RNG restoration occurs.
4. Persist each trajectory's ID/hash/C validity and search count only in the
   fresh remote diagnostic directory.  Stdout contains aggregate counts, scalar
   comparisons and SHA/path evidence.  Exact hits/2560 must match native TRAIN
   `train/perf/em` within native three-decimal rounding tolerance `.0005001`.
   Failure preserves partial diagnostic outputs, never overwrites or restarts
   the original training.

The original per-row C was not saved as an online scorer record.  Outputs are
explicitly labeled independent historical strict-C reconstruction.  Passing this
preflight authenticates all C inputs for a future historical-version R replay;
it neither authenticates probe inference nor computes the requested correlation.
That later replay requires separate resource/version/math admission and must
retain the same trajectory IDs, mask-aware contexts and FP32 reward storage.

Small synthetic fixtures use a character decoder and original dependency-free
strict scorer.  They cover alias normalization, protocol invalidity, observation
exclusion, final-action boundaries, masks, unique gold joins and EM rounding;
they are not evidence that the original tokenizer or full historical inputs have
already passed a real remote preflight.
