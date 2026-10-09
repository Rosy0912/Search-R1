# Recorded debug coverage: local CPU only

This independent entry inspects accepted immutable local backups. It does not alter any training, scoring, runtime, Node, model, GPU, HTTP, RNG, signal, or resource-lock state. Root must publish and read back the source before its first actual execution. No fixtures or data execution were performed during preparation.

Use an absolute new output directory:

```text
python3 -B derive.py --manifest /ABS/inputs.private.json --manifest-sha256 EXACT_SHA256 --output /ABS/NEW/output
```

The input schema is `opsd-ablation-recorded-debug-coverage-inputs-v1`, with five explicit `{path, sha256}` references:

- `aggregate_source`: already published `opsd_ablation_recorded_clipmass_v1/aggregate_clipmass.py`, exact SHA `b7a264119fd18b9ac744f452c86e8deb6db550c318050178e94adb38a1f39c8a`.
- `clipmass_inputs`: the exact previously accepted `seven-group-closed-clipmass-inputs-v1` manifest, including selected group config, native facts, and complete local backup proofs.
- `committed_proof`: that run's `referenced-files-proof.private.json`.
- `clipmass_summary`: that run's `clipmass-publication.json`.
- `clipmass_derived_manifest`: that run's `derived-manifest.json`, binding both output SHA values and the original aggregate source SHA.

All source loading compiles SHA-checked bytes directly under an independent module name; it does not use pyc. The existing helper supplies `FixedBackups`, `stable_json`, `visit`, `opened`, and `finish`. Original sources are unchanged. Input groups are an explicit nonempty accepted subset with exact variant/run/config and ten complete native outers. This preparation is intended for C1; C0 is not rescanned.

The first sequential visit checks exact committed membership, token masks/order and four-distribution top32 arrays (length32, finite logp, unique in-range IDs, actual8192 cap per outer). It also inspects teacher selection/request/parent/child/case/HTTP documents and sampling completions. No KL/clipmass/objective/model/scoring recomputation occurs.

Teacher own generation differs from fixed-D training q. Parent embedded child wait0/release, source100/commonBF16, no-denominator/RNG-isolation and transparent-generation flags are checked. External `runs/phase-exits` files are outside these rawouter backups and are not independently reread. Student raw generation/HTTP contents reuse accepted member SHA rather than being rescanned.

Outer0 headers verify native `SHA256('vocab-v1:42:'+qid)` preselection and its first two valid positions. Complete/cap-skip records are distinct. Second visit: committed NPY only, fresh member SHA, magic/header, little-endian FP32 shape151936, exact body size and every payload value finite. No exponentiation/normalization/KL/gradient reconstruction. Current subset bytes are checked against24GiB; all-seven shared ledger/reservations remain pending.

Compressed visits traverse relevant archives and drain gzip EOF/CRC. Existing whole-archive SHA plus current metadata continuity are reused; only selected members are freshly hashed. The two visits consume disjoint selected members. Nothing is extracted.

Outputs: aggregate `debug-coverage-publication.json`, private CLI/artifact index `details.private.json` (outside Git), and `derived-manifest.json`. Existing output directories are refused. Missing/changed references or invalid identities/counts/arrays/caps/exits reject coverage. No raw QA/qid/token text is published. A subset does not certify70 outers,5632 evaluations, causal effects, teacher quality or future admission.
