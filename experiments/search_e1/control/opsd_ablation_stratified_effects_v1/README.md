# Native512 recorded-score stratified postprocessing

This local stdlib helper implements the explicitly approved reporting contract.
It reads a SHA-bound `opsd-ablation-stratified-native-inputs-v1` manifest whose
`models` is a nonempty subset of the eleven formal labels. Each label supplies
`summary: {path, sha256, bytes}`, `records: {path, sha256, bytes}` and `version`.
Missing labels remain pending. The two zeros may use the same summary and1024-row
records file; each version contributes512 predictions. No node, HTTP, model,
Torch, inference, scorer, raw trace or optimizer read occurs.

The only source reuse is AST extraction of `counts` and `bootstrap` from the
exact frozen native summary SHA
`609e2e3379d2a26d3af0f9ddad522dabf97df0bbbe63cf7a23d0937d52f48877`.
The whole source module is never imported. SHA is computed once per bound file,
and that same byte buffer is parsed. Native counts use integer search histogram
keys; JSON normalization makes the comparison to saved string-keyed JSON exact.
The native accepted schema, role/run/variant/version, source/runtime/cohort/
initial/greedy/study identities, every primary slot,256+256 coverage, mutually
exclusive strict failure labels and all original overall/dataset counts are
checked. Local record SHA/bytes must equal native `private_primary_records`.
Question/row identities are used privately for pairing but never emitted.

The18 prespecified comparisons are the native17 plus BF16zero→FP16zero.
Each has overall512, NQ256 and Hotpot256 paired cells and independent seeded
10000-draw question-unit percentile bootstrap. Intervals are descriptive,
marginal and unadjusted for multiple comparisons; J is a combined intervention.
The final report still needs K/Adam/exposure/precision covariates and separate
execution, numeric, debug and persistence proofs. Pending counts do not signify
training failure or authorize an inference rerun.

Root publishes and byte-verifies this source **before** actual use. There are no
new fixtures or model tests. Run on the existing exact recorded inputs only:

```sh
python3 -B derive.py --manifest /ABS/inputs.json --manifest-sha256 SHA \
  --pure-source /ABS/PUBLISHED/summary.py --pure-source-sha256 609e2e3379d2a26d3af0f9ddad522dabf97df0bbbe63cf7a23d0937d52f48877 \
  --output /ABS/NEW_OUTPUT_DIRECTORY
```

The existing output directory is refused. Outputs are the aggregate
`stratified-paired-summary.json`,54-slot `stratified-paired.csv`, Chinese notes and
an artifact SHA derivation manifest. They contain no raw questions, question/row
IDs or private input paths. Source and input manifests remain separate private
provenance; failure leaves any incomplete output rather than claiming success.
This source has been prepared locally, not executed as an experimental result.

## 2026-10-09 remote-only storage amendment

The latest human storage instruction supersedes copying private per-question inputs to Mac. Run the unchanged derive.py on HKUST against its original remote native summaries/records; copy only the resulting public aggregates and small SHA/exit summaries to Mac. prepare_remote_inputs.py binds only summaries with original accepted controller-complete and full native version coverage, hashes original remote record bytes, and writes a fresh independent analysis input manifest under the study's remote analysis directory. It does not modify an evaluation, sample a model or copy raw records. Missing groups remain pending. No extra review or fixtures are required for this read-only input binding.
