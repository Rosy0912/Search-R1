# Independent final coverage assembler

This CPU-only namespace never trains, samples, loads a model, deserializes tensors
or optimizer state, restores RNG, acquires production resource locks, queries GPU
or HTTP state, signals processes, or edits original training/config/source files.
It uses explicit immutable references, never a mutable latest pointer or a glob of
failed attempts. The original46-file helper manifest remains
`af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314`.

It source-only imports the original stdlib checkpoint, sampling, training metrics,
strict scorer and evaluator modules. Checkpoint acceptance is fresh byte/metadata
and contract checking, **not numeric Adam acceptance**. The separate native
`audit_state.py` output is parsed for its actual four final checkpoint identities,
rank steps/digests, full Adam equality, finite FP32 adapter and CUDA-free scope.

## Production CLI

Publish, read back, then deploy this independent prefix before using it on HKUST4.

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 ORIGINAL_ENV_PYTHON -B assemble.py
  --request EXPLICIT_IMMUTABLE_REQUEST
  --request-sha256 EXACT_SHA
  --output NEW_STUDY_ANALYSIS_PATH/coverage.json
```

The output must be new under the authorized study's `analysis/` directory, with
UID1001. A coverage failure produces `accepted=false`. Complete coverage with
missing independent final requirements produces `coverage_accepted=true`,
`final_accepted=false`, `COVERAGE_ACCEPTED_HOLD_EXTERNAL_EVIDENCE`. It cannot be
reported as final acceptance. No missing required evidence is silently passed.

## Explicit request

`schema=opsd-signal-final-coverage-request-v1`:

- `study_manifest`: original exact study manifest reference.
- `study_terminal`: new `opsd-signal-weight-driver-terminal-v1` output, with the
  same retained tmux pane's actual status0, original supervisor identity and owned
  release. It must bind the separate `native_study_completion` reference.
- `native_study_completion`: original four-group-terminal artifact. Keep this
  original schema for old frozen analysis code; the new stronger terminal does
  not replace it.
- `common_zero`: original shared common-zero wrapper, containing parent/child refs.
- `groups`: exactly ordered G_POS/G_NEG/H_POS/H_NEG. Each contains `variant`,
  original `config`, final `train_ledger`, exactly25 ordered `outers`,
  `protected_checkpoints` (true updates1 and5, each explicit `outer_index` and
  `checkpoint`), and `evaluations` at completed outers15/25.
- Each outer row contains `outer_index`, `outer_manifest`, `outer_complete`,
  `end_checkpoint`, `sampling_complete`, `summary`, `trainer_exit`, `token_audit`.
  Formal evaluation rows contain `completed_outer`, `parent`, `child`.

Cross-outer checks include exact previous checkpoint/old adapter, common TRAIN
cursor/cohort, fixed sampling raw model identity, source/runtime/config, full
512x5 unique slots, frozen strict preupdate TRAIN scoring, original B/D miner,
P/K, real cumulative updates=sum(ceil(K/32)), exposure=sum(K), actual trainer
wait0/release, all committed batch summaries and25 ledger rows. K0 means no
updates/exposure, cursor0, unchanged declared adapter and loss/weightNA; an
unsupported zero-token audit is not a training failure and is never fabricated.

The native final TRAIN ledger has no top-level config SHA. Its original body
binds run_id/variant/25; each of its25 rows must bind the exact original config
SHA/run/outer index. The rows must additionally equal the reconstructed summaries.

For native512, the raw outer14/24 checkpoint must match each actual evaluated
FP32 adapter. The original evaluator recomputes all strict final traces with its
unchanged protocol. Exactly512 formal slots, eight logical worker/result pairs,
actual worker waits0, complete64 slices, matched process identities, forward
scaling/dtype declarations and parent release are required. Null-hook/zero-effect
control rows are excluded. Total formal slots are4608; TRAIN slots are256000.

### Token audit modes

K>0 uses the independently published token audit, not a new loss parser.

- `mode=recompute`: `helper_root`, `source_manifest`, `publication`,
  `published_manifest_sha256`. The original `audit.audit` executes read-only.
- Saved output: `output`, `execution`, `helper_root`, `source_manifest`,
  `publication`. It must be the actual accepted native token-audit stdout, with
  original source/config/checkpoint and committed update counts. Every referenced
  score/token/update/batch file is freshly hashed; the exact ordered inventory
  SHA and exposures/tokens must equal that stdout. Failed/uncommitted attempts
  are not selected.
- K0: `token_audit=null` only. Its metadata/lossNA checks are separate.

An execution record is a small immutable actual-run evidence object containing
`accepted=true`, `actual_wait_returncode=0`, exact `stdout`, `source_manifest`,
`publication`, and `torch_imported=false` for stdlib callers. Native numeric Adam
readback may import Torch CPU; this assembler still never does. Existing raw
stdout is not overwritten; if transport recorded these values under another
schema, build an independent evidence adapter with the original actual wait/log
refs. Do not invent a wait code or use a helper CLI's rc as the training rc.

### Separate final requirements

`external_requirements` is optional for coverage but required for final acceptance:

- `final_numeric_adam`: map of four actual `audit_state.py` execution packets.
  Original native output schemas/fields are parsed, not generic envelope claims.
- `protected_and_outer_end_backups`: packets for every distinct protectedu1/u5
  and100 outer-end checkpoint. Each has the actual transfer/probe stdout and
  `target_marker_copy`, a byte-identical **small** copy of the HKUST1 accepted
  marker. Parse the source/target/tar actual return codes and marker/member SHA
  inventory against original K4 CP bytes. This copies audit metadata, not weights.
- `published_source`: nonempty list of actual source overlay/publication specs
  (`helper_root`, `source_manifest`, `publication`). All listed source bytes must
  match published/read-back manifests. Original training publication is checked
  independently per group; do not substitute a document HEAD for deployed source.
- `full_vocab_scope`: actual shared `ledger` and optional `sampled_arithmetic`
  execution packets. The24GiB cap and fresh saved file identities are checked;
  existing complete/cap_skip counts come from token audits. Arithmetic remains
  explicitly sampled, never all positions/allNPY. top32 cannot reconstruct fullKL.
- `effect_and_relationship_analysis`: `effects`, `train_relationship`,
  `heldout_relationship` native CLI artifact packets. `mode=native_cli_artifact`,
  `actual_stdout` names the true captured CLI stdout; `output` names the report
  artifact. `execution.stdout` must equal `actual_stdout`, never a relabeled report
  path. Effects/heldout native stdout contains `output`; TRAIN native stdout is
  the `complete.json` file reference, resolved to `complete.outputs.summary`.
  The complete/native request/config/endpoints/100outers chain is freshly read.
  Preserve native output schemas. Each
  supplies an independent `input_binding` object with
  `schema=opsd-signal-analysis-input-binding-v1`, `input_format_bridge_only=true`,
  exact `original_output`, study_id, four configs, four final_checkpoints, and nine
  formal_evaluations. Actual report scope4608/100 and native endpoint hits must
  match. Statistical math remains in the already published analysis helpers.

No sampled arithmetic is represented as complete full-vocabulary analysis. No
regional loss contribution is represented as gradient attribution or causality.
No TRAIN curve is represented as fixed heldout or postupdate EM.

## Independent TRAIN input bridge

The native sampling receipt contains a pure `calculate` body without config/run
fields. The old analysis collector needs these fields. This bridge copies that
body, appends the explicit original config/run and original sampling receipt,
and writes a new analysis file. It never substitutes the whole sampling receipt
for the TRAIN summary or changes its mathematical contents. Final coverage still
recomputes the original body from all2560 strict-scored rows.

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 ORIGINAL_ENV_PYTHON -B bridge.py
  --config ORIGINAL_CONFIG --config-sha256 EXACT_SHA
  --sampling-complete ORIGINAL_SAMPLING_RECEIPT --sampling-complete-sha256 EXACT_SHA
  --output NEW_STUDY_ANALYSIS_PATH/train-summary-bridge.json
```

## CPU fixtures

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHON -B -m unittest discover
  -s experiments/search_e1/control/opsd_signal_weight_completion_v1
  -p test_completion.py -v
```

Fixtures use small temporary synthetic payloads only. They exercise immutable
SHA/exclusive writes, actual wait scope, source-only compilation, unchanged
native TRAIN bridging, fresh committed-only token inventories, K0 and exposure
guards using original checkpoint/training-metrics code, exact numeric schemas,
backup member/marker identity, native512 worker/raw checkpoint guards and cap
scope. Mocked model-free formal rows do not establish a real evaluation result.

Local fixtures default to the unchanged sibling `opsd_signal_weight_v1` directory.
For an independent Node deployment, set `OPSD_COMPLETION_TEST_ORIGINAL_HELPER_ROOT`
to the immutable **original** helper path from the production config; its exact
manifest/full source inventory must still match af16. This only locates a test
dependency and never changes the production request or helper source.
