# Independent final Chinese report renderer

This namespace only reads bounded JSON metadata/aggregate reports, verifies
explicit immutable SHA bindings, and renders CSV/Markdown/Agg PNG/SVG. It never
loads models/tensors/Torch, trains, generates, recomputes statistical estimates,
queries GPU/HTTP state, signals processes or changes production inputs. The
original training and analysis math remain untouched. NumPy/Matplotlib already
exist in the authorized local/Node runtimes; this helper installs nothing.

## Formal CLI — after complete study and final evidence

Publish and fresh-readback this source manifest first. Run with the original
HKUST4 environment, UID1001, CUDA_VISIBLE_DEVICES empty, and an exclusive new
output directory under the authorized study's `analysis/` directory:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 ORIGINAL_ENV_PYTHON -B report.py
  --request EXACT_REPORT_REQUEST
  --request-sha256 EXACT_SHA
  --output-dir NEW_REMOTE_STUDY_ANALYSIS_DIRECTORY
```

The formal CLI refuses local Mac output. Only small aggregate CSV/JSON/Markdown
and these plots may later be copied locally. Full rollout, token, per-question
prediction, model, tensor and optimizer assets remain HKUST. Do not copy raw
inputs to Mac for this renderer.

The JSON request has `schema=opsd-signal-final-report-request-v1`, `study_id`, and:

- `source`: `{helper_root, source_manifest:{path,sha256}, publication:{path,sha256}}`.
  Publication must be accepted `search-e1`, source_commit=remote_readback_sha,
  exact manifest SHA; every payload SHA and Python inventory is fresh-verified.
- `coverage`: native CLI artifact packet below for the actual final coverage.
  Its output must be `opsd-signal-final-coverage-v1`, final_accepted=true,
  4 complete groups,100 complete outers,256000 TRAIN slots,4608 native test slots,
  the original study ID and actual four final raw checkpoints. The actual
  assembler CLI appends its `request` reference to its output report; this is
  bound to explicit `coverage_request`, not guessed from the pure function.
- `coverage_request`: actual accepted assembler request `{path,sha256}`.
- `configs`, `train_ledgers`: maps of the four G_POS/G_NEG/H_POS/H_NEG original
  references. Each native25-row ledger is separately bound to its config/run,
  all outer indices0–24,2560 denominator, preupdate scope, actual counters and
  each accepted coverage row/checkpoint. The original native ledger has no
  top-level schema/config SHA; the renderer preserves this native format.
- `common_zero`: actual common-zero wrapper reference.
- `evaluations`: exactly nine `{parent,child,summary}` immutable refs, commonzero
  plus the eight outer15/25 postraw endpoints. These must equal final coverage,
  original parent/child/run/config/model/time identities, summary512 and dataset
  denominators256. Test models are raw outer14/24, never cache/best.
- `analysis`: exactly `effects`, `train_relationship`, `heldout_relationship`,
  each a native CLI artifact packet plus `input_binding`. These must match the
  already accepted final coverage external-analysis evidence. Effects use the
  original schema `opsd-signal-weight-final-effects-v1`; the two primary arrows
  remain G_NEG/25→G_POS/25 and H_NEG/25→H_POS/25. TRAIN uses
  `opsd-signal-train-analysis-summary-v1`,100 complete outers; heldout uses
  `opsd-signal-heldout-response-relations-v1`,4608 original EM slots.
- `debug_retention`: actual small `opsd-signal-debug-retention-v1` stdout ref,
  accepted, full_study_complete,100 outers/800 sampling workers/100 teacher
  parents and original study ID. Actual teacher case counts are not padded.
- `parameter_movement`: actual small `opsd-signal-parameter-movement-v1`
  stdout ref, accepted/final/full_study_complete, original configs, the twelve
  0→15/15→25/0→25 A/B-coordinate comparisons and actual raw endpoint refs.

A native CLI artifact packet is:

```json
{
  "mode": "native_cli_artifact",
  "helper_root": "EXACT_PUBLISHED_DEPLOYED_HELPER_DIRECTORY",
  "source_manifest": {"path": "ABSOLUTE_MANIFEST", "sha256": "EXACT_SHA"},
  "publication": {"path": "ABSOLUTE_PUBLICATION", "sha256": "EXACT_SHA"},
  "actual_stdout": {"path": "ORIGINAL_CAPTURED_CLI_STDOUT", "sha256": "EXACT_SHA"},
  "execution": {"path": "ACTUAL_WAIT0_EXECUTION_PACKET", "sha256": "EXACT_SHA"},
  "output": {"path": "ORIGINAL_OUTPUT_REPORT", "sha256": "EXACT_SHA"}
}
```

The execution packet must have actual_wait_returncode0, torch_imported=false and
the exact stdout/source_manifest/publication references. Effects/heldout/coverage
stdout is an accepted envelope naming `output`. Native TRAIN stdout is instead
the file reference to `complete.json`; the renderer follows the real
`complete.outputs.summary`. It never relabels the report as CLI stdout.
`input_binding` is the independently accepted bridge containing original_output,
study_id,4 exact configs,4 actual final checkpoints and9 formal evaluations.

Every read input is freshly streamed SHA-verified again at the end, even if its
metadata appears unchanged. Inputs are at most32MiB JSON each. This renderer
does not reread raw traces or model tensors; final acceptance belongs to the
strong completed coverage assembler and independent audits, not the plots.

## Outputs and interpretation

- `train-preupdate-em.csv`:100 full-denominator points plus actual K/Adam/loss/
  aggregate selected-weight statistics. Different outer cohorts are explicit.
- `heldout-em.csv`:9 saved endpoints × overall/NQ/HotpotQA.
- `paired-endpoint-effects.csv`:original paired gain/loss/preserved/still-wrong,
  signed delta, actual precomputed CI, exact McNemar and family2 Holm where
  defined. All arrows are right minus left; primary25 is never selected by15 EM.
- `train-selected-relations.csv`:actual eligible-subset Pearson/Spearman/AUC,
  question-block CI when genuinely computed, explicit coverage and NA.
- `heldout-relations.csv`:same-response posthoc relations with metric-specific
  coverage/NA. It does not invent confidence intervals for these associations.
- `parameter-displacement.csv`:saved LoRA A/B coordinates, not effective BA
  model weights, gradients or causal impact. Zero-source B relative norm is NA.
- Three English-label scientific Agg PNG/SVG plots:TRAIN25-point curve; same512
  zero/15/25 overall/NQ/Hotpot;four final25 EM bars and the two primary differences.
- Chinese `report-zh.md`, `small_aggregate.json` and final `complete.json`.

`small_aggregate.json` is explicitly whitelisted, compact UTF-8 JSON and must stay≤256KiB. It contains
only aggregates, not qid/question/response/trace/tensor/per-position weights.
The report states actual retrospective association direction/coverage; fixedB
gold-selected teacher coverage (currently expected239 but use actual data), NA,
collider/eligibility bias, regional loss≠gradient/causality, one seed, no E25 and
no new pass5. Sparse zero/15/25 does not locate exact degradation onset.

Failures preserve any partial output but never write an accepted formal complete
artifact. Output directories are exclusive. Do not overwrite/restart a formal
report to hide an input or rendering error; correct an independent new copy.

## CPU interface and visual fixtures

```text
python3 -B -m unittest discover -s THIS_NAMESPACE -p test_report.py -v
python3 -B report.py --synthetic-fixture --output-dir NEW_SMALL_LOCAL_FIXTURE
```

Fixtures use model-free actual native-format schemas and immutable graph refs.
They test complete100/4608 shape, partial/wrongcounts/wrongrun/wrongendpoint,
changed same-size bytes, actual CLI output resolution and retention/movement
scope. They are not production coverage or real experiment results.

The explicit synthetic fixture has visible plot/report labeling and
`accepted=false, formal_report=false, synthetic_fixture=true` in both CLI and
complete metadata. It must never be submitted as final study acceptance. The
formal command has no fixture flag or alternate validation bypass. Node CPU
fixtures must also run after publication/deployment before formal use.
