# One real native512 endpoint versus common zero

Independent CPU reader. It does not modify training, original helpers, configs,
checkpoints or raw evaluation files. It does not load a model, deserialize a
tensor/optimizer, restore RNG, train, sample, query GPU/HTTP/process state, signal
processes, acquire production locks or run an additional evaluation. The source
`common.py` is a byte-identical copy of published completion7d common.py.

The reader source itself and its request must be published/read back, then deployed
as a new namespace before execution. Do not execute a prepared request before an
actual15/25 parent/child/summary exists. No stub, mutable latest or synthetic test
fixture is a production input. Code, config and small summaries may be local;
real per-question outputs remain on HKUST4.

## CLI and result scope

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 E1_RUN_ID=NEW_EXPLICIT_READER_RUN
  ORIGINAL_ENV_PYTHON -B reader.py
  --request IMMUTABLE_REQUEST --request-sha256 EXACT_SHA
  --output-dir NEW_DIRECTORY_UNDER_STUDY_ANALYSIS
```

Production requires HKUST4 UID1001. It emits a small CLI envelope, a remote
`summary.json` and remote `paired-slots.jsonl`. The latter contains only
validation_index, dataset, zero_em, endpoint_em and four-way outcome, with no
question ID, question/answer text, response, raw trace, token or tensor. No
probability-replay field is copied into original result files. CLI fields and
summary refs/counts are suitable for a small local audit copy.

`accepted=true` proves this particular postraw15/25 native512 evaluation versus
common zero. `full_study_accepted=false` remains explicit. It does not prove all100
outers, eight endpoints, final statistical analysis, numeric Adam equality or a
causal improvement. No new significance test or confidence interval is added.

## Explicit request schema

`schema=opsd-signal-single-endpoint-pair-request-v1`:

- `analysis_run_id`: new independent nonempty run marker, different from all four
  original group run IDs. Supply the same marker in the CLI environment.
- `reader_source`, `original_source`, `completion_source`, `analysis_source`:
  each `{helper_root, source_manifest:{path,sha256}, publication:{path,sha256}}`.
  Actual file bytes and exact Python inventory must match each manifest. Each
  publication must be accepted on search-e1, with40hex source_commit equal to
  remote_readback_sha and the exact source_manifest_sha256. The actual reader
  file must lie under reader_source. The three reused publications are pinned
  respectively to00fec71/af16f6e5,7d9fea2/b36aff53 and235e376/649472b2.
- `study_manifest`: original exact study manifest `{path,sha256}`.
- `configs`: exact map G_POS/G_NEG/H_POS/H_NEG to their four original config
  refs. Each must match its immutable study group, published original helper,
  fixed25outer contract and source100/runtime/common initial/cohort/validation
  identity. No config is changed by this reader.
- `common_zero`: `{config, wrapper, parent, child, summary, evaluation_request}`,
  all explicit `{path,sha256}`. Its config must be the original G_POS config,
  even when the target is G_NEG or H_POS/H_NEG. The original shared wrapper must
  bind that same parent and child. Exact original common zero12source files,
  initial adapter payload,305runtime and512data are freshly checked.
- `target`: `{variant, completed_outer, outer_manifest, outer_complete,
  raw_checkpoint, trainer_exit, parent, child, summary, evaluation_request}`.
  completed_outer is exactly15 or25, not a count of Adam updates. Every reference
  is explicit `{path,sha256}`. Paths must be the corresponding native
  outer00014/00024 and evaluation outer15/25 paths. raw_checkpoint is the actual
  committed end-of-outer checkpoint named updateXXXXXXXX-outer00014/00024. Never
  use best, outer-start sampling merge cache or a mutable pointer.

Reference `bytes` fields are optional; when present they are also checked.

## Reused acceptance and paired arithmetic

1. The original00fec checkpoint `check_outer_complete` validates real update
   accounting ceil(K/32), current and prior complete cursors, config/run and four
   full-state file inventories; no tensors are deserialized. The declared raw
   checkpoint must equal this accepted end CP, bind the actual training manifest
   and outer-old, and reconcile cumulative pair exposure.
2. Published7d `assemble.wait_release(..., count=1, phase='training-running')`
   checks one actual torchrun phase-child wait record, **returncode0**, its durable
   terminal and full release. count1 is not a returncode of1.
3. The unchanged7d `assemble.evaluation` checks the evaluator parent wait0,
   raw14/24 cursor=K, exact original request/source/runtime/data, all eight actual
   logical worker waits0/release, their process bindings/64slices,504FP32 adapter
   digest and252 actual layer scaling declarations. Original native strict
   `aggregate(..., verify_traces=True)` must reconstruct saved summary exactly.
4. Published235e `summary.paired_cases` owns gained/lost/preserved/stillwrong
   arithmetic and overall/NQ/Hotpot right-minus-left delta. Temporary independent
   memory copies receive all FEATURES=None solely to satisfy that function's
   optional metric interface. All probability changes are explicitly NA; no
   probability replay or model is run. Every per-slot label must reconcile with
   the original paired count result. Original arrays are unchanged.

The original native summary alone does not contain paired counts. Original
`result_analysis.analyze` requires all4608 formal predictions and is not called on
this two-endpoint prefix. Final nine-endpoint analysis remains unchanged.

## CPU fixture scope

Run `python -B -m unittest discover -s THIS_DIRECTORY -p 'test_*.py' -v`.
The synthetic fixtures exercise original metadata checkpoint/outer acceptance,
original7d endpoint acceptance and original235e paired arithmetic, including raw
versus sampling-cache paths, unfinished K, wrong/unknown/nonzero waits, wrong
worker64slices, missing512, adapter digest/scaling, release, summary/config pins
and paired dataset arithmetic. Their tiny named payloads are explicitly not
model/optimizer tensors. Only fixture native trace verification is bypassed;
production always invokes the original strict trace verification with True.
Fixture acceptance does not mean any real15/25 endpoint has completed.
