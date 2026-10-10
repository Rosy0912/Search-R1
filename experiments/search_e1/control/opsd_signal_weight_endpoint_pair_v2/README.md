# One real native512 endpoint versus common zero — metadata adapter v2

This is a new namespace copied from frozen endpoint_pair_v1 (64794e8). The old
namespace remains unchanged. The v1 binding incorrectly required each original
config's run-scoped publication gate to be the same file as the study-wide helper
publication. The deployed study has one study-wide helper readback and four
distinct scoped receipts. This made v1 bind_inputs reject valid production
metadata before it examined an endpoint. It was not a training/evaluation failure.

V2 removes only that file-equality condition. The study_manifest publication must
still equal original_source.publication. Every config must still bind the exact
00fec source commit, af16 manifest and helper root. Each config publication is
freshly SHA/bytes checked before and after unchanged native
bound→scoped→publication_admission. The native admission still checks its actual
study/run/variant/output scope, accepted search-e1 publication, matching commit
and remote readback, manifest and all deployed source files. Fresh per-group refs
and admission results are explicitly reported at
sources.config_publications.{G_POS,G_NEG,H_POS,H_NEG}.{reference,admission}.
No endpoint/raw/checkpoint/zero/scoring/paired arithmetic function is changed.

Independent CPU reader. It does not modify training, original helpers, configs,
checkpoints or raw evaluation files. It does not load a model, deserialize a
tensor/optimizer, restore RNG, train, sample, query GPU/HTTP/process state, signal
processes, acquire production locks or run an additional evaluation. The source
`common.py` is a byte-identical copy of published completion7d common.py.

The reader source itself and its request must be published/read back, then deployed
as a new namespace before execution. Do not execute endpoint acceptance before an
actual15/25 parent/child/summary exists. The binding-only preflight below may run
before that point and does not inspect target/common_zero evidence. No stub,
mutable latest or synthetic test
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

### Binding-only preflight (no endpoint output)

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 E1_RUN_ID=NEW_EXPLICIT_READER_RUN
  ORIGINAL_ENV_PYTHON -B reader.py
  --request IMMUTABLE_REQUEST --request-sha256 EXACT_SHA --bind-only
```

Do not pass output-dir. This mode reads the published reader/original/completion/
analysis source manifests and publications, ordered study, four configs, their
distinct publication gates and native runtime/common identity checks. It does
not access target/common_zero endpoint evidence or create any output file. The
same request schema is used; target/common_zero fields may be absent for this
preflight. Its stdout schema is opsd-signal-endpoint-bindings-v2, with
accepted=true meaning only bindings accepted; endpoint_accepted=false,
full_study_accepted=false, no_sampling=true and no_output_files_written=true.
sources.binding_checks preserves the original runtime/common verification output.

Native prepare.verify_runtime fresh checks the305 runtime files and scorer;
verify_common fresh checks cohort/train/validation and initial-adapter payload
files. It does not fresh hash the12 source100 weight files. The preflight therefore
reports source100_weight_payload_fresh_SHA=false, no numeric Adam/tensor
deserialization and no512/EM/endpoint acceptance. Formal verify_zero/target retain
their original separate complete evidence and source checks.

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

Reference `bytes` fields are optional; when present they are also checked. Preserve
the actual native rich raw_checkpoint dictionary, including its recorded stat
fields, when staging an endpoint request. The unchanged completion.evaluation
parent check compares that checkpoint dictionary directly; a hand-reformatted
path/SHA-only version can fail even when it names the same bytes. Do not relax the
original endpoint check or use a partial checkpoint as an end-of-outer input.

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

Additional v2 fixtures call actual bind_inputs using a study-wide publication
and four distinct scoped config receipts. They preserve real original native
config validation and publication bound/scoped/admission, all helper file SHA
checks and the original pinned source manifests/commits. Only runtime/common
asset verifications are fixture stubs because synthetic tests do not contain
real TRAIN169103/305runtime/adapter assets. Negative cases include wrong run,
variant, study, output root, helper root/manifest, source commit/readback,
publication SHA/bytes/acceptance/branch, temporary source-copy mutation and
accumulation8 contract changes. Binding-only mode is tested without endpoint
fields and with no target/zero reads or output creation. No fixture writes to the
original helper or demonstrates real production binding/endpoint acceptance.
