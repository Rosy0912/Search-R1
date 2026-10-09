# Read-only native debug retention audit

This independent CPU namespace reads retained artifacts only. It never calls
Controller initialization, spawn, gate, wait, teacher generation, model loading,
HTTP, GPU, signals, production locks or RNG restoration. Its only production
output is a small JSON object on stdout. It writes no Node output file and prints
no question, response, document, raw ID array or token distribution. All originals
remain under the authorized HKUST4 study.

The reused sampling, Controller recorded-exit/teacher-child checks and
teacher select_cases are exact original source-only imports from the immutable
46-file helper manifest
`af16f6e53e96ac7aba383e51570ef28a1595b206431c6458cc4ac39942b78314`.
No original namespace, scientific function or receipt is edited or monkeypatched.
Original helper publication, all46 source bytes, the exact305 runtime source
manifest/files, the original study manifest and all four config SHA values are
checked before reading any outer. The reader's own source/publication is also
required. Model files are neither loaded nor rehashed: the original teacher
checker verifies their existing source identity and metadata, not new tensor math.

## CLI after publication and byte readback

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 ORIGINAL_ENV_PYTHON -B audit.py
  --request IMMUTABLE_REQUEST --request-sha256 EXACT_SHA
```

Capture its actual stdout and actual SSH/process return code separately.
Exit0 means the referenced retention scope passed; prefix0 is never full study
completion. Errors return2 and accepted=false, without signaling any process.
This helper needs only Python3.10 stdlib, original deployed46-file source,
original frozen305 runtime, and the referenced retained data/receipts. It never
requires Torch/NumPy/Transformers or a GPU in production.

## Immutable request

`schema=opsd-signal-debug-retention-request-v1`:

- `mode`: prefix or final. `currentN`: explicit completed outer count1..100.
  Prefix inputs must be exactly the first N of the serial order G_POS0..24,
  G_NEG0..24, H_POS0..24, H_NEG0..24. Prefix mode always reports
  full_study_complete=false, even when supplied100; final requires100.
- `retention_source`: helper_root, source_manifest and publication refs for this
  independent reader. Publication must be accepted search-e1, real matching
  source_commit/remote_readback_sha and source_manifest_sha256. Actual audit.py
  and all source bytes/inventory must match.
- `study_manifest`: exact original path/SHA reference, fixed SHA86db5c8...
- `configs`: map of exactly G_POS/G_NEG/H_POS/H_NEG to their exact original refs.
  Their four SHA values are frozen in audit.py; all four config/source identities
  are checked even for a one-outer prefix. Four cards/micro1/accumulation8/global32
  is explicitly checked. This tool does not reinterpret an alternate config as
  an original run.
- `outers`: ordered explicit entries. Each has variant, outer_index,
  outer_manifest, sampling_complete, outer_complete, end_checkpoint, trainer_exit,
  teacher_parent. Every ref is a real path/SHA (optional bytes). Native first
  sampling receipt has no top-level variant/config; these are obtained from its
  actual terminal, eight worker receipts and outer manifest, not invented fields.
- `accepted_coverage`: optional final-only actual coverage report ref. If used,
  require accepted coverage100/formal4608, same study/four final raw checkpoints
  and exact100 outer checkpoint identities. final_accepted may still be false;
  debug retention is a separate evidence requirement, not an analysis deadlock.

Each outer must be actually complete: committed K/cursor/ceil(K/32), cumulative
updates/exposure, previous raw adapter, sampling raw identity, config/runtime/
source identities and actual trainer wait0/recorded release. This uses checkpoint
JSON identity only. It does not deserialize or independently validate optimizer,
adapter or RNG state; numeric acceptance remains separate.

## Student generation and HTTP scope

For each outer all eight logical workers are retained, actual two-wave waits0
and release are bound, with exact PID/UID/starttick/run/config/model identities.
Each retained worker has64 exact rank::8 questions times five unique siblings,
320 native rows; merged2560 rows must equal all eight shards in original order.
All rollouts, worker receipts, generation trace, HTTP trace, phase terminal and
provided original logs are freshly hashed. Final end rehash catches changed
content even if filesystem metadata appears stable.

vLLM trace is per active generated turn, not one record per trajectory. Its row
count is the sum of authored policy span counts. For call c the active original
rows have len(policy_spans)>c; index is0..active_count-1 and raw native output text
equals that authored span. Prompt/generated IDs must be actual integer lists;
empty generated IDs/empty authored spans and context-limit zero spans are valid.
No tokenizer or retokenization is used, and token count is not compared to text
character count. The original input hash, finish/stop, rank and raw sampling
model identity are retained and checked.

Original verify_http_trace checks complete request/body/parsed-response records,
consecutive IDs, rank/model and accepted completion. generation_call must name a
real generated call. One vLLM HTTP request may contain several searches; this
audit never equates its request count with all rollout search_calls.

The independent payload guard also requires the exact configured positional URL,
native keyword arguments json/timeout, queries as a nonempty string list (an
individual empty string is valid), topk3/return_scores=true and successful2xx
status. The complete response_body_text must parse to exactly its retained
response_json, including JSON value types, and result must have one passage-list
per actual requested query. This is checked from stored bytes only. The original
frozen vllm_rollout uses timeout300 and batches active searches; frozen HF
SearchAgent uses timeout120 with one query. Neither call's logging fields or
batch lengths are inferred from aggregate search counters. No query or response
text is printed on stdout.

## Teacher diagnostic scope

Use the actual teacher parent/selection/request/child/case/HTTP refs; the original
select_cases is recomputed from original mined pairs. Count=min(8,P), not always8.
P0 produces count0 and no request/child/wait; NA never invents a successful B.
Parent and child bind original config/run/pairs/source100 and actual wait0/recorded
release. The exact original Controller.verify_teacher_child checks original B/D,
frozen strict result and HTTP record identity. All actual case/HTTP bytes are
freshly hashed as part of the final inventory.

HF teacher raw_output_ids is a batch1 full input+new sequence. Its prefix must
match the actual native generate input; native generate calls cover authored
policy spans. Extra tokenizer/decode events are preserved; StopTags may decode
many times, so decode count is never treated as generate count. Context-limit
zero-generate cases, teacher wrong answers and invalid action protocol remain
valid retained diagnostic evidence. Teacher search_calls equals its own HTTP
count. It is autonomous teacher diagnosis, original_training_objective=false,
with zero TRAIN/validation denominator contribution. It is not training q on the
original D history.

## Stdout schema

`schema=opsd-signal-debug-retention-v1`, accepted, full_study_complete, mode,
currentN, complete_outers, sampling_workers, teacher_parents, teacher_cases,
teacher_not_applicable_parents; study_id and study/source/coverage refs;
per_outer compact identity/ref/count rows; referenced_inventory contains fresh
file count/bytes and canonical ordered inventory SHA plus fresh_end_sha_recheck.

Final requires100 complete outers,800 sampling workers and100 teacher parents.
teacher_cases is its actual sum, with NA/insufficient eligible cases retained.
There is no assumed800 teacher case requirement. Prefix cannot be used as final
acceptance. This scope does not audit online token losses, top32 or full-vocab
numeric reconstruction, and gives no new EM, training, success-probability or
causal attribution claim.

## CPU fixtures

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHON -B -m unittest discover
  -s THIS_NAMESPACE -p test_retention.py -v
```

Fixtures call actual original stdlib sampling/controller/teacher selection/
strict-scoring functions on small synthetic native-format data, not a model.
Local tests locate the original sibling helper; Node deployment must set
OPSD_RETENTION_TEST_ORIGINAL_HELPER_ROOT to its immutable original deployed path
and OPSD_RETENTION_TEST_RUNTIME_ROOT to frozen-305. Without the latter, local
fixtures read the original core/QA source bytes from the00fec Git object only.
Runtime fixture305 labels are synthetic dependency scaffolding, not a full
runtime acceptance claim. Production verify_source always requires its true
exact manifest and fresh source bytes. No fixture is a real training/evaluation
or Node retention acceptance.
