# Fixed-weight OPSD diagnostics

Independent diagnostics of the actual source100 → OPSD run. These scripts never
create an optimizer, backpropagate, train, mutate frozen science, resume old RNG,
or launch/stop another process. Deployment and GPU ownership belong to the root
controller. Node outputs, full traces, selected source data, model/adapter weights,
and vocabulary arrays are **not Git source artifacts**.

## CPU cases

`select_cases.py` streams all 169,103×5 original trajectories, 100,974 original
pairs and the original training file. It chooses 32 questions from each cell:
NQ/Hotpot × correct-correct, correct-wrong, all-five-wrong. Selection is the
smallest SHA256(`42|dataset|group|question_id`), so it does not depend on scan or
process order. The output retains all five original siblings, the original pair
where one exists, source training identity, exact line/offset/row hashes, full
file hashes and source metadata. It validates strict frozen protocol rewards.

The hard cell has no successful reference, no privileged teacher generation and
no OPSD loss. It retains all five failures and probes their ordinary distributions;
it never invents a successful teacher. The two pair cells must match an actual
successful reference and the actual mined student with policy provenance.

Run the CPU selector with the actual frozen runtime and untouched source files:

```sh
python select_cases.py --runtime /data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runtime/opsd-preparation3-xformers \
  --rollouts /data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runs/preparation-3-rollouts.jsonl \
  --pairs /data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/runs/preparation-3-pairs.jsonl \
  --train /data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/data/train.jsonl \
  --output /data/shaoshitong/search-e1-opsd-debug-20261008/cases
```

Outputs must be new directories; existing data is never overwritten.

## GPU worker inputs and ownership

`worker.py` requires `--runtime`, `--cases-dir`, `--model`, `--weight-manifest`,
`--retriever-admission`, `--output`, four `--adapter-source100/update1/update5/epoch1`
paths, and `--worker-index` 0, 1 or 2. `--workers` must be3. Root assigns exactly
one GPU with `CUDA_VISIBLE_DEVICES`; the authorized plan is GPU4–6. One BF16 fixed
base per worker hosts four named small adapters. Source100 is the original
`best_adapter` whose252 LoRA-B tensors must all be zero. All adapter tensors are
finite and their CPU file hashes are verified before loading. Model source files
must match the supplied fresh CPU source hash attestation's exact metadata; the
worker does not falsely label that metadata check as a fresh large-weight hash.

The weight manifest has `source100: {path, config_sha256, files: [...]}`. Each file
contains `relative_path, bytes, inode, mtime_ns, ctime_ns, device, sha256,
sha256_origin`. `adapters` maps all four version names to
`{path, adapter_model_sha256, adapter_config_sha256}`. Root's separately maintained
`prepare_weight_manifest.py` produces this schema.

The retriever defaults to `http://10.120.17.101:18005/retrieve`. Admission contains
`accepted:true, retriever_url, pid, uid, starttick, source_identity_sha256,
full_gpu_gate_path, full_gpu_gate_sha256`. The full gate must be accepted and its
`candidate_health` must bind the tested process/source and `gpu_ready:true`.
Historical gate `ready:false` is expected before admission. Actual HTTP `/health`
must be ready and retain the same process/source identity at startup, before each
case and before each real query. Optional `health_identity_paths` maps the four
identity fields to exact dotted payload paths; `health_gpu_ready_path` defaults
to `gpu_ready`. Full gate content is hashed once and its exact metadata checked
subsequently. 18004 requires an explicit `--allow-fallback-18004` and its own
correct admission; it is never silently selected.

The three workers partition the immutable cases by `case_index % 3`. Their first
cases rotate through the three groups, then requested full-vocabulary cases have
priority. Exact execution order is saved. All generations are greedy, so this
changes scheduling only. No scientific learning parameters are introduced.

## Actual generation versus fixed response probes

`teacher-generation.json` is **actual autoregressive generation** by source100,
adapter disabled, with the successful whole reference in its user prompt.
`student-generation-{version}.json` is actual ordinary-prompt generation with
each of source100, update1, update5 and raw epoch1. Traces preserve rendered
prompts, generated IDs before structural cuts, actual policy spans, every search
query, raw top3 document/score payload, available document IDs, full observations
and the inserted480-token information, forced budget messages, termination and
strict protocol EM. Missing document IDs are explicitly marked; content hashes
or ranks must not be presented as corpus IDs. Generation budgets match the frozen
HF agent: greedy, segment500, context8192, four searches, top3.

`tokens-{version}-{trajectory}/tokens.jsonl` is **teacher forcing on the preserved
original student response**, not a newly generated response. Pair cells probe
the exact original mined student. Hard cells probe all five original failed
siblings without privileged teacher or OPSD loss. All response tokens have IDs,
token strings, character offsets, actual policy/information masks, target token
probabilities/logprobs, top32 token probabilities/logprobs and entropy for student,
ordinary source100 and (when applicable) privileged frozen source100.

Forward-KL uses the full vocabulary at every position, FP32 log-softmax from
BF16 fixed-model head logits. Records include unclipped KL, each-vocabulary
contribution upper-clip10 summed KL, saturated contribution count and teacher
probability mass, ordinary-base→student KL, and actual masked trajectory loss
contribution. Policy indices are evaluated **first in original32-position
partitions**, preserving native BF16 head row shapes and tail shape. Nonpolicy
tokens are evaluated separately and receive zero actual loss. Output is sorted
by original token index. The trajectory objective preserves native
`sum(dim=-1).sum()` and compares against the actual frozen `pair_loss`, invoked
in no-grad mode on fixed weights. CPU tests check exact equality and head shapes;
GPU checks record fixed tolerances (atol2e-5/rtol2e-4) and fail on disagreement.
Context excess skips the joint pair; teacher/student responses are never
independently truncated.

Every version's actually generated ordinary response also receives a separate
`actual-student-generation` token probe. A successful-reference case additionally
probes the actual privileged teacher output once with source100. These still
teacher-force the now fixed generated response; they are explicitly labeled
`new_student_generation`/`new_teacher_generation`, not autoregressive losses or
the original training objective. Joint-context skip rules remain unchanged.

## Bounded full vocabulary files

Default: no complete vocabulary arrays. `--full-vocab-case-id` may be repeated;
`--full-vocab-versions source100,epoch1` chooses snapshots.
`--full-vocab-max-bytes` is a **global total cap**, divided evenly among three
workers. Root's approved plan uses at most24GiB and at most one fixed question
per cell, selected from the192 cases. Worker can leave unused quota rather than
exceed the cap. A cap skip does not skip token/top32/full-computed-KL diagnostics.

Requested files contain both FP32 probabilities **and FP32 logprobs**, so
probability underflow does not destroy the log distribution. Six matrices per
pair probe (four for a hard probe), shape `[response_tokens, vocabulary]`, are
stored with metadata, SHA256 and exact bytes. Arrays include nonpolicy tokens.
Allocation reserves complete matrix bytes plus header allowance before writing.
Full arrays are limited to preserved original trajectories; the new generated
response probes retain all token/top32/full-computed KL statistics without
consuming this archive allocation.
All other cases retain top32 and full-computed scalar statistics; these cannot
reconstruct the complete vocabulary distribution. Neither top32 nor the stratified
sample's EM is a substitute for the native512/seven-benchmark result.

## CPU verification

```sh
python -m unittest -v test_select_cases test_token_probe test_admission
```

Selector tests use only synthetic CPU fixtures. Token tests use a small CPU toy
model and extract the audited, byte-identical `encoded_pair`, `pair_loss`, mask
and KL math functions; the archive's old reward is never imported. Tests cover
complete groups/provenance/determinism, rejected fake references and partial
inputs, native head/reduction equality, hard missing teacher, context skip,
bounded FP32 arrays and historical pre-admit gate versus live readiness.
These tests are software checks and are never represented as experimental
training, actual192-case completion or five real OPSD updates.
