# Saved LoRA parameter-space displacement audit

This is an independent, read-only CPU audit for the plan §8.3. It reads saved
FP32 LoRA parameters, not a loaded language model. It never initializes CUDA,
imports Transformers/PEFT, generates responses, trains, restores RNG or Adam,
changes source/checkpoints, or copies tensor payloads to Mac. The CLI emits a
compact JSON audit to stdout only. Run it on HKUST4 UID1001 with explicit empty
`CUDA_VISIBLE_DEVICES`, the original Python environment and `-B`.

## Precisely what is measured

For named saved LoRA coordinates \(z=(A_1,B_1,\ldots,A_{252},B_{252})\), two
endpoints produce:

\[
\Delta_{a\to b}=\|z_b-z_a\|_2,\qquad
R_{a\to b}=\|z_b-z_a\|_2/\|z_a\|_2.
\]

Each FP32 tensor is converted to FP64 **before** subtraction and squaring;
tensor squared norms are summed with `math.fsum`, then square-rooted. The audit
reports combined A/B norms, A-only and B-only norms, and the same metrics for
each of the 36 layers. Relative L2 uses the **source endpoint** parameter norm.
When that norm is zero, the ratio is JSON `null` with
`ZERO_SOURCE_PARAMETER_NORM`; this is expected for common-zero B. A zero delta
with a nonzero source norm has relative L2 zero. Nonfinite inputs/reductions
are errors, never neutral values.

These are **LoRA parameter-space displacements**. They are sensitive to the
factorization coordinates: multiplying A by c and B by 1/c can change these
norms while leaving BA unchanged. No effective model-weight `(alpha/r) BA`
delta, gradient, actual optimizer update-vector or causal EM influence is
computed. A larger displacement does not imply a larger answer-quality loss.
Sparse endpoints cannot locate an exact onset of degradation.

## Reused immutable contracts

`common.py` is an exact independent copy of the published completion helper's
immutable reader/source-only binder. Training modules are source-bound under
an isolated package, before importing Torch. The audit reuses the **original**
`checkpoint.check_checkpoint` and `targets.tensor_digest` without patching them.
There is no `check_initial` function in the original checkpoint module; this
audit explicitly checks the original common-initial manifest, two payload SHAs,
504 finite FP32 tensors, matched native digest and 252 exactly zero B tensors.

The checkpoint reader freshly hashes all four distinct rank-state files and
the shared optimizer file, checks declared native contracts, run/config/outer
counters and adapter payloads. This audit does **not** deserialize Adam or RNG,
does not assert independently verified moment equality, and clearly labels
that scope. Actual adapter bytes are loaded on CPU and their 504-tensor native
digest must match the checkpoint's four-rank declaration. A selected best
adapter, sampling merged cache or adapter outside its raw checkpoint is
refused. A checkpoint with `next_pair_cursor != usable_pair_count` is refused.
Complete K0 endpoints remain possible; no artificial update or token loss is
created.

Source100 twelve files and the original 305-file runtime are freshly hashed
before and after the numerical audit. The same common source/runtime/cohort/
train/validation/initial identities are required across all four configs.
Actual endpoint optimizer updates and cumulative pair exposure are recorded,
and formal endpoints must match their complete coverage rows exactly.

## Request and publication

The request is an immutable JSON file, passed with its own exact SHA:

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 ORIGINAL_ENV_PYTHON -B \
  /REMOTE_PUBLISHED_DIR/movement.py \
  --request /REMOTE_PRIVATE_DIR/request.json --request-sha256 EXACT_SHA
```

The actual `movement.py` must be in the request's freshly verified published
`audit_source.helper_root`; publication must name `search-e1`, accepted true,
the same 40-character source commit/readback SHA and exact source manifest.
Source inventory changes or missing files are errors. No Node output file is
created by this helper; the authorized caller can preserve stdout remotely
and copy only a small audit summary locally.

Top-level request schema:

```json
{
  "schema": "opsd-signal-parameter-movement-request-v1",
  "mode": "final",
  "audit_source": {
    "helper_root": "/REMOTE_PUBLISHED_DIR",
    "source_manifest": {"path": "/REMOTE_PUBLISHED_DIR/source-manifest.json", "sha256": "EXACT"},
    "publication": {"path": "/REMOTE_RECEIPTS/publication.json", "sha256": "EXACT"}
  },
  "study_manifest": {"path": "/REMOTE_RECEIPTS/study-manifest.json", "sha256": "EXACT"},
  "common_initial": {"path": "/REMOTE_RECEIPTS/common-initial-adapter.json", "sha256": "EXACT"},
  "configs": {
    "G_POS": {"path": "/REMOTE_GROUP/config.json", "sha256": "EXACT"},
    "G_NEG": {"path": "/REMOTE_GROUP/config.json", "sha256": "EXACT"},
    "H_POS": {"path": "/REMOTE_GROUP/config.json", "sha256": "EXACT"},
    "H_NEG": {"path": "/REMOTE_GROUP/config.json", "sha256": "EXACT"}
  },
  "endpoints": {
    "G_POS": [
      {"outer_index": 14, "optimizer_updates": 0, "cumulative_pair_exposure": 0,
       "checkpoint": {"path": "/REMOTE_EXACT_RAW_CP/manifest.json", "sha256": "EXACT"}},
      {"outer_index": 24, "optimizer_updates": 0, "cumulative_pair_exposure": 0,
       "checkpoint": {"path": "/REMOTE_EXACT_RAW_CP/manifest.json", "sha256": "EXACT"}}
    ],
    "G_NEG": [], "H_POS": [], "H_NEG": []
  },
  "initial_coverage": {
    "actual_stdout": {"path": "/REMOTE_AUDIT/actual-cli-stdout.json", "sha256": "EXACT"},
    "output": {"path": "/REMOTE_AUDIT/coverage.json", "sha256": "EXACT"},
    "execution": {"path": "/REMOTE_AUDIT/actual-execution.json", "sha256": "EXACT"},
    "helper_root": "/REMOTE_COMPLETION_SOURCE",
    "source_manifest": {"path": "/REMOTE_COMPLETION_SOURCE/source-manifest.json", "sha256": "EXACT"},
    "publication": {"path": "/REMOTE_COMPLETION_RECEIPTS/publication.json", "sha256": "EXACT"}
  }
}
```

This illustration is **not executable**: replace every placeholder, populate
all four arms' actual two-endpoint lists and use actual counters, not zeros.
Final mode requires four original configs with 25 outers, exact eight raw
outer14/24 checkpoint references and the complete assembler's actual
CLI stdout → coverage artifact → input request chain. The bound report must
have 100 unique outer rows, 256000 TRAIN slots and 4608 native512 slots. Each
raw checkpoint and update/exposure counter must match that report; each
formal15/25 evaluation must bind that same raw checkpoint and native tensor
digest. The common-zero endpoint must bind the original initial adapter.

`coverage_accepted=true` is required. `final_accepted` may be false because
analysis itself is an outstanding external deliverable: requiring goal-final
acceptance would create a circular dependency. This audit's
`full_study_complete=true` means **complete parameter-endpoint scope**, not
completion of the overall experiment goal. It is a separate analysis proof;
it does not silently extend the old assembler's external requirements.

Prefix mode has exactly one config key and one protected, already fully
committed checkpoint endpoint. It has no `initial_coverage` field and returns
`full_study_complete=false`. Its sole comparison is common-zero → this actual
checkpoint, labelled by its actual completed outer. It proves a numerical/
interface audit only and is never a formal15/25 curve or full-study result.

## Meaningful CPU fixtures

```text
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHON -B \
  -m unittest discover -s THIS_HELPER_DIR -p 'test_*.py' -v
```

Fixtures use 504 **small** FP32 tensors and real original checkpoint-reader
interfaces, with no model weights. They cover known aggregate and A/B norms,
zero B denominator, equal endpoints, native tensor digest, missing tensor or
rank file, wrong dtype/rank/shape, nonfinite input, wrong outer, uncommitted K,
false native acceptance or incomplete coverage, mutated SHA, cache/best
substitution, FP64 protection against FP32 norm overflow, and a factorization
counterexample where BA is unchanged despite nonzero parameter displacement.
Torch is a required CPU test dependency; missing it is an error, not a SKIP.
Tests do not require Safetensors, Transformers, PEFT, a GPU or a network.
On a standalone Node deployment, set the test-only
`OPSD_MOVEMENT_TEST_TRAINING_ROOT` to the exact published original
`.../training-00fec71/opsd_signal_weight_v1` directory so fixtures reuse its
`checkpoint.py` and `targets.py`. This variable is read only by the tests,
never the production audit. Missing original test source is a hard error.
Production Safetensors loading is separately checked on the original Node
environment, after source publication and deployment.
