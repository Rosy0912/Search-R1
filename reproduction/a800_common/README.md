# Search-G1: A800 four-GPU formal200

This adapter runs the existing formal200 objective from the official
Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`,
with fresh Adam, scheduler, RNG and sampler. It does not load earlier A800
checkpoints or continue the separate H100 run.

| Quantity | A800 placement |
| --- | --- |
| Training GPUs | Physical 0–3, four ranks |
| Retrieval / probe GPUs | Retrieval 5–6, probe 6; physical 4 and 7 excluded |
| Outer batch | 512 questions × 5 responses = 2560 trajectories |
| PPO mini-batch | Global 128, local 32 |
| Actor micro-batch | Local 2, actual global 8 |
| Accumulation per mini-batch | 16 forwards/backwards per rank |
| Old/reference log-prob micro-batch | Global 32, local 8 |
| Optimizer opportunities | 20 per outer; successful Adam writes counted separately |
| Learning rate | 1e-6, 57 completed-outer warmup, then constant |
| Saves | 5, 50, 100, 150, 200 |

One trainer processes fresh0 through200. Checkpoint5 and its actual acceptance
are inline; passing that acceptance continues to6 in the same process. All20
opportunities must be processed and at least one Adam write must return for an
outer and one warmup advance to count. All-skipped or abandoned attempts do not
count; partial writes are retained and explicitly reported.

The canonical common-data manifest is
`0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`.
The original train169028/dev512/stage64/final51713 files are copied without
repartitioning. Training and validation refer to their real file hashes and
ordered IDs. The actual full2560 sampling-context/log-prob gate precedes Adam;
the launch plan labels it prospective until that real gate runs.

`prepare_run.py` binds a fresh namespace to published source, official local
model bytes and canonical data. `run_pipeline.py` starts actual local retrieval,
initial probe calibration, the frozen endpoint, and `train.sh`. Assigned GPUs,
host and process births replace Slurm identity; `slurm_job_id` is null. Historical
eight-GPU smoke tests are not synthesized as successful local evidence.

The local deployment/observation entry is `deploy.py`. It first requests the
published GitHub branch; a direct-download failure can use a Git bundle of the
same already-published commit through the Mac. Credentials, model weights and
environments are never copied into the repository. Existing A800 environment,
official model and retrieval assets remain shared resources; only the new run's
private files and recorded process births are eligible for cleanup.

Model execution uses the existing `/data/search-g1/env/bin/python`, headers in
`/data/search-r1/bootstrap-tools/python-dev`, and isolated per-run Ray,
Triton/Inductor/HF caches. The original model and retrieval hashes are verified
at deployment/runtime. The Mac controller dependency is the user's existing
`Documents/remote_codex_control/control.py` SSH route; deployment changes no
SSH/VPN settings or authentication files.
# Local startup recovery

Ray uses the manifest-bound private `/tmp/g-<execution_id>` root so its Unix
socket path stays within the system limit. The old failed namespace remains
unchanged. A new deployment may specify `--initial-probes-source-base` for a
completed official-model initial0 calibration. Its real labels, fitted heads,
model/data fingerprints and unchanged scientific producer bytes are verified;
copied heads receive a new run version with the original provenance. This does
not inherit trained weights or optimizer state, or claim a new calibration.

## Prospective question-probe teacher warmup

Use `deploy_qprobe.py` for this branch. Its `stage`, `prepare`, and `bind`
actions cannot start the pipeline. The official actor stays fresh; one extra
frozen-teacher outer generates all 5120 canonical question groups × 5 actual
responses, fits the new question-selection head on all 5120 groups, and restores
the actor RNG boundary. The actor then performs 200 successful training outers
with candidate5120/selected512/G5/E64. The extra teacher outer is outside that
budget and performs no actor optimizer or scheduler writes.

The existing G1 binary reward probes remain separate from this new question
selection head. Preparation defaults to the original frozen initial0 source
`/data/shaoshitong/Search-G1-a800-fresh200-7895330a2c584bba9940720e98f006ef`;
`--initial-probes-source-base` can select an explicit source only at `stage`.
The existing adoption verifier still checks its real model/data/producer
provenance before the pipeline can use those reward probes. The new adapter
does not claim that this later GPU-stage adoption has already passed. Reward
probe refresh and save schedules remain 0/50/100/150 and 5/50/100/150/200.

Publish the exact source commit first, then use one small local handle throughout:

```text
python -m reproduction.a800_common.deploy_qprobe stage --handle HANDLE.json \
  --source-commit FULL_PUBLISHED_SHA \
  --published-ref refs/heads/codex/qprobe-teacher-warmup-a800-20261014 \
  --baseline-intent-sha256 21ab1fc56365501b2082808f660dc7f3f130bedc9bb4748c63ada21c6032de3a \
  --teacher-warmup-cache /data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v2
python -m reproduction.a800_common.deploy_qprobe prepare --handle HANDLE.json
python -m reproduction.a800_common.deploy_qprobe feature-plan --handle HANDLE.json
python -m reproduction.a800_common.deploy_qprobe inspect --handle HANDLE.json
python -m reproduction.a800_common.deploy_qprobe bind --handle HANDLE.json
python -m reproduction.a800_common.deploy_qprobe launch --handle HANDLE.json --review-receipt REVIEW.json
```

`stage` verifies the pushed generic branch ref, creates a Git bundle below32MiB
that reuses the original A800 checkout's baseline objects, and installs an
independent source namespace. It sends no untracked data/model/environment
files. `prepare` creates a new fresh200 namespace, copies and SHA-verifies the
six canonical inputs plus original official model manifest and formal method
template directly from the original completed1db base. The original deployment
intent must match the supplied exact SHA. Only source commit, source-map hash
and creation time change in the formal method. The old baseline's ninth input,
its historical source bundle, is excluded from the eight copied inputs.
Preparation uses CPU with CUDA
hidden and preserves BF16, actor local2/global8, accumulation16, reference
local8, formal guards and LR warmup57.

`feature-plan` only prints the separate remote feature job's argv, environment
and progress paths. Run that job on A800 from the new run's `repo`, using the
printed environment and explicit auxiliary physical GPU5 or6 after checking
availability. `precompute_qprobe.py` invokes the existing feature implementation,
checks the actual single visible GPU and refuses an occupied device. It freezes
`CUDA_VISIBLE_DEVICES` to the admitted NVML UUID before importing Torch, then
checks the actual CUDA-runtime UUID against that physical GPU. It writes
`preparation/question-features-progress.json` with completed question counts,
PID birth and exit status, then an actual `features-ready.json` after full cache
verification. `inspect` returns these bounded summaries plus the existing
deployment observation. `bind` follows completed feature preparation and fixes
the immutable teacher cache path from the stage intent and output paths under
the new remote run.

The teacher cache defaults to the stable remote namespace
`/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v2`, so a later fresh
actor launch can reuse accepted trajectories. Choose another explicit
`--teacher-warmup-cache` only when creating a new intent. Its real path and
parent must be owned by the current remote user and contain no symlink. The
schema2 scientific identity binds model/data, question sampling/configuration,
retrieval and relevant producer source. Existing identity/chunks are never
overwritten or erased; a mismatch requires an explicitly selected new namespace.
Reused chunks receive the new run's lineage while the teacher cache preserves
its original provenance. `inspect` reads completion evidence from that bound
shared path.

The final `launch` reuses `deploy.py`'s original `REMOTE_LAUNCH_PREPARED` and its
compatible intent/receipt schema. Local launch requires the same published HEAD,
a clean checkout, the full committed and disk runtime source maps, and exact
committed bytes of both deployment modules. The imported launcher text must
also equal the committed snippet; the remote gate compares the submitted
snippet/hash with its verified published deployment module before executing it.
It requires an actual independent code-review
JSON receipt with role `a800_qprobe_independent_code_review`, `approved: true`,
`independent: true`, distinct nonempty `reviewer` and `implementation_actor`,
the exact `source_commit` and complete `source_sha256` map, a past positive
`reviewed_unix`, and `checks` containing exactly these seven keys set to true:

```json
{
  "official_model_and_canonical_data": true,
  "frozen_teacher_5120_x_5": true,
  "bootstrap_all_5120_groups": true,
  "exploration_64": true,
  "formal_200_not_201": true,
  "four_rank_bf16_local2_global8_accum16_ref8": true,
  "unchanged_guards_lr_warmup57": true
}
```

The adapter never manufactures that review. Its launch gate independently
checks the bound E64, all5120 bootstrap limit, remote teacher cache, canonical
feature cache and four-rank recipe before the original launcher performs its
actual process/GPU admission. Every mutation has an exclusive local/remote
claim. A timeout or failure preserves the same namespace and forbids automatic
retry; use `inspect` and diagnose the real state. The local source bundle is
deleted only after A800 returns the matching SHA receipt. Keep a failed relay
bundle until its target is verified. No SSH configuration or credentials change.
