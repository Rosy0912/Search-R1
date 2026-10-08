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
