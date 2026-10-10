# Historical individual TRAIN reward reconstruction

This is an independent diagnostic on the existing Search-G1 Baai A800 node.
No original online per-trajectory reward was persisted. Every resulting value is
`R_reconstructed`, never an original online recorded reward. Iteration averages
and calibration subsets are not substitutes for the requested individual pairs.

The complete accepted input scope currently is TRAIN1–146: 146 × 2560 trajectories.
147–150 were not yet committed at the audited snapshot and must not be invented.
C is computed independently by the published bounded-stream CPU V2 helper. R can
start as soon as C's first complete accepted step exists, and waits for each
following C step if necessary. Outputs stay entirely on the original node:

- Suggested new parent: `/data/shaoshitong/Search-G1-reward-reconstruction-20261010-1ec36152611a`.
- C source/output: that parent's `raw-C` directory.
- R source, immutable configuration, logs, per-row `pairs-R-reconstructed.jsonl`,
  status, summaries and `memory.jsonl`: a new independent sibling R directory.
- HF cache and temporary files explicitly use that new output's `cache` and `tmp`.
- Mac may retain source/configuration/docs and small gates or step summaries only.
  Do not copy traces, weights, hidden features, per-row pair files or model cache.

## Exact mechanics and identity

Fresh original source SHA binds the original reward `_score_probe`, probe
`_load`, `_last_hidden`, `_commit_ctx` and KK partition functions. Their exact
AST bodies run in an isolated namespace; original server/trainer modules are
not started/imported. Sources remain unchanged.

Each step is the originally committed attempt, policy snapshot and probe version.
Each complete raw trace is SHA-streamed and its 2560 actor rows independently
canonical-hashed against the native gate. Initial-model tokenizer decoding,
response information-mask segments, answer validity, authored search counts and
all decoded/mask SHA values must match the independently accepted C rows.
Original attention lengths include prompt, observation and correction tokens.
Original world4 KK equal-size permutation, including tie handling and partition
order, restores the original reward request order. No index zip after balance.

All2560 requests, including invalid and wrong outputs, remain present. No prompt
deduplication or selective scoring changes padding/batch shape. The corresponding
historical version's own tokenizer/encoder/head are used. There is one complete
commit pass (640 micro4 batches), then one complete boundary pass (640 micro4):
left padding and left truncation4096, `add_special_tokens=False`, BF16 HF SDPA,
`model.model`, `use_cache=False`, exact head layer−1 hook, padded final position,
FP32 CPU hidden vectors and one `predict_proba` call per full2560 feature matrix.
No new generation, retrieval, HTTP, probe reload or training update occurs.

The actual original environment whitelist plus source defaults bind:
`PENALTY=.1, MAX_SEARCH=2, wrong_floor=.05, bal_ground=.6,
bal_nosearch_bonus=.5, w_cost=.02, wrong_ground_scale=0, probe_base=0,
probe_gate=true, probe_alpha_cost=false`. The frozen `_score_probe` executes.
Its scalar is rounded to FP32 at native terminal response-token placement.
C is strict final policy-span binary EM; it is not the `.05` wrong floor.

Each reconstructed step must match original exact `actual-updates` FP32
`critic/rewards/mean` within2e−6, and printed score min/max/mean and EM within
0.0005001. Aggregate agreement is necessary consistency only: it does not prove
unknown historical online per-row bitwise equality. Per-row pairs are exported
only to the independent remote directory after this gate. Pearson sufficient
statistics use every individual C/R pair in the accepted step, never step means.

## Resource contract and launch

Only own auxiliary GPU5 is considered, physical UUID
`GPU-21508d9d-bd43-ff94-7bee-1227ec7b007b`. It shares the original retriever;
it is not a dedicated idle GPU. Do not use training0–3, excluded4/7 or production
probe6. Do not modify original process ledgers, locks, routes or monitors.

Launch in a new session (`setsid`, or an equivalent direct new-session exec),
`CUDA_VISIBLE_DEVICES=5`, original Node environment, original CPU numeric thread
settings. Before Torch import: approved immutable config, published source
manifest, fresh full historical encoder/head SHA, actual run owner and exact
GPUUUID/original retriever PID/starttick/SID/UID, no foreign process,
actual free≥32768MiB, and a new independent replay lock. No free-memory snapshot
by itself is treated as allocation authorization. Current human authorization
allows this diagnostic within the existing run's resources; root writes the
explicit config authorization `user_existing_run_auxiliary_full_reward_diagnostic`.

Own allocation is capped at24576MiB via CUDA allocator and direct-process
sampling; candidate actual free must stay≥8192MiB. A separate remote2s observer
accepts only the bound original retriever and the exact new replay process.
A violation stops this own diagnostic through a HOLD; it never signals anyone.
The original `monitor_only` occupancy sampler and ownership checks are left
unchanged; a separate SID is not a member of an original component's session.

CLI:

```
python -B replay_gpu.py --config <immutable_remote_config> --config-sha256 <SHA> \
  --source-manifest <published_manifest.json> --source-manifest-sha256 <SHA>
```

Config fields are visible in `replay_gpu.py`: original base/run/node/SHA,
C base and published C helper path/SHA, accepted full-encoder CPU preflight
path/SHA, original reward-parameter identity proof path/SHA, remote fresh output,
start/end steps, reward constants, GPU index/UUID, initial/running free floors
and own memory cap. This implementation never starts itself from preparation.

Run all CPU tests locally and in original Node environment before deployment.
After actual GPU launch, verify two complete real step summaries and native gates;
PID liveness or first model initialization is not completion. Measure the first
complete iteration time before estimating duration. Preserve failed partials and
own private error log. Full-scope correlation is available only after every
requested step has a complete accepted per-row artifact and SHA.
