# Search-G1: question-only diversity probe

This branch starts from `codex/search-g1-mixed-first-200` at
`b7f91f33c00e54cdc192b31dd72ae95f67132254`. It adds an opt-in selector **before
rollout**. It does not change the current running experiment, deploy a new job,
or claim improved validation EM. The original mixed-first path remains available.

## One outer iteration

1. Draw 5,120 unique canonical training questions with the saved source sampler.
2. Read fixed question-only features; score them with the frozen current CPU head.
3. Uniformly select `exploration_questions=E` first, without replacement. Select
   the highest predicted diversity `512-E` questions from the remaining pool.
4. Generate exactly five complete real search trajectories for each selected
   question: 512 questions, 2,560 trajectories. Never refill using observed EM.
5. Keep all groups, including all-wrong and all-correct, for the original Search-G1
   reward, advantage and PPO update. Strict terminal EM supplies probe labels;
   it does not replace Search-G1's shaped training reward.
6. Following a successful outer update, fit the question head once on the observed
   pre-update actor labels and recent replay. Use that updated head next iteration.
   An attempt rejected entirely by existing policy guards does not advance the
   outer counter or train the head; it consumes its candidate stream and has an audit.

`E` is an experiment hyperparameter, not a hard-coded 64. The CLI exposes
`0 / 64 / 128 / 256`; the runtime supports integer `0..512`. Each ablation keeps
512 questions and five responses. `E=0` is allowed and has no unbiased random
exploration slice; metrics explicitly mark that slice unavailable.

## Prediction target and loss

For five binary strict terminal EM labels `e_j`, let `K=sum(e_j)`. The observed
mean and **sample standard deviation (ddof=1)** are

$$m(K)=K/5,\qquad s(K)=\sqrt{K(5-K)/20}.$$

| K | Mean | Sample std |
|---:|---:|---:|
| 0 | 0.0 | 0 |
| 1 | 0.2 | 0.447214 |
| 2 | 0.4 | 0.547723 |
| 3 | 0.6 | 0.547723 |
| 4 | 0.8 | 0.447214 |
| 5 | 1.0 | 0 |

A LayerNorm → Linear(64) → GELU → Linear(6) head predicts `p(K=k | question)`.
It induces mean regression and std regression without violating binary moment
constraints:

$$\hat m=\sum_kp_km(k),\qquad \hat s=\sum_kp_ks(k),\qquad
\widehat{\mathrm{mixed}}=1-p_0-p_5.$$

Rank by `expected_std` (the conditional expectation of the five-response sample
std). **Do not** rank by predictive entropy, `sqrt(mean*(1-mean))`, or
`sqrt(expected_variance)`. For example, `p_0=p_5=0.5` has predicted mean 0.5 but
expected within-group std **0**: either all five fail or all five succeed.

$$L=CE(K,p)+0.1(\hat m-m)^2+0.1((\hat s-s)/\sqrt{0.3})^2.$$

Head defaults: Adam LR `3e-4`, two epochs per successful outer, batch 256, hidden
64, last 16 successful outer rounds, recency half-life 4, seed `20261013`.
The last linear layer starts at zero: an untrained head ties every question and
selection tie breaking is independent of actor RNG. This avoids treating random
initial predictions as meaningful rankings. The first actual labels make it
adaptive. Recency replay addresses policy drift; it does not eliminate it or
turn selectively observed labels into an unbiased training-population estimate.
No naïve inverse propensity correction or class-balanced probability claim is made.

## Feature cost and input isolation

The prepare command encodes each training prompt once using the frozen official
`Qwen/Qwen2.5-3B-Instruct`, revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, decoder layer 27 (zero-based), last
nonpadding prompt token. It runs the base transformer without an LM head;
it does not generate CoT, search or an answer. The full base forward executes;
this implementation does not claim early exit at layer 27.

The feature namespace binds model files, revision, tokenizer, exact question
text, prompt extraction, compute precision, layer, pooling, canonical train
manifest and file hashes. Runtime question scoring receives only `qid/question`,
never answers, observed rewards or rollout text. Only the selected questions
are materialized as native training inputs. Validation/stage/final questions
are excluded by canonical split verification. The common data manifest is
`0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`.

For 169,028 questions and 2,048-dimensional FP16 features, the feature array is
approximately 660 MiB, plus the index. Keep it, raw audits, head snapshots,
actor/optimizer states and all output on explicitly named remote storage.
Only source, documentation and small synthetic test fixtures belong on the Mac.

## Prospective preparation (remote only; no job launched by these commands)

Stage this branch into a **new** common8 task's `BASE/repo` **before preparing
its run manifest and method**, then use the existing common8 prepare/admission
pipeline. Binding rejects source bytes changed after preparation; it does not
backdate the method or source inventory. Do not bind or mutate an already claimed,
admitted, running or completed run. Set the variables below to actual verified
remote paths, not the local workstation.

```sh
python -m search_g1.qprobe_prepare precompute \
  --data "$BASE/data" --model "$OFFICIAL_MODEL" \
  --official-manifest "$OFFICIAL_MANIFEST" \
  --cache-dir "$REMOTE_FEATURES" --device cuda:0 --batch-size 4
python -m search_g1.qprobe_prepare verify-cache --cache-dir "$REMOTE_FEATURES"
python -m reproduction.common8.qprobe_bind \
  --base "$BASE" --cache-dir "$REMOTE_FEATURES" \
  --retriever-version "$IMMUTABLE_INDEX_MANIFEST_SHA" \
  --exploration-questions 64 --seed 20261013
```

Precompute requires an available, explicitly allocated GPU if `cuda:0` is chosen.
It must not occupy another experiment's GPU. CPU is supported, slower. Model
loading is local-files-only and never writes a download cache to the Mac.
Run binding performs full feature-cache readback, freezes the exploration/head
configuration and verifies the already prepared runtime source inventory. Launch afterward with
this repo's existing resource admission and common8 pipeline; no alternate
shortcut bypasses optimizer or policy-safety checks.

Optional bootstrap accepts complete, authentic **all-candidate** history,
including rejected all-wrong groups, with five unique trajectory IDs and the
actual strict EM vector for every group. Each audit must bind its actual actor,
source commit, sampling/scoring contract and retrieval version. An external
provenance assertion cannot relabel a different protocol. No future-policy
labels, evaluation questions or accepted-only exports are admitted.

```sh
python -m search_g1.qprobe_prepare bootstrap \
  --data "$BASE/data" --cache-dir "$REMOTE_FEATURES" \
  --audit "$ALL_CANDIDATE_AUDIT" --provenance "$LINEAGE_PROOF_JSON" \
  --config reproduction/common8/qprobe-head-defaults.json \
  --output "$REMOTE_BOOTSTRAP_HEAD"
```

A fresh actor requires a bootstrap head with `completed_outer=0` and the exact
head configuration (including seed); add `--initial-head-checkpoint` when
binding. History from a later actor is not a shortcut to initializing a fresh
actor. Legacy audits missing actual protocol evidence are rejected; the supported
fallback is a cold head. Later actors require a completed paired receipt, below.
The defaults JSON must be adjusted consistently if a different seed is selected.

## Checkpoint and recovery

Each saved checkpoint contains the head and its Adam/replay/RNG state, feature
namespace, source sampler order/cursor/pending rows, actor identity, real outer
and physical optimizer counters, and worker/driver RNG. A post-checkpoint boundary
receipt outside the immutable actor directory pins the old reward probe **after**
any scheduled 50-iteration refresh. This prevents restoring actor 50 with the
pre-refresh reward probe or with a newer live manifest.

The common8 `prepare_run --restore-receipt` route accepts a
`QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS` receipt from
`artifacts/question-probe/completed-pairs/global_step_N/source-receipt.json`.
The native common8 preparation entry requires the original eight-rank layout.
The four-rank local preparation entry can bind a four-rank pair, but also needs
a restoration-aware method document; the default fresh deployment template is
not automatically converted or deployed. Both paths still require actual GPU
admission and restoration checks.
The new execution preserves the source run/actor lineage and selector configuration;
its output directory is new. It restores the authoritative paired head, not an
arbitrary `snapshot_path` or bootstrap head. Source hashes, all worker counters,
scheduler, candidate order, pinned reward version and head/actor identity must
agree. Worker RNG is captured again at the completed post-evaluation/refit
boundary, outside the immutable actor directory; recovery uses this later RNG
with the checkpoint model/Adam state. Historical skipped attempts are counted separately from completed outers. A
checkpoint already at 200 is an inspection artifact, not a new training resume.
For generic recovery, final verification distinguishes saved-prefix state from
the newly measured suffix; missing historical raw logs or evaluations are
explicitly reported and never reconstructed as fresh measurements.

Publication validates full file SHA and saved CPU state. **Actual CUDA/FSDP
restore acceptance is a separate runtime gate**, requiring model, optimizer,
scheduler, RNG and allocation readback. CPU fixtures cannot establish GPU restart
correctness. Changing E or the head configuration during exact restoration is
rejected; that is a separate experiment, not an exact resume.

## Evidence, comparisons and limitations

Per-attempt prediction ledgers precede rollout and label discovery. Selected
strict-EM vectors and prequential NLL/mean/std error are saved before head fit.
Exploration and predicted-selection slices are reported separately. Saved
head/selection receipts are immutable and checked on restore; failed head
publication rolls back its optimizer, replay and RNG. Existing Search-G1 reward
probes continue their original 50-iteration update schedule. Actor learning rate,
512×5 budget, mini-batch 128, 20 optimizer **opportunities**, warmup, and the three
policy/numerical protections are unchanged. Actual writes can be fewer if the
existing guards skip opportunities; the logs report actual counts.

Compare uniform 512, original post-rollout mixed-first and E=0/64/128/256 using
both equal successful-update budgets and equal total cost. Count feature
precompute, head fit, candidates/rollouts, search calls and wall time. This method
removes extra candidate rollouts; it does **not** guarantee a fourfold end-to-end
speedup or higher EM. The current question-only predictor has a one-round label
lag and limited knowledge of actor drift. Its calibration and selection benefit
must be measured in the real experiment.

The implementation includes CPU tests of the mathematical target, learning,
leakage boundary, exact sampling budget, native DataProto collection, failure
handling, paired restoration and unchanged legacy configuration. No new GPU
training was started for this implementation request.

## Animation

[Open the standalone interactive demonstration](visualizations/question-diversity-probe-demo.html).
Native editable Canvas: `visualizations/question-diversity-probe.canvas.tsx`.
Six phases show candidates → question encoding → K-distribution → random/top
selection → five actual EM labels → separate actor and head updates. Controls
switch E and K, pause/advance/replay, and respect reduced-motion preferences.
All depicted predictions and outcomes are synthetic explanatory examples.

### MP4 export

The same six scenes can be exported as a 52-second, 1920×1080, 30 fps H.264
video with Chinese explanatory captions and no audio. Export uses the existing
standalone demonstration and never starts training or accesses model data.
Frames stream through memory into FFmpeg; no full frame sequence is stored.

```sh
node docs/visualizations/export-question-diversity-video.mjs /your/output/question-diversity-probe.mp4
```

Requirements: Node.js, Playwright, Chrome/Chromium, FFmpeg and FFprobe. The shared
visualization helper supports `PLAYWRIGHT_MODULE_PATH` and
`PLAYWRIGHT_EXECUTABLE_PATH`; the exporter also accepts `FFMPEG` and `FFPROBE`.
Appending `--preview` writes six scene previews instead of a video. A successful
export writes a sidecar `*-video-verification.json` with source/output SHA256,
scene checks, dimensions, duration and decoded frame count. Verify the exported
video visually as well; this receipt is not evidence of training performance.

The `--pace brisk` profile runs for 30 seconds, doubles the in-scene animation
speed, shortens transitions to 250 ms and uses concise captions. An optional
`--music` file is mixed as quiet AAC stereo at 48 kHz, targeting -24 LUFS.
The included composer creates an original soft piano melody/arpeggio with a
subtle warm pad, without downloaded samples, voices or percussion (Python and
NumPy required). Its 30-second score has smooth fades at both ends.

```sh
python3 docs/visualizations/compose-question-probe-music.py --output /your/output/music.wav
node docs/visualizations/export-question-diversity-video.mjs /your/output/question-probe-brisk-music.mp4 --pace brisk --music /your/output/music.wav
```
