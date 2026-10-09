# C1 + P + R: one hundred online outers, native512 every ten

This independent CPU supervisor starts one new PR training run from source
archive GRPO100. It never restores GRPO Adam. The training config stays frozen
at `outer_iterations=100`. Its controller runs to outer10, returns real0 with
`TRAIN_BOUNDARY`, and releases all GPU children. The supervisor evaluates the
actual raw outer9 checkpoint on the original heldout512. It then calls the same
training run with `--resume --until-outer20`, restoring full Adam/RNG/cursor state.
This continues through100; ten evaluations yield **5120 primary predictions**.
The first10 + first512 completion is recorded immediately and the pipeline keeps
going without restarting training from zero.

PR combines C1's LR1e-6/fixed teacher/unit pair weighting with P's FP32 master,
gradient and optimizer state plus FP16 main training/sampling, and R's
deterministic uniform student choice among all five siblings. There is no M
mixture, W weighting, or extra GRPO stage. The separately frozen training source
defines the scientific settings and owns training admission and actual state.

The evaluator reuses the accepted native HF SDPA protocol: ordinary greedy,
BF16 source100 base plus unmerged FP32 raw adapter, repetition1.05, EOS151645 or
151643, context8192,500 new tokens/turn,5turn/4search/top3/observation480. Eight
physical workers each evaluate `[rank::8]` (64 questions, seed42); no privileged
reference is supplied. Full HTTP, raw generated IDs and strict final-action
scoring remain Node-only. Eight original hook null controls are extra and never
count in512. No new source baseline and no FP16 deployment assessment is added;
the same source/runtime/data/protocol accepted BF16zero239 baseline is reused.

## CPU preparation and publication

```sh
python -B prepare_pipeline.py --training-config B/receipts/training-config.json \
  --training-config-sha256 SHA --base B/periodic \
  --helper-root B/source/PUBLISHED/opsd_pr100_periodic_v1 \
  --output B/periodic/receipts/pipeline-template.json
```

This creates ten separate eval templates and one NOT_READY pipeline. Root fills
each eval template's source commit/manifest/publication/storage (all referring
to this immutable new directory), plus the pipeline source/publication,
`startup_previous_driver_terminal`, and `baseline_summary`/`bf16_zero`. Root
writes final templates to independent files and replaces the template refs in
one new final pipeline config with `state=READY`. Planned observer refs are
present but their nonexistent SHA stays null until actual startup; no fake
checkpoint/terminal/observer hash is introduced during preparation.

The previous original eight-card driver must have an actual accepted0 terminal.
Fresh native GPU/oracle/transition admission still rejects any current foreign
or unreleased work. The supervisor holds only `pipeline.lock`; training and
evaluation child controllers take the same existing GPU/study/group leases.
The parent never takes a child GPU lock and never starts both phases concurrently.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 E1_RUN_ID=PIPELINE_RUN \
  python -B supervisor.py --pipeline-config FINAL_CONFIG \
  --pipeline-config-sha256 SHA
```

The source-bound scoped CPU role uses exact entry `supervisor.py`, actual
`--pipeline-config` and SHA argv, and the pipeline run marker. A new independent
memory observer is bound to the immutable training config and recorded before
training. No old observer is reused. Each boundary has actual Popen wait0 and
full release evidence, immutable status snapshot/terminal, raw checkpoint,
separate final eval config, strict per-dataset EM and paired question bootstrap
against the source baseline. Curves are `pipeline/curve-through-NNN.json`.

Ten selected evaluation-boundary checkpoint mirrors run as independent CPU
children on the same HKUST node. They never block a training/evaluation start.
Only those ten checkpoints are mirrored; original training retention and logs
are managed by the new training policy. Mirror receipts state separate inodes,
full file SHA and actual exit status; they are not off-node disaster recovery.
No large asset is copied to the Mac. The pipeline waits for mirror completion
only at its final deliverable boundary.

CPU checks are only the five new periodic template/binding cases:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s THIS_DIR \
  -p test_periodic.py -v
```

Actual GPU training, boundary state or EM is not established by source/template
preparation. On failure all actual wait/error/partial result files remain; the
supervisor sends no signals. It does not support silent parent restart against
an existing pipeline directory: a genuine future fault requires an explicit
independent continuation rather than duplicate slot/Adam execution.
