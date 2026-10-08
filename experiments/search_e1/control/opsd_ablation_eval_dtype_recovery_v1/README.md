# Independent dtype and failed-C0 control recovery

Original seven scientific configurations, runtime, source100 and all first-C0 sampling/mining/teacher diagnostic records remain immutable. This independent driver resumes only the exact failed-before-Adam C0 output. It verifies actual original pane rc2, original child rc1, exact process absence, full owned/opaque release, empty checkpoint inventory and original outer0 identities before replacing the dead driver CPU role. The existing observer is reused after fresh identity/memory checks. The other six groups remain fresh.

Every group trainer dispatch uses the separately published FP32 outer-old repair overlay. The native trainer, checkpoint/scientific identity guards, locks, wait results and actual optimizer logic are unchanged. Evaluator and driver source are deployed under an independent path whose final directory name remains `opsd_ablation_eval_v1`, as required by the original training admission contract. New publication receipts use the repository namespace `opsd_ablation_eval_dtype_recovery_v1`.

Three focused CPU boundary regressions passed; the original 25 evaluator fixtures were previously accepted and are not repeated merely for this control recovery. Actual GPU execution and updated-weight/effect acceptance must be recorded separately. No completed old failure or prepared source is counted as a training update.

Below is the historical native evaluator description, retained as interface documentation.

# Reverse ablation: immediate per-group native512 evaluation

This is a new independent evaluator and CPU sequential study driver. It does not
execute the obsolete 200-outer results plan, train a model, signal any process,
or select a best checkpoint. Publication/readback and the new storage admission
must precede execution. A NOT_READY study/config cannot start model work.

## Exactly eleven fixed models

Each C0/C1/M/R/W/P/J group is trained by its separately frozen training
controller. Immediately after that group's actual foreground return code 0,
`TRAIN_COMPLETE` with ten committed outer chains, and complete owned process/GPU
release, the driver evaluates its raw outer9 endpoint on all 512 heldout questions.
It does not wait for all seven groups to finish. C0 is followed once by the common
BF16-zero and FP16-zero evaluations; P and J additionally evaluate their own FP16
deployed raw10 models. Seven BF16 raw10 + two common zeros + P/J FP16 = **5632
primary predictions**. Hook/zero-effect controls are separate extra generations.

Main BF16 evaluation loads the same source100 base as BF16 plus the actual 504
finite FP32 LoRA tensors **without merging**. Every worker verifies the loaded
adapter tensor digest against this group's committed checkpoint and its change
from the shared initial adapter. Initial 252 B tensors must be exactly zero.

FP16 deployment is a separate arithmetic condition: an isolated empty-CUDA CPU
child loads source100 as FP32, loads that adapter as FP32, merges in FP32, then
performs **one** FP16 cast. It saves a complete deployed HF model and manifest.
The FP16 inference worker loads this actual merged FP16 model and verifies its
full tensor digest. It does not substitute FP16 base + unmerged FP32 adapter.
For FP16-zero, the pre-cast merged FP32 tensors must equal the unmodified FP32
source tensors. Nonfinite or overflowed casts are rejected.

## Native protocol and evidence

Eight physical workers map one-to-one onto eight logical ranks. Rank r evaluates
`validation[r::8]`, exactly 64 questions, with seed42 for every rank. Validation
is the original SHA-bound 512 unique IDs/texts, NQ256 + Hotpot256. Each ordinary
question calls the frozen `SearchAgent.run(question, temperature=0)` directly.
HF SDPA, repetition1.05, EOS `[151645,151643]`, context8192, new500, five turns,
four searches, top3 and observation480 remain unchanged. No privileged reference
is supplied. The original raw result and last authored span list go to the exact
frozen strict scorer. Earlier answers and retrieved information cannot become a
final answer. Balanced-tag diagnostics are not the reward.

The first question of each rank has a plain-versus-hooked result/RNG null-effect
control; BF16-zero also has an enabled-zero-versus-disabled adapter control.
All controls are explicitly excluded from 512. Transparent scoped hooks preserve
the original generate return, kwargs, processor list and Response, while recording
the raw generate IDs before native stop-tag cropping and complete HTTP queries/
responses. Private questions, golds, responses and token/HTTP traces stay in node
artifacts, never Git.

Completion checks freshly SHA-bind the current training helpers, unchanged305
runtime, data/cohort source, ten complete actual512×5 sampling outputs, original
versioned miner/pairs/weights, pair encoding/order cursor, actual phase exit
receipts and final native checkpoint files. Historical checkpoints are checked
by their small manifests; the final native `check_checkpoint` follows any reviewed
shared-optimizer payload references. This evaluator does not invent an Adam
tensor-semantics audit from counters or claim that source100 GRPO Adam was restored.

Summaries report strict EM, NQ/Hotpot EM, invalid final action versus valid wrong
answer, executed-search counts, paired gained/lost cases and a seeded 10,000-draw
95% percentile bootstrap over **512 question units**, never five siblings. Final
cross-group comparison requires all eleven exact512 model slots and common
source/runtime/data/initial identities. K, update count and exposure differences
remain covariates; counts and bootstrap intervals do not automatically establish
algorithmic causality. Historical source239/512 is diagnostic, not a match-hits
gate or a value to copy into new results.

## Explicit configuration interface

CPU template only:

```sh
python -B prepare.py --training-config GROUP_CONFIG --training-config-sha256 SHA \
  --base NEW_EVAL_BASE --helper-root PUBLISHED_EVAL_SOURCE --role group --output EVAL_TEMPLATE
```

Use `--role common_zero` once with C0. Template creation does not start a GPU.
Root supplies each independent eval run's publication/readback and output-storage
receipts, source commit/manifest and **new group** observer references:

- `gates.memory_observer`: that training group's planned immutable binding path;
  its SHA is null in the template and filled from the actual new observer at the
  successful training boundary.
- `gates.memory_observer_config`: the actual immutable group training config fileproof.
- `gates.memory_observer_source`: its published `memory_observer.py` fileproof.

The observer binding must name that exact group/config, with actual PID/UID/tick,
independent observer environment marker and fresh current `sample_unix <=30s`.
The stopped old200 observer cannot stand in for a new group observer.

Standalone evaluator configuration additionally fills
`training.terminal` (normalized actual successful foreground rc0/controller,
run/config, status and full-release references) and `training.raw10_checkpoint`.
The standalone command is:

```sh
CUDA_VISIBLE_DEVICES='' E1_RUN_ID=EVAL_RUN python -B controller.py \
  --config FINAL_EVAL_CONFIG --config-sha256 SHA
```

The **sequential study entry** takes one immutable JSON manifest with:

```json
{
  "schema": "opsd-ablation-seven-sequential-study-v1",
  "state": "READY",
  "run_id": "independent 32 lowercase hex",
  "study_id": "same study as all seven configs",
  "helper_root": "/absolute/published/opsd_ablation_eval_v1",
  "helper_source_manifest": {"path": "/absolute/source-manifest.json", "sha256": "SHA"},
  "source_commit": "published 40 hex",
  "publication": {"path": "/absolute/driver-publication.json", "sha256": "SHA"},
  "output_root": "/absolute/new-independent-study-driver/runs",
  "groups": [
    {"variant": "C0", "training_config": {"path": "/absolute/C0-config", "sha256": "SHA"}, "evaluation_template": {"path": "/absolute/C0-eval-template", "sha256": "SHA"}},
    {"variant": "C1", "training_config": {"path": "/absolute/C1-config", "sha256": "SHA"}, "evaluation_template": {"path": "/absolute/C1-eval-template", "sha256": "SHA"}},
    {"variant": "M", "training_config": {"path": "/absolute/M-config", "sha256": "SHA"}, "evaluation_template": {"path": "/absolute/M-eval-template", "sha256": "SHA"}},
    {"variant": "R", "training_config": {"path": "/absolute/R-config", "sha256": "SHA"}, "evaluation_template": {"path": "/absolute/R-eval-template", "sha256": "SHA"}},
    {"variant": "W", "training_config": {"path": "/absolute/W-config", "sha256": "SHA"}, "evaluation_template": {"path": "/absolute/W-eval-template", "sha256": "SHA"}},
    {"variant": "P", "training_config": {"path": "/absolute/P-config", "sha256": "SHA"}, "evaluation_template": {"path": "/absolute/P-eval-template", "sha256": "SHA"}},
    {"variant": "J", "training_config": {"path": "/absolute/J-config", "sha256": "SHA"}, "evaluation_template": {"path": "/absolute/J-eval-template", "sha256": "SHA"}}
  ],
  "common_zero_template": {"path": "/absolute/C0-common-zero-eval-template", "sha256": "SHA"}
}
```

Eval templates have terminal/raw10 fields null, with publication/storage and
observer config/source references ready and the new binding path fixed. All run IDs and output roots must be distinct and
nonoverlapping. Source/data/cohort/initial and helper/runtime hashes must be
common across the seven groups. The driver first runs the actual frozen training
controller, waits its actual return code and full release, then fills **only**
terminal/raw10 references and the planned observer binding's actual SHA in a separate final eval config. It checks the complete
ten-outer chain before spawning that evaluator. The parent is CPU-only and holds
none of the shared/study locks its child needs.

Before each group, the CPU driver verifies the dedicated observer-startup gate:
published source, new storage/transition admission, oracle identity and empty GPU
resources still apply. It starts the published CPU-only `memory_observer.py`,
waits for its immutable group/config binding and fresh sample, and only then starts
training. An existing binding is reused only with the same live, exact source-bound
observer; it is never overwritten or restarted. The group role registry records
the driver and observer's actual UID/PID/starttick/parent/argv/environment marker,
canonical argv SHA, published source and immutable input. These two exact CPU roles
are the only additional ownership exclusions; known owned/run/descendant matches
and unknown opaque processes retain priority. Observer bootstrap accepts driver
only; normal training/evaluation gates require both roles.

Observers remain alive without signals. Each group's actual foreground completion
freezes its own memory end snapshot; later global samples from that persistent
observer are not reported as this group's training minima. External driver release
anchors only the known phase controllers, so its independent CPU observer is not
mistaken for a training descendant. Normal training/evaluation controllers retain
their own ancestry anchors and all release checks.

```sh
CUDA_VISIBLE_DEVICES='' E1_RUN_ID=STUDY_DRIVER_RUN python -B study_runner.py \
  --study-manifest STUDY_MANIFEST --study-manifest-sha256 SHA
```

Eight evaluator leases are original five shared leases + training-group lease +
study lease + independent results lease. The child rechecks admission after locks.
Actual eight GPU UUIDs must match, unused cards need20GiB direct free, actual owned
running cards need1GiB, oracle onlyGPU7 <=1GiB, and the CPU parent cannot have a
CUDA context. Ownership uses the reviewed full scanner with known PID/tick and
actual parent-chain/environment checks, never input-path substring ownership.
Quick exit/partial spawn records real rc/UNKNOWN and rejects; read uncertainty
does not invent child failure or release leases around live GPU children. There
are no signals, GPU pilots, silent partial-slot reuse, or implicit reruns.

All final group and common-zero summaries feed an immutable
`opsd-ablation-eleven-fixed-model-inputs-v1` manifest. The study driver constructs
this automatically; `summary.py --compare-manifest PATH
--compare-manifest-sha256 SHA --output PATH` is the equivalent CPU aggregation
entry. Its full raw trace re-scoring occurs in each per-run collector; the final
cross-group pass checks the SHA-bound accepted summary/private512 counts rather
than claiming another independent re-score of all5632 raw traces.

## CPU checks and current limit

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_ablation_eval_v1 -p 'test_*.py' -v
```

25 necessary CPU fixtures, zero skips: actual frozen strict QA, native hook
transparency, old timestamp-valid pyc bypass, sequential source-only runtime clone
binding, actual producer ten-round file schema/miner identity, tail/K0 counters,
early/overwritten final binding refusal, new observer bootstrap/exact two-role binding,
real quick-exit/nonzero and release-read
uncertainty handling, GPU ownership/free negatives, exact512/5632 denominator,
exclusive failures/paired question bootstrap, 504 FP32 adapter/Bzero and actual
CPU FP32-to-FP16 once-cast overflow rejection. Node-layout CPU tests may set
`OPSD_ABLATION_TEST_RUNTIME` and `OPSD_ABLATION_TEST_TRAINING_HELPER` explicitly;
fallback is the local frozen305 and new training namespace, with no skips.

These are source/CPU-engineering checks, not real GPU native/HF controls, resource
admission, completed ten-outers, 5632 predictions or effect reproduction. Root is
the sole publisher/deployer/executor. New storage scope remains pending until its
explicit accepted receipt exists.
