# M4: transparent native512 fixed-weight evaluation

This independent helper calls the **actual frozen `search_e1.rollout.SearchAgent.run`**. It does not copy that state machine, train a model, create an optimizer, merge an adapter, or change generation options. Its scoped hooks capture actual tokenizer inputs, `model.generate` input/output IDs and original kwargs, actual HF logits-processor classes/properties, and the complete existing `requests.post` response. Hooks return the same original BatchEncoding, Tensor/ModelOutput, processor-list, and Response objects and restore attributes on exceptions. Per-token processed scores are not collected by this helper.

The cohort is the original 512 heldout questions: 256 NQ and 256 Hotpot, SHA `75785f9c9b32c1b35878f6922be57923f431c52d11aa410c380f159d69febd76`. Each of eight logical ranks processes `validation[rank::8]` in its original order, 64 questions, with seed42. Logical rank `r` uses physical worker `r % 3`, on GPU4–6; every actual model call remains one question/batch1. Within each rank, the phase's two fixed models alternate per question with separate same42 RNG streams. All four small adapters are loaded on the same frozen BF16 source100 model and retain FP32 LoRA tensors. No parameters are updated. Loading additional adapters can consume initialization RNG, so logical evaluation streams are seeded after loading; actual before/after RNG digests are retained instead of assuming no consumption.

Generation inherits the actual source model `generation_config.json`. Native temperature0 supplies `do_sample=False`, max_new_tokens500 and its original pad/stopping kwargs; **repetition_penalty1.05 and the EOS list `[151645,151643]` remain inherited**. There is no replacement GenerationConfig and no new logits processor. Full generation config, actual processor list, native token outputs, final authored spans and strict reward are saved. The native protocol parser takes a list of authored spans; diagnostic validity and strict EM are CPU rechecked during aggregation. Whole-transcript tag balance is not substituted for the native last-action rule.

## Phases and acceptance

1. `endpoints`: all512 source100 (actual saved zero-B `best_adapter`) and raw epoch1. Each logical rank/version's first actual question has an extra unhooked run and a hooked run from the same RNG state. Native result bytes/fields and post-RNG digests must match. The hooked result is the normal512 slot; the extra unhooked result is explicitly excluded from the denominator. This is part of the complete run, not a separate pilot/restart.
2. Exact source239/512 and epoch203/512 are required before `early` can start. A mismatch saves complete output and `HOLD_ENDPOINT_MISMATCH`; it does not swap routes, rerun the entire evaluation, or start update1/5 automatically.
3. `early`: all512 update1 and update5. The final four-model summary verifies 2048 unique model/question slots and reports paired flips. Already accepted endpoints are reused, never rerun by `--resume-run`.

Original native evaluation only saved aggregate history, not per-question results, in the examined `runs/opsd`, logs and receipt directories. Matching 239/203 confirms aggregate fidelity; it cannot prove the historical correct-question sets are identical. All four M4 models use the same admitted resident18005 route. This differs from the original native18004 endpoint; the accepted numeric/HTTP gate is bound and the route change is explicit. That gate does not prove all future retrieval queries are bitwise identical. An actual admission/identity error fails/HOLDs rather than silently falling back.

## Strict startup and release gates

The controller imports no torch/CUDA. Before spawning any GPU worker it requires:

- Latest helper source manifest, Git commit/push and remote readback attestation.
- M1's complete192 unique case IDs and four versions, three final COMPLETE64 statuses/results hashes, three actual child exit0 codes, original source/weight final checks and matching fixed manifest.
- Actual M1 controller exit0 from the preserved owned tmux pane, bound to PID33834, UID1005, starttick922457688 and run `894021c8f55f436f96bb3cb146ab05bb`. Mere process disappearance does not satisfy the gate.
- All owned M1 PIDs/ticks and same-run descendants/orphans absent. No signals are sent.
- The shared M1 `experiment.lock` plus the independent M4 lock acquired and held across all phases. Lock files are not removed.
- Fixed source/cohort/science/adapters and admitted resident health/source/PID/tick unchanged.
- Actual GPU4–6 UUIDs, no compute process on those GPUs, memory.free at least20000MiB each and output disk at least100GiB. Other GPUs/processes are untouched.

A missing prerequisite writes `NOT_READY`, exits3 and starts no worker. Phase worker failures preserve partial traces/status/logs and HOLD; no controller signal stops another worker. Already spawned children are waited to actual terminal status before resource locks are released. At a phase transition, a bounded55-second resource-release wait allows GPU cleanup; it neither reruns questions nor starts a new pilot.

## Root-supplied proof schemas

`publication_proof` (absolute path plus SHA in the config):

```json
{
  "accepted": true,
  "source_manifest_sha256": "<SHA of this helper source-manifest.json>",
  "source_commit": "<full 40-character commit>",
  "remote_readback_sha": "<same full commit after actual push/readback>",
  "branch": "search-e1",
  "verified_unix": 0
}
```

Use an actual positive `verified_unix`, not the placeholder. The helper cannot infer publication from a local checkout or old branch. Root must write this proof after committing/pushing the final source/config preparation and verifying remote readback.

`m1_release_receipt`:

```json
{
  "accepted": true,
  "m1_run_id": "894021c8f55f436f96bb3cb146ab05bb",
  "controller": {"pid": 33834, "uid": 1005, "starttick": 922457688, "exit_code": 0},
  "controller_final": {"path": "<M1 output/controller-final.json>", "sha256": "<actual>"},
  "tmux_exit_proof": {"path": "<normalized actual terminal proof>", "sha256": "<actual>"},
  "completed_cases": 192,
  "source_and_weight_end_verified": true,
  "debug_source_manifest_sha256": "<actual flat 15-file gpu-source-manifest SHA>",
  "weight_manifest_sha256": "<actual complete fixed-weight manifest SHA>"
}
```

The normalized `tmux_exit_proof` must contain identical `m1_run_id` and `controller`, `pane_dead:true`, `pane_dead_status:0`, positive `observed_unix`, `actual_tmux_reading` with the saved final reading, and `preservation_receipt:{path,sha256}`. The preservation SHA is pinned to `d872cfc0da2329e319167f6d8d6ae85ac48ce309d21fd83d92c6b75bf61ed45b`. This is the actual Oct8 owned-window remain-on-exit evidence; the controller was still live when it was written. It is not itself an exit0 proof.

## CPU tests, configuration and launch

Local CPU fixtures, including the actual preserved frozen QA parser/core and a complete three-worker1024-slot trace merge:

```bash
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_native512_eval -p 'test_*.py' -v
```

On another checkout set `M4_TEST_FROZEN_RUNTIME` to the original preserved runtime. Missing runtime skips the three frozen-source fixtures explicitly; a full acceptance test run must have zero skips. No fixture makes an HTTP request or imports CUDA. Mock tensors/ModelOutputs/Responses are only fixtures and never become experimental outputs.

Root deploys the final9 Python files plus `source-manifest.json` into a new independent source directory after publication. The manifest's `files` maps each flat Python filename to its actual SHA; no other Python files may appear in that directory. Root then creates the explicit config with CPU preparation only:

```bash
/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python -B /data/shaoshitong/search-e1-opsd-native512-20261008/source/opsd_native512_eval/prepare_config.py --m1-release-receipt <actual-release-json> --publication-proof <actual-publication-json> --retriever-admission <bound-resident-admission-json> --output-root /data/shaoshitong/search-e1-opsd-native512-20261008 --output-config /data/shaoshitong/search-e1-opsd-native512-20261008/receipts/config.json
```

Review the produced config and its printed SHA. Only root starts the complete controller after M1 actually releases resources:

```bash
/data/shaoshitong/search-e1-opsd-from-archive100-8gpu-20261007/env/bin/python -B /data/shaoshitong/search-e1-opsd-native512-20261008/source/opsd_native512_eval/controller.py --config /data/shaoshitong/search-e1-opsd-native512-20261008/receipts/config.json --config-sha256 <printed-actual-SHA>
```

Worker startup requires an immutable launch ticket bound to a live exact controller; a manual worker invocation cannot bypass publication/ownership.

`--resume-run <absolute existing native512-run>` is only for a fully complete/accepted endpoint phase when the early phase directory does not yet exist. It checks all complete endpoint files again and skips no partial question sets. Partial early work needs a separately reviewed recovery, not `--resume-run` relabeling. A source/config change requires a new explicit run and cannot reuse the same accepted receipt silently.

Node-only outputs include per-slot actual prompt/token/HTTP trace, strict answer/protocol, native unhooked spotcheck, RNG evidence, status/logs, immutable tickets, phase child terminal statuses and the complete four-model summary. Raw questions, gold answers, model outputs and tokens are not intended for Git. Root publishes only source/config identities and compact counts/proofs.
