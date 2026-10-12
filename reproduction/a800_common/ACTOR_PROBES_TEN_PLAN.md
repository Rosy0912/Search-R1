# Actor probes every ten complete outer updates

Status: implementation in an isolated checkout; **not deployed or launched**.
Predecessor run: `1db662001a0f4d998928bcf6f5f33c9d`. Its fixed source and
after50 reward schedule remain unchanged. Admission for this experiment requires
that predecessor training, final evaluation, checkpoint acceptance and owned
cleanup have completed, followed by resource and source verification.

The persisted deployment intent must include the accepted predecessor's base,
run/execution/source identity and full-acceptance/cleanup receipt hashes.
Admission pins the specific human-authorized predecessor, checks all200 actual
complete update records, final seven-source/NQ3610 counts, milestone receipts,
preserved source, recorded process births and surviving owned sessions. Another
already-finished experiment cannot substitute for this predecessor. If it needs
recovery, the successor gate requires an explicitly verified recovery-chain
extension rather than accepting a self-selected different parent.

On 2026-10-12 the user clarified that “independent LLM Head” means the existing
two binary probes, not a vocabulary projection. Preserve the answer-state
evidence-deletion sensitivity target and prompt-state closed-book target, their
existing question-grouped classifier selection, calibration data and G1 formula.

## Parameter ownership and live representation path

* GRPO owns the actor backbone, generation head, Adam and outer scheduler.
* The reference remains the pinned official model; no probe optimizer owns it.
* Probe fitting owns only the two independent classifier heads, including their
  standardization parameters. Their serialized fitted state and fitting RNG
  provenance belong to the probe version, not the GRPO optimizer.
* For each reward batch, the driver constructs exactly the existing Xa and Xp
  contexts. An actor-worker RPC performs a no-grad forward on the **current
  FSDP actor**, before its PPO update, and returns detached FP32 last-token
  representations. The RPC preserves module modes and existing gradients and
  cannot invoke Adam or the scheduler. Heads score these representations on the
  driver; the old frozen encoder server must not supply predictions in this mode.
* Pad requests to a multiple of four and use equal forward counts on all ranks,
  so root/layer FSDP collectives remain matched. Remove duplicate padding rows
  before head scoring. BF16 autocast and actor microbatch two are preserved.
* Bind every batch to run/execution, logical iteration, attempt identity, actual
  optimizer-write count and head version. A retried incomplete outer may have
  retained partial Adam writes; do not confuse it with the preceding saved actor.
  No refresh or actor mutation is concurrent with reward prediction.

## Ten-iteration refresh and ordering

Initialize both heads from genuine official-fresh actor calibration at step0;
do not blindly reuse the predecessor's frozen-teacher adoption admission.
After complete outers10,20,...,190, fit new heads using that actual actor state.
Calibration is the existing reserved300 training questions, with exclusions
against dev512, stage64 and final51713 independently verified. Calibration
generation, typed evidence-deletion labels and hidden extraction use an immutable
temporary export of the actor just completed, on admitted auxiliary GPU6.
The live training actor remains untouched. This temporary export is a fitting
input, **not** the encoder used for the next ten reward batches.

Write a candidate version under an independent attempt directory; require finite
features and fitted values, both labels/classes, both grouped held-out folds,
data and actor hashes, and unchanged actor/generation/reference/optimizer proofs.
Only then atomically publish its manifest. Both heads switch together. Failed
refresh cannot silently continue with an unbound or stale version.

For a complete outer: finish all20 opportunities; commit the outer scheduler;
record actual completion; when due, export the actor and fit/publish probes;
when a milestone is due, save model plus resumable state bound to the published
probe version; enter the next outer. At step100 save an immutable actor export,
refresh heads from that export, bind the checkpoint to the new probe state, then
enter101 with binary EM. Probe scores after100 are observation only and cannot
modify any correct1/wrong0 reward, advantage or masking rule. No terminal200
refresh is needed because it supplies no subsequent training reward.

## Manifest and compatibility

Add explicit opt-in `g1_through100_then_binary` and `actor_detached_binary_heads_v1`
with refresh interval10. Missing fields retain existing behavior, including the
running after50 experiment and frozen-encoder initial admissions. New fields must
propagate through preparation, persisted deployment intent, actual configuration,
worker source admission, reward evidence, checkpoint receipts and final verifier.
Unknown or partially specified variants fail admission. Save5/50/100/150/200;
intermediate fitting exports remain remote and may be released only after their
probe/version evidence and checkpoint retention obligations are satisfied.

## Full-state checkpoint and recovery

At milestones save canonical actor weights and four-rank optimizer/scheduler/RNG
state, driver RNG, sampler generator state and consumed epoch/batch cursor,
recoverable attempt counters, actual Adam-write totals, head artifacts/fitting
state and version manifest. Hash every file and publish a final receipt only
after all ranks have durably saved. Checkpoint step and probe actor step differ
at step5 (head0) and must be stated explicitly. An initial launch still starts
fresh official weights, Adam, scheduler, RNG and sampler. Resume is a separate
identity/namespace bound to a validated parent and original logical200 chain.
If full state is unavailable, report weights-only state loss and lost unsaved
updates explicitly; do not count them twice or restart as fresh0.

Implemented: at first-five acceptance, deserialize and reload the actual saved
rank-local Adam/scheduler/counters plus training and rollout RNG, with before/after
state digests and unchanged parameters/gradients/reference; reload the saved
driver sampler/counters/reward/RNG in the same process without creating an
iterator. For a future new-process recovery, a replacement DataLoader iterator
must undo its extra base-seed draw, finish the saved epoch (even when exhausted),
then enter the next shuffled epoch normally. CPU tests cover persisted state
restoration after deliberate state perturbation, sampler tail, next-epoch order
and driver RNG. Cross-namespace canonical model loading components are now
implemented, while the complete admitted production resume entry and actual
A800 validation remain pending; a same-process roundtrip is not proof that
the separate production entry has been exercised.

The first-five roundtrip now also reads the actual canonical actor checkpoint,
checks the receipt and all bound file hashes, canonical key/shape/FP32 master
finite state and tied-head equality, and loads it through the same FSDP full-state
graph as export. Each rank agrees on read/device admission before entering the
weight-loading collectives. The restored actor's parameter and gradient digests
must match the actual pre-load actor, while reference parameters stay unchanged.
A CPU mechanics test creates a new worker object, restores serialized model and
Adam/scheduler/accounting/RNG state, and requires its next stochastic update to
exactly match uninterrupted execution. This does not assert four-GPU FSDP or a
cross-namespace production resume; those still require actual admission.

The first-five reload has a four-rank preflight before canonical parameter
loading, including the complete rank set and shared receipt hash. Parameter
release/offload also reaches one collective disposition before the next state
check. A real four-process CPU test injects a missing rank receipt and one
rank's release error; no peer enters a later model/update collective. This
does not promise recovery from a failed FSDP-internal CUDA/transport collective;
such a failure remains fatal and requires preserved fault evidence.

Cross-namespace restoration components admit only an explicit same-experiment
root, exact data/model/recipe and a pinned full-state receipt. Before mutation,
they verify the parent's preserved source files, canonical checkpoint and Head
hashes, actual full-state counters, and absence of its trainer PID births or
session survivors. Only namespace identity is rebound; moments, scheduler,
RNG/sampler and two Head bytes are preserved. The actor-only RPC verifies the
loaded worker state and records a RESTORE ledger anchor. Driver restoration
preserves the unfinished data epoch and reward adaptation state. These are
components, not a complete recovery launcher: deployment admission, durable
chain/partial-loss accounting and final verifier integration remain pending.
The ordinary fresh0 pipeline explicitly rejects a recovery manifest.

Counter-chain auditing reads the fully returned driver attempts and actual
update rows, rederives every segment, and checks exact saved write/attempt
offsets at namespace transitions. The retained logical intervals must cover
1..200 once. Parent updates beyond the selected checkpoint remain explicit
discarded evidence; their writes cannot be inherited or double-counted. G1
and binary reward phases stay separate. This scalar audit deliberately has no
full-acceptance flag: independent rank/scheduler/source/checkpoint/Head binding,
rank-local incomplete tails and the complete final/cleanup audit are separate
mandatory evidence, and production recovery integration remains pending.

The counter audit also handles an exit after the durable all-rank driver return
but before its reward/scalar row is written. Only the final complete attempt may
have this missing scalar, and it must lie beyond the retained checkpoint. Keep
its original attempt as an explicit unscored discarded tail and count its lost
Adam writes; never synthesize reward or inherit those writes. Interior scalar
holes, multiple unscored completions and altered summaries fail the audit. Nine
focused scalar tests pass, and independent static review found no definite new
issue. This still requires a nonempty scored prefix and a retained update in
the namespace; immediate failed recovery branches and partial rank-local tails
need separate preserved evidence, not invented retained segments.

The new read-only rank-tail component reads stable owned newline-terminated
driver, rank-worker and scheduler logs without writing to the run. It derives
the last durable driver return and retains an unreturned opportunity prefix on
each rank. Recorded Adam write positions are lower-bound evidence; coalesce
matching four-rank receipts once and never count a new complete outer without
the actual driver return, even if all four scheduler COMPLETE rows exist.
Keep trailing fragment bytes/hash and both bound and unbound fatal scheduler
snapshots/REJECTED rows for fault review. Historical scheduler births/source-map bindings are checked;
actual process exit, source-file bytes, complete monitor/probability proof,
canonical/RNG/Head acceptance and restore launch remain independent gates.
Fifteen focused scalar tail/counter tests pass. Independent static review found
and then confirmed repair of a bound-fatal fault flag omission. This component still requires
the namespace's scored/retained prefix; failed zero-progress branches remain a
separate pending case, and production chain integration has not been exercised.

The driver recovery path is now connected to the training loop behind an
explicit separate launch-admission receipt. Before actor loading it validates
the saved driver sampler/RNG, Head bytes and exact saved write/attempt origin.
Each worker preflight and collective confirmation binds the child identity and
saved attempt cursor. The driver checks all four actual RPC returns, unchanged
fixed reference, then restores its saved state and verifies the resulting digest.
It uses the saved iterator once, preserving epoch/batch indices and undoing the
replacement iterator's extra seed draw, then creates new epochs normally. No
repeated baseline or artificial first-five run is inserted during recovery.
Namespace-local update and attempt writers begin at the saved K+1/cursor+1,
reject old parent rows/counters, and retain the original parent files separately.
Twenty focused CPU tests cover this path and existing writer/accounting behavior;
the actual Ray serialization check requires target dependencies unavailable on
the Mac. The separate production launcher, real rank/tail chain reader and
complete final verifier integration still remain mandatory before recovery use.

A separate remote-only recovery preparation entry now derives a candidate
manifest from the actual saved driver and an explicitly selected experiment
root. Source and original canonical data must already be staged in an independent
A800 namespace; preparation rejects existing launch/log/checkpoint state and
requires the executing module to be the pinned published child source. It checks
the canonical shared products and original official-model manifest, preserved
parent source/reference/full-state/Head hashes, prior restore-origin proof when
applicable, and absence of parent trainer births/session survivors. It re-reads
the real canonical actor tensors on CPU (435 FP32 finite tensors) and preserves
post-checkpoint discarded driver/rank evidence. Only then may it create the
child's private Ray root and durably publish the candidate manifest, followed
by a final preparation marker bound to its actual bytes. A manifest-write
failure cannot publish a prepared marker. Old driver ledgers and dangling
links are rejected before preparation, in addition to rank and claim state.

The child's physical Adam/attempt origin comes from the saved payload, never
the parent's later observed counters. Recipe, BF16/journal mode, retrieval
assets/physical placement and reserved calibration IDs remain bound. Initial
constructor recipe fields still describe the original official-root recipe;
`run_start_semantics` and `optimizer_semantics` explicitly state full-state resume.
No new fresh0 or initial300 admission is claimed, no Head refit is performed,
and preparation has `production_admission=False`. Twenty-two focused CPU
manifest/restore/reader tests pass, including the real file-hash verifier on
explicit tiny scalar fixtures, durable publication order and failure handling,
and old driver-ledger/dangling-link rejection. These do not verify real model
weights; actual remote preparation is not exercised.
Child restored-Head endpoint/resource/guard-plan admission, production launcher,
zero-progress fault branches and complete final chain acceptance remain pending.

The separate restored-Head staging entry now requires the exact completed
preparation marker/candidate SHA and saved origin, clean pinned child commit
and complete runtime SHA map, original data hashes and actual parent recovery
context. It validates the saved fitted binary classifiers on CPU, copies only
the two immutable saved Head files and rebinds namespace paths/identity while
preserving the version, fitting actor step, data and parameter hashes. Both
Heads and their archived version manifest are durable before the live manifest
is published atomically and exclusively. The final staging marker binds both
manifests and the preparation/parent checkpoint hashes and grants no production
admission. A partial copy cannot publish a live version or completed marker;
an existing live version cannot be replaced. Four small scalar-file tests cover
5/50/100/150, corruption/source symlinks, interrupted copy and exclusive live
publication; these do not exercise actual fitted Heads, endpoint or A800 state.

The opt-in pipeline now starts a CPU-only binary Head service instead of the
frozen encoder service. Its two fitted classifiers are hash-checked and loaded
together; health, suspend/resume and synthetic readout smoke responses bind the
actual producer identity and live version. It does not load a CUDA encoder or
claim task accuracy from the synthetic vectors. Production reward features still
come through the current actor's detached-representation RPC. Both admission
and training verification recheck all input hashes, the complete runtime source
map, Git HEAD and clean source before querying the actual endpoint. Fresh0 and
saved K=5/50/100/150 classifier versions are handled explicitly; no new initial0
calibration is claimed for a restored version.

Recovery preparation also preserves exact candidate bytes in a separate,
immutable manifest before the mutable pipeline run manifest and completion
marker. Saved-Head staging binds that immutable SHA and the pre-launch run bytes.
The latest 22 focused CPU tests cover service/HTTP identity, hash and version
rejection, input/source rechecks, legacy admission, durable preparation order and
saved-Head staging. Joblib deserialization uses RAM fixtures on this Mac; native
fitted Heads, current-actor RPCs, resource/guard admission and the separate
production recovery launcher still require actual A800 verification. Independent
static re-review found no new definite defect after the source/input gate fix.

Restored namespaces now have a distinct prospective guard-plan role. It freezes
the saved K origin and inherited Adam writes separately from zero new child
writes, binds the immutable candidate, parent full-state receipt, both archived
Heads, child endpoint/resource/retrieval/release evidence and actual recipe, and
requires full restoration before rollout. The historical reader makes no live
process or probability-health claim. Fresh0 guard artifacts cannot impersonate
this role. The actual worker threshold consumer admits this schema only for the
explicit recovery binding and role. Independent review found and prompted fixes
for the worker schema dispatch, retrieval/release content checks and exact GPU6
probe placement. The separate production recovery launcher and full final chain
verifier remain pending; this component itself does not launch a trainer.

The existing scorer still uses the current actor's detached RPC features after
the separately admitted full state restore; staging never calls a frozen
teacher for reward predictions or fits any new classifier.

Published-source staging can run while the predecessor trains. It uses a separate
source-only directory and an offline Git bundle based on the preserved runtime34
repository, without modifying that repository, moving data/weights/environments,
claiming GPUs or starting services. The new Git HEAD, clean tree and complete
runtime SHA map must match the actually pushed commit. A source-staging receipt
is not a run manifest, training admission or proof of native algorithm behavior.
The full child deployment and its independent review still precede any trainer.

A separate trainer entry now dispatches explicitly to restored-origin guards and
a pre-restore admission reconstructed from the actual frozen inputs. The receipt
binds the complete guard/hash set, saved parent origin, source/configuration and
actual pipeline birth. Training claim, configuration exec and driver restoration
consume that same receipt; claim hashes, saved counters, owned trainer ledger and
actual PID births must agree before loading actor state. It does not claim a new
optimizer update, restored actor or final chain acceptance. The ordinary fresh
entry rejects recovery binding. The separate owned recovery pipeline and full
final chain acceptance remain required before production recovery is exercised.
CPU mechanics and constructed scalar admissions are not native A800 evidence.

The real scheduler and attempt readers now admit namespace-local intervals
beginning at the actual saved origin. They read the bound parent manifest and
full-state receipt hashes plus the child launch/restore proof, require four
child-bound confirmations and the saved policy-rejection count, and independently
bind the first RESTORE row to source/contract and an owned trainer PID birth.
Its actual scheduler counter/LR, write total and attempt cursor must agree.
The existing BEGIN/COMMIT/COMPLETE/ABORT, warmup, write receipts and per-attempt
40 monitors/20 opportunities remain mandatory. Counter and monitor increments
report only work performed in this namespace; inherited totals are explicit.
Constructed scalar-file tests cover resume at5/50/100, zero-origin four/eight
rank regressions and wrong origin/LR/birth/hash/parent-prefix rejection. This
does not verify canonical/RNG tensor bytes or incomplete tails and is not the
complete chain/final acceptance; the admitted recovery launcher and those final
checks still require implementation and actual A800 execution.

## Required evidence before launch

1. Focused tests:100/101 reward boundary,10-step refresh boundaries, strict
   manifest rejection, stale/mixed head versions and incomplete outer handling.
2. Actual feature/gradient isolation: only classifier parameters change; actor
   backbone/generation head/reference and GRPO optimizer/scheduler hashes agree
   across probe fitting. Features returned by the RPC have no grad_fn.
3. Four-rank real RPC: equal collectives, padding/order identity, finite FP32
   returned features, same current-actor identity as rollout and recomputation.
4. Genuine calibration: no dev/final leakage, unchanged target/context semantics,
   both grouped folds, actual atomically published head/data/actor hash binding.
5. Special-token and sampling/recomputed probability chain; reuse compatible
   proofs only after explicit source-contract review. Do not add a full HF
   vocabulary probability diagnostic or unrelated extra evaluation.
6. Full-state checkpoint save/reload, exact logical scheduler position and
   sampler/RNG/partial-count continuation. Probe version survives restore.
7. A sub-agent with no parent context independently reviews new code and the
   actual pinned deployment, and any discovered errors are repaired before launch.

## Production contract and completion

A800 training0–3, retrieval5–6/probe6, never4/7. BF16 actor/reference/rollout,
FP32 master/reduce/selectedLP.512x5 trajectories; mini128/epoch1;20 opportunities,
actor local2/global8/accum16; old/ref local8/global32. LR1e-6/warmup57 complete
outers, first consumedLR0; entropy/refKL0; DualClip3/clip.2/samplingKL.2/
negative-tail.01/exp-bound30/FP64 globalL2clip1. The original manifest bytes/order
SHA is `0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`.
Train169028/dev512(NQ256+Hotpot256)/stage64(NQ32+Hotpot32)/final51713,
seven sources includingNQ3610. No split changes.

The same trainer runs1..200; short-run acceptance is its first five, not another
launch. All20 processed and at least one returned Adam write count as a complete
outer; report opportunities/successes/skips/retained partials separately, rank
replicas once. Publish current source and necessary scripts/configuration to the
private repository promptly; verify remote SHA/blob. No models/environment/QA
or token output in Git or on the Mac. Thirty-minute bounded monitoring and hourly
delivered-anchor reports remain; rewards for1..100 and101..200 have separate
statistics. Complete the goal only after200 actual complete outers, final51713
ordered IDs/binaryEM/F1/seven counts/NQ3610, full canonical finite/hash checks,
valid recovery chain and only-owned cleanup. Report degraded effects honestly.
