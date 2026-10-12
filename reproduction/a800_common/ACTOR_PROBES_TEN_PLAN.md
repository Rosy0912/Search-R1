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
