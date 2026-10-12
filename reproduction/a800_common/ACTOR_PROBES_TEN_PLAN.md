# Current actor binary probes, refreshed every ten complete outers

Status: implementation and independent static review in an isolated checkout.
The complete retained-chain and post-cleanup readers are now implemented.
Actual A800 algorithm/deployment admission remains pending; no new trainer has
started. The predecessor is run `1db662001a0f4d998928bcf6f5f33c9d`, source
`34e63b7b12f4056503803905c1b3dd1be95f0adc`. Keep its runtime and after50 reward
schedule unchanged until its200/final51713/canonical/owned-cleanup acceptance.

## Experiment and ownership

Start from official fresh Qwen/Qwen2.5-3B-Instruct revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Do not inherit the predecessor's
weights, Adam, scheduler, RNG or sampler. Use an independent namespace and
identity. Outers1–100 use the original G1 formula;101–200 use strict binary EM,
correct1/wrong0. The human clarified that the two independent binary classifier
probes are retained; they are not the vocabulary generation Head.

GRPO owns the actor backbone, generation Head, Adam and outer scheduler. The
fixed reference remains official. Probe fitting owns only the two classifier
Heads and their standardization parameters. Preserve the answer-state evidence
sensitivity target, prompt-state closed-book target, question-grouped selection,
reserved300 calibration data, and existing G1 formula.

For every reward batch, construct the existing Xa/Xp contexts. A no-grad RPC
on the current FSDP actor returns detached FP32 last-token representations;
the driver scores them with the two classifier Heads. No frozen encoder supplies
production predictions. Pad requests to a multiple of four, preserve matched
forward counts across all ranks, then remove duplicate padding rows. Preserve
BF16 autocast, actor microbatch2, module modes, existing gradients, and
Python/NumPy/Torch RNG on both success and failure. No probe operation may invoke
actor Adam or scheduler. Bind the readout to run/execution, logical iteration,
attempt, actual Adam-write count, row order and fixed dual-Head version.

## Refresh and iteration ordering

Fit genuine official-fresh actor Heads atstep0. Do not reuse the predecessor's
frozen-teacher initial admission. After complete10,20,...,190, fit the new pair
from an immutable export of the actor just completed, on admitted auxiliaryGPU6.
The export supplies calibration generation/labels/features; production readouts
remain on the current training actor. The live actor/reference/Adam are untouched.
Use reserved300 training questions only, with dev512/stage64/final51713 exclusion
and unchanged data hashes. Require finite fitted values, both classes, grouped
held-out folds, typed evidence-deletion labels and genuine calibration arrays.

Publish each candidate under a unique attempt directory. Durably write both
Head files and their manifest before atomically switching the live pair. Keep
one fixed version throughout each outer. Preserve and compare driver RNG plus
all four actor/reference parameters, gradients, modes and RNG; actor comparison
also covers Adam/scheduler and both rollout RNG streams. A failed refresh must
retain evidence and cannot silently continue with a stale Head version.

For every outer: process all20 opportunities; commit scheduler/completion only
when at least one Adam write returned; refresh when due; save model plus full
resumable state when a milestone is due; then enter the next outer. At100, export
actor100, refresh Head100, save checkpoint100 bound to that pair, then enter101
with binary EM. After100 probe outputs are observations and cannot change the
binary reward, advantage or masking. No terminal200 refresh is needed.

## Recipe and exact inputs

Use the actual common products and verify manifest bytes/order SHA256
`0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`.
Train169028; dev512 NQ/Hotpot256each; stage64 32each; final51713 seven sources,
includingNQ3610. Do not repartition or deduplicate canonical occurrences.

A800 train0–3; retrieval5–6/probe6; never use or stop foreignGPU4/7. Each outer
512questions×G5=2560trajectories, PPOmini128/epoch1/20opportunities. Actorlocal2,
global8/accum16; old/refglobal32/local8. BF16 actor/reference/rollout and FP32
master/reduction/selected-log-probability. LR1e-6, fixed57 complete-outer linear
warmup then constant; firstouter actualLR0. Entropy/refKLcoeff0, low_var KL,
clip.2, globalL2clip1 inFP64. Preserve DualClip3, sampling-policyKL.2, negative
ratio-tail.01, finite checks and exp-bound30, original group advantage/state mask
and loss denominator. No extra independent HF full-probability forward.

Four rank replicas count once. Record processed opportunities, actual successful
Adam writes, skips and partial/inflight evidence separately. All20 processed plus
at least one returned write count one complete outer and one scheduler advance.
Fully skipped or abandoned outers do not increase200. Preserve real partial
updates; do not invent a rollback. G1/binary reward windows remain separate.

Hydra-generated run YAML/logging lives under the remote namespace's
`logs/trainer-hydra`, with `hydra.job.chdir=false`. Repository output files would
break its exact runtime source map. Local artifacts remain code/config/docs and
small scalar/hash audits; model/state/environment/raw outputs stay remote.

## Remote preparation, independent review and one launch

Use `a800_common.deploy_actor_probes` with a persisted small deployment handle.
After the exact predecessor's real acceptance and owned cleanup, derive that
handle from its existing input hash metadata and the exact latest published
source-only staging receipt. The predecessor's real200/final/canonical receipts
and process births are checked remotely before creating the new namespace.
Copy the eight original data/official/method files only within A800; clone the
verified pinned source there. No local QA, model or environment copy is required.

`--prepare` performs CPU input/source/official-weight-file/configuration/space
verification and writes a prepared deployment receipt. It does not fit Heads,
allocate GPU, start services or a trainer, or claim native algorithm acceptance.
`--inspect-prepared` returns actual small metadata for a no-parent-context
sub-agent source and actual deployment review. Record that genuine review with
the prepared receipt SHA and exact run/execution/source identity; the production
entry has no automatic accepted-review writer.

Only then use `--launch`. The remote wrapper rechecks the real reviewed bytes,
configuration, model/input identity and reserved disk receipt before the existing
exclusive Popen claim. The fresh pipeline repeats admission before its GPU lock
and native allocation. Separate exclusive local prepare/launch claims survive an
SSH timeout: inspect the same handle; never retry the uncertain mutation.
The existing after50 predecessor does not use these opt-in fresh actor-probe
gates. Native actor RPC, gradient/RNG isolation and first-five acceptance still
must run in the new same-process training chain after launch.

## Checkpoints and fault recovery

Save5/50/100/150/200 canonical actor weights plus four rank-local full Adam,
scheduler, counters and training/rollout RNG; driver RNG, sampler generator,
permutation and consumed epoch/batch cursor; reward adaptation state; exact dual
Heads and version. Publish full-state completion only after every file is durable
and hashed. Step5 has actor5/Head0. Keep all actual fitting exports and original
calibration artifacts through post-cleanup byte verification.

The same trainer continues1..200, including the first-five gate. At5 actually
reload canonical weights and persisted worker/driver state, require equal
before/after digests and unchanged reference, then continue6 in that process.
CPU mechanics and collective failure fixtures are not proof of native A800
FSDP/GRPO behavior. Native first-five evidence must still be produced and checked.

Recovery is a separate namespace and full-state entry at savedK+1. Verify the
latest complete canonical435finite/hash checkpoint and full state, original
source/data/Heads, actual parent trainer/service exit, and independent review of
original failure plus candidate source/deployment. Freeze a candidate manifest,
stage only the saved Head bytes, admit restored-origin guards, and restore
moments/scheduler/RNG/sampler before rollout. Preserve the unfinished epoch and
undo the replacement DataLoader iterator's extra seed draw. Record inherited
writes separately from zero initial child writes. No fresh0 launcher may stand
in for resume. A completed trainer needs final scoring/acceptance repair instead
of replaying earlier training. Weights-only or zero-retained-progress failure
branches require explicit fault-specific adaptation and truthful state-loss/
original evidence handling; the full-state entry must reject incompatible state.

## Actual final chain and cleanup acceptance

`verify_actor_probe_chain` discovers actual parent manifests and retained owners;
reads original namespace source/input bytes; validates probability, monitor,
scheduler, readout, full-state/canonical receipts and refresh isolation; reconstructs
logical1..200 once; keeps discarded returned and rank-local unreturned lower-bound
writes separate. All-skip retries add opportunities without adding complete outers.
Each adopted ten-outer pair must trace to its real original calibration and fit
owner; restored archives preserve exactly the saved pair/version/actor step.
Revalidate genuine calibration arrays/fitted Heads and root same-process first5.
Rebuild actual final51713 ordered IDs, strict binary EM/F1 and seven-source/NQ
counts from the unchanged canonical data. No synthetic score is measured accuracy.

The pre-cleanup proof explicitly leaves owned cleanup and goal completion false.
Seal fixed training/scalar logs, original calibration/encoder files and all
historical owned process births. Allow dynamic sampler/inventory/pipeline/cleanup
records to finish. A separate `--after-cleanup` read-only audit requires the actual
finished pipeline stage, all preserved births still covered by the ledger, exited
exact births and owned sessions, released owned CUDA contexts, absent private
corpus cache, and preserved original assets. Rehash source, model/full-state/
Heads/calibration/fixed evidence, actual final scores, and original root-five GPU
byte prefix. Appended samples are allowed; an altered prefix is rejected.

Ownership follows PID/start ticks/SID/UID. An ordinary shared pipeline shell
session is not wholly owned. A different session-leader birth using the same
numeric PID/SID belongs to another task; do not signal it or treat it as a leak.
A missing old leader can leave real old-session children, which remain detectable.
The live cleanup writer also uses exact births for NVML PID-only release checks.

## Deployment and completion

Commit/push the exact private `shaoshitong/Search-R1` source, verify remote SHA
and changed GitHub blob bytes, then stage an independent source-only copy via
the existing physical control route. Source staging may occur while the
predecessor trains; it claims no GPU or service and leaves runtime34 unchanged.
Verify staged GitHEAD/clean tree/full runtime map twice and obtain independent
actual-record deployment review. Source-only staging is not production admission.

After exact predecessor completion, create one bound fresh experiment intent,
verify remote data/model/environment/output paths and assigned resources, then
perform the reviewed production launch. Confirm at least two actual progress
measurements and one complete iteration; check native first5 then continue the
same process. Healthy monitoring remains bounded every30minutes; formal hours
publish scalar/hash evidence and deliver reports before advancing the anchor.
SSH timeout is observation failure and never authorizes duplicate execution.

Only actual new200, final51713, canonical435finite/hash, valid retained recovery
chain, exact owned cleanup and an actual user report permit goal completion.
Report any regression honestly. Preserve source backup and remote byte validation
before releasing this task's resources. No global Ray stop, foreign task cleanup,
auth/SSH/VPN changes, or models/environments in Git.
