# M3 fixed-weight continuation candidate

Local candidate only. No real preparation, model, GPU, HTTP, launch or training has
been executed by its author. CPU preparation remains in its separately frozen
sibling directory. Root publishes/backs up both, supplies actual data/config,
reviews resource admission, then launches this worker after M4 resource release.
Candidate includes the independent CPU controller below; author has not invoked
it or any worker. There are no signals, optimizer, backward or new training.

```sh
python3 -B -m unittest discover -s experiments/search_e1/control/opsd_teacher_counterfactual_execution -p test_execution.py -v
# Root only, after reviewed publication, external budget/controller admission:
python3 -B worker.py --config CONFIG --config-sha256 SHA \
  --ticket TICKET --ticket-sha256 SHA --phase pilot --worker-index 0
# same run/output/config, a NEW remaining-phase ticket, accepted pilot receipt:
python3 -B worker.py --config CONFIG --config-sha256 SHA \
  --ticket REMAINING_TICKET --ticket-sha256 SHA --phase remaining --worker-index 0
```

34 synthetic CPU fixtures pass with zero skips. Actual hashed frozen SearchAgent
class definitions supply the comparator: kwargs keys/values and StopTags start,
contexts, kept output/spans, native search/480 observation, turn4 forced once,
fifth-search refusal, invalid answer/think, resumed search budgets, overrun actual
IDs versus cut/posthoc IDs, and context rejection. Other fixtures cover processor
order/in-place/raw clone/original returned object/exception restoration, teacher
adapter scope and six branches' four-checkpoint forcing, fixed16/48 no repeats,
changed resource proof, foreign GPU/PID reuse/expired/notready admission, exclusive
output, and CPU-only worker import. They do not substitute for GPU equivalence.

## Execution semantics

Each selected case consumes the CPU preparation's unique active A/B states for
original vLLM history and available actual HF source100/epoch1 histories; aliases
execute once. Terminal boundaries and unavailable donors retain explicit skip
records. Original observations remain byte-exact context and are never fetched
again. Each state gets disabled-adapter frozen teacher original/none/final-block
removed/mismatched reference generations, plus ordinary source100 and epoch1
students. No extra opportunities: inherit next turn, completed searches, remaining
native five-turn/four-search budget and pending forced text. Actual joint HF
context must fit8192 with500 reserve under EACH condition, no truncation.

Worker requires actual native torch2.4.0/transformers4.46.3/peft0.13.2.
Generation uses native greedy kwargs, inherited actual GenerationConfig including
repetition penalty1.05 and EOS IDs151645/151643, SDPA/BF16 base and FP32 PEFT
adapters. Source100 zero-B252 and all finite adapter tensors are checked. No new
weights or teacher updates. All parameters initially require_grad=False; PEFT adapter switching can adjust
its flags, while strict no_grad wraps every inference/probe and no optimizer or
backward exists. This guarantees fixed weights. no_grad wraps
inference/probes. Teacher adapter disabled scope exits before ordinary students.

`score_tap` temporarily wraps actual LogitsProcessorList.__call__ in this worker's
single-thread process, invokes its original operation exactly once and returns
its ORIGINAL tensor. It clones raw scores before in-place processing. Actual
first generation step per turn saves raw/processed FP32 log_softmax top32,
entropy, actual top32 scores/logits, processed greedy target ID and raw logp/rank,
plus processor class order. Isolated tag-token scores/ranks are labelled as
individual IDs, never a joint tag/action probability.
No extra generate kwargs/output_scores are supplied. Emitted first ID must match
that actual processed greedy decision. Top32 is not a full distribution artifact;
processor attributes are available in inherited generation_config, not separately
claimed as complete processor configuration. GPU preflight compares plain versus
tapped generation IDs AND post-RNG state from same pre-RNG, and enabled zero-B
versus disabled raw last-position head logits exactly. Controls are additional,
excluded from case/branch scores. Source actual HF traces are rehashed, template/
context-token count/actual-ID decode/observation-ID replay/state hashes checked.
Stored source HF input IDs are unavailable; regenerated input IDs are explicitly
current-tokenizer reconstructions, never claimed historic input-ID evidence.
Original vLLM probabilities/IDs remain unavailable and unknown.

For every new common branch response, four fixed students source100/update1/
update5/epoch1 are teacher-forced through frozen token_probe and native pair_loss
comparison. Teacher forcing uses original successful reference for EVERY branch,
not the intervention reference; reference variants affect the rollout, while the
fixed training-style scoring object stays comparable. Only newly emitted policy
spans are valid for this supplemental branch loss; inherited policy tokens remain
causal context and masked. Chunks32, per-valid-token mean, forward teacher→student
KL, component upper-cap10 and frozen privileged teacher are retained. This is
branch loss on new responses, not historical online training loss nor original
complete-trajectory objective. Skipped context/mask probes keep their reason.
Actual generation IDs, before-cut decode, kept posthoc encoding, branch ownership,
reference condition/hash, aliases/state identity and retrieval JSON all persist.
Existing accepted M1 source100/epoch1 responses also get four-checkpoint common-
response cross-forcing without any new rollout. Original mined y gets source100
fixed-response remove-final/mismatched reference probes (original/none M1 probes
are reused, not silently regenerated). These latter summaries explicitly replace
the probe default successful-reference label with counterfactual reference.
New OWN responses cannot be mean-compared as if common history. q invariance can
be tested across the four same-response forcing artifacts offline.

## Root configuration/admission contract

Config schema `m3-fixed-counterfactual-v1`, workers3/seed42, unique run_id,
approved_gpu_uuids(3), model, four adapters, weight_manifest, runtime,
retriever_admission and exact18005 URL, output NEW run directory. SHA/path objects:
`cpu_helper_manifest`, `debug_helper_manifest`, `m4_helper_manifest`,
`execution_manifest`, `preparation_summary`; remaining phase additionally
`pilot_receipt`. All dependencies' manifest files are freshly hashed; source12/
adapter identity is checked using existing helpers and pinned M4 weight labels.
Preparation's helper hashes must match the exact frozen CPU manifest. Actual
states output hash must match preparation summary; accepted64/16 cohort required.

Root's external controller reserves/budgets GPU4–6, owns locks and verifies actual
M1/M4 terminal receipts/UID/PID/starttick/run/UUID, no foreign compute process,
headroom and retrieval gate. It writes worker-specific ticket schema
`m3-external-resource-ticket-v1`: accepted, run_id, phase, worker_index, owner
{pid,uid,starttick}, gpu_uuid, config_sha256, expiry_unix, m4_fully_released=true,
root_lock_held=true, plus SHA/path proofs `m4_release_proof`, `lock_proof`,
`gpu_inventory_proof` each accepted for this NEW run. **The independent
controller below produces these proofs from actual gates.** The worker checks
proof hashes/live owner/expiry before loading and each case/request; actual CUDA
UUID matches ticket. A static boolean/proof alone must NOT be treated by root as
an independent live GPU/lock/release audit. Tickets must cover the declared
runtime budget; expiry aborts, no automatic retry, renewal or continuation.

Pilot admission requires same-run16 unique completed IDs plus explicit accepted
state_reconstruction and null_effect; remaining48 exclude pilot entirely. Root
must independently inspect/merge three worker completions, actual context skips,
state replay and GPU controls before producing this receipt. The independent controller performs
the acceptance below; worker alone cannot assert pilot acceptance. Worker phase subdirectories are
exclusive; failure preserves evidence and cannot overwrite/retry the phase.
Fresh publication manifest binding and all three workers actual process identities
remain root controller duties. The independent controller below implements this contract.

Cost upper envelope before alias/context skips:64×3 histories×2 states×6 branches
=2304 generation branches, each followed by4 supplemental probes, plus at most
64×(8 existing-response cross-probes +2 original-y reference probes). Root should
review the explicit budget before launch. Fixed16 pilot comes first; no efficacy
or teacher-quality conclusion follows from this code or its fixtures.

## Complete independent controller (supersedes the external-contract limitation)

`controller.py` now implements the external CPU controller; the earlier worker
contract describes the files it produces. Root alone deploys/starts it. It imports
only verified CPU helpers, waits20s for actual M1/M4 release, and never starts or
reruns either prior experiment. It reads the accepted full2048 M4 final, both
phase child rc0/status UID/PID/startticks and every saved native512 result, freshly
recomputes cohort/native scores and239/203 endpoint gate, verifies old controller
identities are gone and private-run descendants/orphans absent. A preserved M4
pane must actually report dead/status0/paneID/PID/command hash under the same live
tmux-server UID/PID/starttick; disappearance/emptystatus cannot pass.

It acquires the exact D/M1 experiment.lock, M4 output experiment.lock, plus its own
M3 parent experiment.lock; never steals/unlocks another process. Actual selected
GPU4–6 UUIDs must match M4, selected compute empty and memory.free≥20480MiB before
each phase, disk≥100GiB. GPU0–3 resident/GPU7 oracle are observed only incidentally
in inventory, never acquired, configured or signalled. During workers only exact
owned UID/PID/starttick/UUID compute is allowed. Resident health plus real candidate
process identity and observer process identity/source/current≤5s, no foreign
history/current compute, are checked each monitor and each worker case/request.

After pilot children really exit0 and fully release, CPU acceptance demands all16
fixed cases, exact prepared branch/skip inventory, every required probe summary,
all finite/native scalar/region reconstruction, executor state/context replay,
frozen QA strict EM/final validity, and all four common-response teacher q numeric
invariance. It checks each worker's actual identity and GPU raw-zero-B/plain/tap
null-effect evidence. Acceptance does not require good rewards or favorable
counterfactual effects. It automatically proceeds to the same run's remaining48
with accepted pilot SHA; never repeats pilot or resumes partial GPU work.

90s worker leases renew every10s from fresh immutable actual-resource evidence.
Worker renewals retain same owner/run/config identity; inventory≤30s. Any spawn,
identity/resource/numeric or child failure makes HOLD, stops lease renewal, waits
owned children to exit normally and their own CUDA/private-run remnants to
release; no signal, stop-other-job or automatic GPU retry. Dynamic tickets are
private runtime evidence, not scientific hyperparameters. Healthy workers are
not stopped merely because an immutable initial ticket grew old.

Controller additional config keys: uid1005, python, physical_gpus[4,5,6],
`debug_root`, `m4_output_root`, `m4_run`, `publication_proof` SHA/path (actual
search-e1 commit==remote_readback_sha, source_manifest_sha256 matches candidate),
`m4_config` SHA/path, `m4_tmux_binding` SHA/path, `runtime_manifest` SHA/path
(actual305 SHA3c05be8e…), `region_helper_manifest` SHA/path. Helper manifest files
can live in receipts: provide `debug_helper_source`, `m4_helper_source`,
`region_helper_source`, `cpu_helper_source` actual source directories explicitly
when different from manifest parent. Supports flat original15 manifest as well
as wrapped `files` manifests. All Python inventory/source SHA is checked. Runtime
305 and source12/adapter identities are freshly hashed at start and each phase.
The accepted CPU prep source cases are freshly verified and hash-selected64 order
reconstructed; inputs remain192-case subsets, never original845k full scans.

`m4_tmux_binding` must be captured while original pane was live/preserved:
`{m4_run,preserved_before_exit:true,pane_id,pane_pid,pane_start_command_sha256,
 tmux_server:{pid,uid,starttick}}`. Without a real preserved binding, HOLD/wait;
never invent it from absent processes. Existing completion receipts remain
unchanged. Resident observer config: `retriever_observer_path` (dynamic summary),
`retriever_observer_identity:{pid,uid,starttick}`, and pinned
`retriever_observer_source_sha256`. These are actual root-observed identities.

```sh
# Root only after reviewed publication/config, zero real invocation by author:
E1_RUN_ID=<config-run-id> python3 -B controller.py --config CONFIG --config-sha256 SHA
```

CPU-only final fixture count is34, with actual immutable native comparator and
fresh observer/PID reuse/foreign/stale, preserved pane real-status0, missing
terminal refusal and exact unordered orphan identity filtering. Actual12-source streaming hashes reject changed-content fixtures. No real models,
Torch, GPU, HTTP or actual preparation/evaluation were executed by author.

## Final independent review fixes

M4 helper source MUST be the actual published V2, not V1. The exact same V2
owned_process_audit scanner is invoked with known UID/PID/startticks, run ID,
execution/output roots, independent CPU exclusion proofs and ancestry anchors.
Every complete raw audit is preserved. M3's actual live controller carries its
own E1_RUN_ID; only its exact live identity's run-match blocker is separately
classified expected-still-running. It is never falsely claimed exited, while
all other owned/opaque/zombie/UID-changed/uncertain known processes still block.
There is no fallback to an environ-only/list scanner.

Frozen M2 region math originally equates teacher presence with successful-reference
label. Changed-reference probes deliberately keep an honest counterfactual label.
The independent numeric_regions AST overlay changes ONLY the has_teacher predicate
to explicit teacher presence; every mask/finite/clip/scalar/region formula remains
frozen. The original summary/status stays unchanged, overlay records its scope,
and success is never inferred. Actual frozen region fixtures exercise this case.

Live renewed lease/candidate/observer checks occur before every actual generate,
forcing/probe, direct reference probe, zero-B/control forward/generate, GPU model
load and adapter load. One in-flight call returns naturally; its next GPU call
cannot bypass a failed guard merely because no search was generated. Completed
generate/probe progress is recorded independently of final case completion.
Spawn failure and monitor failure both latch HOLD, revoke own worker leases,
never renew any partial admitted work, and wait started children normally. Actual
or uncertain owned descendants/own CUDA allocation keep this controller's locks
held until release; independent opaque blockers remain failures and can never
be silently treated as foreign/exited. No signal or retry.

Config also carries `independent_cpu_exclusion_proofs` SHA/path records as in V2;
absence cannot be replaced with assumed empty environment. `m4_helper_source`
binds the actual V2 source directory. Optional preserved `pane_process` identity
is passed as known-owned along with controllers and all six phase children.
Kernel EUID is read from /proc/status; /proc directory ownership is not treated
as process UID. 34 CPU fixtures include real V2 audit/self/owned/Z/opaque/PIDreuse,
actual frozen region changed-ref, no-search expired guard, second-Popen failure
with zero renew and wait, and kernel UID fixtures. No GPU/HTTP/real model was run.

Release-read uncertainty is explicit: proc/kernel/scanner/nvidia-smi read errors
write UNCERTAIN and keep this controller's locks held through segmented CPU retries.
Only actual known identities absent/reused plus ownCUDA empty and solely unrelated
opaque blockers may produce HOLD and release this controller's locks. No read
exception is swallowed as release; disk-proof-write failure also cannot grant it.
